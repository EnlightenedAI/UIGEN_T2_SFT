# 01_data_profiling.py
"""
UIGEN-T2 数据集画像脚本

修改记录：
  (1) 横纵轴范围根据数据实际分布自动确定（全量展示，不截断异常长尾）
  (2) Token 数使用实际 Tokenizer 真实计算（支持环境变量 TOKENIZER_PATH）
  (3) 质量评估阈值手动设置（THRESHOLDS 集中管理）
  (4) reasoning 只设下限，不设上限（长 reasoning = 高质量思维链）
  (5) total_tok_max = 8162（8192 - 30 模板开销）
  (6) 近重复检测：MinHash+LSH + sentence-transformers
      语义去重修复：使用 FAISS 精确暴力搜索，完美处理对角线并保证 O(N^2) 精确计算
  (7) 相似度分布探查 + n-gram 敏感性分析
  (8) 低质量特征多维度识别
  (9) 修复分位数展示，补充 P5、P95
  (10) 升级相似度探查可视化（KDE曲线 + 警戒遮罩 + 高亮标签）
  (11) 修复 ylim 获取时机问题（在所有绘图元素添加完毕后获取）
  (12) 删除重复的 import seaborn
  (13) 新增全量相似度分布统计：每条记录的最大相似度分位数分析 + 阈值敏感性扫描
       MinHash 精确计算每个候选的 Jaccard；语义去重收集 max_cosine_per_record
       新增 Figure 5 全量相似度分布可视化；将统计结果保存到 profile_data
"""

import json
import re
import hashlib
import warnings
import random as _random
warnings.filterwarnings("ignore")

from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("data",    exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi":  150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size": 10,
})
PALETTE = sns.color_palette("husl", 8)
FIELDS  = ["prompt", "reasoning", "response"]

# ─────────────────────────────────────────────────────────────────────────────
# 手动阈值配置
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "response_tok_min":  1200,
    "response_tok_max":  4100,
    "reasoning_tok_min": 800,
    "reasoning_tok_max": None,
    "total_tok_max":     8162,
}

TEMPLATE_OVERHEAD = 30

# ─────────────────────────────────────────────────────────────────────────────
# 近重复检测配置
# ─────────────────────────────────────────────────────────────────────────────
MINHASH_THRESHOLD      = 0.4
MINHASH_NUM_PERM       = 128
MINHASH_NGRAM_PROMPT   = 4
MINHASH_NGRAM_RESPONSE = 6

SEMANTIC_THRESHOLD     = 0.99
SEMANTIC_BATCH_SIZE    = 256
SBERT_MODEL_PATH       = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SIMILARITY_PROBE_SIZE  = 2000
SIMILARITY_PROBE_PAIRS = 5000
SEED                   = 42

_random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────
def safe_hist(ax, data: np.ndarray, bins: int, color, alpha: float,
              force_xmin: float = 0.0, force_xmax: float = None,
              x_hi_pct: float = 100.0):
    """
    绘制直方图并返回实际使用的 (xmin, xmax)。
    x_hi_pct=100.0 表示展示全量数据不截断。
    注意：调用方在此函数返回后再调用 ax.get_ylim() 才能得到正确值。
    """
    xmin = force_xmin
    xmax = (float(np.percentile(data, x_hi_pct))
            if force_xmax is None else force_xmax)
    if xmax <= xmin:
        xmax = xmin + 1
    ax.hist(data, bins=bins, range=(xmin, xmax),
            color=color, alpha=alpha, edgecolor="none")
    ax.set_xlim(xmin, xmax)
    return xmin, xmax


def get_ylim_top(ax) -> float:
    """
    安全获取当前 ax 的 y 轴上限。
    matplotlib 在添加完所有绘图元素后才会更新 autoscale，
    此函数强制触发 autoscale 再读取，避免读到默认值 1.0。
    """
    ax.autoscale(axis='y')
    return ax.get_ylim()[1]


def quantile_stats(arr) -> Dict:
    a = np.array(arr, dtype=np.float64)
    if len(a) == 0:
        return {}
    return {
        "mean":   float(np.mean(a)),
        "median": float(np.median(a)),
        "std":    float(np.std(a)),
        "min":    float(np.min(a)),
        "p1":     float(np.percentile(a,  1)),
        "p5":     float(np.percentile(a,  5)),
        "p10":    float(np.percentile(a, 10)),
        "p25":    float(np.percentile(a, 25)),
        "p75":    float(np.percentile(a, 75)),
        "p90":    float(np.percentile(a, 90)),
        "p95":    float(np.percentile(a, 95)),
        "p99":    float(np.percentile(a, 99)),
        "max":    float(np.max(a)),
    }


def print_stats_table(stats_dict: Dict[str, Dict], title: str):
    print(f"\n  ── {title} ──")
    print(f"  {'Field':12s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} "
          f"{'P5':>8s} {'P10':>8s} {'P25':>8s} {'P75':>8s} {'P90':>8s} "
          f"{'P95':>8s} {'P99':>8s} {'Max':>8s}")
    print("  " + "-" * 110)
    for f, s in stats_dict.items():
        print(f"  {f:12s} "
              f"{s['mean']:>8.0f} {s['median']:>8.0f} {s['std']:>8.0f} "
              f"{s['p5']:>8.0f} {s['p10']:>8.0f} {s['p25']:>8.0f} "
              f"{s['p75']:>8.0f} {s['p90']:>8.0f} {s['p95']:>8.0f} "
              f"{s['p99']:>8.0f} {s['max']:>8.0f}")


def print_similarity_stats(arr: np.ndarray, label: str,
                            threshold: float, field: str = ""):
    """
    打印相似度数组的完整分位数表，帮助选择过滤阈值。
    arr 中 -1 表示"找不到候选/对角线"，统计时排除。
    """
    valid = arr[arr >= 0]
    if len(valid) == 0:
        print(f"    {label}: no valid data")
        return

    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 100]
    vals        = {p: float(np.percentile(valid, p)) for p in percentiles}

    print(f"\n    ── {label} (n={len(valid):,}, field={field or 'N/A'}) ──")
    print(f"    {'Stat':12s}  {'Value':>10s}    Threshold={threshold}")
    print(f"    {'Mean':12s}  {float(np.mean(valid)):>10.4f}")
    print(f"    {'Std':12s}  {float(np.std(valid)):>10.4f}")
    print(f"    {'-' * 40}")
    for p in percentiles:
        flag   = "***" if vals[p] >= threshold else "   "
        marker = " ← ~threshold" if abs(vals[p] - threshold) < 0.02 else ""
        print(f"    P{p:<9.1f}  {vals[p]:>10.4f}  {flag}{marker}")

    print(f"\n    超过阈值 {threshold} 的记录数（阈值敏感性扫描）:")
    scan_thresholds = sorted(set([
        round(max(min(threshold + delta, 1.0), 0.0), 2)
        for delta in [-0.10, -0.05, 0.0, +0.05, +0.10]
    ]))
    for thr in scan_thresholds:
        n   = int((valid >= thr).sum())
        bar = "█" * min(int(n / len(valid) * 50), 50)
        marker = " ← current" if abs(thr - threshold) < 0.001 else ""
        print(f"      ≥ {thr:.2f}: {n:>7,} / {len(valid):,} "
              f"({n / len(valid) * 100:5.2f}%)  {bar}{marker}")


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def count_duplicates(hashes: List[str]) -> Tuple[int, int]:
    cnt = Counter(hashes)
    return (sum(1 for v in cnt.values() if v > 1),
            sum(v - 1 for v in cnt.values() if v > 1))


def char_ngrams(text: str, n: int) -> Set[str]:
    text = text.lower().strip()
    if len(text) < n:
        return {text} if text else {"<empty>"}
    return {text[i: i + n] for i in range(len(text) - n + 1)}


def has_html_structure(text: str) -> bool:
    return bool(
        re.search(r"<html[\s>]",       text, re.I) or
        re.search(r"<!DOCTYPE\s+html", text, re.I) or
        re.search(r"<body[\s>]",       text, re.I)
    )


def has_valid_html_tags(text: str) -> bool:
    tags = re.findall(r"<([a-zA-Z][a-zA-Z0-9]*)", text)
    return len(set(tags)) >= 3


def has_unclosed_code_block(text: str) -> bool:
    return text.count("```") % 2 != 0


def reasoning_is_repetitive(text: str, window: int = 50,
                             rep_ratio: float = 0.4) -> bool:
    if len(text) < window * 3:
        return False
    chunks = [text[i: i + window]
              for i in range(0, len(text) - window, window)]
    if len(chunks) < 3:
        return False
    rep_count = 0
    for a, b in zip(chunks[:-1], chunks[1:]):
        sa, sb = set(a), set(b)
        if not (sa | sb):
            continue
        if len(sa & sb) / len(sa | sb) > 0.8:
            rep_count += 1
    return (rep_count / (len(chunks) - 1)) > rep_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Step 1：加载 tokenizer
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading tokenizer ...")
print("=" * 60)

TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH, trust_remote_code=True,
)
print(f"  Tokenizer : {TOKENIZER_PATH}")
print(f"  Vocab size: {tokenizer.vocab_size:,}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2：加载数据集
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 2: Loading dataset ...")
ds      = load_dataset("Tesslate/UIGEN-T2", split="train")
records = list(ds)
N       = len(records)
print(f"Total records: {N:,}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3：字段完整性
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Field completeness check ...")
completeness = {}
for f in FIELDS:
    missing = sum(
        1 for r in records if not r.get(f) or str(r[f]).strip() == ""
    )
    completeness[f] = {
        "total": N, "missing": missing,
        "present": N - missing,
        "present_pct": (N - missing) / N * 100,
    }
    print(f"  {f:12s}: present={N - missing:,}  missing={missing:,}  "
          f"({(N - missing) / N * 100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4：字符长度 & 真实 Token 数
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Computing character lengths and real token counts ...")

BATCH = 512

def batch_tokenize(texts: List[str]) -> List[int]:
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    return [len(ids) for ids in enc["input_ids"]]

char_lens:  Dict[str, List[int]] = {}
token_lens: Dict[str, List[int]] = {}

for f in FIELDS:
    texts = [str(r.get(f) or "") for r in records]
    char_lens[f] = [len(t) for t in texts]
    tok_counts: List[int] = []
    for start in tqdm(range(0, N, BATCH), desc=f"  tokenize [{f}]"):
        tok_counts.extend(batch_tokenize(texts[start: start + BATCH]))
    token_lens[f] = tok_counts

total_token_lens: List[int] = [
    token_lens["prompt"][i] +
    token_lens["reasoning"][i] +
    token_lens["response"][i]
    for i in range(N)
]

total_with_template: List[int] = [
    v + TEMPLATE_OVERHEAD for v in total_token_lens
]

# ─────────────────────────────────────────────────────────────────────────────
# Step 5：统计摘要
# ─────────────────────────────────────────────────────────────────────────────
char_stats        = {f: quantile_stats(char_lens[f])  for f in FIELDS}
token_stats       = {f: quantile_stats(token_lens[f]) for f in FIELDS}
total_token_stats = quantile_stats(total_token_lens)

print_stats_table(char_stats,  "Character-length statistics")
print_stats_table(token_stats,
                  f"Token-length statistics ({TOKENIZER_PATH}, no template)")

print(f"\n  ── Total token statistics (prompt+reasoning+response, no template) ──")
s = total_token_stats
print(f"  Mean={s['mean']:.0f}  Median={s['median']:.0f}  Std={s['std']:.0f}  "
      f"P5={s['p5']:.0f}  P10={s['p10']:.0f}  P25={s['p25']:.0f}  "
      f"P75={s['p75']:.0f}  P90={s['p90']:.0f}  P95={s['p95']:.0f}  "
      f"P99={s['p99']:.0f}  Max={s['max']:.0f}")

MAX_SEQ_LEN    = 8192
EFFECTIVE_MAX  = THRESHOLDS["total_tok_max"]
n_exceed_raw   = sum(1 for v in total_token_lens    if v > EFFECTIVE_MAX)
n_exceed_templ = sum(1 for v in total_with_template if v > MAX_SEQ_LEN)
print(f"\n  Records with bare total_tokens > {EFFECTIVE_MAX}: "
      f"{n_exceed_raw:,}  ({n_exceed_raw / N * 100:.1f}%)")
print(f"  Records with total+template    > {MAX_SEQ_LEN}: "
      f"{n_exceed_templ:,}  ({n_exceed_templ / N * 100:.1f}%)")
print(f"  (template overhead = {TEMPLATE_OVERHEAD} tokens)")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6：阈值配置 & 预估过滤量
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Threshold configuration & expected filter counts ...")
print("\n  Current THRESHOLDS:")
for k, v in THRESHOLDS.items():
    print(f"    {k:25s} = "
          f"{v if v is not None else 'None (no upper limit)'}")

checks = [
    ("response  tok < min",
     token_lens["response"], THRESHOLDS["response_tok_min"], None),
    ("response  tok > max",
     token_lens["response"], None, THRESHOLDS["response_tok_max"]),
    ("reasoning tok < min",
     token_lens["reasoning"], THRESHOLDS["reasoning_tok_min"], None),
    ("total+tmpl tok > 8192",
     total_with_template, None, MAX_SEQ_LEN),
]

print("\n  Expected filter counts:")
for label, arr, lo, hi in checks:
    if lo is not None:
        n = sum(1 for v in arr if v < lo)
        print(f"    {label:35s}: {n:>6,}  ({n / N * 100:.2f}%)")
    elif hi is not None:
        n = sum(1 for v in arr if v > hi)
        print(f"    {label:35s}: {n:>6,}  ({n / N * 100:.2f}%)")

if THRESHOLDS["reasoning_tok_max"] is None:
    print(f"    {'reasoning tok > max':35s}: "
          f"skipped (no upper limit set)")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7：Prompt 主题分析
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Prompt topic analysis ...")

TOPIC_KEYWORDS = {
    "Landing Page":     ["landing page", "hero section", "call to action", "cta"],
    "Dashboard":        ["dashboard", "analytics", "chart", "graph", "metric", "kpi"],
    "E-commerce":       ["product", "cart", "shop", "store", "checkout", "price",
                         "e-commerce", "ecommerce"],
    "Form / Auth":      ["form", "login", "signup", "register", "authentication",
                         "input field", "password"],
    "Portfolio":        ["portfolio", "resume", "cv", "personal website", "showcase"],
    "Blog / Article":   ["blog", "article", "post", "news", "magazine"],
    "Mobile UI":        ["mobile", "app", "ios", "android", "responsive"],
    "Admin Panel":      ["admin", "panel", "settings", "management", "control"],
    "Navigation":       ["navigation", "navbar", "menu", "sidebar", "breadcrumb"],
    "Card / Component": ["card", "component", "widget", "modal", "dialog", "tooltip"],
}

def classify_prompt(text: str) -> str:
    tl = text.lower()
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in tl for kw in kws):
            return topic
    return "Other"

topic_counts: Counter = Counter()
for r in tqdm(records, desc="  topic classify"):
    topic_counts[classify_prompt(r.get("prompt", ""))] += 1

print("\n  Prompt topic distribution:")
for topic, cnt in topic_counts.most_common():
    print(f"    {topic:25s}: {cnt:>6,}  ({cnt / N * 100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Step 8：Response 类型分析
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Response type analysis ...")

def analyze_response_type(text: str) -> Dict[str, bool]:
    t = text or ""
    return {
        "has_html_tag":  bool(re.search(r"<html[\s>]", t, re.I)),
        "has_body_tag":  bool(re.search(r"<body[\s>]", t, re.I)),
        "has_head_tag":  bool(re.search(r"<head[\s>]", t, re.I)),
        "has_js":        bool(re.search(r"<script[\s>]", t, re.I)),
        "has_css_style": bool(re.search(r"<style[\s>]", t, re.I)),
        "has_tailwind":  bool(re.search(r"tailwind", t, re.I)),
        "has_bootstrap": bool(re.search(r"bootstrap", t, re.I)),
        "has_react":     bool(re.search(r"react|jsx|useState|useEffect", t)),
        "has_vue":       bool(re.search(r"v-bind|v-model|vue\.js|createApp", t)),
        "has_svg":       bool(re.search(r"<svg[\s>]", t, re.I)),
        "has_canvas":    bool(re.search(r"<canvas[\s>]", t, re.I)),
    }

resp_type_agg: Dict[str, int] = defaultdict(int)
for r in tqdm(records, desc="  resp type"):
    for k, v in analyze_response_type(r.get("response", "")).items():
        if v:
            resp_type_agg[k] += 1

print("\n  Response feature presence:")
for k, v in sorted(resp_type_agg.items(), key=lambda x: -x[1]):
    print(f"    {k:20s}: {v:>6,}  ({v / N * 100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Step 9：Reasoning 分析
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Reasoning field analysis ...")
reasoning_present = sum(
    1 for r in records if r.get("reasoning", "").strip()
)
print(f"  Reasoning present: {reasoning_present:,} / {N:,} "
      f"({reasoning_present / N * 100:.1f}%)")
corr_tok = np.corrcoef(
    token_lens["reasoning"], token_lens["response"]
)[0, 1]
print(f"  Pearson corr (reasoning_tokens vs response_tokens "
      f"[as Quality Proxy]): {corr_tok:.4f}")

reas_arr     = np.array(token_lens["reasoning"], dtype=float)
reas_nonzero = reas_arr[reas_arr > 0]
print(f"\n  Reasoning token distribution (non-zero, n={len(reas_nonzero):,}):")
print(f"  P5={np.percentile(reas_nonzero, 5):.0f}  "
      f"P10={np.percentile(reas_nonzero,10):.0f}  "
      f"P25={np.percentile(reas_nonzero,25):.0f}  "
      f"P50={np.percentile(reas_nonzero,50):.0f}  "
      f"P75={np.percentile(reas_nonzero,75):.0f}  "
      f"P90={np.percentile(reas_nonzero,90):.0f}  "
      f"P95={np.percentile(reas_nonzero,95):.0f}  "
      f"P99={np.percentile(reas_nonzero,99):.0f}  "
      f"Max={np.max(reas_nonzero):.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 10：重复与近重复检测
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] Duplicate & near-duplicate detection ...")

# ── 10-A：精确重复（MD5）─────────────────────────────────────────────────────
print("\n  [10-A] Exact duplicate detection (MD5) ...")

prompt_hashes   = [md5(r.get("prompt",   "")) for r in records]
response_hashes = [md5(r.get("response", "")) for r in records]

pg, pr = count_duplicates(prompt_hashes)
rg, rr = count_duplicates(response_hashes)
print(f"    Exact dup prompts  : {pr:,} records in {pg:,} groups "
      f"({pr / N * 100:.2f}%)")
print(f"    Exact dup responses: {rr:,} records in {rg:,} groups "
      f"({rr / N * 100:.2f}%)")

hash_counter_prompt  = Counter(prompt_hashes)
hash_counter_resp    = Counter(response_hashes)
exact_dup_prompt_idx = {
    i for i, h in enumerate(prompt_hashes)
    if hash_counter_prompt[h] > 1
}
exact_dup_resp_idx   = {
    i for i, h in enumerate(response_hashes)
    if hash_counter_resp[h] > 1
}

# ── 10-B：相似度分布探查 + n-gram 敏感性分析 ─────────────────────────────────
print(f"\n  [10-B] Similarity distribution probe ...")

probe_indices = _random.sample(range(N), min(SIMILARITY_PROBE_SIZE, N))
probe_prompts = [str(records[i].get("prompt",   "") or "") for i in probe_indices]
probe_resps   = [str(records[i].get("response", "") or "") for i in probe_indices]

_random.seed(SEED)
fixed_pairs = [
    (_random.randint(0, len(probe_indices) - 1),
     _random.randint(0, len(probe_indices) - 1))
    for _ in range(SIMILARITY_PROBE_PAIRS)
]
fixed_pairs = [(a, b) for a, b in fixed_pairs if a != b]

minhash_available        = False
minhash_dup_prompt_count = 0
minhash_dup_resp_count   = 0
minhash_dup_prompt_idx_set: Set[int] = set()
minhash_dup_resp_idx_set:   Set[int] = set()
minhash_jacc_prompt_arr  = np.array([], dtype=np.float32)
minhash_jacc_resp_arr    = np.array([], dtype=np.float32)
jaccard_prompt_arr       = np.array([])
jaccard_resp_arr         = np.array([])

try:
    from datasketch import MinHash, MinHashLSH

    def make_minhash(text: str, num_perm: int, ngram: int) -> "MinHash":
        m = MinHash(num_perm=num_perm)
        for ng in char_ngrams(text, ngram):
            m.update(ng.encode("utf-8"))
        return m

    print(f"\n    n-gram sensitivity (prompt, {len(fixed_pairs)} pairs):")
    print(f"    {'n':>4s}  {'P50':>8s}  {'P90':>8s}  {'P95':>8s}  "
          f"{'P99':>8s}  {'>0.5':>7s}  {'>0.7':>7s}")
    print("    " + "-" * 58)
    for ng in [2, 3, 4, 5, 6, 8]:
        mh_list   = [make_minhash(t, MINHASH_NUM_PERM, ng)
                     for t in probe_prompts]
        jacc_vals = np.array([
            mh_list[a].jaccard(mh_list[b]) for a, b in fixed_pairs
        ])
        marker = " ←" if ng == MINHASH_NGRAM_PROMPT else ""
        print(f"    {ng:>4d}  "
              f"{np.percentile(jacc_vals,50):>8.4f}  "
              f"{np.percentile(jacc_vals,90):>8.4f}  "
              f"{np.percentile(jacc_vals,95):>8.4f}  "
              f"{np.percentile(jacc_vals,99):>8.4f}  "
              f"{(jacc_vals>0.5).mean()*100:>6.1f}%  "
              f"{(jacc_vals>0.7).mean()*100:>6.1f}%{marker}")

    probe_mh_prompt = [
        make_minhash(t, MINHASH_NUM_PERM, MINHASH_NGRAM_PROMPT)
        for t in tqdm(probe_prompts, desc="    probe MinHash prompt")
    ]
    jaccard_prompt_arr = np.array([
        probe_mh_prompt[a].jaccard(probe_mh_prompt[b])
        for a, b in fixed_pairs
    ])

    probe_mh_resp = [
        make_minhash(t, MINHASH_NUM_PERM, MINHASH_NGRAM_RESPONSE)
        for t in tqdm(probe_resps, desc="    probe MinHash response")
    ]
    jaccard_resp_arr = np.array([
        probe_mh_resp[a].jaccard(probe_mh_resp[b])
        for a, b in fixed_pairs
    ])

    minhash_available = True

except ImportError:
    print("    [SKIP] datasketch not installed.")

# 语义相似度探查
semantic_available         = False
semantic_dup_count         = 0
semantic_dup_idx_set:   Set[int]  = set()
semantic_dup_examples: List[Dict] = []
cosine_arr             = np.array([])
max_cosine_per_record  = np.full(N, -1.0, dtype=np.float32)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    import faiss

    sbert = SentenceTransformer(SBERT_MODEL_PATH)

    print(f"\n    Semantic cosine probe ({len(probe_indices)} samples) ...")
    probe_embs  = sbert.encode(
        probe_prompts, batch_size=SEMANTIC_BATCH_SIZE,
        show_progress_bar=True, normalize_embeddings=True,
        convert_to_numpy=True,
    )
    emb_t_probe = torch.tensor(probe_embs)
    cosine_arr  = np.array([
        float(torch.dot(emb_t_probe[a], emb_t_probe[b]).item())
        for a, b in fixed_pairs
    ])

    print(f"\n    Cosine distribution ({len(cosine_arr)} pairs):")
    for thr in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        n      = (cosine_arr > thr).sum()
        marker = " ← current" if abs(thr - SEMANTIC_THRESHOLD) < 0.001 else ""
        print(f"      cosine > {thr:.2f}: {n:>5}  "
              f"({n / len(cosine_arr) * 100:.1f}%){marker}")

    del emb_t_probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    semantic_available = True

except ImportError:
    print("    [SKIP] sentence-transformers or faiss not installed.")

# ── 探查可视化（修复版）──────────────────────────────────────────────────────
probe_plot_data = []   # 收集所有要画的 (label, arr, threshold, ng_info, color)

if len(jaccard_prompt_arr) > 0:
    probe_plot_data.append((
        "MinHash Jaccard\n(prompt)",
        jaccard_prompt_arr,
        MINHASH_THRESHOLD,
        f"n-gram={MINHASH_NGRAM_PROMPT}  pairs={len(jaccard_prompt_arr):,}",
        PALETTE[0],
    ))

if len(jaccard_resp_arr) > 0:
    probe_plot_data.append((
        "MinHash Jaccard\n(response)",
        jaccard_resp_arr,
        MINHASH_THRESHOLD,
        f"n-gram={MINHASH_NGRAM_RESPONSE}  pairs={len(jaccard_resp_arr):,}",
        PALETTE[4],
    ))

if len(cosine_arr) > 0:
    probe_plot_data.append((
        "Semantic Cosine\n(prompt)",
        cosine_arr,
        SEMANTIC_THRESHOLD,
        f"{SBERT_MODEL_PATH.split('/')[-1]}  pairs={len(cosine_arr):,}",
        PALETTE[2],
    ))

# 诊断打印：明确告知哪些数组有数据
print(f"\n    Probe plot data availability:")
print(f"      jaccard_prompt_arr : {len(jaccard_prompt_arr):>6,} pairs  "
      f"{'✓' if len(jaccard_prompt_arr) > 0 else '✗ (datasketch missing?)'}")
print(f"      jaccard_resp_arr   : {len(jaccard_resp_arr):>6,} pairs  "
      f"{'✓' if len(jaccard_resp_arr) > 0 else '✗ (datasketch missing?)'}")
print(f"      cosine_arr         : {len(cosine_arr):>6,} pairs  "
      f"{'✓' if len(cosine_arr) > 0 else '✗ (sentence-transformers missing?)'}")

if len(probe_plot_data) == 0:
    print("\n    [SKIP] No similarity probe data available.")
    print("           请安装: pip install datasketch sentence-transformers faiss-cpu")
    print("           fig5_similarity_probe_upgraded.png 不会生成")
else:
    n_cols  = len(probe_plot_data)
    fig_p, axes_p = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5.5))
    if n_cols == 1:
        axes_p = [axes_p]

    fig_p.suptitle(
        "UIGEN-T2 Similarity Distribution & Filter Thresholds (Probe Sample)",
        fontsize=14, fontweight="bold", y=1.05,
    )

    for ax, (label, arr, threshold, ng_info, color) in zip(axes_p, probe_plot_data):
        plot_min = max(float(arr.min()) - 0.05, 0.0)

        sns.histplot(arr, bins=80, color=color, alpha=0.6,
                     kde=True, line_kws={"linewidth": 2},
                     ax=ax, binrange=(plot_min, 1.0))
        ax.set_xlim(plot_min, 1.05)

        ax.axvline(threshold, color="red", ls="--", lw=2.5,
                   label=f"Filter Threshold: {threshold}")
        ax.axvspan(threshold, 1.05, color="red", alpha=0.1,
                   label="Filtered Region")

        for pct, ls in [(90, ":"), (95, "-."), (99, "--")]:
            val = float(np.percentile(arr, pct))
            ax.axvline(val, color="grey", ls=ls, lw=1.2,
                       label=f"P{pct}={val:.3f}")

        ax.set_title(f"{label}\n({ng_info})", fontsize=11)
        ax.set_xlabel("Similarity Score", fontsize=11)
        ax.set_ylabel("Pair Count", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")

        ylim_top  = get_ylim_top(ax)
        n_right   = (arr > threshold).sum()
        pct_right = n_right / len(arr) * 100
        ax.text(
            min(threshold + 0.02, 0.98),
            ylim_top * 0.85,
            f"Filtered:\n{n_right} pairs\n({pct_right:.1f}%)",
            color="darkred", fontsize=10, fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round,pad=0.4",
                      fc="white", ec="red", lw=1.5, alpha=0.95),
        )

    plt.tight_layout()
    plt.savefig("figures/fig5_similarity_probe_upgraded.png",
                bbox_inches="tight")
    plt.close()
    print("\n    Saved: figures/fig5_similarity_probe_upgraded.png")

# ── 10-C：全量近重复检测 ──────────────────────────────────────────────────────
print(f"\n  [10-C] Full-dataset near-duplicate detection ...")
if minhash_available:
    prompt_texts   = [str(r.get("prompt",   "") or "") for r in records]
    response_texts = [str(r.get("response", "") or "") for r in records]

    def build_lsh_near_dups(
        texts: List[str], field_name: str,
        threshold: float, num_perm: int, ngram: int,
    ) -> Tuple[int, Set[int], np.ndarray]:
        """
        返回 (dup_count, dup_idx_set, max_jaccard_per_record)

        max_jaccard_per_record[i]:
          - 若记录 i 先于所有候选插入（即它是第一个），值为 0.0
          - 若 LSH 找到候选，对所有候选精确计算 Jaccard，取最大值
          - 未找到候选且已插入 LSH 的记录，值为 0.0
        """
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        minhashes_all: List["MinHash"] = []

        for text in tqdm(texts,
                         desc=f"    MinHash [{field_name}] n={ngram}"):
            m = MinHash(num_perm=num_perm)
            for ng in char_ngrams(text, ngram):
                m.update(ng.encode("utf-8"))
            minhashes_all.append(m)

        dup_idx: Set[int] = set()
        max_jacc_per_record = np.zeros(len(texts), dtype=np.float32)

        for i, m in enumerate(tqdm(
            minhashes_all, desc=f"    LSH [{field_name}]"
        )):
            candidates = lsh.query(m)
            if candidates:
                # 对所有候选精确计算 Jaccard，取最大值
                best_jacc = 0.0
                for cand_key in candidates:
                    cand_idx = int(cand_key.split("_")[1])
                    jacc     = m.jaccard(minhashes_all[cand_idx])
                    if jacc > best_jacc:
                        best_jacc = jacc
                max_jacc_per_record[i] = best_jacc
                if best_jacc >= threshold:
                    dup_idx.add(i)
            else:
                lsh.insert(f"id_{i}", m)

        return len(dup_idx), dup_idx, max_jacc_per_record

    minhash_dup_prompt_count, minhash_dup_prompt_idx_set, minhash_jacc_prompt_arr = \
        build_lsh_near_dups(
            prompt_texts, "prompt",
            MINHASH_THRESHOLD, MINHASH_NUM_PERM, MINHASH_NGRAM_PROMPT,
        )
    minhash_dup_resp_count, minhash_dup_resp_idx_set, minhash_jacc_resp_arr = \
        build_lsh_near_dups(
            response_texts, "response",
            MINHASH_THRESHOLD, MINHASH_NUM_PERM, MINHASH_NGRAM_RESPONSE,
        )

    print(f"    Near-dup prompts   (Jac≥{MINHASH_THRESHOLD}): "
          f"{minhash_dup_prompt_count:,}  "
          f"({minhash_dup_prompt_count / N * 100:.2f}%)")
    print(f"    Near-dup responses (Jac≥{MINHASH_THRESHOLD}): "
          f"{minhash_dup_resp_count:,}  "
          f"({minhash_dup_resp_count / N * 100:.2f}%)")

# ── 语义近重复（全量，基于 FAISS 的精确暴力搜索）────────────────────────────
if semantic_available:
    from sentence_transformers import SentenceTransformer
    import faiss

    sbert        = SentenceTransformer(SBERT_MODEL_PATH)
    prompt_texts = [str(r.get("prompt", "") or "") for r in records]

    print(f"\n    Encoding all {N:,} prompts ...")
    # 注意：normalize_embeddings=True 必须保留，这是使用 Inner Product 计算 Cosine 的前提
    embeddings = sbert.encode(
        prompt_texts, batch_size=SEMANTIC_BATCH_SIZE,
        show_progress_bar=True, normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print(f"\n    Building FAISS IndexFlatIP (Exact Brute-force) ...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # 暴力内积计算

    # 如果有 GPU 且想进一步加速，可以解开下方注释：
    # res = faiss.StandardGpuResources()
    # index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(embeddings)

    print(f"    Pairwise cosine search (threshold={SEMANTIC_THRESHOLD}) ...")
    # k=2: 每个向量最近的邻居是它自己（相似度为1.0，排第0位），真正的近邻在第1位
    D, I = index.search(embeddings, k=2)

    # 提取与其他记录的最大余弦相似度及对应索引
    max_cosine_per_record[:] = D[:, 1]
    nearest_neighbors_idx = I[:, 1]

    # 收集去重结果
    for i in tqdm(range(N), desc="    collecting semantic dups"):
        max_sim = max_cosine_per_record[i]
        neighbor_idx = nearest_neighbors_idx[i]

        if max_sim >= SEMANTIC_THRESHOLD:
            semantic_dup_idx_set.add(i)

            # 收集示例 (保证不重复收集 a->b 和 b->a)
            if len(semantic_dup_examples) < 50 and i > neighbor_idx:
                semantic_dup_examples.append({
                    "idx_a":    int(neighbor_idx),
                    "idx_b":    int(i),
                    "cosine":   round(float(max_sim), 4),
                    "prompt_a": prompt_texts[neighbor_idx][:120],
                    "prompt_b": prompt_texts[i][:120],
                })

    semantic_dup_count = len(semantic_dup_idx_set)
    print(f"    Semantic near-dup (cos≥{SEMANTIC_THRESHOLD}): "
          f"{semantic_dup_count:,}  "
          f"({semantic_dup_count / N * 100:.2f}%)")

    if semantic_dup_examples:
        print(f"\n    Top 5 similar pairs:")
        for ex in sorted(semantic_dup_examples[:5],
                         key=lambda x: -x["cosine"]):
            pa = ex["prompt_a"].replace("\n", " ")[:45]
            pb = ex["prompt_b"].replace("\n", " ")[:45]
            print(f"    {ex['cosine']:.4f}  {pa}  |  {pb}")

    del embeddings

# ── 10-D：汇总 ───────────────────────────────────────────────────────────────
print("\n  [10-D] Deduplication summary:")
all_dup_idx = (exact_dup_prompt_idx | exact_dup_resp_idx |
               minhash_dup_prompt_idx_set | minhash_dup_resp_idx_set |
               semantic_dup_idx_set)

rows = [
    ("Exact dup prompt (MD5)",   pr, pr / N * 100),
    ("Exact dup response (MD5)", rr, rr / N * 100),
]
if minhash_available:
    rows += [
        (f"MinHash near-dup prompt  (Jac≥{MINHASH_THRESHOLD})",
         minhash_dup_prompt_count, minhash_dup_prompt_count / N * 100),
        (f"MinHash near-dup response(Jac≥{MINHASH_THRESHOLD})",
         minhash_dup_resp_count,   minhash_dup_resp_count   / N * 100),
    ]
if semantic_available:
    rows += [
        (f"Semantic near-dup (cos≥{SEMANTIC_THRESHOLD})",
         semantic_dup_count, semantic_dup_count / N * 100),
    ]
rows += [("Union (all methods)",
          len(all_dup_idx), len(all_dup_idx) / N * 100)]

for label, cnt, pct in rows:
    print(f"    {label:55s}: {cnt:>6,}  ({pct:.2f}%)")

# ── 10-E：全量相似度分布分位数统计 ───────────────────────────────────────────
print("\n  [10-E] Similarity score distribution (full dataset) ...")
print("         (每条记录与其他记录的最大相似度分位数分析)\n")

if minhash_available:
    print_similarity_stats(
        minhash_jacc_prompt_arr,
        label="MinHash Jaccard – Prompt",
        threshold=MINHASH_THRESHOLD,
        field="prompt",
    )
    print_similarity_stats(
        minhash_jacc_resp_arr,
        label="MinHash Jaccard – Response",
        threshold=MINHASH_THRESHOLD,
        field="response",
    )

if semantic_available:
    print_similarity_stats(
        max_cosine_per_record,
        label="Semantic Cosine – Prompt",
        threshold=SEMANTIC_THRESHOLD,
        field="prompt",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Step 11：低质量特征识别
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] Low-quality feature identification ...")

quality_flags = []
for i, r in enumerate(tqdm(records, desc="  quality flags")):
    resp_tok  = token_lens["response"][i]
    reas_tok  = token_lens["reasoning"][i]
    resp_text = r.get("response",  "") or ""
    reas_text = r.get("reasoning", "") or ""

    flags: Dict[str, bool] = {}

    flags["response_tok_short"]  = resp_tok < THRESHOLDS["response_tok_min"]
    flags["response_tok_long"]   = resp_tok > THRESHOLDS["response_tok_max"]
    flags["reasoning_tok_short"] = reas_tok < THRESHOLDS["reasoning_tok_min"]

    flags["no_html_structure"]   = not has_html_structure(resp_text)
    flags["too_few_html_tags"]   = not has_valid_html_tags(resp_text)
    flags["unclosed_code_block"] = has_unclosed_code_block(resp_text)

    flags["reasoning_empty"]      = reas_tok == 0
    flags["reasoning_repetitive"] = reasoning_is_repetitive(reas_text)

    flags["exact_dup_prompt"]   = i in exact_dup_prompt_idx
    flags["exact_dup_response"] = i in exact_dup_resp_idx
    flags["minhash_near_dup"]   = (i in minhash_dup_prompt_idx_set or
                                   i in minhash_dup_resp_idx_set)
    flags["semantic_near_dup"]  = i in semantic_dup_idx_set

    flags["exceed_seq_len"] = total_with_template[i] > MAX_SEQ_LEN

    flags["any_issue"] = any(flags.values())
    quality_flags.append(flags)

flag_summary = {
    k: sum(f[k] for f in quality_flags) for k in quality_flags[0]
}

print("\n  Quality flag summary:")
groups = [
    ("── Token length", [
        "response_tok_short", "response_tok_long", "reasoning_tok_short",
    ]),
    ("── HTML quality", [
        "no_html_structure", "too_few_html_tags", "unclosed_code_block",
    ]),
    ("── Reasoning quality", [
        "reasoning_empty", "reasoning_repetitive",
    ]),
    ("── Duplicates", [
        "exact_dup_prompt", "exact_dup_response",
        "minhash_near_dup", "semantic_near_dup",
    ]),
    ("── Sequence length", ["exceed_seq_len"]),
    ("── Summary",         ["any_issue"]),
]
for group_name, keys in groups:
    print(f"\n  {group_name}")
    for k in keys:
        v = flag_summary.get(k, 0)
        print(f"    {k:33s}: {v:>6,}  ({v / N * 100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Step 12：可视化
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] Generating figures ...")

field_colors = {
    "prompt":    PALETTE[0],
    "reasoning": PALETTE[2],
    "response":  PALETTE[4],
}

# ── Figure 1：字符长度 & Token 数 ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle(
    f"UIGEN-T2 Field Length Distributions\n"
    f"Token counts: {TOKENIZER_PATH} (no template)",
    fontsize=13, fontweight="bold", y=1.02,
)
for row, f in enumerate(FIELDS):
    char_arr  = np.array(char_lens[f],  dtype=float)
    token_arr = np.array(token_lens[f], dtype=float)

    ax = axes[row][0]
    _, xmax_c = safe_hist(ax, char_arr, bins=80,
                          color=field_colors[f], alpha=0.85)
    ax.axvline(char_stats[f]["median"], color="red",  ls="--", lw=1.8,
               label=f"Median ({char_stats[f]['median']:.0f})")
    ax.axvline(char_stats[f]["p10"],    color="grey", ls=":",  lw=1.3,
               label=f"P10 ({char_stats[f]['p10']:.0f})")
    ax.axvline(char_stats[f]["p90"],    color="grey", ls=":",  lw=1.3,
               label=f"P90 ({char_stats[f]['p90']:.0f})")
    ax.set_title(f"{f} – Character Length")
    ax.set_xlabel(f"Characters  [0 ~ Max={xmax_c:.0f}]")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    ax = axes[row][1]
    _, xmax_t = safe_hist(ax, token_arr, bins=80,
                          color=field_colors[f], alpha=0.6)
    ax.axvline(token_stats[f]["median"], color="red",  ls="--", lw=1.8,
               label=f"Median ({token_stats[f]['median']:.0f})")
    ax.axvline(token_stats[f]["p10"],    color="grey", ls=":",  lw=1.3,
               label=f"P10 ({token_stats[f]['p10']:.0f})")
    ax.axvline(token_stats[f]["p90"],    color="grey", ls=":",  lw=1.3,
               label=f"P90 ({token_stats[f]['p90']:.0f})")
    ax.set_title(f"{f} – Token Length")
    ax.set_xlabel(f"Tokens  [0 ~ Max={xmax_t:.0f}]")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig1_length_distribution.png", bbox_inches="tight")
plt.close()
print("  Saved: figures/fig1_length_distribution.png")

# ── Figure 1b：total_tokens 分布 ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
total_arr  = np.array(total_token_lens, dtype=float)
xmax_total = max(float(np.max(total_arr)), MAX_SEQ_LEN * 1.05)
safe_hist(ax, total_arr, bins=100, color=PALETTE[6], alpha=0.8,
          force_xmin=0.0, force_xmax=xmax_total)
ax.axvline(EFFECTIVE_MAX, color="orange", ls="--", lw=2,
           label=f"total_tok_max={EFFECTIVE_MAX} "
                 f"(8192-{TEMPLATE_OVERHEAD})")
ax.axvline(MAX_SEQ_LEN, color="red", ls="-.", lw=1.5,
           label=f"MAX_SEQ_LEN={MAX_SEQ_LEN}")
ax.axvline(float(np.median(total_arr)), color="green", ls="-", lw=1.8,
           label=f"Median ({np.median(total_arr):.0f})")
ax.set_title("Total Token Length Distribution (bare, no template)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Total Tokens")
ax.set_ylabel("Count")
ax.legend(fontsize=9)
ylim_top_1b = get_ylim_top(ax)
ax.text(EFFECTIVE_MAX * 1.01, ylim_top_1b * 0.8,
        f"{n_exceed_raw:,}\n({n_exceed_raw / N * 100:.1f}%)\nexceed filter",
        color="orange", fontsize=9, va="top")
plt.tight_layout()
plt.savefig("figures/fig1b_total_token_distribution.png",
            bbox_inches="tight")
plt.close()
print("  Saved: figures/fig1b_total_token_distribution.png")

# ── Figure 2：主题分布 & Response 特征 ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("UIGEN-T2 Content Analysis", fontsize=14)
topics = [t for t, _ in topic_counts.most_common()]
counts = [topic_counts[t] for t in topics]
colors = sns.color_palette("tab10", len(topics))
bars   = axes[0].barh(topics[::-1], counts[::-1], color=colors[::-1])
axes[0].set_title("Prompt Topic Distribution")
axes[0].set_xlabel("Count")
for bar, cnt in zip(bars, counts[::-1]):
    axes[0].text(
        bar.get_width() + N * 0.003,
        bar.get_y() + bar.get_height() / 2,
        f"{cnt:,}  ({cnt / N * 100:.1f}%)",
        va="center", fontsize=8,
    )
feat_labels = list(resp_type_agg.keys())
feat_vals   = [resp_type_agg[k] for k in feat_labels]
sorted_idx  = np.argsort(feat_vals)
axes[1].barh(
    [feat_labels[i] for i in sorted_idx],
    [feat_vals[i]   for i in sorted_idx],
    color=PALETTE[3], alpha=0.8,
)
axes[1].set_title("Response Feature Presence")
axes[1].set_xlabel("Count")
for i, idx in enumerate(sorted_idx):
    axes[1].text(
        feat_vals[idx] + N * 0.003, i,
        f"{feat_vals[idx] / N * 100:.1f}%",
        va="center", fontsize=8,
    )
plt.tight_layout()
plt.savefig("figures/fig2_topic_distribution.png", bbox_inches="tight")
plt.close()
print("  Saved: figures/fig2_topic_distribution.png")

# ── Figure 3：质量指标 ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("UIGEN-T2 Quality Indicators",
             fontsize=14, fontweight="bold", y=1.01)
ax_flags = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax_reas  = plt.subplot2grid((2, 2), (1, 0))
ax_resp  = plt.subplot2grid((2, 2), (1, 1))
plt.subplots_adjust(hspace=0.45, wspace=0.3)

flag_keys = [k for k in flag_summary if k != "any_issue"]
flag_pcts = [flag_summary[k] / N * 100 for k in flag_keys]
bar_colors_q = [
    "#e74c3c" if p > 5 else "#f39c12" if p > 1 else "#2ecc71"
    for p in flag_pcts
]
bars = ax_flags.bar(flag_keys, flag_pcts,
                    color=bar_colors_q, alpha=0.88, width=0.6)
ax_flags.set_title("Quality Flag Rates (% of total)", fontsize=12)
ax_flags.set_ylabel("Percentage (%)")
ax_flags.set_ylim(0, max(flag_pcts) * 1.35 if flag_pcts else 1)
ax_flags.set_xticks(range(len(flag_keys)))
ax_flags.set_xticklabels(flag_keys, rotation=25, ha="right")
for bar, key, pct in zip(bars, flag_keys, flag_pcts):
    ax_flags.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(flag_pcts) * 0.02,
        f"{flag_summary[key]:,}\n({pct:.1f}%)",
        ha="center", va="bottom", fontsize=8, fontweight="bold",
    )
from matplotlib.patches import Patch
ax_flags.legend(
    handles=[
        Patch(facecolor="#e74c3c", alpha=0.88, label=">5%  High"),
        Patch(facecolor="#f39c12", alpha=0.88, label="1-5% Medium"),
        Patch(facecolor="#2ecc71", alpha=0.88, label="<1%  Low"),
    ],
    loc="upper right", fontsize=9,
)

reas_tok_arr    = np.array(token_lens["reasoning"], dtype=float)
thresh_reas_min = THRESHOLDS["reasoning_tok_min"]
reas_xmax       = float(np.max(reas_tok_arr))
safe_hist(ax_reas, reas_tok_arr, bins=100, color=PALETTE[2], alpha=0.85,
          force_xmin=0.0, force_xmax=reas_xmax)
ax_reas.axvline(thresh_reas_min, color="red", ls="--", lw=2,
                label=f"min={thresh_reas_min} tok")
ax_reas.set_title("Reasoning Token Distribution\n(no upper limit)",
                  fontsize=11)
ax_reas.set_xlabel(f"Tokens  [0 ~ Max={reas_xmax:.0f}]")
ax_reas.set_ylabel("Count")
ax_reas.legend(fontsize=9)
reas_ylim_top   = get_ylim_top(ax_reas)
n_below_reas    = sum(1 for v in reas_tok_arr if v < thresh_reas_min)
ax_reas.text(
    thresh_reas_min * 1.03, reas_ylim_top * 0.85,
    f"← below:\n{n_below_reas} ({n_below_reas / N * 100:.1f}%)",
    color="red", fontsize=9,
)

resp_tok_arr    = np.array(token_lens["response"], dtype=float)
thresh_resp_min = THRESHOLDS["response_tok_min"]
thresh_resp_max = THRESHOLDS["response_tok_max"]
resp_xmax       = max(float(np.max(resp_tok_arr)), thresh_resp_max * 1.05)
safe_hist(ax_resp, resp_tok_arr, bins=100, color=PALETTE[4], alpha=0.75,
          force_xmin=0.0, force_xmax=resp_xmax)
ax_resp.axvline(thresh_resp_min, color="#e74c3c", ls="--", lw=2,
                label=f"min={thresh_resp_min} tok")
ax_resp.axvline(thresh_resp_max, color="#f39c12", ls="--", lw=2,
                label=f"max={thresh_resp_max} tok")
ax_resp.axvline(float(np.median(resp_tok_arr)), color="#27ae60",
                ls="-", lw=1.8,
                label=f"Median={np.median(resp_tok_arr):.0f}")
ax_resp.set_title("Response Token Distribution", fontsize=11)
ax_resp.set_xlabel(f"Tokens  [0 ~ {resp_xmax:.0f}]")
ax_resp.set_ylabel("Count")
ax_resp.legend(fontsize=9, loc="upper right")
resp_ylim_top = get_ylim_top(ax_resp)
n_below_resp  = sum(1 for v in resp_tok_arr if v < thresh_resp_min)
n_above_resp  = sum(1 for v in resp_tok_arr if v > thresh_resp_max)
ax_resp.text(
    thresh_resp_min + resp_xmax * 0.01, resp_ylim_top * 0.75,
    f"below:\n{n_below_resp} ({n_below_resp / N * 100:.1f}%)",
    color="#e74c3c", fontsize=8, va="top",
)
ax_resp.text(
    thresh_resp_max - resp_xmax * 0.01, resp_ylim_top * 0.75,
    f"above:\n{n_above_resp} ({n_above_resp / N * 100:.1f}%)",
    color="#f39c12", fontsize=8, va="top", ha="right",
)

plt.savefig("figures/fig3_quality_indicators.png", bbox_inches="tight")
plt.close()
print("  Saved: figures/fig3_quality_indicators.png")

# ── Figure 4：重复分析 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("UIGEN-T2 Duplicate Analysis",
             fontsize=13, fontweight="bold")

method_labels = ["Exact\n(MD5)"]
prompt_cnts   = [pr]
resp_cnts     = [rr]
if minhash_available:
    method_labels.append(f"MinHash\n(Jac≥{MINHASH_THRESHOLD})")
    prompt_cnts.append(minhash_dup_prompt_count)
    resp_cnts.append(minhash_dup_resp_count)
if semantic_available:
    method_labels.append(f"Semantic\n(cos≥{SEMANTIC_THRESHOLD})")
    prompt_cnts.append(semantic_dup_count)
    resp_cnts.append(0)

x     = np.arange(len(method_labels))
width = 0.35
bars1 = axes[0].bar(x - width / 2, prompt_cnts, width,
                    label="Prompt",   color=PALETTE[0], alpha=0.85)
bars2 = axes[0].bar(x + width / 2, resp_cnts,   width,
                    label="Response", color=PALETTE[4], alpha=0.85)
axes[0].set_title("Duplicate Counts by Method")
axes[0].set_ylabel("Record Count")
axes[0].set_xticks(x)
axes[0].set_xticklabels(method_labels)
axes[0].legend(fontsize=9)
all_cnts  = prompt_cnts + resp_cnts
y_max_bar = max(max(all_cnts) * 1.25, 10)
axes[0].set_ylim(0, y_max_bar)
for bar, cnt in zip(list(bars1) + list(bars2), prompt_cnts + resp_cnts):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + y_max_bar * 0.01,
        f"{cnt:,}\n({cnt / N * 100:.2f}%)",
        ha="center", va="bottom", fontsize=8,
    )

dup_sets_ordered = [
    ("Exact prompt\n(MD5)",   exact_dup_prompt_idx),
    ("Exact response\n(MD5)", exact_dup_resp_idx),
]
if minhash_available:
    dup_sets_ordered += [
        (f"MinHash prompt\n(Jac≥{MINHASH_THRESHOLD})",
         minhash_dup_prompt_idx_set),
        (f"MinHash resp\n(Jac≥{MINHASH_THRESHOLD})",
         minhash_dup_resp_idx_set),
    ]
if semantic_available:
    dup_sets_ordered += [
        (f"Semantic\n(cos≥{SEMANTIC_THRESHOLD})", semantic_dup_idx_set),
    ]

running_union: Set[int] = set()
bar_labels, marginal_vals = [], []
for name, idx_set in dup_sets_ordered:
    marginal_vals.append(len(idx_set - running_union))
    bar_labels.append(name)
    running_union |= idx_set

total_dup_union = len(running_union)
bar_labels_full = bar_labels + ["Clean\n(no dup)"]
marginal_full   = marginal_vals + [N - total_dup_union]
colors_bar_full = (
    [PALETTE[i % len(PALETTE)] for i in range(len(bar_labels))] +
    ["#cccccc"]
)
y_pos = np.arange(len(bar_labels_full))
hbars = axes[1].barh(y_pos, marginal_full,
                     color=colors_bar_full, alpha=0.85, edgecolor="none")
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(bar_labels_full, fontsize=9)
axes[1].set_xlabel("Record Count")
axes[1].set_title(
    f"Marginal Coverage\n"
    f"(dup union={total_dup_union:,}="
    f"{total_dup_union / N * 100:.2f}%)",
    fontsize=10,
)
x_max_bar = max(max(marginal_full) * 1.20, 10)
axes[1].set_xlim(0, x_max_bar)
for hbar, val in zip(hbars, marginal_full):
    axes[1].text(
        hbar.get_width() + x_max_bar * 0.01,
        hbar.get_y() + hbar.get_height() / 2,
        f"{val:,}  ({val / N * 100:.2f}%)",
        va="center", ha="left", fontsize=8,
    )
plt.tight_layout()
plt.savefig("figures/fig4_duplicate_analysis.png", bbox_inches="tight")
plt.close()
print("  Saved: figures/fig4_duplicate_analysis.png")

# ── Figure 5：全量相似度分布（精确，每条记录的最大相似度）─────────────────────
print("\n  Generating full-dataset similarity distribution figures ...")

sim_plot_configs = []
if minhash_available and len(minhash_jacc_prompt_arr) > 0:
    # 只展示有候选的记录（max_jacc > 0）
    valid_prompt_jacc = minhash_jacc_prompt_arr[minhash_jacc_prompt_arr > 0]
    if len(valid_prompt_jacc) > 0:
        sim_plot_configs.append((
            valid_prompt_jacc,
            f"MinHash Jaccard – Prompt\n"
            f"(n-gram={MINHASH_NGRAM_PROMPT}, "
            f"records with any candidate={len(valid_prompt_jacc):,})",
            MINHASH_THRESHOLD,
            PALETTE[0],
        ))

if minhash_available and len(minhash_jacc_resp_arr) > 0:
    valid_resp_jacc = minhash_jacc_resp_arr[minhash_jacc_resp_arr > 0]
    if len(valid_resp_jacc) > 0:
        sim_plot_configs.append((
            valid_resp_jacc,
            f"MinHash Jaccard – Response\n"
            f"(n-gram={MINHASH_NGRAM_RESPONSE}, "
            f"records with any candidate={len(valid_resp_jacc):,})",
            MINHASH_THRESHOLD,
            PALETTE[1],
        ))

if semantic_available:
    # max_cosine_per_record 初始值 -1，过滤掉 -1 即可
    valid_cosine = max_cosine_per_record[max_cosine_per_record >= 0]
    if len(valid_cosine) > 0:
        sim_plot_configs.append((
            valid_cosine,
            f"Semantic Cosine – Prompt\n"
            f"(max cosine to any other record, n={len(valid_cosine):,})",
            SEMANTIC_THRESHOLD,
            PALETTE[2],
        ))

if sim_plot_configs:
    n_cols  = len(sim_plot_configs)
    fig5, axes5 = plt.subplots(1, n_cols, figsize=(8 * n_cols, 6.5))
    if n_cols == 1:
        axes5 = [axes5]

    fig5.suptitle(
        "UIGEN-T2  Full-Dataset Similarity Distributions\n"
        "(each record's max similarity score to any other record)",
        fontsize=13, fontweight="bold", y=1.03,
    )

    for ax5, (arr, label, thr, color) in zip(axes5, sim_plot_configs):
        plot_min = max(float(arr.min()) - 0.02, 0.0)

        # 主直方图 + KDE
        sns.histplot(
            arr, bins=100, color=color, alpha=0.55, kde=True,
            line_kws={"linewidth": 2.5, "color": color},
            ax=ax5, binrange=(plot_min, 1.0),
        )
        ax5.set_xlim(plot_min, 1.02)

        # 阈值竖线 + 阴影
        ax5.axvline(thr, color="red", ls="--", lw=2.5,
                    zorder=5, label=f"Threshold = {thr}")
        ax5.axvspan(thr, 1.02, color="red", alpha=0.08,
                    label="Would be filtered")

        # 分位数辅助线
        pct_style = {
            "P50": ("green",      ":"),
            "P90": ("darkorange", ":"),
            "P95": ("purple",     "-."),
            "P99": ("brown",      "--"),
        }
        for pct_name, pct_n in [("P50", 50), ("P90", 90),
                                  ("P95", 95), ("P99", 99)]:
            pv    = float(np.percentile(arr, pct_n))
            c_pct, ls_pct = pct_style[pct_name]
            ax5.axvline(pv, color=c_pct, ls=ls_pct, lw=1.5,
                        label=f"{pct_name}={pv:.3f}")

        ax5.set_title(label, fontsize=11, fontweight="bold")
        ax5.set_ylabel("Record Count", fontsize=11)
        ax5.legend(fontsize=9, loc="upper left")

        # 所有绘图元素完成后获取正确 ylim
        ylim_top5  = get_ylim_top(ax5)
        n_filtered = int((arr >= thr).sum())
        pct_filt   = n_filtered / len(arr) * 100
        ax5.text(
            min(thr + 0.015, 0.97),
            ylim_top5 * 0.88,
            f"Filtered:\n{n_filtered:,}\n({pct_filt:.2f}%)",
            color="darkred", fontsize=10, fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round,pad=0.4",
                      fc="white", ec="red", lw=1.5, alpha=0.95),
        )

        # x 轴标签嵌入分位数速查表
        pct_summary = "  ".join(
            f"P{p}={np.percentile(arr, p):.3f}"
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        )
        ax5.set_xlabel(
            f"Max Similarity Score\n[{pct_summary}]",
            fontsize=8.5,
        )

    plt.tight_layout()
    plt.savefig("figures/fig5_fulldata_similarity_distribution.png",
                bbox_inches="tight")
    plt.close()
    print("  Saved: figures/fig5_fulldata_similarity_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# Step 13：保存画像数据
# ─────────────────────────────────────────────────────────────────────────────
print("\n[13] Saving profile data ...")

# ── 构建相似度分布统计摘要（供 JSON 保存）────────────────────────────────────
similarity_distributions: Dict = {}

if minhash_available:
    valid_pj = minhash_jacc_prompt_arr[minhash_jacc_prompt_arr > 0]
    valid_rj = minhash_jacc_resp_arr[minhash_jacc_resp_arr > 0]
    similarity_distributions["minhash_jacc_prompt"] = {
        "n_total":            int(len(minhash_jacc_prompt_arr)),
        "n_with_candidate":   int(len(valid_pj)),
        "quantile_stats":     quantile_stats(valid_pj) if len(valid_pj) else {},
        "threshold":          MINHASH_THRESHOLD,
        "n_above_threshold":  int((minhash_jacc_prompt_arr >= MINHASH_THRESHOLD).sum()),
    }
    similarity_distributions["minhash_jacc_response"] = {
        "n_total":            int(len(minhash_jacc_resp_arr)),
        "n_with_candidate":   int(len(valid_rj)),
        "quantile_stats":     quantile_stats(valid_rj) if len(valid_rj) else {},
        "threshold":          MINHASH_THRESHOLD,
        "n_above_threshold":  int((minhash_jacc_resp_arr >= MINHASH_THRESHOLD).sum()),
    }

if semantic_available:
    valid_cos = max_cosine_per_record[max_cosine_per_record >= 0]
    similarity_distributions["semantic_cosine_prompt"] = {
        "n_total":            int(len(max_cosine_per_record)),
        "n_valid":            int(len(valid_cos)),
        "quantile_stats":     quantile_stats(valid_cos) if len(valid_cos) else {},
        "threshold":          SEMANTIC_THRESHOLD,
        "n_above_threshold":  int((max_cosine_per_record >= SEMANTIC_THRESHOLD).sum()),
        "threshold_sensitivity": {
            str(round(max(min(SEMANTIC_THRESHOLD + d, 1.0), 0.0), 2)):
                int((valid_cos >= round(
                    max(min(SEMANTIC_THRESHOLD + d, 1.0), 0.0), 2
                )).sum())
            for d in [-0.10, -0.05, 0.0, +0.05, +0.10]
        },
    }

profile_data = {
    "n_total":                 N,
    "completeness":            completeness,
    "char_stats":              char_stats,
    "token_stats":             token_stats,
    "total_token_stats":       total_token_stats,
    "topic_counts":            dict(topic_counts),
    "resp_type_agg":           dict(resp_type_agg),
    "corr_tok":                float(corr_tok),
    "quality_flag_summary":    flag_summary,
    "thresholds":              THRESHOLDS,
    "template_overhead":       TEMPLATE_OVERHEAD,
    "dedup_config": {
        "minhash_threshold":      MINHASH_THRESHOLD,
        "minhash_ngram_prompt":   MINHASH_NGRAM_PROMPT,
        "minhash_ngram_response": MINHASH_NGRAM_RESPONSE,
        "semantic_threshold":     SEMANTIC_THRESHOLD,
    },
    "similarity_distributions": similarity_distributions,
    "dup_indices": {
        "exact_prompt":    sorted(exact_dup_prompt_idx),
        "exact_response":  sorted(exact_dup_resp_idx),
        "minhash_prompt":  sorted(minhash_dup_prompt_idx_set),
        "minhash_resp":    sorted(minhash_dup_resp_idx_set),
        "semantic_prompt": sorted(semantic_dup_idx_set),
    },
    "semantic_dup_examples":   semantic_dup_examples,
    "per_record_token_lens": {
        "prompt":              token_lens["prompt"],
        "reasoning":           token_lens["reasoning"],
        "response":            token_lens["response"],
        "total":               total_token_lens,
        "total_with_template": total_with_template,
    },
}

with open("data/profile_data.json", "w") as f:
    json.dump(profile_data, f, indent=2)

print("  Saved: data/profile_data.json")

# ─────────────────────────────────────────────────────────────────────────────
# 完成
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Done] Data profiling complete.")
print("=" * 60)
print("  THRESHOLDS 说明：")
print(f"    response_tok_min  = {THRESHOLDS['response_tok_min']}")
print(f"    response_tok_max  = {THRESHOLDS['response_tok_max']}")
print(f"    reasoning_tok_min = {THRESHOLDS['reasoning_tok_min']}")
print(f"    reasoning_tok_max = None  (不设上限)")
print(f"    total_tok_max     = {THRESHOLDS['total_tok_max']}  "
      f"(8192 - {TEMPLATE_OVERHEAD} 模板开销)")
print("=" * 60)
print("  输出文件：")
print("    figures/fig1_length_distribution.png")
print("    figures/fig1b_total_token_distribution.png")
print("    figures/fig2_topic_distribution.png")
print("    figures/fig3_quality_indicators.png")
print("    figures/fig4_duplicate_analysis.png")
print("    figures/fig5_fulldata_similarity_distribution.png")
print("    figures/fig5_similarity_probe_upgraded.png")
print("    data/profile_data.json")
print("=" * 60)