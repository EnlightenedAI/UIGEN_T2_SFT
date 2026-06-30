# 02_data_cleaning.py
"""
UIGEN-T2 数据集清洗脚本

清洗规则：
  R1   response  token 数 < response_tok_min
  R2   response  token 数 > response_tok_max
  R3   response 缺失基本 HTML 结构标签
  R4   reasoning token 数 < reasoning_tok_min (含空=0，不设上限)
  R5   prompt 精确重复 (MD5，保留首次出现)
  R6   prompt/response 近重复 (MinHash，来自01预计算)
  R7   prompt 语义近重复 (sentence-transformers，来自01预计算)
  R8   prompt 过短 (< 20 chars)
  R9   total_tok + template_overhead > 8192

兼容性：
  - 兼容旧版 profile_data.json（无 total_with_template 字段）
  - 自动从 total + template_overhead 计算

验证集：从清洗后数据分层抽样 1000 条，Clean/Raw 完全共用。
"""

import json
import re
import hashlib
import random
from collections import defaultdict
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from tqdm import tqdm
import os

os.makedirs("data",    exist_ok=True)
os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi":  150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size": 10,
})
PALETTE = sns.color_palette("husl", 8)

SEED     = 42
VAL_SIZE = 1000

random.seed(SEED)
np.random.seed(SEED)

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

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────
def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def has_html_structure(text: str) -> bool:
    return bool(
        re.search(r"<html[\s>]",       text, re.I) or
        re.search(r"<!DOCTYPE\s+html", text, re.I) or
        re.search(r"<body[\s>]",       text, re.I)
    )


def classify_prompt(text: str) -> str:
    tl = text.lower()
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in tl for kw in kws):
            return topic
    return "Other"


def stratified_sample(
    items:      List,
    total_size: int,
    key_fn:     Callable,
    seed:       int = SEED,
) -> Tuple[List, List]:
    """
    分层抽样，严格返回 total_size 条。
    名额不足时将余量补给其他组，保证总数准确。
    返回 (sampled, remaining)。
    """
    if len(items) <= total_size:
        return list(items), []

    groups: Dict[str, List] = defaultdict(list)
    for item in items:
        groups[key_fn(item)].append(item)

    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)

    n_topics  = len(groups)
    base      = total_size // n_topics
    remainder = total_size  % n_topics

    sorted_topics = sorted(groups.keys(), key=lambda t: -len(groups[t]))
    quota: Dict[str, int] = {
        topic: base + (1 if rank < remainder else 0)
        for rank, topic in enumerate(sorted_topics)
    }

    # 名额不足时将差额补给其他组
    deficit = 0
    for topic in sorted_topics:
        available = len(groups[topic])
        if quota[topic] > available:
            deficit      += quota[topic] - available
            quota[topic]  = available
    for topic in sorted_topics:
        if deficit <= 0:
            break
        can_give = len(groups[topic]) - quota[topic]
        if can_give > 0:
            give         = min(can_give, deficit)
            quota[topic] += give
            deficit      -= give

    sampled  : List = []
    remaining: List = []
    for topic, group in groups.items():
        q = quota[topic]
        sampled.extend(group[:q])
        remaining.extend(group[q:])

    rng.shuffle(sampled)
    rng.shuffle(remaining)

    assert len(sampled) == total_size, (
        f"stratified_sample: expected {total_size}, got {len(sampled)}"
    )
    return sampled, remaining


def apply_rules(
    record:               Dict,
    idx:                  int,
    token_lens_response:  List[int],
    token_lens_reasoning: List[int],
    total_with_template:  List[int],
    thresholds:           Dict,
    seen_prompt_hashes:   Set[str],
    near_dup_indices:     Set[int],
) -> Tuple[bool, str]:
    """
    依次应用 R1~R9，返回 (keep, reason)。

    规则说明：
      R4 只检查下限，不检查上限（reasoning 不设 max）
      R9 使用 total_with_template（含模板开销）与 8192 比较
    """
    prompt   = record.get("prompt",    "") or ""
    response = record.get("response",  "") or ""

    resp_tok  = token_lens_response[idx]
    reas_tok  = token_lens_reasoning[idx]
    total_tok = total_with_template[idx]

    # R8：prompt 过短
    if len(prompt.strip()) < 20:
        return False, "R8_prompt_too_short"

    # R1：response token 数低于下限
    if resp_tok < thresholds["response_tok_min"]:
        return False, "R1_response_tok_short"

    # R2：response token 数高于上限
    if resp_tok > thresholds["response_tok_max"]:
        return False, "R2_response_tok_long"

    # R3：response 缺失基本 HTML 结构
    if not has_html_structure(response):
        return False, "R3_no_html_structure"

    # R4：reasoning token 数低于下限（含空=0，不设上限）
    if reas_tok < thresholds["reasoning_tok_min"]:
        return False, "R4_reasoning_tok_short"

    # R5：prompt 精确重复（MD5，保留首次出现）
    h = md5(prompt)
    if h in seen_prompt_hashes:
        return False, "R5_exact_dup_prompt"
    seen_prompt_hashes.add(h)

    # R6/R7：近重复（MinHash + 语义，01预计算，O(1) 查表）
    if idx in near_dup_indices:
        return False, "R6R7_near_dup"

    # R9：total + template > 8192
    # 兼容处理：None 或非 int 视为超长直接过滤
    if not isinstance(total_tok, int) or total_tok > 8192:
        return False, "R9_exceed_seq_len"

    return True, ""


def save_jsonl(data: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps({
                "prompt":    rec.get("prompt",    ""),
                "reasoning": rec.get("reasoning", ""),
                "response":  rec.get("response",  ""),
            }, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):,} records → {path}")


def get_char_lens(data: List[Dict], field: str) -> np.ndarray:
    return np.array(
        [len(str(r.get(field, "") or "")) for r in data], dtype=float
    )


def safe_hist(ax, data: np.ndarray, bins: int, color, alpha: float,
              force_xmin: float = 0.0, force_xmax: float = None,
              x_hi_pct: float = 99.0):
    xmin = force_xmin
    xmax = (float(np.percentile(data, x_hi_pct))
            if force_xmax is None else force_xmax)
    if xmax <= xmin:
        xmax = xmin + 1
    ax.hist(data, bins=bins, range=(xmin, xmax),
            color=color, alpha=alpha, edgecolor="none")
    ax.set_xlim(xmin, xmax)
    return xmin, xmax


def field_stats(data: List[Dict], field: str) -> Dict:
    a = get_char_lens(data, field)
    return {
        "mean":   float(np.mean(a)),
        "median": float(np.median(a)),
        "p10":    float(np.percentile(a, 10)),
        "p90":    float(np.percentile(a, 90)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 1：加载原始数据集
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading raw dataset ...")
print("=" * 60)

ds      = load_dataset("Tesslate/UIGEN-T2", split="train")
records = list(ds)
N_raw   = len(records)
print(f"Raw records: {N_raw:,}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2：从画像结果读取预计算数据
#
# 兼容性处理：
#   新版 01 脚本会保存 total_with_template 字段；
#   旧版没有此字段，自动从 total + template_overhead 计算，无需重跑 01。
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 2: Loading profile_data.json ...")

with open("data/profile_data.json", "r") as f:
    profile_data = json.load(f)

per_record           = profile_data["per_record_token_lens"]
token_lens_prompt    : List[int] = per_record["prompt"]
token_lens_reasoning : List[int] = per_record["reasoning"]
token_lens_response  : List[int] = per_record["response"]
total_token_lens     : List[int] = per_record["total"]

assert len(total_token_lens) == N_raw, (
    f"Token list length {len(total_token_lens)} != records {N_raw}. "
    "Please re-run 01_data_profiling.py first."
)

THRESHOLDS:        Dict = profile_data["thresholds"]
TEMPLATE_OVERHEAD: int  = profile_data.get("template_overhead", 30)

# ── 兼容旧版 profile_data.json ────────────────────────────────────────────────
_twt = per_record.get("total_with_template")
if _twt is None or any(v is None for v in _twt):
    print(f"  [INFO] 'total_with_template' not found or contains None, "
          f"computing from total + template_overhead({TEMPLATE_OVERHEAD}) ...")
    total_with_template: List[int] = [
        v + TEMPLATE_OVERHEAD for v in total_token_lens
    ]
else:
    total_with_template = [int(v) for v in _twt]

assert len(total_with_template) == N_raw, \
    "total_with_template length mismatch."
assert all(isinstance(v, int) for v in total_with_template), \
    "total_with_template contains non-int values."

# ── reasoning_tok_max 兼容 None ───────────────────────────────────────────────
# 旧版 THRESHOLDS 可能有 reasoning_tok_max=1500，新版为 None
# 02 脚本不使用此字段（R4 只检查下限），此处仅做提示
if THRESHOLDS.get("reasoning_tok_max") is not None:
    print(f"  [INFO] THRESHOLDS['reasoning_tok_max'] = "
          f"{THRESHOLDS['reasoning_tok_max']} (旧版配置，02脚本已忽略上限，"
          f"reasoning 不设上限)")

print(f"\n  Thresholds (from profile_data.json):")
for k, v in THRESHOLDS.items():
    note = ""
    if k == "reasoning_tok_max":
        note = "  ← 02脚本不使用（不设上限）" if v is not None else ""
    print(f"    {k:25s} = "
          f"{v if v is not None else 'None (no upper limit)'}{note}")
print(f"  Template overhead: {TEMPLATE_OVERHEAD} tokens")

n_exceed = sum(1 for v in total_with_template if v > 8192)
print(f"  Records with total+template > 8192: "
      f"{n_exceed:,} ({n_exceed / N_raw * 100:.1f}%)")

# ── 近重复索引（01预计算）────────────────────────────────────────────────────
dup_indices_raw : Dict     = profile_data.get("dup_indices", {})
near_dup_indices: Set[int] = set()
near_dup_indices.update(dup_indices_raw.get("minhash_prompt",  []))
near_dup_indices.update(dup_indices_raw.get("minhash_resp",    []))
near_dup_indices.update(dup_indices_raw.get("semantic_prompt", []))
near_dup_indices.update(dup_indices_raw.get("exact_response",  []))

dedup_cfg = profile_data.get("dedup_config", {})
print(f"\n  Near-dup indices from 01:")
print(f"    minhash_prompt   "
      f"(n={dedup_cfg.get('minhash_ngram_prompt','?')}, "
      f"Jac≥{dedup_cfg.get('minhash_threshold','?')}): "
      f"{len(dup_indices_raw.get('minhash_prompt',  [])):>6,}")
print(f"    minhash_resp     "
      f"(n={dedup_cfg.get('minhash_ngram_response','?')}, "
      f"Jac≥{dedup_cfg.get('minhash_threshold','?')}): "
      f"{len(dup_indices_raw.get('minhash_resp',    [])):>6,}")
print(f"    semantic_prompt  "
      f"(cos≥{dedup_cfg.get('semantic_threshold','?')}):        "
      f"{len(dup_indices_raw.get('semantic_prompt', [])):>6,}")
print(f"    exact_response   (MD5):                    "
      f"{len(dup_indices_raw.get('exact_response',  [])):>6,}")
print(f"    union            :                         "
      f"{len(near_dup_indices):>6,}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3：对全量原始数据应用清洗规则
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 3: Applying cleaning rules R1~R9 to all records ...")

seen_hashes  : Set[str]       = set()
cleaned_all  : List[Dict]     = []
rule_counter : Dict[str, int] = defaultdict(int)

for i, record in enumerate(tqdm(records, desc="  filtering")):
    keep, reason = apply_rules(
        record               = record,
        idx                  = i,
        token_lens_response  = token_lens_response,
        token_lens_reasoning = token_lens_reasoning,
        total_with_template  = total_with_template,
        thresholds           = THRESHOLDS,
        seen_prompt_hashes   = seen_hashes,
        near_dup_indices     = near_dup_indices,
    )
    if keep:
        cleaned_all.append(record)
    else:
        rule_counter[reason] += 1

N_clean   = len(cleaned_all)
N_dropped = N_raw - N_clean

print(f"\n  Raw:     {N_raw:,}")
print(f"  Cleaned: {N_clean:,}  ({N_clean / N_raw * 100:.1f}%)")
print(f"  Dropped: {N_dropped:,}  ({N_dropped / N_raw * 100:.1f}%)")

RULE_LABELS = {
    "R1_response_tok_short": "R1   response tok < min",
    "R2_response_tok_long":  "R2   response tok > max",
    "R3_no_html_structure":  "R3   no HTML structure",
    "R4_reasoning_tok_short":"R4   reasoning tok < min (no upper limit)",
    "R5_exact_dup_prompt":   "R5   exact dup prompt (MD5)",
    "R6R7_near_dup":         "R6+R7 near-dup (MinHash+Semantic)",
    "R8_prompt_too_short":   "R8   prompt < 20 chars",
    "R9_exceed_seq_len":     "R9   total+template > 8192",
}

print("\n  Drop breakdown by rule:")
print(f"  {'Rule':58s} {'Count':>8s} {'%raw':>8s}")
print("  " + "-" * 78)
for rule_key in [
    "R8_prompt_too_short",   "R1_response_tok_short",
    "R2_response_tok_long",  "R3_no_html_structure",
    "R4_reasoning_tok_short","R5_exact_dup_prompt",
    "R6R7_near_dup",         "R9_exceed_seq_len",
]:
    cnt   = rule_counter.get(rule_key, 0)
    label = RULE_LABELS.get(rule_key, rule_key)
    print(f"  {label:58s} {cnt:>8,}  ({cnt / N_raw * 100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4：从清洗后数据中划出共享验证集
#
# val_shared 来自 cleaned_all（质量有保证），
# Clean 和 Raw 两组训练完全共用，loss 可直接横向对比。
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nStep 4: Splitting val_shared from cleaned data "
      f"(VAL_SIZE={VAL_SIZE}) ...")

assert N_clean > VAL_SIZE, (
    f"Cleaned dataset ({N_clean}) too small for VAL_SIZE={VAL_SIZE}. "
    "Please relax cleaning thresholds."
)

val_shared, clean_train = stratified_sample(
    items      = cleaned_all,
    total_size = VAL_SIZE,
    key_fn     = lambda r: classify_prompt(r.get("prompt", "")),
    seed       = SEED,
)

assert len(val_shared)  == VAL_SIZE,              \
    f"val_shared size mismatch: {len(val_shared)}"
assert len(clean_train) == N_clean - VAL_SIZE,    \
    f"clean_train size mismatch: {len(clean_train)}"

print(f"  cleaned_all:  {N_clean:,}")
print(f"  val_shared:   {len(val_shared):,}  "
      f"← 来自清洗后数据，Clean/Raw 完全共用")
print(f"  clean_train:  {len(clean_train):,}")

# 主题分布报告
topic_clean_cnt: Dict[str, int] = defaultdict(int)
topic_val_cnt  : Dict[str, int] = defaultdict(int)
for r in cleaned_all:
    topic_clean_cnt[classify_prompt(r.get("prompt", ""))] += 1
for r in val_shared:
    topic_val_cnt[classify_prompt(r.get("prompt", ""))] += 1

print(f"\n  val_shared topic distribution:")
print(f"  {'Topic':25s} {'Clean':>8s} {'Val':>6s} {'Val%':>6s}")
print("  " + "-" * 50)
for topic in sorted(topic_clean_cnt.keys(),
                    key=lambda t: -topic_clean_cnt[t]):
    n_c = topic_clean_cnt[topic]
    n_v = topic_val_cnt.get(topic, 0)
    print(f"  {topic:25s} {n_c:>8,} {n_v:>6,} {n_v / n_c * 100:>5.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5：构建 Raw 对照训练集
#
# 从原始数据中排除 val_shared 包含的样本后随机采样，
# 数量与 clean_train 严格相等（排除数量差异的影响）。
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 5: Building raw train set ...")

val_prompt_md5: Set[str] = {
    md5(r.get("prompt", "")) for r in val_shared
}
raw_pool = [
    r for r in records
    if md5(r.get("prompt", "")) not in val_prompt_md5
]

rng_raw = random.Random(SEED + 1)
rng_raw.shuffle(raw_pool)
raw_train = raw_pool[:len(clean_train)]

print(f"  raw_pool  (原始排除val): {len(raw_pool):,}")
print(f"  raw_train (等量采样):    {len(raw_train):,}")
print(f"  clean_train:             {len(clean_train):,}")
print(f"  val_shared (共用):       {len(val_shared):,}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6：保存数据集
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 6: Saving datasets ...")

save_jsonl(clean_train, "data/train_clean.jsonl")
save_jsonl(raw_train,   "data/train_raw.jsonl")
save_jsonl(val_shared,  "data/val_shared.jsonl")

print(f"\n  {'File':35s} {'Size':>8s}  Note")
print("  " + "-" * 68)
print(f"  {'data/train_clean.jsonl':35s} {len(clean_train):>8,}  "
      f"Clean 训练集（清洗后）")
print(f"  {'data/train_raw.jsonl':35s}   {len(raw_train):>8,}  "
      f"Raw 训练集（未清洗，等量）")
print(f"  {'data/val_shared.jsonl':35s}  {len(val_shared):>8,}  "
      f"共享验证集（来自清洗后数据）★")
print(f"\n  ★ Clean / Raw 两组训练均使用 val_shared 评估，loss 可直接对比。")
print(f"    03_train.py 中 val_file 固定指向 data/val_shared.jsonl。")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7：清洗前后对比统计（字符维度）
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 7: Before/After comparison statistics (char-level) ...")

FIELDS = ["prompt", "reasoning", "response"]

print(f"\n  {'':25s} {'Mean':>10s} {'Median':>10s} "
      f"{'P10':>10s} {'P90':>10s}")
print("  " + "-" * 68)
for split_name, data in [
    ("Raw   train", raw_train),
    ("Clean train", clean_train),
    ("Val  shared", val_shared),
]:
    print(f"\n  [{split_name}]")
    for field in FIELDS:
        s = field_stats(data, field)
        print(f"    {field:12s}  "
              f"{s['mean']:>10.0f} {s['median']:>10.0f} "
              f"{s['p10']:>10.0f} {s['p90']:>10.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 8：Figure 5 —— 清洗前后字符长度分布对比
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 8: Generating figures ...")

COLOR_RAW   = "#e88080"
COLOR_CLEAN = "#6ec98f"

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "UIGEN-T2: Before vs After Cleaning — Field Length Distributions (chars)\n"
    f"Raw n={len(raw_train):,}  →  Clean n={len(clean_train):,}  "
    f"(dropped {N_dropped:,} = {N_dropped / N_raw * 100:.1f}%)  "
    f"|  Shared val={VAL_SIZE} (from cleaned data)",
    fontsize=13, fontweight="bold", y=1.01,
)

for col, field in enumerate(FIELDS):
    raw_lens   = get_char_lens(raw_train,   field)
    clean_lens = get_char_lens(clean_train, field)
    # x_max      = float(np.percentile(raw_lens, 99))
    x_max      = float(np.max(raw_lens))

    # 上行：Raw
    ax_raw = axes[0][col]
    safe_hist(ax_raw, raw_lens, bins=60, color=COLOR_RAW, alpha=0.85,
              force_xmin=0.0, force_xmax=x_max)
    median_raw = float(np.median(raw_lens))
    ax_raw.axvline(median_raw, color="darkred", ls="--", lw=1.6,
                   label=f"Median: {median_raw:,.0f}")
    ax_raw.set_title(f"{field} — Raw  (n={len(raw_lens):,})", fontsize=11)
    ax_raw.set_xlabel(f"Characters  [0 ~ Max={x_max:,.0f}]")
    ax_raw.set_ylabel("Count")
    ax_raw.legend(fontsize=9)

    # 下行：Clean
    ax_clean = axes[1][col]
    safe_hist(ax_clean, clean_lens, bins=60, color=COLOR_CLEAN, alpha=0.85,
              force_xmin=0.0, force_xmax=x_max)
    median_clean = float(np.median(clean_lens))
    ax_clean.axvline(median_clean, color="darkgreen", ls="--", lw=1.6,
                     label=f"Median: {median_clean:,.0f}")
    ax_clean.set_title(f"{field} — Clean  (n={len(clean_lens):,})", fontsize=11)
    ax_clean.set_xlabel(f"Characters  [0 ~ Max={x_max:,.0f}]")
    ax_clean.set_ylabel("Count")
    ax_clean.legend(fontsize=9)

    # 两行之间标注 median 变化量
    delta = median_clean - median_raw
    sign  = "+" if delta >= 0 else ""
    fig.text(
        (col + 0.5) / 3, 0.505,
        f"Δ median = {sign}{delta:,.0f}",
        ha="center", va="center", fontsize=9, color="navy",
        bbox=dict(boxstyle="round,pad=0.3",
                  fc="lightyellow", ec="navy", lw=0.8),
    )

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("figures/fig5_before_after_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved: figures/fig5_before_after_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Step 9：Figure 6 —— 规则命中率柱状图
# ─────────────────────────────────────────────────────────────────────────────
rule_order = [
    "R8_prompt_too_short",   "R1_response_tok_short",
    "R2_response_tok_long",  "R3_no_html_structure",
    "R4_reasoning_tok_short","R5_exact_dup_prompt",
    "R6R7_near_dup",         "R9_exceed_seq_len",
]
rule_vals    = [rule_counter.get(k, 0) for k in rule_order]
rule_pcts    = [v / N_raw * 100 for v in rule_vals]
short_labels = [RULE_LABELS.get(k, k) for k in rule_order]

fig, ax = plt.subplots(figsize=(13, 6))
bar_colors = [
    "#e74c3c" if p > 5 else "#f39c12" if p > 1 else "#3498db"
    for p in rule_pcts
]
bars = ax.bar(range(len(rule_order)), rule_pcts,
              color=bar_colors, alpha=0.88, width=0.6)
ax.set_xticks(range(len(rule_order)))
ax.set_xticklabels(short_labels, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("% of raw records filtered")
ax.set_title(
    f"Records Filtered per Rule  "
    f"(raw={N_raw:,}, dropped={N_dropped:,} = {N_dropped / N_raw * 100:.1f}%)\n"
    f"R8→R1→R2→R3→R4→R5→R6R7→R9  |  R4: reasoning 不设上限",
    fontsize=11, fontweight="bold",
)
y_top = max(rule_pcts) * 1.35 if max(rule_pcts) > 0 else 1.0
ax.set_ylim(0, y_top)

for bar, cnt, pct in zip(bars, rule_vals, rule_pcts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + y_top * 0.02,
        f"{cnt:,}\n({pct:.1f}%)",
        ha="center", va="bottom", fontsize=8, fontweight="bold",
    )

from matplotlib.patches import Patch
ax.legend(
    handles=[
        Patch(facecolor="#e74c3c", alpha=0.88, label=">5%  High"),
        Patch(facecolor="#f39c12", alpha=0.88, label="1-5% Medium"),
        Patch(facecolor="#3498db", alpha=0.88, label="<1%  Low"),
    ],
    loc="upper right", fontsize=9,
)
plt.tight_layout()
plt.savefig("figures/fig6_rule_filter_rates.png", bbox_inches="tight")
plt.close()
print("  Saved: figures/fig6_rule_filter_rates.png")

# ─────────────────────────────────────────────────────────────────────────────
# 完成
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Done] Data cleaning complete.")
print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
print(f"  │  File                         Size      Note                 │")
print(f"  ├──────────────────────────────────────────────────────────────┤")
print(f"  │  data/train_clean.jsonl     {len(clean_train):>7,}    Clean 训练集          │")
print(f"  │  data/train_raw.jsonl       {len(raw_train):>7,}    Raw 训练集（等量）    │")
print(f"  │  data/val_shared.jsonl      {len(val_shared):>7,}    共享验证集 ★          │")
print(f"  └──────────────────────────────────────────────────────────────┘")
print(f"\n  ★ val_shared 来自清洗后数据，Clean / Raw 两组完全共用。")
print(f"    03_train.py 中 val_file 统一指向 data/val_shared.jsonl。")
print(f"\n  Figures:")
print(f"    figures/fig5_before_after_comparison.png")
print(f"    figures/fig6_rule_filter_rates.png")