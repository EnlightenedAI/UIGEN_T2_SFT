# 03_train.py
"""
UIGEN-T2 SFT 训练脚本

使用:
  python 03_train.py --mode raw
  python 03_train.py --mode clean

两组实验：
  - 完全相同的超参数
  - 完全相同的验证集 data/val_shared.jsonl（来自清洗后数据）
  - 唯一变量：训练集质量（raw vs clean）
"""

import os
import json
import argparse
import random
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    set_seed,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["raw", "clean"], required=True,
                    help="Training mode: raw or clean data")
parser.add_argument("--model_name",
                    default="Qwen/Qwen3-0.6B",
                    help="Base model path")
parser.add_argument("--max_steps", type=int, default=600,
                    help="Maximum training steps")
parser.add_argument("--output_dir", default=None,
                    help="Output directory (auto-generated if not specified)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# 配置（两组实验完全相同的超参数）
# ─────────────────────────────────────────────────────────────────────────────
SEED          = 42
MODEL_NAME    = args.model_name
MODE          = args.mode
MAX_SEQ_LEN   = 9182
MAX_STEPS     = args.max_steps
BATCH_SIZE    = 2
GRAD_ACCUM    = 4
LR            = 2e-5
LR_SCHEDULER  = "cosine"
WARMUP_RATIO  = 0.05
LOGGING_STEPS = 10
EVAL_STEPS    = 50
SAVE_STEPS    = 200
FP16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

OUTPUT_DIR = (args.output_dir or
              f"/nfs-data/sdd/pyzhu/UIGEN_T2_SFT/checkpoints/{MODE}")
LOG_FILE   = f"logs/train_{MODE}.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs",     exist_ok=True)
os.makedirs("figures",  exist_ok=True)

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print(f"Training Mode: {MODE.upper()}")
print(f"Model:         {MODEL_NAME}")
print(f"Max Steps:     {MAX_STEPS}")
print(f"Batch Size:    {BATCH_SIZE} × {GRAD_ACCUM} = "
      f"{BATCH_SIZE * GRAD_ACCUM} effective")
print(f"Learning Rate: {LR}")
print(f"Max Seq Len:   {MAX_SEQ_LEN}")
print(f"FP16: {FP16}  BF16: {BF16}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class UIGenDataset(Dataset):
    """
    格式：
      <|im_start|>user
      {prompt}<|im_end|>
      <|im_start|>assistant
      <think>
      {reasoning}
      </think>
      {response}<|im_end|>

    Label mask 策略：
      只对 assistant 内容部分计算 loss（prompt + assistant header 设为 -100）。
      通过在 full_enc 中搜索最后一次出现的 assistant header token ids 定位边界，
      避免二次 tokenize 的 BPE 边界偏移问题。

    reasoning 为空时格式退化为：
      <|im_start|>assistant
      {response}<|im_end|>
      （R4 清洗规则已过滤 reasoning 过短的样本，训练集中此情况极少）
    """

    _assistant_header = "<|im_start|>assistant\n"

    def __init__(
        self,
        data_path:   str,
        tokenizer,
        max_seq_len: int = 4096,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Dict] = []

        # 预计算 assistant header token ids（整个生命周期只做一次）
        self._assistant_header_ids: List[int] = tokenizer.encode(
            self._assistant_header, add_special_tokens=False
        )
        self._header_len = len(self._assistant_header_ids)

        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line.strip()))

        print(f"  Loaded {len(self.samples):,} samples from {data_path}")
        print(f"  Assistant header: {repr(self._assistant_header)}")
        print(f"  Header token ids: {self._assistant_header_ids} "
              f"(len={self._header_len})")

        # 训练前验证 label mask 正确性
        self._validate_samples(n_check=3)

    def __len__(self) -> int:
        return len(self.samples)

    def _format(self, rec: Dict) -> str:
        """构建完整的对话字符串。"""
        prompt    = rec.get("prompt",    "")
        reasoning = rec.get("reasoning", "")
        response  = rec.get("response",  "")

        if reasoning.strip():
            assistant_content = (
                f"<think>\n{reasoning}\n</think>\n{response}"
            )
        else:
            assistant_content = response

        return (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
        )

    def _find_prompt_len(self, input_ids: List[int]) -> int:
        """
        在 input_ids 中搜索最后一次出现的 assistant header token ids，
        返回 prompt 部分长度（含 header，这部分 label 设为 -100）。

        搜索最后一次：防止 prompt 内容中恰好含有相同 token 序列。
        未找到时返回 0（退化为对全序列计算 loss，安全兜底）。
        """
        header_ids = self._assistant_header_ids
        header_len = self._header_len
        for i in range(len(input_ids) - header_len, -1, -1):
            if input_ids[i: i + header_len] == header_ids:
                return i + header_len
        return 0

    def _validate_samples(self, n_check: int = 3):
        """
        训练前抽查 label mask，打印统计信息供人工核验。
        重点检查：
          - label token 占比（应在 50%~90% 之间，过低说明 prompt 太长）
          - 是否被截断（total == max_seq_len 说明触发了截断）
          - 是否有 0 个 label token（assistant 被完全截断，无训练信号）
        """
        print(f"\n  Label mask validation ({n_check} samples):")
        print(f"  {'idx':>4s}  {'total':>7s}  {'prompt':>7s}  "
              f"{'label':>7s}  {'label%':>6s}  {'truncated':>9s}")
        print("  " + "-" * 54)
        for i in range(min(n_check, len(self.samples))):
            item      = self[i]
            total     = len(item["input_ids"])
            n_label   = sum(1 for l in item["labels"] if l != -100)
            n_prompt  = total - n_label
            truncated = (total >= self.max_seq_len)
            flag      = "YES ⚠" if truncated else "no"
            print(f"  {i:>4d}  {total:>7d}  {n_prompt:>7d}  "
                  f"{n_label:>7d}  {n_label / total * 100:>5.1f}%  "
                  f"{flag:>9s}")
            if n_label == 0:
                warnings.warn(
                    f"Sample {i}: 0 label tokens! "
                    "Assistant content may be entirely truncated. "
                    f"(total={total}, max_seq_len={self.max_seq_len})",
                    stacklevel=2,
                )

    def __getitem__(self, idx: int) -> Dict:
        rec       = self.samples[idx]
        full_text = self._format(rec)

        # 一次 tokenize，含截断
        full_enc = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = full_enc["input_ids"]

        # 定位 label mask 边界
        prompt_len = self._find_prompt_len(input_ids)
        if prompt_len > 0:
            labels = [-100] * prompt_len + list(input_ids[prompt_len:])
        else:
            # 安全退化：未找到 header（理论上不应发生）
            warnings.warn(
                f"[idx={idx}] Assistant header not found in input_ids. "
                "Falling back to full sequence as labels. "
                f"Check if the template format matches the tokenizer.",
                stacklevel=2,
            )
            labels = list(input_ids)

        assert len(input_ids) == len(labels), (
            f"[idx={idx}] Length mismatch: "
            f"input_ids={len(input_ids)}, labels={len(labels)}"
        )

        return {
            "input_ids":      input_ids,
            "attention_mask": full_enc["attention_mask"],
            "labels":         labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Callback：记录 train / val loss
# ─────────────────────────────────────────────────────────────────────────────
class LossLoggerCallback(TrainerCallback):
    """
    将每步 train loss 和每个 eval 点的 val loss 记录到 jsonl 文件。
    训练结束后可用于绘制 loss 曲线和 Clean vs Raw 对比分析。
    """

    def __init__(self, log_file: str):
        self.log_file     = log_file
        self.train_losses: List[Dict] = []
        self.eval_losses:  List[Dict] = []
        open(log_file, "w").close()   # 清空旧文件

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        if "loss" in logs:
            entry = {
                "step":       step,
                "train_loss": logs["loss"],
                "lr":         logs.get("learning_rate", 0),
            }
            self.train_losses.append(entry)
            with open(self.log_file, "a") as f:
                f.write(json.dumps({"type": "train", **entry}) + "\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        step     = state.global_step
        val_loss = metrics.get("eval_loss")
        if val_loss is not None:
            entry = {"step": step, "eval_loss": val_loss}
            self.eval_losses.append(entry)
            with open(self.log_file, "a") as f:
                f.write(json.dumps({"type": "eval", **entry}) + "\n")
            print(f"\n  [Eval @ step {step}] Val Loss = {val_loss:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 主训练流程
# ─────────────────────────────────────────────────────────────────────────────

# 1. Tokenizer & Model
print("\n[1] Loading tokenizer and model ...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=(torch.bfloat16 if BF16
           else torch.float16 if FP16
           else torch.float32),
    device_map="auto",
)
model.config.use_cache = False
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model loaded: {n_params / 1e6:.0f}M parameters")

# 2. 数据集
#    train_file 根据 mode 切换，val_file 两组完全相同
print("\n[2] Loading datasets ...")
train_file = f"data/train_{MODE}.jsonl"
val_file   = "data/val_shared.jsonl"   # ← 两组完全共用，loss 可直接对比

train_dataset = UIGenDataset(train_file, tokenizer, max_seq_len=MAX_SEQ_LEN)
val_dataset   = UIGenDataset(val_file,   tokenizer, max_seq_len=MAX_SEQ_LEN)

print(f"\n  Train: {len(train_dataset):,}  ({MODE})")
print(f"  Val:   {len(val_dataset):,}  (val_shared — 来自清洗后数据，两组共用)")

# 3. Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    return_tensors="pt",
)

# 4. 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    fp16=FP16,
    bf16=BF16,
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    seed=SEED,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    label_names=["labels"],
)

# 5. Callback & Trainer
loss_logger = LossLoggerCallback(log_file=LOG_FILE)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[loss_logger],
)

# 6. 训练
print(f"\n[3] Starting training ({MODE.upper()}) ...")
trainer.train()

print("\n[4] Training complete.")
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
print(f"  Model saved → {os.path.join(OUTPUT_DIR, 'final')}")
print(f"  Loss log   → {LOG_FILE}")
print(f"  Train pts  : {len(loss_logger.train_losses)}")
print(f"  Eval  pts  : {len(loss_logger.eval_losses)}")

# 7. 绘制 loss 曲线
if loss_logger.eval_losses:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Training Curves — {MODE.upper()}\n"
        f"Val = val_shared (来自清洗后数据，Clean/Raw 共用)",
        fontsize=13, fontweight="bold",
    )

    # Train loss
    if loss_logger.train_losses:
        steps  = [e["step"]       for e in loss_logger.train_losses]
        losses = [e["train_loss"] for e in loss_logger.train_losses]
        axes[0].plot(steps, losses, color="#3498db", lw=1.2, alpha=0.8)
        axes[0].set_title(f"Train Loss ({MODE.upper()})")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

    # Val loss
    val_steps  = [e["step"]      for e in loss_logger.eval_losses]
    val_losses = [e["eval_loss"] for e in loss_logger.eval_losses]
    axes[1].plot(val_steps, val_losses,
                 color="#e74c3c", lw=2.0, marker="o", markersize=4)
    axes[1].set_title(
        f"Val Loss ({MODE.upper()}) — val_shared\n"
        f"Final: {val_losses[-1]:.4f}"
    )
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].annotate(
        f"Final: {val_losses[-1]:.4f}",
        xy=(val_steps[-1], val_losses[-1]),
        xytext=(-60, 10), textcoords="offset points",
        fontsize=9, color="#e74c3c",
        arrowprops=dict(arrowstyle="->", color="#e74c3c"),
    )

    plt.tight_layout()
    curve_path = f"figures/loss_curve_{MODE}.png"
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()
    print(f"  Loss curve → {curve_path}")