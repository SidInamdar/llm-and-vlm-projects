"""
train_cs_chatbot_instrumented.py
Llama 3.1 8B Instruct · LoRA r=64 · RTX 5090
Maximum visibility build: per-layer LoRA drift, token efficiency,
gradient health, activation norms — all logged to TensorBoard + MLflow.
"""

import os, math, time
from pathlib import Path
import torch
import numpy as np
import mlflow
from collections import defaultdict
from transformers import (
    AutoModelForCausalLM, AutoTokenizer
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from torch.utils.tensorboard import SummaryWriter
from datasets import load_from_disk

# Resolve repo root relative to this file:
# this file  → projects/llama-sft/llama31-sft-monitoring-pipeline.py
# repo root  → ../../  (llm-and-vlm-projects/)
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR  = "./checkpoints/cs-chatbot-lora"
MAX_SEQ_LEN = 512
TB_LOG_DIR   = "./logs/tb"
RUN_NAME     = "lora-r64-lr2e4-ep3"

SYSTEM_PROMPT = (
    "You are a professional customer support assistant. "
    "Be empathetic, concise, and resolution-focused. "
    "If unable to resolve, offer to escalate to a human agent."
)

# ── 1. Tokenizer ──────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── 2. Model ──────────────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False,
    device_map={"": 0},
)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# ── 3. LoRA ───────────────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# Snapshot initial LoRA weights — we'll measure drift from this baseline
# Keys: param_name → initial L2 norm of the weight tensor
initial_lora_norms = {
    name: param.data.float().norm().item()
    for name, param in model.named_parameters()
    if param.requires_grad and ("lora_A" in name or "lora_B" in name)
}

# ── 4. Dataset ────────────────────────────────────────────────────────────────
# Pass in your already-prepared Dataset objects directly.
# Both must have "instruction" and "response" columns.
# Replace this with your actual dataset variable (74K samples, unsplit):
full_dataset = load_from_disk(
    str(_REPO_ROOT / "datasets" / "processed" / "bitext_multiwoz_sft_dataset")
)

# Split — 95% train, 5% val — split BEFORE formatting so val is clean
split       = full_dataset.train_test_split(test_size=0.05, seed=42)
train_split = split["train"]   # ~70,300 samples
val_split   = split["test"]    # ~3,700  samples

print(f"Train: {len(train_split)} | Val: {len(val_split)}")

def fmt(s):
    msgs = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": s["instruction"]},
        {"role": "assistant", "content": s["response"]},
    ]
    return {"text": tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False)}

ds = {
    "train":      train_split.map(fmt, remove_columns=train_split.column_names),
    "validation": val_split.map(fmt,   remove_columns=val_split.column_names),
}

# ── 5. Training args ──────────────────────────────────────────────────────────
args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10,
    report_to="tensorboard",
    logging_dir=TB_LOG_DIR,
    dataloader_num_workers=4,
    max_grad_norm=1.0,
    remove_unused_columns=False,
    # Moved here from SFTTrainer (required by newer trl)
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
)

# ── 6. Loss masking collator ──────────────────────────────────────────────────
RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tokenizer)

# ── 7. Visibility hooks ───────────────────────────────────────────────────────

class VisibilityCallback:
    """
    Hooks into the trainer to log deeper diagnostics every N steps.

    Logs to both TensorBoard (live) and MLflow (persistent).

    What it tracks:
      - token_efficiency      : fraction of batch tokens that are NOT masked
      - lora_drift/{name}     : L2 distance each LoRA matrix has moved from init
      - grad_norm/{module}    : per-module gradient norms (q/k/v/o/gate/up/down)
      - lora_rank_utilization : effective rank (nuclear norm / Frobenius norm ratio)
        approximates how many singular dimensions are actually being used
      - gpu_mem_gb            : allocated VRAM at log time
      - steps_per_second      : throughput
    """

    def __init__(self, tb_writer, log_every=50, deep_every=200):
        self.tb          = tb_writer
        self.log_every   = log_every    # lightweight metrics
        self.deep_every  = deep_every   # expensive metrics (rank utilization)
        self._last_time  = time.time()
        self._last_step  = 0

    # ── called by trainer after each optimizer step ──────────────────────────
    def on_step_end(self, args, state, control, model, **kwargs):
        step = state.global_step
        if step % self.log_every != 0:
            return

        # ── Token efficiency ─────────────────────────────────────────────────
        # Injected by the patched collator (see below); falls back gracefully
        if hasattr(model, "_last_token_efficiency"):
            eff = model._last_token_efficiency
            self.tb.add_scalar("training/token_efficiency", eff, step)
            mlflow.log_metric("token_efficiency", eff, step=step)

        # ── LoRA weight drift ────────────────────────────────────────────────
        # For each LoRA matrix, compute how far it has moved from initialization.
        # A matrix that hasn't moved = the adapter isn't learning that module.
        # Very large drift = possible instability or overfitting signal.
        drift_by_type = defaultdict(list)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" not in name and "lora_B" not in name:
                continue
            current_norm = param.data.float().norm().item()
            init_norm    = initial_lora_norms.get(name, 1.0)
            drift        = abs(current_norm - init_norm) / max(init_norm, 1e-8)
            self.tb.add_scalar(f"lora_drift/{name}", drift, step)

            # Aggregate by module type for the summary chart
            for tag in ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]:
                if tag in name:
                    drift_by_type[tag].append(drift)
                    break

        for tag, drifts in drift_by_type.items():
            self.tb.add_scalar(f"lora_drift_mean/{tag}", np.mean(drifts), step)

        # ── Per-module gradient norms ────────────────────────────────────────
        # Tells you which modules are receiving the strongest learning signal.
        # Useful to detect: dead modules (near-zero grad), exploding modules.
        grad_by_module = defaultdict(list)
        for name, param in model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            gnorm = param.grad.float().norm().item()
            for tag in ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]:
                if tag in name:
                    grad_by_module[tag].append(gnorm)
                    break

        for tag, gnorms in grad_by_module.items():
            self.tb.add_scalar(f"grad_norm_by_module/{tag}", np.mean(gnorms), step)

        # ── GPU memory ───────────────────────────────────────────────────────
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated(0) / 1e9
            self.tb.add_scalar("system/gpu_mem_allocated_gb", mem_gb, step)
            mlflow.log_metric("gpu_mem_allocated_gb", mem_gb, step=step)

        # ── Throughput ───────────────────────────────────────────────────────
        now  = time.time()
        sps  = (step - self._last_step) / max(now - self._last_time, 1e-6)
        self.tb.add_scalar("system/steps_per_second", sps, step)
        self._last_time = now
        self._last_step = step

        # ── Deep diagnostics (expensive, less frequent) ──────────────────────
        if step % self.deep_every == 0:
            self._log_rank_utilization(model, step)
            self._log_layer_activation_norms(model, step)

    def _log_rank_utilization(self, model, step):
        """
        Effective rank = nuclear_norm / frobenius_norm.
        For a rank-r matrix: max value approaches r when all singular values equal.
        Low effective rank = adapter has collapsed to fewer dims than r.
        High = adapter is using its full capacity.
        """
        for name, param in model.named_parameters():
            if "lora_B" not in name or not param.requires_grad:
                continue
            # Only compute for reasonably sized matrices
            if param.numel() > 1_000_000:
                continue
            W = param.data.float()
            try:
                S = torch.linalg.svdvals(W)
                nuclear_norm   = S.sum().item()
                frobenius_norm = W.norm().item()
                eff_rank = nuclear_norm / max(frobenius_norm, 1e-8)
                self.tb.add_scalar(f"lora_eff_rank/{name}", eff_rank, step)
            except Exception:
                pass  # SVD can fail on degenerate matrices; skip silently

    def _log_layer_activation_norms(self, model, step):
        """
        Register forward hooks briefly, run a tiny dummy batch,
        log mean activation norm per layer, then remove hooks.
        Detects saturating layers (very high norms) or dead layers (near zero).
        NOTE: This adds ~1-2s overhead every deep_every steps. Disable if slow.
        """
        hooks      = []
        act_norms  = {}

        def make_hook(layer_name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    act_norms[layer_name] = output.detach().float().norm().item()
                elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                    act_norms[layer_name] = output[0].detach().float().norm().item()
            return hook

        # Hook every 4th transformer layer to keep overhead low
        for i, layer in enumerate(model.base_model.model.model.layers):
            if i % 4 == 0:
                h = layer.register_forward_hook(make_hook(f"layer_{i}"))
                hooks.append(h)

        # Tiny dummy forward pass (single token, no grad)
        try:
            dummy = torch.tensor([[1]], device=next(model.parameters()).device)
            with torch.no_grad():
                model(dummy)
            for lname, norm in act_norms.items():
                self.tb.add_scalar(f"activation_norm/{lname}", norm, step)
        except Exception:
            pass
        finally:
            for h in hooks:
                h.remove()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Log eval metrics to MLflow whenever the trainer evaluates."""
        step = state.global_step
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v, step=step)

    def on_train_end(self, args, state, control, **kwargs):
        mlflow.log_metric("best_eval_loss", state.best_metric or 0)
        self.tb.flush()
        self.tb.close()


# ── Token efficiency patch ────────────────────────────────────────────────────
# Subclass the collator to attach token efficiency to the model object
# so the callback can read it without touching the training loop internals.
class InstrumentedCollator(DataCollatorForCompletionOnlyLM):
    def __init__(self, *args, model_ref=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_ref = model_ref

    def __call__(self, features):
        batch = super().__call__(features)
        labels = batch.get("labels")
        if labels is not None and self._model_ref is not None:
            total    = labels.numel()
            unmasked = (labels != -100).sum().item()
            eff      = unmasked / max(total, 1)
            self._model_ref._last_token_efficiency = eff
        return batch


# ── 8. Wire everything together ───────────────────────────────────────────────
os.makedirs(TB_LOG_DIR, exist_ok=True)
tb_writer  = SummaryWriter(log_dir=TB_LOG_DIR)
vis_cb     = VisibilityCallback(tb_writer, log_every=10, deep_every=200)
instr_coll = InstrumentedCollator(
    RESPONSE_TEMPLATE, tokenizer=tokenizer, model_ref=model
)

# Custom trainer that injects the callback into the right hooks
from transformers import TrainerCallback

class ProxyCallback(TrainerCallback):
    def __init__(self, cb): self.cb = cb
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Explicitly capture model so VisibilityCallback always receives it
        self.cb.on_step_end(args, state, control, model=model, **kwargs)
    def on_evaluate(self, args, state, control, **kwargs):
        self.cb.on_evaluate(args, state, control, **kwargs)
    def on_train_end(self, args, state, control, **kwargs):
        self.cb.on_train_end(args, state, control, **kwargs)

# ── 9. Train ──────────────────────────────────────────────────────────────────
mlflow.set_experiment("cs-chatbot-sft")

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_params({
        "model":         MODEL_NAME,
        "lora_r":        64,
        "lora_alpha":    128,
        "learning_rate": 2e-4,
        "epochs":        3,
        "batch_size":    32,
        "seq_len":       MAX_SEQ_LEN,
        "target_modules": "q/k/v/o_proj+gate/up/down_proj",
    })

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,   # replaces deprecated tokenizer=
        data_collator=instr_coll,
        callbacks=[ProxyCallback(vis_cb)],
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR + "/final")
    tokenizer.save_pretrained(OUTPUT_DIR + "/final")
    mlflow.log_artifacts(OUTPUT_DIR + "/final", artifact_path="lora-adapter")

print("Training complete.")
print(f"TensorBoard: tensorboard --logdir {TB_LOG_DIR} --port 6006")
print(f"MLflow UI:   mlflow ui --port 5000")