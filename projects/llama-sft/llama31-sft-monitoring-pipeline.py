"""
train_cs_chatbot_instrumented.py
Llama 3.1 8B Instruct · LoRA r=64 · RTX 5090
Maximum visibility build: per-layer LoRA drift, token efficiency,
gradient health, activation norms — all logged to TensorBoard + MLflow.

Python 3.12.13 | transformers >= 4.45 | peft >= 0.10 | NO trl dependency
Called by: projects/llama-sft/submit_llama_sft.sh (SLURM)
"""

# Must be set before any transformers/datasets import to prevent
# network calls to huggingface.co on air-gapped / no-internet nodes.
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"

import time
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ── Repo root (this file lives at projects/llama-sft/) ────────────────────────
# parents[0] = llama-sft/
# parents[1] = projects/
# parents[2] = llm-and-vlm-projects/
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Config ────────────────────────────────────────────────────────────────────
# Local model path — no HF hub string, no network needed
MODEL_NAME  = str(_REPO_ROOT / "models" / "checkpoints" / "meta-llama--Llama-3.1-8B-Instruct")
OUTPUT_DIR  = "./checkpoints/cs-chatbot-lora"
MAX_SEQ_LEN = 512
TB_LOG_DIR  = "./logs/tb"
RUN_NAME    = "lora-r64-lr2e4-ep3"

# Llama 3 assistant header — marks where response starts for loss masking
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"

SYSTEM_PROMPT = (
    "You are a professional customer support assistant. "
    "Be empathetic, concise, and resolution-focused. "
    "If unable to resolve, offer to escalate to a human agent."
)

# ── 1. Tokenizer ──────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

# Pre-tokenise the assistant header once — reused in every tokenize_and_mask call
_ASSISTANT_HEADER_IDS: list[int] = tokenizer.encode(
    ASSISTANT_HEADER, add_special_tokens=False
)
_HEADER_LEN = len(_ASSISTANT_HEADER_IDS)

# ── 2. Model ──────────────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,                    # was torch_dtype=
    attn_implementation="sdpa",              # was "flash_attention_2" — not installed
    use_cache=False,
    device_map={"": 0},
)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# ── 3. LoRA ───────────────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# Store actual initial weight tensors for true drift: ‖W - W₀‖ / ‖W₀‖
# Storing .clone().cpu() is negligible RAM for r=64 LoRA matrices.
initial_lora_weights: dict[str, torch.Tensor] = {
    name: param.data.clone().cpu()
    for name, param in model.named_parameters()
    if param.requires_grad and ("lora_A" in name or "lora_B" in name)
}

# ── 4. Dataset ────────────────────────────────────────────────────────────────
full_dataset = load_from_disk(
    str(_REPO_ROOT / "datasets" / "processed" / "bitext_multiwoz_sft_dataset")
)

split       = full_dataset.train_test_split(test_size=0.05, seed=42)
train_split = split["train"]
val_split   = split["test"]

print(f"Train: {len(train_split)} | Val: {len(val_split)}")


def tokenize_and_mask(sample: dict) -> dict:
    """
    Applies chat template, tokenises, then masks all tokens up to and
    including the assistant header so loss is only computed on the response.

    Done at map() time so masking runs once on CPU, not on GPU every step.
    """
    msgs = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": sample["instruction"]},
        {"role": "assistant", "content": sample["response"]},
    ]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
    )
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,       # collator handles padding per-batch
        return_tensors=None, # return plain lists — datasets prefers this
    )

    input_ids = enc["input_ids"]
    labels    = list(input_ids)

    # Mask everything up to and including the assistant header
    mask_until = len(input_ids)   # fallback: mask entire sequence
    for j in range(len(input_ids) - _HEADER_LEN):
        if input_ids[j : j + _HEADER_LEN] == _ASSISTANT_HEADER_IDS:
            mask_until = j + _HEADER_LEN
            break

    for k in range(mask_until):
        labels[k] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": enc["attention_mask"],
        "labels":         labels,
    }


# num_proc=4: map is CPU-bound and embarrassingly parallel
ds = {
    "train": train_split.map(
        tokenize_and_mask,
        remove_columns=train_split.column_names,
        num_proc=4,
        desc="Tokenising train",
    ),
    "validation": val_split.map(
        tokenize_and_mask,
        remove_columns=val_split.column_names,
        num_proc=4,
        desc="Tokenising val",
    ),
}
ds["train"].set_format("torch")
ds["validation"].set_format("torch")

# ── 5. Collator ───────────────────────────────────────────────────────────────
# Masking is already done at map() time.
# The collator only needs to pad sequences to the same length per batch
# and track token efficiency for the visibility callback.

class InstrumentedCollator:
    """
    Pads input_ids / attention_mask / labels to the longest sequence in
    the batch. Labels padded with -100 so padding tokens don't contribute
    to loss. Attaches token_efficiency to model for VisibilityCallback.
    """

    def __init__(self, tokenizer, model_ref=None):
        self.tokenizer  = tokenizer
        self._model_ref = model_ref

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_ids      = [torch.tensor(f["input_ids"])      for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        labels         = [torch.tensor(f["labels"])         for f in features]

        input_ids      = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0,
        )
        labels         = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100,
        )

        if self._model_ref is not None:
            total    = labels.numel()
            unmasked = (labels != -100).sum().item()
            self._model_ref._last_token_efficiency = unmasked / max(total, 1)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── 6. Training args ──────────────────────────────────────────────────────────
args = TrainingArguments(
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
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10,
    report_to="tensorboard",
    logging_dir=TB_LOG_DIR,
    dataloader_num_workers=0,   # data is in-memory; workers add IPC overhead
    max_grad_norm=1.0,
    remove_unused_columns=False,
)

# ── 7. Visibility callback ────────────────────────────────────────────────────

class VisibilityCallback:
    """
    Deep training diagnostics — logs to TensorBoard (live) and MLflow (persistent).

    Metrics emitted:
      training/token_efficiency      — fraction of non-masked tokens per batch
      lora_drift/{name}              — true L2 drift: ‖W - W₀‖ / ‖W₀‖
      lora_drift_mean/{module}       — per-module-type mean drift
      grad_norm_by_module/{module}   — per-module gradient norms (pre-zero_grad)
      lora_eff_rank/{name}           — nuclear_norm / frobenius_norm
      activation_norm/layer_{i}      — sampled every deep_every steps
      system/gpu_mem_allocated_gb    — VRAM in use
      system/steps_per_second        — training throughput
    """

    def __init__(self, tb_writer: SummaryWriter, log_every: int = 10, deep_every: int = 200):
        self.tb                                 = tb_writer
        self.log_every                          = log_every
        self.deep_every                         = deep_every
        self._last_time                         = time.time()
        self._last_step                         = 0
        # Gradient norms populated via backward hooks (before zero_grad wipes them)
        self._last_grad_norms: dict[str, float] = {}

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """
        Register persistent gradient hooks on all trainable params.
        on_step_end fires AFTER optimizer.zero_grad() so param.grad is None there.
        Hooks fire during backward — capturing actual gradient values.
        """
        if model is None:
            return
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            def _make_hook(n: str):
                def _hook(grad: torch.Tensor):
                    if grad is not None:
                        self._last_grad_norms[n] = grad.float().norm().item()
                return _hook
            param.register_hook(_make_hook(name))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step % self.log_every != 0:
            return

        # ── Token efficiency ─────────────────────────────────────────────────
        if hasattr(model, "_last_token_efficiency"):
            eff = model._last_token_efficiency
            self.tb.add_scalar("training/token_efficiency", eff, step)
            mlflow.log_metric("token_efficiency", eff, step=step)

        # ── True LoRA weight drift ───────────────────────────────────────────
        drift_by_type: dict[str, list[float]] = defaultdict(list)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" not in name and "lora_B" not in name:
                continue
            if name not in initial_lora_weights:
                continue
            W0    = initial_lora_weights[name].to(param.device)
            drift = (param.data.float() - W0.float()).norm().item()
            drift /= max(W0.float().norm().item(), 1e-8)
            self.tb.add_scalar(f"lora_drift/{name}", drift, step)
            for tag in ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"):
                if tag in name:
                    drift_by_type[tag].append(drift)
                    break

        for tag, drifts in drift_by_type.items():
            self.tb.add_scalar(f"lora_drift_mean/{tag}", float(np.mean(drifts)), step)

        # ── Per-module gradient norms (from backward hooks) ──────────────────
        grad_by_module: dict[str, list[float]] = defaultdict(list)
        for name, gnorm in self._last_grad_norms.items():
            for tag in ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"):
                if tag in name:
                    grad_by_module[tag].append(gnorm)
                    break
        for tag, gnorms in grad_by_module.items():
            self.tb.add_scalar(f"grad_norm_by_module/{tag}", float(np.mean(gnorms)), step)

        # ── GPU memory ───────────────────────────────────────────────────────
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated(0) / 1e9
            self.tb.add_scalar("system/gpu_mem_allocated_gb", mem_gb, step)
            mlflow.log_metric("gpu_mem_allocated_gb", mem_gb, step=step)

        # ── Throughput ───────────────────────────────────────────────────────
        now = time.time()
        sps = (step - self._last_step) / max(now - self._last_time, 1e-6)
        self.tb.add_scalar("system/steps_per_second", sps, step)
        self._last_time = now
        self._last_step = step

        # ── Deep diagnostics (expensive, less frequent) ──────────────────────
        if step % self.deep_every == 0:
            self._log_rank_utilization(model, step)
            self._log_layer_activation_norms(model, step)

    def _log_rank_utilization(self, model, step: int):
        """
        Effective rank = nuclear_norm / frobenius_norm.
        Low  → adapter collapsed to fewer dims than r.
        High → adapter fully utilising its rank budget.
        """
        for name, param in model.named_parameters():
            if "lora_B" not in name or not param.requires_grad:
                continue
            if param.numel() > 1_000_000:
                continue
            W = param.data.float()
            try:
                S            = torch.linalg.svdvals(W)
                nuclear_norm = S.sum().item()
                frob_norm    = W.norm().item()
                eff_rank     = nuclear_norm / max(frob_norm, 1e-8)
                self.tb.add_scalar(f"lora_eff_rank/{name}", eff_rank, step)
            except Exception:
                pass

    def _log_layer_activation_norms(self, model, step: int):
        """
        Hooks every 4th transformer layer, runs a single dummy token in
        eval + inference_mode (avoids gradient_checkpointing conflicts),
        logs activation norms, removes hooks, restores training mode.
        Overhead: ~1-2s per call.
        """
        hooks: list     = []
        act_norms: dict = {}

        def _make_hook(layer_name: str):
            def _hook(module, input, output):
                t = output[0] if isinstance(output, tuple) else output
                if isinstance(t, torch.Tensor):
                    act_norms[layer_name] = t.detach().float().norm().item()
            return _hook

        try:
            base_layers = model.base_model.model.model.layers
        except AttributeError:
            return

        for i, layer in enumerate(base_layers):
            if i % 4 == 0:
                hooks.append(layer.register_forward_hook(_make_hook(f"layer_{i}")))

        was_training = model.training
        model.eval()
        try:
            dummy = torch.tensor([[1]], device=next(model.parameters()).device)
            with torch.inference_mode():
                model(dummy)
            for lname, norm in act_norms.items():
                self.tb.add_scalar(f"activation_norm/{lname}", norm, step)
        except Exception:
            pass
        finally:
            for h in hooks:
                h.remove()
            if was_training:
                model.train()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        step = state.global_step
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v, step=step)

    def on_train_end(self, args, state, control, **kwargs):
        mlflow.log_metric("best_eval_loss", state.best_metric or 0.0)
        self.tb.flush()
        self.tb.close()


# ── 8. ProxyCallback ──────────────────────────────────────────────────────────

class ProxyCallback(TrainerCallback):
    """Bridges VisibilityCallback (plain class) into the TrainerCallback protocol."""

    def __init__(self, cb: VisibilityCallback):
        self.cb = cb

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.cb.on_train_begin(args, state, control, model=model, **kwargs)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.cb.on_step_end(args, state, control, model=model, **kwargs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.cb.on_evaluate(args, state, control, metrics=metrics, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        self.cb.on_train_end(args, state, control, **kwargs)


# ── 9. Wire everything together ───────────────────────────────────────────────
os.makedirs(TB_LOG_DIR, exist_ok=True)

tb_writer = SummaryWriter(log_dir=TB_LOG_DIR)
vis_cb    = VisibilityCallback(tb_writer, log_every=10, deep_every=200)
collator  = InstrumentedCollator(tokenizer=tokenizer, model_ref=model)

# ── 10. Train ─────────────────────────────────────────────────────────────────
mlflow.set_experiment("cs-chatbot-sft")

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_params({
        "model":                "Llama-3.1-8B-Instruct",
        "lora_r":               64,
        "lora_alpha":           128,
        "learning_rate":        2e-4,
        "epochs":               3,
        "effective_batch_size": 32,     # per_device(8) × grad_accum(4)
        "per_device_batch":     8,
        "grad_accum_steps":     4,
        "seq_len":              MAX_SEQ_LEN,
        "target_modules":       "q/k/v/o_proj+gate/up/down_proj",
    })

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[ProxyCallback(vis_cb)],
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR + "/final")
    tokenizer.save_pretrained(OUTPUT_DIR + "/final")
    mlflow.log_artifacts(OUTPUT_DIR + "/final", artifact_path="lora-adapter")

print("Training complete.")
print(f"TensorBoard: tensorboard --logdir {TB_LOG_DIR} --port 6006")
print(f"MLflow UI:   mlflow ui --port 5000")