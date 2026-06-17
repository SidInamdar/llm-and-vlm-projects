"""
qwen25-ppo-smoke-test.py
Minimal end-to-end PPO loop on Qwen2.5-0.5B-Instruct (single GPU).

Validates that PPOTrainer, the reward wrapper, LoRA, and generation all work
correctly BEFORE scaling to the full 7B 2-GPU setup.

Usage:
    uv run python projects/qwen25-rlhf-ppo/qwen25-ppo-smoke-test.py
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlflow
import torch
import torch.nn as nn
from datasets import load_from_disk
from peft import LoraConfig
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

try:
    from trl import PPOConfig, PPOTrainer
except ImportError:
    from trl.experimental.ppo import PPOConfig, PPOTrainer

# ── Import reward wrapper from sibling file ───────────────────────────────────
# Filenames use hyphens (repo convention) so standard import is not possible.
_PROJECT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "reward_wrapper", str(_PROJECT_DIR / "qwen25-ppo-reward-wrapper.py")
)
_reward_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reward_mod)
load_reward_model = _reward_mod.load_reward_model

# ── Reward Wrapper for Tokenization ───────────────────────────────────────────
class RewardModelWrapper(nn.Module):
    def __init__(self, reward_model, reward_tokenizer, policy_tokenizer, device):
        super().__init__()
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.policy_tokenizer = policy_tokenizer
        self.device = device

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.reward_model, name)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Decode policy tokens
        full_texts = self.policy_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # Re-encode with reward tokenizer
        inputs = self.reward_tokenizer(
            full_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Compute rewards
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            
        if hasattr(outputs, "logits"):
            return outputs.logits
        elif isinstance(outputs, tuple):
            return outputs[0]
        return outputs

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PROMPTS_PATH = _REPO_ROOT / "datasets" / "processed" / "alpaca_ppo_prompts"

# ── Config ────────────────────────────────────────────────────────────────────
POLICY_MODEL_HUB = "Qwen/Qwen2.5-0.5B-Instruct"
_LOCAL_POLICY = _REPO_ROOT / "models" / "checkpoints" / "Qwen--Qwen2.5-0.5B-Instruct"

NUM_STEPS = 10
BATCH_SIZE = 4
MINI_BATCH_SIZE = 2
TB_LOG_DIR = str(_PROJECT_DIR / "logs" / "tb-smoke-test")
MLFLOW_EXPERIMENT = "qwen25-rlhf-ppo-smoke-test"


def main() -> None:
    print("=" * 60)
    print(" PPO Smoke Test — Qwen2.5-0.5B-Instruct (single GPU)")
    print("=" * 60)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  VRAM  : {vram:.1f} GB")
    else:
        print("  ⚠ No GPU — running on CPU (will be slow)")

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    model_path = str(_LOCAL_POLICY) if _LOCAL_POLICY.exists() else POLICY_MODEL_HUB
    print(f"\n[1/6] Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.padding_side != "left":
        print(f"  ⚠ Fixing padding_side: {tokenizer.padding_side} → left")
        tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  ⚠ Set pad_token = eos_token ({tokenizer.eos_token})")

    # ── 2. Policy model ───────────────────────────────────────────────────────
    print(f"\n[2/6] Loading policy model (0.5B + LoRA)...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": device},
        torch_dtype=torch.float16,
    )

    # ── 3. Reward and Value Models ────────────────────────────────────────────
    print(f"\n[3/6] Loading reward and value models...")
    
    # Value Model (required by TRL 1.5.1)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    value_model.config.pad_token_id = tokenizer.pad_token_id

    # Reward Model
    reward_model, reward_tokenizer = load_reward_model(device=device)
    reward_wrapper = RewardModelWrapper(reward_model, reward_tokenizer, tokenizer, device)

    # ── 4. Dataset ────────────────────────────────────────────────────────────
    print(f"\n[4/6] Loading prompts from {TRAIN_PROMPTS_PATH}...")
    if not TRAIN_PROMPTS_PATH.exists():
        print("  ✗ Prompts not found. Run qwen25-ppo-dataset-prep.py first.")
        sys.exit(1)

    ds = load_from_disk(str(TRAIN_PROMPTS_PATH))
    n_needed = min(BATCH_SIZE * NUM_STEPS, len(ds))
    ds = ds.select(range(n_needed))
    print(f"  Using {len(ds)} prompts for smoke test")

    def tokenize_fn(example: dict) -> dict:
        enc = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=128,
            padding=False,
        )
        return {"input_ids": enc["input_ids"]}

    ds = ds.map(tokenize_fn, remove_columns=["prompt"])
    ds.set_format("torch", columns=["input_ids"])

    # ── 5. PPO config + trainer ───────────────────────────────────────────────
    print(f"\n[5/6] Setting up PPOTrainer...")
    os.makedirs(TB_LOG_DIR, exist_ok=True)
    
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        per_device_train_batch_size=BATCH_SIZE,
        mini_batch_size=MINI_BATCH_SIZE,
        num_ppo_epochs=2,
        gamma=1.0,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        kl_coef=0.2,
        max_grad_norm=1.0,
        report_to="tensorboard",
        logging_dir=TB_LOG_DIR,
        max_steps=NUM_STEPS,
    )

    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=None,
        reward_model=reward_wrapper,
        train_dataset=ds,
        value_model=value_model,
        peft_config=lora_config,
    )

    # ── 6. Training loop ──────────────────────────────────────────────────────
    print(f"\n[6/6] Running PPO training...\n")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="smoke-test"):
        ppo_trainer.train()

    print("\n" + "=" * 60)
    print(" ✓ SMOKE TEST PASSED")
    print(f"   PPO steps completed.")
    print(f"   TensorBoard logs: {TB_LOG_DIR}")

    # ── Post-training generation samples ──────────────────────────────────────
    print("\n Sample generations (post-training):")
    sample_prompts = [
        "What is machine learning?",
        "Explain gravity in simple terms.",
        "Write a short poem about the ocean.",
    ]
    for prompt in sample_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=64, temperature=0.7, do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        response = text[len(prompt):].strip()
        print(f"  Q: {prompt}")
        print(f"  A: {response[:150]}...")
        print()

    print("=" * 60)
    print(" Smoke test complete. Safe to proceed to 7B 2-GPU training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
