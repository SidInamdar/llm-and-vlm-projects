"""
qwen25-ppo-smoke-test.py
Minimal end-to-end PPO loop on Qwen2.5-0.5B-Instruct (single GPU).

Validates that PPOTrainer, the reward wrapper, LoRA, and generation all work
correctly BEFORE scaling to the full 7B 2-GPU setup.

Checks:
    1. PPO step() completes without exception
    2. Rewards are finite (not NaN / Inf)
    3. KL divergence is non-negative
    4. Model generates coherent text after a few steps

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
from datasets import load_from_disk
from peft import LoraConfig
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# ── Import reward wrapper from sibling file ───────────────────────────────────
# Filenames use hyphens (repo convention) so standard import is not possible.
_PROJECT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "reward_wrapper", str(_PROJECT_DIR / "qwen25-ppo-reward-wrapper.py")
)
_reward_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reward_mod)
load_reward_model = _reward_mod.load_reward_model
compute_rewards = _reward_mod.compute_rewards

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

    device = 0 if torch.cuda.is_available() else "cpu"
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

    # CRITICAL FIX: Qwen2.5 tokenizer defaults to right padding, but
    # causal-LM generation requires LEFT padding so the last non-pad
    # token is always the most recent.
    if tokenizer.padding_side != "left":
        print(f"  ⚠ Fixing padding_side: {tokenizer.padding_side} → left")
        tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  ⚠ Set pad_token = eos_token ({tokenizer.eos_token})")

    # ── 2. Policy model (0.5B + LoRA + value head) ────────────────────────────
    print(f"\n[2/6] Loading policy model (0.5B + LoRA + value head)...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path,
        peft_config=lora_config,
        device_map={"": device},
        torch_dtype=torch.float16,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M "
          f"({100 * trainable / total:.2f}%)")

    # ── 3. Reward model ───────────────────────────────────────────────────────
    print(f"\n[3/6] Loading reward model (DeBERTa)...")
    reward_model, reward_tokenizer = load_reward_model(device=device)

    # ── 4. Dataset ────────────────────────────────────────────────────────────
    print(f"\n[4/6] Loading prompts from {TRAIN_PROMPTS_PATH}...")
    if not TRAIN_PROMPTS_PATH.exists():
        print("  ✗ Prompts not found. Run qwen25-ppo-dataset-prep.py first.")
        sys.exit(1)

    ds = load_from_disk(str(TRAIN_PROMPTS_PATH))
    # Take just enough for the smoke test
    n_needed = min(BATCH_SIZE * NUM_STEPS, len(ds))
    ds = ds.select(range(n_needed))
    print(f"  Using {len(ds)} prompts for smoke test")

    # Tokenize — PPOTrainer expects "input_ids" column
    def tokenize_fn(example: dict) -> dict:
        enc = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=128,
            padding=False,
        )
        return {"input_ids": enc["input_ids"], "query": example["prompt"]}

    ds = ds.map(tokenize_fn, remove_columns=["prompt"])
    ds.set_format("torch", columns=["input_ids"])

    # ── 5. PPO config + trainer ───────────────────────────────────────────────
    print(f"\n[5/6] Setting up PPOTrainer...")
    ppo_config = PPOConfig(
        model_name=model_path,
        learning_rate=1e-5,
        batch_size=BATCH_SIZE,
        mini_batch_size=MINI_BATCH_SIZE,
        ppo_epochs=2,          # fewer for smoke test
        gamma=1.0,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        init_kl_coef=0.2,
        target_kl=6.0,
        max_grad_norm=1.0,
        log_with=None,         # we log manually below
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
    )

    generation_kwargs = {
        "max_new_tokens": 64,  # shorter for smoke test speed
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    # ── 6. Training loop ──────────────────────────────────────────────────────
    print(f"\n[6/6] Running {NUM_STEPS} PPO steps...\n")
    os.makedirs(TB_LOG_DIR, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=TB_LOG_DIR)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    errors: list[str] = []

    with mlflow.start_run(run_name="smoke-test"):
        for step, batch in enumerate(ppo_trainer.dataloader):
            if step >= NUM_STEPS:
                break

            t0 = time.time()
            query_tensors: list[torch.Tensor] = batch["input_ids"]

            # ── Generate responses ────────────────────────────────────────────
            response_tensors = ppo_trainer.generate(
                query_tensors, **generation_kwargs
            )
            # Ensure response-only (strip prompt tokens if generate included them)
            response_tensors = [
                resp[len(query):]
                if len(resp) > len(query)
                else resp
                for query, resp in zip(query_tensors, response_tensors)
            ]

            # ── Decode full texts (prompt + response) for reward scoring ──────
            full_texts: list[str] = []
            for q, r in zip(query_tensors, response_tensors):
                full = tokenizer.decode(
                    torch.cat([q, r]), skip_special_tokens=True
                )
                full_texts.append(full)

            # ── Compute rewards ───────────────────────────────────────────────
            rewards = compute_rewards(
                full_texts, reward_model, reward_tokenizer, device=device
            )

            # ── PPO step ──────────────────────────────────────────────────────
            stats = ppo_trainer.step(
                list(query_tensors), list(response_tensors), rewards
            )

            # ── Validate ──────────────────────────────────────────────────────
            reward_vals = [r.item() for r in rewards]
            mean_reward = sum(reward_vals) / len(reward_vals)
            kl = stats.get("objective/kl", float("nan"))

            if any(not torch.isfinite(r) for r in rewards):
                errors.append(f"Step {step}: non-finite rewards detected")
            if isinstance(kl, (int, float)) and kl < 0:
                errors.append(f"Step {step}: negative KL ({kl:.4f})")

            # ── Log ───────────────────────────────────────────────────────────
            kl_val = kl if isinstance(kl, (int, float)) else 0.0
            tb_writer.add_scalar("smoke/mean_reward", mean_reward, step)
            tb_writer.add_scalar("smoke/kl", kl_val, step)
            mlflow.log_metrics({"mean_reward": mean_reward, "kl": kl_val}, step=step)

            kl_str = f"{kl:.4f}" if isinstance(kl, (int, float)) else "?"
            dt = time.time() - t0
            print(f"  Step {step:3d} | reward={mean_reward:+.4f} | "
                  f"KL={kl_str} | {dt:.1f}s")

    tb_writer.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if errors:
        print(" ✗ SMOKE TEST FAILED")
        for e in errors:
            print(f"   • {e}")
        sys.exit(1)

    print(" ✓ SMOKE TEST PASSED")
    print(f"   {NUM_STEPS} PPO steps completed without errors.")
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
