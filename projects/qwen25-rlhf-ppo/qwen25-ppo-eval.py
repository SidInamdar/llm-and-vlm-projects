"""
qwen25-ppo-eval.py
Evaluate a model on the 100 held-out Alpaca prompts.

Generates responses, scores them with the DeBERTa reward model, and saves
the reward distribution as JSON.  Used three times:
    1. Baseline — unmodified SFT model (no LoRA)
    2. Post-main PPO — after training with init_kl_coef=0.2
    3. Post-ablation — after training with init_kl_coef=0.0

Usage:
    # Baseline (no LoRA adapter)
    python qwen25-ppo-eval.py --eval-name baseline

    # Post-main PPO
    python qwen25-ppo-eval.py \\
        --model-path models/checkpoints/qwen25-7b-ppo-main \\
        --eval-name post-main

    # Post-ablation
    python qwen25-ppo-eval.py \\
        --model-path models/checkpoints/qwen25-7b-ppo-ablation \\
        --eval-name post-ablation
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Import reward wrapper ─────────────────────────────────────────────────────
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
EVAL_PROMPTS_PATH = _REPO_ROOT / "datasets" / "processed" / "alpaca_ppo_eval_prompts"

BASE_MODEL_HUB = "Qwen/Qwen2.5-7B-Instruct"
_LOCAL_BASE = _REPO_ROOT / "models" / "checkpoints" / "Qwen--Qwen2.5-7B-Instruct"

RESULTS_DIR = _PROJECT_DIR / "eval_results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate model on held-out prompts")
    p.add_argument("--model-path", type=str, default=None,
                    help="Path to LoRA adapter checkpoint (omit for baseline)")
    p.add_argument("--eval-name", type=str, required=True,
                    help="Name for this evaluation (baseline, post-main, post-ablation)")
    p.add_argument("--device", type=int, default=0,
                    help="GPU device index")
    p.add_argument("--batch-size", type=int, default=8,
                    help="Generation batch size")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(f" Held-Out Evaluation — {args.eval_name}")
    print("=" * 60)

    device = args.device

    # ── 1. Load eval prompts ──────────────────────────────────────────────────
    print(f"\n[1/4] Loading held-out prompts from {EVAL_PROMPTS_PATH}...")
    if not EVAL_PROMPTS_PATH.exists():
        print("  ✗ Eval prompts not found. Run qwen25-ppo-dataset-prep.py first.")
        sys.exit(1)

    eval_ds = load_from_disk(str(EVAL_PROMPTS_PATH))
    prompts: list[str] = eval_ds["prompt"]
    print(f"  Loaded {len(prompts)} held-out prompts")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    base_path = str(_LOCAL_BASE) if _LOCAL_BASE.exists() else BASE_MODEL_HUB

    print(f"\n[2/4] Loading model...")
    print(f"  Base  : {base_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map={"": device},
        torch_dtype=torch.float16,
    )

    if args.model_path:
        adapter_path = args.model_path
        # If relative, resolve from repo root
        if not Path(adapter_path).is_absolute():
            adapter_path = str(_REPO_ROOT / adapter_path)
        print(f"  LoRA  : {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("  LoRA  : None (baseline)")
        model = base_model

    model.eval()

    # ── 3. Generate responses ─────────────────────────────────────────────────
    print(f"\n[3/4] Generating responses (batch_size={args.batch_size})...")
    generation_kwargs = {
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    all_responses: list[str] = []
    all_full_texts: list[str] = []

    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i : i + args.batch_size]
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        for j, prompt in enumerate(batch_prompts):
            full_text = tokenizer.decode(outputs[j], skip_special_tokens=True)
            response = full_text[len(prompt):].strip()
            all_responses.append(response)
            all_full_texts.append(full_text)

        done = min(i + args.batch_size, len(prompts))
        print(f"  Generated {done}/{len(prompts)}", end="\r")

    print(f"  Generated {len(all_responses)} responses")

    # ── 4. Score with reward model ────────────────────────────────────────────
    print(f"\n[4/4] Scoring with reward model...")
    reward_model, reward_tokenizer = load_reward_model(device=device)
    rewards = compute_rewards(
        all_full_texts, reward_model, reward_tokenizer, device=device
    )
    reward_values = [r.item() for r in rewards]

    # ── Summary stats ─────────────────────────────────────────────────────────
    import statistics

    mean_r = statistics.mean(reward_values)
    std_r = statistics.stdev(reward_values) if len(reward_values) > 1 else 0.0
    min_r = min(reward_values)
    max_r = max(reward_values)
    median_r = statistics.median(reward_values)

    print(f"\n{'─' * 50}")
    print(f" Reward Distribution — {args.eval_name}")
    print(f"{'─' * 50}")
    print(f"  Mean   : {mean_r:+.4f}")
    print(f"  Std    : {std_r:.4f}")
    print(f"  Median : {median_r:+.4f}")
    print(f"  Min    : {min_r:+.4f}")
    print(f"  Max    : {max_r:+.4f}")
    print(f"{'─' * 50}")

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "eval_name": args.eval_name,
        "model_path": args.model_path or "baseline (no LoRA)",
        "num_prompts": len(prompts),
        "summary": {
            "mean": mean_r,
            "std": std_r,
            "median": median_r,
            "min": min_r,
            "max": max_r,
        },
        "per_sample": [
            {
                "prompt": prompts[i],
                "response": all_responses[i],
                "reward": reward_values[i],
            }
            for i in range(len(prompts))
        ],
    }

    out_path = RESULTS_DIR / f"{args.eval_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # ── Sample outputs ────────────────────────────────────────────────────────
    print(f"\n Top 3 responses (by reward):")
    sorted_indices = sorted(range(len(reward_values)), key=lambda i: reward_values[i], reverse=True)
    for rank, idx in enumerate(sorted_indices[:3]):
        print(f"  [{rank + 1}] reward={reward_values[idx]:+.4f}")
        print(f"      Q: {prompts[idx][:80]}...")
        print(f"      A: {all_responses[idx][:120]}...")
        print()

    print(f" Bottom 3 responses (by reward):")
    for rank, idx in enumerate(sorted_indices[-3:]):
        print(f"  [{rank + 1}] reward={reward_values[idx]:+.4f}")
        print(f"      Q: {prompts[idx][:80]}...")
        print(f"      A: {all_responses[idx][:120]}...")
        print()

    print("=" * 60)
    print(f" Evaluation complete — {args.eval_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
