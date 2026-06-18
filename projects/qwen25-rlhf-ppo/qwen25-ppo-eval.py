import argparse
import json
import os
import sys
import torch
from pathlib import Path
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

# ── Paths relative to parent foobar directory ─────────────────────────────────
REWARD_MODEL_PATH = "llm-and-vlm-projects/models/checkpoints/OpenAssistant--reward-model-deberta-v3-large-v2"
REWARD_MODEL_LENGTH = 512

POLICY_MODEL_NAME = "Qwen--Qwen2.5-7B-Instruct"
POLICY_MODEL_PATH = f"llm-and-vlm-projects/models/checkpoints/{POLICY_MODEL_NAME}"

EVAL_PROMPTS_PATH = "llm-and-vlm-projects/datasets/processed/alpaca_ppo_eval_prompts"
RESULTS_DIR = "llm-and-vlm-projects/projects/qwen25-rlhf-ppo/eval_results"


def load_reward_model() -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    import sys
    if getattr(sys, "_cached_reward_model", None) is not None:
        return sys._cached_reward_model, sys._cached_reward_tokenizer

    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH,
        num_labels=1,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
    n_params = sum(p.numel() for p in reward_model.parameters()) / 1e6
    print(f"  Reward model loaded on device ({n_params:.1f}M params, frozen)")
    sys._cached_reward_model = reward_model
    sys._cached_reward_tokenizer = reward_tokenizer
    return reward_model, reward_tokenizer 


@torch.no_grad()
def compute_rewards(
    texts: list[str], 
    reward_model: AutoModelForSequenceClassification,
    reward_tokenizer: AutoTokenizer,
    device: torch.device | int = 0,
    batch_size: int = 8, 
) -> list[torch.Tensor]:
    all_rewards : list[torch.Tensor] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = reward_tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=REWARD_MODEL_LENGTH,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        outputs = reward_model(**encodings)
        logits = outputs.logits.squeeze(-1)
        for j in range(logits.shape[0]):
            all_rewards.append(logits[j].detach().float())

    return all_rewards


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

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # ── 1. Load eval prompts ──────────────────────────────────────────────────
    print(f"\n[1/4] Loading held-out prompts from {EVAL_PROMPTS_PATH}...")
    eval_ds = load_from_disk(str(EVAL_PROMPTS_PATH))
    prompts: list[str] = eval_ds["prompt"]
    print(f"  Loaded {len(prompts)} held-out prompts")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print(f"\n[2/4] Loading model...")
    print(f"  Base  : {POLICY_MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if args.model_path:
        adapter_path = args.model_path
        if not Path(adapter_path).exists() and not Path(adapter_path).is_absolute():
            fallback_path = Path("llm-and-vlm-projects") / adapter_path
            if fallback_path.exists():
                adapter_path = str(fallback_path)
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
    reward_model, reward_tokenizer = load_reward_model()
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
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

    out_path = Path(RESULTS_DIR) / f"{args.eval_name}.json"
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
