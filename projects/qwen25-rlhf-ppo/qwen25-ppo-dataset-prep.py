"""
qwen25-ppo-dataset-prep.py
Download tatsu-lab/alpaca, extract prompts, split into train + 100-prompt
held-out eval subset.

Prompts = instruction (+ input if non-empty). The "output" field is ignored —
PPO generates its own responses.

Outputs:
    datasets/processed/alpaca_ppo_prompts        — training prompts
    datasets/processed/alpaca_ppo_eval_prompts   — 100 fixed held-out prompts

Usage:
    uv run python projects/qwen25-rlhf-ppo/qwen25-ppo-dataset-prep.py
"""

from pathlib import Path

from datasets import load_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = _REPO_ROOT / "datasets" / "processed"
TRAIN_SAVE_PATH = PROCESSED_DIR / "alpaca_ppo_prompts"
EVAL_SAVE_PATH = PROCESSED_DIR / "alpaca_ppo_eval_prompts"

# Raw dataset — pre-downloaded by qwen25-ppo-download.sh on a login node.
ALPACA_HUB = "tatsu-lab/alpaca"
_LOCAL_ALPACA = _REPO_ROOT / "datasets" / "raw" / "tatsu-lab--alpaca"

SEED = 42
EVAL_SIZE = 100


def extract_prompt(example: dict) -> dict:
    """Combine instruction + input (if non-empty) into a single prompt string."""
    prompt = example["instruction"].strip()
    inp = example.get("input", "").strip()
    if inp:
        prompt += f"\n\nInput: {inp}"
    return {"prompt": prompt}


def main() -> None:
    print("=" * 60)
    print(" Alpaca Dataset Preparation for PPO")
    print("=" * 60)

    # ── Download / load ─────────────────────────────────────────────────────────
    dataset_path = str(_LOCAL_ALPACA) if _LOCAL_ALPACA.exists() else ALPACA_HUB
    print(f"\n[1/3] Loading tatsu-lab/alpaca from {dataset_path}...")
    ds = load_dataset(dataset_path, split="train")
    print(f"  Total examples: {len(ds)}")

    # ── Extract prompts (drop output) ─────────────────────────────────────────
    print("\n[2/3] Extracting prompts...")
    ds = ds.map(extract_prompt, remove_columns=ds.column_names, num_proc=4)
    print(f"  Extracted {len(ds)} prompts")

    # ── Split: 100 held-out eval, rest for training ───────────────────────────
    print("\n[3/3] Splitting into train + held-out eval...")
    split = ds.train_test_split(test_size=EVAL_SIZE, seed=SEED)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  Train: {len(train_ds)} | Eval (held-out): {len(eval_ds)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(TRAIN_SAVE_PATH))
    eval_ds.save_to_disk(str(EVAL_SAVE_PATH))

    print(f"\nSaved:")
    print(f"  Train prompts : {TRAIN_SAVE_PATH}")
    print(f"  Eval prompts  : {EVAL_SAVE_PATH}")

    # ── Preview ───────────────────────────────────────────────────────────────
    print(f"\nSample held-out prompts:")
    for i in range(min(5, len(eval_ds))):
        preview = eval_ds[i]["prompt"][:120].replace("\n", " ")
        print(f"  [{i}] {preview}...")

    print("\n" + "=" * 60)
    print(" Dataset preparation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
