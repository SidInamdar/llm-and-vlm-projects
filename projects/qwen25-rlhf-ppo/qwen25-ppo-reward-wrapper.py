"""
qwen25-ppo-reward-wrapper.py
Wrapper for OpenAssistant/reward-model-deberta-v3-large-v2.

The reward model is a DeBERTa-v3-large with a single-output classification head
(num_labels=1).  It outputs **raw logits** (not probabilities) — higher = better.

Provides:
    load_reward_model(device)  → (model, tokenizer)
    compute_rewards(texts, model, tokenizer, device) → List[torch.Tensor]

The List[torch.Tensor] return format is what PPOTrainer.step() expects for the
``rewards`` argument — one scalar tensor per sample.

Can also be run standalone to smoke-test the reward model:
    uv run python projects/qwen25-rlhf-ppo/qwen25-ppo-reward-wrapper.py
"""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
REWARD_MODEL_HUB = "OpenAssistant/reward-model-deberta-v3-large-v2"
_LOCAL_REWARD_PATH = (
    _REPO_ROOT / "models" / "checkpoints" / "OpenAssistant--reward-model-deberta-v3-large-v2"
)

# DeBERTa-v3-large has a 512-token context window.
REWARD_MAX_LENGTH = 512


def load_reward_model(
    device: torch.device | int = 0,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load the frozen reward model and its tokenizer onto *device*.

    Checks for a local cache under ``models/checkpoints/`` first; falls back
    to the HuggingFace Hub identifier if not found.
    """
    model_path = str(_LOCAL_REWARD_PATH) if _LOCAL_REWARD_PATH.exists() else REWARD_MODEL_HUB

    print(f"Loading reward model from: {model_path}")
    reward_tokenizer = AutoTokenizer.from_pretrained(model_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        device_map={"": device},
        torch_dtype=torch.float16,
    )

    # Freeze — reward model is never trained.
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    n_params = sum(p.numel() for p in reward_model.parameters()) / 1e6
    print(f"  Reward model loaded on device {device} ({n_params:.1f}M params, frozen)")
    return reward_model, reward_tokenizer


@torch.no_grad()
def compute_rewards(
    texts: list[str],
    reward_model: AutoModelForSequenceClassification,
    reward_tokenizer: AutoTokenizer,
    device: torch.device | int = 0,
    batch_size: int = 8,
) -> list[torch.Tensor]:
    """
    Score a list of texts with the reward model.

    Returns
    -------
    list[torch.Tensor]
        One **scalar** float32 tensor per input text.  This is the exact format
        PPOTrainer.step() expects for its ``rewards`` argument.

    Notes
    -----
    The DeBERTa reward model uses a ``num_labels=1`` classification head.
    ``model(**inputs).logits`` has shape ``(batch, 1)``; we squeeze and
    detach to scalar tensors.
    """
    all_rewards: list[torch.Tensor] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = reward_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=REWARD_MAX_LENGTH,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        outputs = reward_model(**encodings)
        # outputs.logits shape: (batch_size, 1) → squeeze to (batch_size,)
        logits = outputs.logits.squeeze(-1)

        for j in range(logits.size(0)):
            all_rewards.append(logits[j].detach().float())

    return all_rewards


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(" Reward Model Smoke Test")
    print("=" * 60)

    _device: torch.device | int = 0 if torch.cuda.is_available() else "cpu"
    rm, rt = load_reward_model(device=_device)

    samples = [
        "The capital of France is Paris. It is known for the Eiffel Tower and rich cultural heritage.",
        "asdkjh asdkjh asdk the the the the the the the the",
        "I appreciate your question. Let me provide a detailed and helpful response to that.",
    ]

    rewards = compute_rewards(samples, rm, rt, device=_device)

    print("\nResults:")
    for text, reward in zip(samples, rewards):
        print(f"  Reward: {reward.item():+.4f}  |  {text[:70]}...")

    # Sanity: good text should score higher than garbage
    assert rewards[0].item() > rewards[1].item(), (
        "Sanity check failed: coherent text scored lower than garbage"
    )
    print("\n✓ Reward model wrapper works correctly.")
