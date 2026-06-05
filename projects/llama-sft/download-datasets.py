import os
from typing import Any

import requests
from huggingface_hub import snapshot_download

from datasets import Dataset, DatasetDict, load_dataset

_MULTIWOZ_BASE = "https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.2"


def download_bitext(output_dir: str, cache_dir: str | None = None) -> None:
    """
    Download the Bitext customer support LLM chatbot training dataset
    and save it to disk.

    Args:
        output_dir: Root directory where the dataset will be saved.
        cache_dir: Optional cache directory for HuggingFace.
    """
    dataset_id = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    print(f"Downloading {dataset_id}...")
    ds = load_dataset(dataset_id, cache_dir=cache_dir)

    dest_path = os.path.join(output_dir, "bitext")
    os.makedirs(dest_path, exist_ok=True)
    ds.save_to_disk(dest_path)
    print(f"Saved {dataset_id} to {dest_path}")


def _extract_state(frame: dict[str, Any]) -> dict[str, Any]:
    """Extract the state sub-dict from a dialogue frame, with safe fallbacks."""
    if "state" not in frame:
        return {
            "active_intent": "",
            "requested_slots": [],
            "slots_values": {
                "slots_values_name": [],
                "slots_values_list": [],
            },
        }
    state = frame["state"]
    return {
        "active_intent": state["active_intent"],
        "requested_slots": state["requested_slots"],
        "slots_values": {
            "slots_values_name": list(state["slot_values"].keys()),
            "slots_values_list": list(state["slot_values"].values()),
        },
    }


def _extract_slot(slot: dict[str, Any]) -> dict[str, Any]:
    """Extract a normalised slot dict from a raw MultiWOZ slot."""
    has_copy = "copy_from" in slot
    return {
        "slot": slot["slot"],
        "value": "" if has_copy else slot["value"],
        "start": slot.get("start", -1),
        "exclusive_end": slot.get("exclusive_end", -1),
        "copy_from": slot.get("copy_from", ""),
        "copy_from_value": slot["value"] if has_copy else [],
    }


def _extract_dialogue_acts(
    mapped_acts: dict[str, Any],
    turn_id: str,
) -> dict[str, Any]:
    """Build the dialogue_acts sub-dict for a single turn."""
    turn_acts = mapped_acts.get(turn_id, {})

    dialog_act = [
        {
            "act_type": act_type,
            "act_slots": {
                "slot_name": [s for s, _ in act_slots],
                "slot_value": [v for _, v in act_slots],
            },
        }
        for act_type, act_slots in turn_acts.get("dialog_act", {}).items()
    ]

    span_info = [
        {
            "act_type": s[0],
            "act_slot_name": s[1],
            "act_slot_value": s[2],
            "span_start": s[3],
            "span_end": s[4],
        }
        for s in turn_acts.get("span_info", [])
    ]

    return {"dialog_act": dialog_act, "span_info": span_info}


def process_multiwoz_split(
    split_prefix: str,
    data_files: dict[str, Any],
    dialogue_acts: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Process the MultiWOZ raw dialogue JSON files for a given split prefix.

    Args:
        split_prefix: The prefix of the split (e.g. 'train', 'dev', 'test').
        data_files: A dictionary mapping filenames/keys to their loaded JSON content.
        dialogue_acts: Dialogue acts mapping from dialogue ID to acts.

    Returns:
        A list of processed records representing dialogue samples.
    """
    records: list[dict[str, Any]] = []
    file_keys = sorted(k for k in data_files if k.startswith(split_prefix))

    for key in file_keys:
        for dialogue in data_files[key]:
            mapped_acts = dialogue_acts.get(dialogue["dialogue_id"], {})
            record = {
                "dialogue_id": dialogue["dialogue_id"],
                "services": dialogue["services"],
                "turns": [
                    {
                        "turn_id": turn["turn_id"],
                        "speaker": turn["speaker"],
                        "utterance": turn["utterance"],
                        "frames": [
                            {
                                "service": frame["service"],
                                "state": _extract_state(frame),
                                "slots": [_extract_slot(slot) for slot in frame["slots"]],
                            }
                            for frame in turn["frames"]
                        ],
                        "dialogue_acts": _extract_dialogue_acts(mapped_acts, turn["turn_id"]),
                    }
                    for turn in dialogue["turns"]
                ],
            }
            records.append(record)
    return records


def _build_multiwoz_urls() -> list[tuple[str, str]]:
    """Build the list of (key, url) pairs for all MultiWOZ v2.2 files."""
    urls: list[tuple[str, str]] = [
        ("dialogue_acts", f"{_MULTIWOZ_BASE}/dialog_acts.json"),
    ]
    urls += [
        (f"train_{i:03d}", f"{_MULTIWOZ_BASE}/train/dialogues_{i:03d}.json") for i in range(1, 18)
    ]
    urls += [(f"dev_{i:03d}", f"{_MULTIWOZ_BASE}/dev/dialogues_{i:03d}.json") for i in range(1, 3)]
    urls += [
        (f"test_{i:03d}", f"{_MULTIWOZ_BASE}/test/dialogues_{i:03d}.json") for i in range(1, 3)
    ]
    return urls


def download_multiwoz(output_dir: str) -> None:
    """
    Download MultiWOZ v2.2 dialogue files directly from GitHub and build
    the dataset.

    Args:
        output_dir: Root directory where the dataset will be saved.
    """
    print("Downloading MultiWOZ v2.2 files...")

    data_files: dict[str, Any] = {}
    for name, url in _build_multiwoz_urls():
        print(f"  Downloading {name}...")
        r = requests.get(url)
        r.raise_for_status()
        data_files[name] = r.json()

    dialogue_acts = data_files["dialogue_acts"]

    print("Building MultiWOZ dataset splits...")
    ds = DatasetDict(
        {
            "train": Dataset.from_list(process_multiwoz_split("train", data_files, dialogue_acts)),
            "validation": Dataset.from_list(
                process_multiwoz_split("dev", data_files, dialogue_acts)
            ),
            "test": Dataset.from_list(process_multiwoz_split("test", data_files, dialogue_acts)),
        }
    )

    dest_path = os.path.join(output_dir, "multiwoz")
    os.makedirs(dest_path, exist_ok=True)
    ds.save_to_disk(dest_path)
    print(f"Saved MultiWOZ v2.2 to {dest_path}")


DATASET_REGISTRY: dict[str, Any] = {
    "bitext": download_bitext,
    "multiwoz": download_multiwoz,
}


# ── Model download ────────────────────────────────────────────────────────────

def download_model(
    repo_id: str,
    models_dir: str,
    hf_token: str | None = None,
) -> None:
    """
    Download a HuggingFace model and its tokenizer to the local models/ directory.

    Weights are saved to:   <models_dir>/checkpoints/<model-slug>/
    Tokenizer is saved to:  <models_dir>/tokenizers/<model-slug>/

    The model slug is the repo_id with '/' replaced by '--', e.g.
    'meta-llama/Llama-3.1-8B-Instruct' → 'meta-llama--Llama-3.1-8B-Instruct'.

    Uses ``huggingface_hub.snapshot_download`` so no GPU or torch is required
    at download time — safe to call in the CPU dataset-prep stage.

    Args:
        repo_id:    HuggingFace model repo ID, e.g. 'meta-llama/Llama-3.1-8B-Instruct'.
        models_dir: Root models directory (e.g. llm-and-vlm-projects/models/).
        hf_token:   Optional HuggingFace token for gated models (reads
                    HF_TOKEN env var automatically if not provided).
    """
    from transformers import AutoTokenizer

    slug = repo_id.replace("/", "--")
    weights_dir = os.path.join(models_dir, "checkpoints", slug)
    tokenizer_dir = os.path.join(models_dir, "tokenizers", slug)

    # ── Model weights ─────────────────────────────────────────────────────────
    if os.path.isdir(weights_dir) and any(
        f.endswith((".safetensors", ".bin", ".pt")) for f in os.listdir(weights_dir)
    ):
        print(f"Model weights already present at {weights_dir} — skipping download.")
    else:
        print(f"Downloading model weights: {repo_id}")
        print(f"  → destination: {weights_dir}")
        os.makedirs(weights_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=weights_dir,
            token=hf_token or os.environ.get("HF_TOKEN"),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(f"  ✓ Weights saved to {weights_dir}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    if os.path.isdir(tokenizer_dir) and os.listdir(tokenizer_dir):
        print(f"Tokenizer already present at {tokenizer_dir} — skipping download.")
    else:
        print(f"Downloading tokenizer: {repo_id}")
        print(f"  → destination: {tokenizer_dir}")
        os.makedirs(tokenizer_dir, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(
            repo_id,
            token=hf_token or os.environ.get("HF_TOKEN"),
        )
        tok.save_pretrained(tokenizer_dir)
        print(f"  ✓ Tokenizer saved to {tokenizer_dir}")


def download_dataset(
    name: str,
    output_dir: str,
    cache_dir: str | None = None,
) -> None:
    """
    Generic dataset downloader that downloads the requested dataset by name.

    Args:
        name: Name of the dataset to download ('bitext', 'multiwoz').
        output_dir: Target root directory for raw datasets.
        cache_dir: Optional cache directory (for Hugging Face datasets).

    Raises:
        ValueError: If the dataset name is not supported.
    """
    name_clean = name.strip().lower()
    if name_clean not in DATASET_REGISTRY:
        supported = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Dataset '{name}' is not supported. Choose from: {supported}")

    download_fn = DATASET_REGISTRY[name_clean]
    if name_clean == "bitext":
        download_fn(output_dir, cache_dir)
    else:
        download_fn(output_dir)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # Resolve paths relative to this file:
    # this file  → projects/llama-sft/download-datasets.py
    # repo root  → ../../  (llm-and-vlm-projects/)
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _RAW_DIR = str(_REPO_ROOT / "datasets" / "raw")
    _MODELS_DIR = str(_REPO_ROOT / "models")

    _DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    parser = argparse.ArgumentParser(
        description="Download SFT datasets and/or the base model from HuggingFace"
    )
    parser.add_argument(
        "--dataset",
        choices=["bitext", "multiwoz", "all", "none"],
        default="all",
        help="Which dataset(s) to download (default: all; use 'none' to skip datasets)",
    )
    parser.add_argument(
        "--output-dir",
        default=_RAW_DIR,
        help=f"Root directory for raw datasets (default: {_RAW_DIR})",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        metavar="REPO_ID",
        help=f"HuggingFace model repo ID to download (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--models-dir",
        default=_MODELS_DIR,
        help=f"Root directory for model weights and tokenizers (default: {_MODELS_DIR})",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip model download (datasets only)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help="HuggingFace token for gated models (falls back to HF_TOKEN env var)",
    )
    args_cli = parser.parse_args()

    # ── Datasets ──────────────────────────────────────────────────────────────
    if args_cli.dataset != "none":
        targets = (
            list(DATASET_REGISTRY.keys())
            if args_cli.dataset == "all"
            else [args_cli.dataset]
        )
        for name in targets:
            download_dataset(name=name, output_dir=args_cli.output_dir)

    # ── Model ─────────────────────────────────────────────────────────────────
    if not args_cli.no_model:
        download_model(
            repo_id=args_cli.model,
            models_dir=args_cli.models_dir,
            hf_token=args_cli.hf_token,
        )
