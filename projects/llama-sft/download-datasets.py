import os
from typing import Any

import requests

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


#download_dataset(name='bitext',output_dir='/home/siddhesh/Documents/repos/llm-and-vlm-projects/datasets/raw')
download_dataset(name='multiwoz', output_dir='/home/siddhesh/Documents/repos/llm-and-vlm-projects/datasets/raw')
