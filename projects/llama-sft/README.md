# llama-sft

> Supervised fine-tuning of Llama 3 on customer-support and task-oriented dialogue datasets.

---

## Goal

Fine-tune Llama 3 using SFT on the Bitext customer-support dataset and MultiWOZ v2.2 dialogue dataset to produce a model capable of multi-domain task-oriented conversations.

## Setup

```bash
# From repo root
uv sync

# Download datasets
uv run python projects/llama-sft/dataset-download.py --dataset all

# Run training (when implemented)
uv run python projects/llama-sft/train.py --config projects/llama-sft/config.yaml
```

## Data

| Split | Path | Notes |
|-------|------|-------|
| Bitext (train) | `datasets/raw/bitext/` | Customer support LLM chatbot training data |
| MultiWOZ (train/val/test) | `datasets/raw/multiwoz/` | Multi-domain Wizard-of-Oz dialogues v2.2 |

See [`datasets/README.md`](../../datasets/README.md) for download instructions.

## Model

| Checkpoint | Path | Notes |
|------------|------|-------|
| _(to be added)_ | `models/checkpoints/<run>/` | |

See [`models/README.md`](../../models/README.md) for download instructions.

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| _(to be added)_ | | |

## File Structure

```
llama-sft/
├── dataset-download.py   ← download datasets via CLI
├── llama3-sft.ipynb      ← exploration notebook
└── README.md
```
