# datasets/

This directory holds all datasets consumed by projects in this repo.
**The actual files are git-ignored.** Only structure and this README are tracked.

---

## Directory Structure

```
datasets/
├── raw/          ← original, immutable downloads — never modified
└── processed/    ← transformed / tokenised outputs ready for training
```

> **Rule:** Scripts always read from `raw/` and write to `processed/`. Never edit raw files in-place.

---

## Datasets

| Dataset | Version | Source | Stored in | Consumed by |
|---------|---------|--------|-----------|-------------|
| Bitext Customer Support | latest | [HuggingFace](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | `datasets/raw/bitext/` | `projects/llama-sft/` |
| MultiWOZ | v2.2 | [GitHub](https://github.com/budzianowski/multiwoz) | `datasets/raw/multiwoz/` | `projects/llama-sft/` |

---

## Downloading Data

### Using the project download script

```bash
# Download all datasets for llama-sft
uv run python projects/llama-sft/dataset-download.py --dataset all

# Download a specific dataset
uv run python projects/llama-sft/dataset-download.py --dataset bitext
```

### HuggingFace Hub (manual)

```bash
huggingface-cli download <org>/<dataset> --local-dir datasets/raw/<name>
```

### DVC (if configured)

```bash
dvc pull
```

---

## Adding a New Dataset

1. Create a subdirectory under `datasets/raw/<dataset-name>/`.
2. Add a row to the table above (source, version, consuming project).
3. Write a preprocessing script in the consuming project (e.g. `projects/<name>/preprocess.py`) that outputs to `datasets/processed/<dataset-name>/`.
4. Do **not** commit the files themselves — only commit this README update.
