# llm-and-vlm-projects

> Building models that hallucinate less than I do. Eventually.

A monorepo for LLM and VLM research projects. Data and model weights are **shared** across projects; code is **project-scoped**.

---

## Repository Layout

```
repo/
├── projects/          ← one directory per experiment / paper
│   └── llama-sft/
│
├── datasets/          ← entirely git-ignored; created on the fly
│   ├── raw/
│   └── processed/
│
├── models/            ← entirely git-ignored; created on the fly
│   ├── checkpoints/
│   ├── configs/
│   └── tokenizers/
│
└── shared/            ← flat utility modules reused across projects
    ├── mlm_dataset.py
    ├── clm_dataset_download.py
    ├── roberta.py
    ├── metrics.py
    ├── mlm_benchmark.py
    └── tests/
```

## Key Conventions

| Rule | Rationale |
|------|-----------|
| Each project is self-contained under `projects/<name>/` | Namespacing prevents experiment bleed |
| `datasets/` and `models/` are entirely git-ignored | Use project scripts to download on the fly |
| `shared/` is the only cross-project import boundary | No copy-pasting utilities |

## Getting Started

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync the environment
uv sync

# Run a project script
uv run python projects/llama-sft/dataset-download.py --dataset all
```

## Asset Management

Large files (datasets, checkpoints) are tracked via **DVC** or stored on **HuggingFace Hub**.
These directories are entirely git-ignored and populated on the fly by project download scripts.
