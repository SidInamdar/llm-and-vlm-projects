# llm-and-vlm-projects

> Building models that hallucinate less than I do. Eventually.

A monorepo for LLM and VLM research projects. Data and model weights are **shared** across projects; code is **project-scoped**.

---

## Repository Layout

```
repo/
├── projects/          ← one directory per experiment / paper
│   ├── vlm_quantization/
│   └── llm_routing/
│
├── data/              ← NOT tracked by git (see data/README.md)
│   ├── raw/
│   └── processed/
│
├── models/            ← NOT tracked by git (see models/README.md)
│   ├── checkpoints/
│   └── configs/
│
├── shared/            ← utilities reused across projects
│   ├── dataloaders/
│   ├── metrics/
│   └── viz/
│
└── notebooks/         ← exploration only; no imports from projects/
```

## Key Conventions

| Rule | Rationale |
|------|-----------|
| Each project is self-contained under `projects/<name>/` | Namespacing prevents experiment bleed |
| `data/` and `models/` are git-ignored | Use DVC / HF Hub / S3 for assets |
| `shared/` is the only cross-project import boundary | No copy-pasting utilities |
| `notebooks/` never imports from `projects/` | One-way dependency only |

## Getting Started

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync the environment
uv sync

# Run a project script
uv run python projects/vlm_quantization/train.py
```

## Asset Management

Large files (datasets, checkpoints) are tracked via **DVC** or stored on **HuggingFace Hub**.
See [`data/README.md`](data/README.md) and [`models/README.md`](models/README.md) for download instructions.
