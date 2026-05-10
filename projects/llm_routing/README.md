# llm_routing

> One-line description of this project.

---

## Goal

_What are you trying to achieve? State the hypothesis or research question._

## Setup

```bash
# From repo root
uv sync
uv run python projects/llm_routing/train.py --config projects/llm_routing/config.yaml
```

## Data

| Split | Path | Notes |
|-------|------|-------|
| train | `data/processed/<name>/train/` | |
| val   | `data/processed/<name>/val/`   | |

See [`data/README.md`](../../data/README.md) for download instructions.

## Model

| Checkpoint | Path | Notes |
|------------|------|-------|
| best | `models/checkpoints/<run>/` | |

See [`models/README.md`](../../models/README.md) for download instructions.

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| | | |

## File Structure

```
llm_routing/
├── train.py
├── eval.py
├── config.yaml
└── README.md
```
