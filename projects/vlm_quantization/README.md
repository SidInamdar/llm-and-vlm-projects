# vlm_quantization

> One-line description of this project.

---

## Goal

_What are you trying to achieve? State the hypothesis or research question._

## Setup

```bash
# From repo root
uv sync
uv run python projects/vlm_quantization/train.py --config projects/vlm_quantization/config.yaml
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
vlm_quantization/
├── train.py        ← training entry point
├── eval.py         ← evaluation entry point
├── config.yaml     ← hyperparameters and paths
└── README.md
```
