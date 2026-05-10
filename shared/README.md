# shared/

Cross-project utilities. Anything that would otherwise be copy-pasted between projects lives here.

---

## Modules

| Module | Purpose |
|--------|---------|
| `dataloaders/` | Generic dataset wrappers, collation functions, samplers |
| `metrics/` | Evaluation metrics (perplexity, ROUGE, VQA accuracy, …) |
| `viz/` | Plotting helpers, attention maps, weight distribution charts |

---

## Import Convention

```python
# In any project script
from shared.dataloaders import CaptionDataset
from shared.metrics import vqa_accuracy
from shared.viz import plot_attention
```

The repo root must be on `PYTHONPATH`, or you run via `uv run` from the repo root (preferred).

---

## Rules

- `shared/` **must not** import from `projects/` — dependency flows one way only.
- Keep modules small and focused. If something is only used by one project, it belongs in that project.
- Every public function should have a docstring and type hints.
- Add tests under `shared/<module>/tests/` when logic is non-trivial.
