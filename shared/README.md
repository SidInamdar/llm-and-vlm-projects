# shared/

Cross-project utilities. Anything that would otherwise be copy-pasted between projects lives here.

---

## Modules

| Module | Purpose |
|--------|---------|
| `mlm_dataset.py` | Tokenises sentences into a HuggingFace Dataset for MLM training |
| `clm_dataset_download.py` | Generic dataset downloader with registry (Bitext, MultiWOZ) |
| `roberta.py` | Download, save, and reload RoBERTa model + tokenizer |
| `metrics.py` | MLM perplexity and token-length distribution analysis |
| `mlm_benchmark.py` | Verification sentence mining and fill-mask benchmarking |

---

## Import Convention

```python
# In any project script
from shared import tokenise_sentences, load_roberta
from shared import compute_mlm_perplexity, token_length_distribution
from shared import mine_verification_sentences, run_benchmarks
from shared import download_dataset

# Or import from specific modules
from shared.metrics import compute_mlm_perplexity
from shared.roberta import load_roberta
```

The repo root must be on `PYTHONPATH`, or you run via `uv run` from the repo root (preferred).

---

## Rules

- `shared/` **must not** import from `projects/` — dependency flows one way only.
- Keep modules small and focused. If something is only used by one project, it belongs in that project.
- Every public function should have a docstring and type hints.
- Add tests under `shared/tests/` when logic is non-trivial.
