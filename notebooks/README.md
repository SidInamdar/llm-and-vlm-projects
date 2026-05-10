# notebooks/

Exploratory notebooks — **not production code**.

## Convention

- Notebooks live here and **nowhere else**.
- Notebooks may import from `shared/` but **never from `projects/`**.
- If you find yourself writing reusable logic in a notebook, move it to `shared/`.
- Name notebooks with a date prefix: `YYYY-MM-DD_topic.ipynb`.

## Running

```bash
uv run jupyter lab
```
