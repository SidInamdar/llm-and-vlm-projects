# AI Agent Instructions — llm-and-vlm-projects

This file is written for AI coding assistants. Read it before touching any code.

---

## Repo Purpose

A **research monorepo** for LLM and VLM experiments. The core design tension is:
- **Data and model weights** are shared across projects.
- **Code is project-scoped** — each experiment lives in its own directory.

---

## Toolchain

| Tool | Version / Notes |
|------|----------------|
| Python | 3.12 (pinned in `.python-version`) |
| Package manager | `uv` — use `uv run`, `uv add`, `uv sync` |
| Linter / formatter | `ruff` — run `uv run ruff check .` and `uv run ruff format .` |
| Test runner | `pytest` — `uv run pytest`; test paths are under `shared/` |
| Build backend | `hatchling` |

**Never use `pip` directly.** Always `uv add <pkg>` to manage deps.

---

## Directory Layout and Rules

```
repo/
├── projects/<name>/   ← experiment code (self-contained)
├── data/              ← git-ignored assets; only README and .gitkeep tracked
│   ├── raw/           ← immutable downloads, never edit in-place
│   └── processed/     ← outputs of preprocessing scripts
├── models/            ← git-ignored weights
│   ├── checkpoints/   ← saved weights (.pt, .safetensors, GGUF, …)
│   └── configs/       ← YAML/JSON configs ARE committed to git
├── shared/            ← cross-project utilities; installable Python package
│   ├── dataloaders/
│   ├── metrics/
│   └── viz/
└── notebooks/         ← exploration only
```

### `projects/`
- Each project is a **flat directory**: `train.py`, `eval.py`, `config.yaml`, `README.md`.
- Projects may import from `shared/` but **never from other projects/**.
- When creating a new project, copy the README template from an existing one and fill in the dataset/checkpoint tables.
- Add a project-level `requirements.txt` or extend `pyproject.toml` `[project.optional-dependencies]` only for heavy GPU deps that not every project needs.

### `shared/`
- This is a proper Python package (`from shared.metrics import ...`).
- It is installed editably via `uv sync` — no `sys.path` hacks needed.
- `shared/` **must not** import from `projects/` — one-way dependency only.
- Every public symbol must have a docstring and type hints.
- Non-trivial logic gets a test under `shared/<module>/tests/`.

### `data/` and `models/`
- **Never commit weight files or raw datasets.** The gitignore covers `*.pt`, `*.safetensors`, `*.ckpt`, `*.bin`, `data/raw/*`, `data/processed/*`, `models/checkpoints/*`.
- `models/configs/` **is** committed — it is the reproducibility record.
- When a new dataset or checkpoint is introduced, update the registry table in `data/README.md` or `models/README.md` respectively.

### `notebooks/`
- Notebooks may import from `shared/` only.
- Name notebooks `YYYY-MM-DD_short_description.ipynb`.
- If reusable logic emerges in a notebook, extract it to `shared/`.

---

## Running Code

```bash
# Sync environment (run this after any pyproject.toml change)
uv sync --extra dev

# Run a project entry point
uv run python projects/vlm_quantization/train.py --config projects/vlm_quantization/config.yaml

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Tests
uv run pytest

# Jupyter
uv run jupyter lab
```

---

## Adding a New Project

1. `mkdir -p projects/<project_name>`
2. Copy README template from `projects/vlm_quantization/README.md`, fill tables.
3. Write `train.py`, `eval.py`, `config.yaml`.
4. If the project needs its own dependencies, add them with `uv add --optional <project_name> <pkg>`.
5. Register any datasets/checkpoints it uses in `data/README.md` and `models/README.md`.

---

## Adding to `shared/`

1. Add the module file under `shared/<module>/`.
2. Export from `shared/<module>/__init__.py`.
3. Add docstring + type hints to every public function.
4. Write tests under `shared/<module>/tests/test_<module>.py`.
5. Update `shared/README.md` module table.

---

## Git Hygiene

- Do **not** `git add` anything under `data/`, `models/checkpoints/`, `.venv/`, `wandb/`, `mlruns/`, `outputs/`.
- `models/configs/` and `models/README.md` are the exception — commit these.
- Commit `.gitkeep` files to preserve empty directory structure.
- Keep commits scoped: one project or one shared utility per commit.

---

## Common Mistakes to Avoid

| Mistake | Correct approach |
|---------|-----------------|
| `import sys; sys.path.insert(...)` | `uv sync` installs `shared/` editably; just import directly |
| Putting reusable code in `projects/` | Move it to `shared/` |
| Importing from `projects/` inside `shared/` | Never — breaks the one-way dependency rule |
| Editing files in `data/raw/` | Write preprocessing scripts that output to `data/processed/` |
| Committing `.pt` or `.safetensors` files | They are gitignored; push to HF Hub or S3 instead |
| Using `pip install` | Use `uv add <pkg>` to keep `pyproject.toml` and `uv.lock` in sync |

---

## Key Files to Read First

When starting work in this repo, read in order:

1. This file (`AGENTS.md`) — you are here
2. `README.md` — high-level overview
3. `data/README.md` — what datasets exist and how to get them
4. `models/README.md` — what checkpoints exist and where they came from
5. `projects/<target>/README.md` — the specific experiment you are working on
6. `shared/README.md` — available utilities
