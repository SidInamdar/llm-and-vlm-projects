# data/

This directory holds all datasets consumed by projects in this repo.
**The actual files are git-ignored.** Only structure and this README are tracked.

---

## Directory Structure

```
data/
├── raw/          ← original, immutable downloads — never modified
└── processed/    ← transformed / tokenised outputs ready for training
```

> **Rule:** Scripts always read from `raw/` and write to `processed/`. Never edit raw files in-place.

---

## Datasets

| Dataset | Version | Source | Stored in | Consumed by |
|---------|---------|--------|-----------|-------------|
| _(add entries as datasets are added)_ | | | | |

### Template row

```
| <name> | <version/date> | <URL or HF repo> | data/raw/<subdir>/ | projects/<name>/ |
```

---

## Downloading Data

### HuggingFace Hub

```bash
# Example — replace with the real dataset ID
huggingface-cli download <org>/<dataset> --local-dir data/raw/<name>
```

### DVC (if configured)

```bash
dvc pull
```

### Manual

Follow dataset-specific instructions and place files under `data/raw/<dataset-name>/`.

---

## Adding a New Dataset

1. Create a subdirectory under `data/raw/<dataset-name>/`.
2. Add a row to the table above (source, version, consuming project).
3. Write a preprocessing script in the consuming project (e.g. `projects/<name>/preprocess.py`) that outputs to `data/processed/<dataset-name>/`.
4. Do **not** commit the files themselves — only commit this README update.
