# models/

This directory holds model checkpoints and architecture configs.
**The actual weight files are git-ignored.** Only structure, configs committed to git, and this README are tracked.

---

## Directory Structure

```
models/
├── checkpoints/   ← saved model weights (.pt / .safetensors / GGUF / …)
└── configs/       ← architecture / training configs that produced each checkpoint
```

---

## Checkpoint Registry

| Checkpoint | Base Model | Project | Training Data | Date | Hub Link |
|------------|-----------|---------|--------------|------|----------|
| _(add entries as checkpoints are produced)_ | | | | | |

### Template row

```
| models/checkpoints/<name>/ | <base> | projects/<project>/ | datasets/processed/<data>/ | YYYY-MM-DD | <HF or S3 URL> |
```

---

## Downloading Checkpoints

### HuggingFace Hub

```bash
huggingface-cli download <org>/<model> --local-dir models/checkpoints/<name>
```

### DVC

```bash
dvc pull models/checkpoints/<name>
```

---

## Architecture Configs

Configs in `models/configs/` are **committed to git** and version-controlled.
They are the source of truth for reproducing a run — link every checkpoint row to its config file.

```
models/configs/
└── vlm_quantization_v1.yaml   ← example
```

---

## Adding a New Checkpoint

1. Save the checkpoint under `models/checkpoints/<run-name>/`.
2. Copy or symlink the config used to `models/configs/<run-name>.yaml`.
3. Push weights to HF Hub or S3, then add a row to the registry above.
4. Do **not** commit the weight files — only commit configs and README updates.
