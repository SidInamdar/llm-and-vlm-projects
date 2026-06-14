# Qwen2.5 RLHF PPO Fine-Tuning

PPO-based RLHF pipeline for **Qwen2.5-7B-Instruct** with a DeBERTa reward model,
trained on Alpaca prompts. Compares KL-regularised PPO against an ablation without
KL penalty (to demonstrate reward hacking).

---

## Models

| Role | Model | GPU | Notes |
|------|-------|-----|-------|
| Policy | Qwen/Qwen2.5-7B-Instruct | 1 | LoRA r=8 + value head |
| Reference | Qwen/Qwen2.5-7B-Instruct | 0 | Frozen, fp16 |
| Reward | OpenAssistant/reward-model-deberta-v3-large-v2 | 0 | Frozen, eval mode |
| Smoke test | Qwen/Qwen2.5-0.5B-Instruct | 0 | Single-GPU validation |

## Dataset

| Dataset | Source | Split |
|---------|--------|-------|
| tatsu-lab/alpaca | HuggingFace Hub | 52K train prompts + 100 held-out eval |

Prompts = `instruction` + `input` (if non-empty). The `output` field is ignored —
PPO generates its own responses.

## Checkpoints

| Run | Path | Description |
|-----|------|-------------|
| Main PPO | `models/checkpoints/qwen25-7b-ppo-main/` | 300 steps, init_kl_coef=0.2 |
| Ablation | `models/checkpoints/qwen25-7b-ppo-ablation/` | 300 steps, init_kl_coef=0.0 |

---

## Files

| File | Purpose |
|------|---------|
| `qwen25-ppo-dataset-prep.py` | Download Alpaca, extract prompts, split train/eval |
| `qwen25-ppo-reward-wrapper.py` | DeBERTa reward model wrapper (load + score) |
| `qwen25-ppo-smoke-test.py` | 0.5B single-GPU end-to-end PPO validation |
| `qwen25-ppo-training.py` | 7B 2-GPU PPO training (main + ablation) |
| `qwen25-ppo-eval.py` | Held-out evaluation (generate + score + save) |

---

## Three Runs

1. **Baseline** — generate on 100 held-out prompts with unmodified SFT model,
   score with reward model, save reward distribution.

2. **Main PPO** — 300 steps with `init_kl_coef=0.2`. KL regularisation prevents
   the policy from drifting too far from the reference.

3. **Ablation** — identical to (2) but `init_kl_coef=0.0`. Without KL penalty,
   expect reward hacking: repetition, generic phrases, degenerate outputs that
   score high on the reward model but are low-quality.

For runs 2 and 3, re-run evaluation on the same 100 held-out prompts after
training for a before/after comparison.

---

## Usage

### Full pipeline via SLURM

```bash
# From the foobar/ root:
sbatch job_run_scripts/qwen25-rlhf-ppo/qwen25-ppo-run.sh
```

### Individual steps (manual)

```bash
# 1. Dataset preparation (needs internet)
uv run python projects/qwen25-rlhf-ppo/qwen25-ppo-dataset-prep.py

# 2. Smoke test (single GPU)
uv run python projects/qwen25-rlhf-ppo/qwen25-ppo-smoke-test.py

# 3. Baseline evaluation
uv run python projects/qwen25-rlhf-ppo/qwen25-ppo-eval.py --eval-name baseline

# 4. Main PPO training (2 GPUs)
uv run python projects/qwen25-rlhf-ppo/qwen25-ppo-training.py \
    --init-kl-coef 0.2 --run-name ppo-main

# 5. Post-main evaluation
uv run python projects/qwen25-rlhf-ppo/qwen25-ppo-eval.py \
    --model-path models/checkpoints/qwen25-7b-ppo-main --eval-name post-main

# 6. Ablation training (2 GPUs)
uv run python projects/qwen25-rlhf-ppo/qwen25-ppo-training.py \
    --init-kl-coef 0.0 --run-name ppo-ablation

# 7. Post-ablation evaluation
uv run python projects/qwen25-rlhf-ppo/qwen25-ppo-eval.py \
    --model-path models/checkpoints/qwen25-7b-ppo-ablation --eval-name post-ablation
```

---

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| learning_rate | 1e-5 |
| batch_size | 16 |
| mini_batch_size | 4 |
| ppo_epochs | 4 |
| gamma | 1.0 |
| cliprange | 0.2 |
| cliprange_value | 0.2 |
| vf_coef | 0.1 |
| init_kl_coef | 0.2 (main) / 0.0 (ablation) |
| target_kl | 6.0 |
| max_grad_norm | 1.0 |

### Generation Config (Rollouts)

| Parameter | Value |
|-----------|-------|
| max_new_tokens | 128 |
| temperature | 0.7 |
| top_p | 0.9 |
| do_sample | True |
| padding_side | left (fixed from Qwen2.5 default of right) |

---

## Logging

- **TensorBoard**: `tensorboard --logdir projects/qwen25-rlhf-ppo/logs/`
- **MLflow**: `mlflow ui --port 5000`

Per-step metrics: `mean_reward`, `kl_divergence`, `policy_loss`, `value_loss`, `clip_fraction`

---

## Known Issues

- **Qwen2.5 padding**: defaults to right-padding. All scripts fix this to
  left-padding for generation.
- **clip_fraction**: may not be directly exposed by PPOTrainer depending on
  trl version. The training script checks stats keys and logs NaN if missing.
- **Cross-GPU tensors**: PPOTrainer may need tensor movement between GPU 0
  (ref model) and GPU 1 (policy). If issues arise, check the smoke test first
  on single GPU to isolate the problem.
