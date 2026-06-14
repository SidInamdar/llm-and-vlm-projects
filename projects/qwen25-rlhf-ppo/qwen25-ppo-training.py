"""
qwen25-ppo-training.py
Full 7B 2-GPU RLHF PPO training pipeline.

GPU 0 : reward model (DeBERTa, frozen) + reference model (Qwen2.5-7B, fp16, frozen)
GPU 1 : policy model  (Qwen2.5-7B + LoRA + value head)

Runs ``--num-steps`` PPO steps (default 300) and saves the LoRA adapter.

Usage:
    # Main run (KL regularised)
    python qwen25-ppo-training.py --init-kl-coef 0.2 --run-name ppo-main

    # Ablation run (no KL — expect reward hacking / degenerate outputs)
    python qwen25-ppo-training.py --init-kl-coef 0.0 --run-name ppo-ablation
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlflow
import torch
from datasets import load_from_disk
from peft import LoraConfig
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# ── Import reward wrapper from sibling file ───────────────────────────────────
_PROJECT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "reward_wrapper", str(_PROJECT_DIR / "qwen25-ppo-reward-wrapper.py")
)
_reward_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reward_mod)
load_reward_model = _reward_mod.load_reward_model
compute_rewards = _reward_mod.compute_rewards

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PROMPTS_PATH = _REPO_ROOT / "datasets" / "processed" / "alpaca_ppo_prompts"
CHECKPOINTS_DIR = _REPO_ROOT / "models" / "checkpoints"

# ── Model identifiers ────────────────────────────────────────────────────────
POLICY_MODEL_HUB = "Qwen/Qwen2.5-7B-Instruct"
_LOCAL_POLICY = _REPO_ROOT / "models" / "checkpoints" / "Qwen--Qwen2.5-7B-Instruct"

MLFLOW_EXPERIMENT = "qwen25-rlhf-ppo"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen2.5-7B RLHF PPO training")
    p.add_argument("--init-kl-coef", type=float, default=0.2,
                    help="Initial KL penalty coefficient (0.0 for ablation)")
    p.add_argument("--run-name", type=str, default="ppo-main",
                    help="Run name for MLflow / TensorBoard / checkpoint dir")
    p.add_argument("--num-steps", type=int, default=300,
                    help="Total PPO training steps")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print(f" Qwen2.5-7B RLHF PPO Training — {args.run_name}")
    print(f" init_kl_coef={args.init_kl_coef}  steps={args.num_steps}")
    print("=" * 70)

    # ── Verify 2 GPUs ─────────────────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"  ✗ Need 2 GPUs, found {n_gpus}. Use the smoke test for single-GPU.")
        sys.exit(1)

    for i in range(2):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f"  GPU {i}: {name} ({vram:.1f} GB)")

    model_path = str(_LOCAL_POLICY) if _LOCAL_POLICY.exists() else POLICY_MODEL_HUB
    save_dir = CHECKPOINTS_DIR / f"qwen25-7b-{args.run_name}"
    tb_log_dir = str(_PROJECT_DIR / "logs" / f"tb-{args.run_name}")

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # CRITICAL FIX: Qwen2.5 tokenizer defaults to right padding.
    # Causal-LM generation requires left padding.
    if tokenizer.padding_side != "left":
        print(f"  ⚠ Fixing padding_side: {tokenizer.padding_side} → left")
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  ⚠ Set pad_token = eos_token ({tokenizer.eos_token})")

    # ── 2. Policy model on GPU 1 ──────────────────────────────────────────────
    print(f"\n[2/7] Loading policy model (7B + LoRA + value head) → GPU 1...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path,
        peft_config=lora_config,
        device_map={"": 1},
        torch_dtype=torch.float16,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M "
          f"({100 * trainable / total:.2f}%)")

    # ── 3. Reference model on GPU 0 (frozen, no LoRA) ─────────────────────────
    print(f"\n[3/7] Loading reference model (7B, frozen, fp16) → GPU 0...")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    ref_total = sum(p.numel() for p in ref_model.parameters()) / 1e6
    print(f"  Reference model: {ref_total:.1f}M params (frozen)")

    # ── 4. Reward model on GPU 0 ──────────────────────────────────────────────
    print(f"\n[4/7] Loading reward model (DeBERTa) → GPU 0...")
    reward_model, reward_tokenizer = load_reward_model(device=0)

    # ── 5. Dataset ────────────────────────────────────────────────────────────
    print(f"\n[5/7] Loading training prompts...")
    if not TRAIN_PROMPTS_PATH.exists():
        print("  ✗ Prompts not found. Run qwen25-ppo-dataset-prep.py first.")
        sys.exit(1)

    ds = load_from_disk(str(TRAIN_PROMPTS_PATH))
    print(f"  Total prompts: {len(ds)}")

    def tokenize_fn(example: dict) -> dict:
        enc = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=128,
            padding=False,
        )
        return {"input_ids": enc["input_ids"], "query": example["prompt"]}

    ds = ds.map(tokenize_fn, remove_columns=["prompt"])
    ds.set_format("torch", columns=["input_ids"])

    # ── 6. PPO config + trainer ───────────────────────────────────────────────
    print(f"\n[6/7] Setting up PPOTrainer...")
    ppo_config = PPOConfig(
        model_name=model_path,
        learning_rate=1e-5,
        batch_size=16,
        mini_batch_size=4,
        ppo_epochs=4,
        gamma=1.0,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        init_kl_coef=args.init_kl_coef,
        target_kl=6.0,
        max_grad_norm=1.0,
        log_with=None,  # we log manually
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=ds,
    )

    generation_kwargs = {
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    # ── 7. Training loop ──────────────────────────────────────────────────────
    print(f"\n[7/7] Starting PPO training ({args.num_steps} steps)...\n")
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "model": POLICY_MODEL_HUB,
            "init_kl_coef": args.init_kl_coef,
            "num_steps": args.num_steps,
            "batch_size": 16,
            "mini_batch_size": 4,
            "ppo_epochs": 4,
            "lora_r": 8,
            "lora_alpha": 16,
            "learning_rate": 1e-5,
            "cliprange": 0.2,
            "target_kl": 6.0,
        })

        global_step = 0
        epoch = 0

        while global_step < args.num_steps:
            epoch += 1
            for batch in ppo_trainer.dataloader:
                if global_step >= args.num_steps:
                    break

                t0 = time.time()
                query_tensors: list[torch.Tensor] = batch["input_ids"]

                # ── Generate responses ────────────────────────────────────────
                response_tensors = ppo_trainer.generate(
                    query_tensors, **generation_kwargs
                )
                # Ensure response-only (strip prompt if included)
                response_tensors = [
                    resp[len(query):]
                    if len(resp) > len(query)
                    else resp
                    for query, resp in zip(query_tensors, response_tensors)
                ]

                # ── Decode for reward scoring ─────────────────────────────────
                full_texts: list[str] = []
                for q, r in zip(query_tensors, response_tensors):
                    full = tokenizer.decode(
                        torch.cat([q, r]), skip_special_tokens=True
                    )
                    full_texts.append(full)

                # ── Compute rewards (GPU 0) ───────────────────────────────────
                rewards = compute_rewards(
                    full_texts, reward_model, reward_tokenizer, device=0
                )

                # ── PPO step ──────────────────────────────────────────────────
                stats = ppo_trainer.step(
                    list(query_tensors), list(response_tensors), rewards
                )

                # ── Extract metrics ───────────────────────────────────────────
                reward_vals = [r.item() for r in rewards]
                mean_reward = sum(reward_vals) / len(reward_vals)
                kl = stats.get("objective/kl", 0.0)
                policy_loss = stats.get("ppo/loss/policy", 0.0)
                value_loss = stats.get("ppo/loss/value", 0.0)

                # clip_fraction: fraction of samples where |ratio - 1| > cliprange
                # PPOTrainer may expose this as "ppo/policy/clipfrac".
                # If not present, we report 0 with a warning on the first step.
                clip_frac = stats.get("ppo/policy/clipfrac", None)
                if clip_frac is None:
                    clip_frac = stats.get("ppo/policy/clipfraction", None)
                if clip_frac is None:
                    # Fallback: not directly available without modifying PPOTrainer internals.
                    # Log NaN so it's visible that this metric is missing.
                    if global_step == 0:
                        print("  ⚠ clip_fraction not found in PPOTrainer stats — "
                              "logging NaN. Check stats keys for your trl version.")
                        print(f"    Available keys: {sorted(stats.keys())}")
                    clip_frac = float("nan")

                # ── Log to TensorBoard + MLflow ───────────────────────────────
                metrics = {
                    "mean_reward": mean_reward,
                    "kl_divergence": kl if isinstance(kl, (int, float)) else 0.0,
                    "policy_loss": policy_loss if isinstance(policy_loss, (int, float)) else 0.0,
                    "value_loss": value_loss if isinstance(value_loss, (int, float)) else 0.0,
                    "clip_fraction": clip_frac if isinstance(clip_frac, (int, float)) else 0.0,
                }

                for name, val in metrics.items():
                    tb_writer.add_scalar(f"ppo/{name}", val, global_step)
                mlflow.log_metrics(metrics, step=global_step)

                # GPU memory
                if torch.cuda.is_available():
                    for gpu_idx in range(2):
                        mem_gb = torch.cuda.memory_allocated(gpu_idx) / 1e9
                        tb_writer.add_scalar(
                            f"system/gpu{gpu_idx}_mem_gb", mem_gb, global_step
                        )

                dt = time.time() - t0
                kl_str = f"{kl:.4f}" if isinstance(kl, (int, float)) else "?"
                print(
                    f"  Step {global_step:4d}/{args.num_steps} | "
                    f"reward={mean_reward:+.4f} | KL={kl_str} | "
                    f"policy_loss={policy_loss:.4f} | "
                    f"{dt:.1f}s"
                )

                global_step += 1

        # ── Save ──────────────────────────────────────────────────────────────
        print(f"\nSaving LoRA adapter to {save_dir}...")
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        mlflow.log_artifacts(str(save_dir), artifact_path="lora-adapter")

    tb_writer.close()

    print("\n" + "=" * 70)
    print(f" Training complete — {args.run_name}")
    print(f"  Steps      : {global_step}")
    print(f"  Checkpoint : {save_dir}")
    print(f"  TensorBoard: tensorboard --logdir {tb_log_dir}")
    print(f"  MLflow     : mlflow ui --port 5000")
    print("=" * 70)


if __name__ == "__main__":
    main()
