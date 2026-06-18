import argparse
import os
import sys
import time
import mlflow
import torch
import torch.nn as nn
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl.experimental.ppo import PPOTrainer, PPOConfig

# ── Paths relative to parent foobar directory ─────────────────────────────────
REWARD_MODEL_PATH = "llm-and-vlm-projects/models/checkpoints/OpenAssistant--reward-model-deberta-v3-large-v2"
REWARD_MODEL_LENGTH = 512

POLICY_MODEL_NAME = "Qwen--Qwen2.5-7B-Instruct"
POLICY_MODEL_PATH = f"llm-and-vlm-projects/models/checkpoints/{POLICY_MODEL_NAME}"

TRAIN_PROMPTS_PATH = "llm-and-vlm-projects/datasets/processed/alpaca_ppo_prompts"
CHECKPOINTS_DIR = "llm-and-vlm-projects/models/checkpoints"


def load_reward_model() -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    import sys
    if getattr(sys, "_cached_reward_model", None) is not None:
        return sys._cached_reward_model, sys._cached_reward_tokenizer

    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH,
        num_labels=1,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
    n_params = sum(p.numel() for p in reward_model.parameters()) / 1e6
    print(f"  Reward model loaded on device ({n_params:.1f}M params, frozen)")
    sys._cached_reward_model = reward_model
    sys._cached_reward_tokenizer = reward_tokenizer
    return reward_model, reward_tokenizer 


class RewardModelWrapper(nn.Module):

    def __init__(self,):
        super().__init__()
        self.reward_model, self.reward_tokenizer = load_reward_model()
        self.policy_tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
        self.policy_tokenizer.padding_side = "left"
        self.policy_tokenizer.pad_token = self.policy_tokenizer.eos_token
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.base_model_prefix = "dummy_backbone"
        self.__dict__["dummy_backbone"] = self

    def score(self, x):
        return x

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.reward_model, name)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, **kwargs):
        full_texts = self.policy_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        inputs = self.reward_tokenizer(
            full_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=REWARD_MODEL_LENGTH,
        )
        dev = next(self.reward_model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            rewards = outputs.logits
        
        rewards_expanded = rewards.unsqueeze(1).expand(-1, input_ids.shape[1], -1)
        
        from types import SimpleNamespace
        return SimpleNamespace(hidden_states=[rewards_expanded])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen2.5-7B RLHF PPO training")
    p.add_argument("--init-kl-coef", type=float, default=0.2,
                    help="Initial KL penalty coefficient")
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

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading tokenizer from {POLICY_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Policy model ───────────────────────────────────────────────────────
    print(f"\n[2/7] Loading policy model (7B + LoRA)...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    policy_model = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # ── 3. Value model ────────────────────────────────────────────────────────
    print(f"\n[3/7] Loading value model (7B + LoRA)...")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        POLICY_MODEL_PATH,
        num_labels=1,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    value_model.config.pad_token_id = tokenizer.pad_token_id

    value_lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="SEQ_CLS",
    )
    value_model = get_peft_model(value_model, value_lora_config)

    # ── 4. Reward wrapper ─────────────────────────────────────────────────────
    print(f"\n[4/7] Setting up Reward Model Wrapper...")
    reward_wrapper = RewardModelWrapper()

    # ── 5. Dataset ────────────────────────────────────────────────────────────
    print(f"\n[5/7] Loading training prompts...")
    ds = load_from_disk(str(TRAIN_PROMPTS_PATH))
    print(f"  Total prompts: {len(ds)}")

    def tokenize_fn(example: dict) -> dict:
        enc = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=128,
            padding=False,
        )
        return {"input_ids": enc["input_ids"]}

    ds = ds.map(tokenize_fn, remove_columns=["prompt"])
    ds.set_format("torch", columns=["input_ids"])

    # ── 6. PPO config + trainer ───────────────────────────────────────────────
    print(f"\n[6/7] Setting up PPOTrainer...")
    save_dir = f"{CHECKPOINTS_DIR}/qwen25-7b-{args.run_name}"
    tb_log_base = os.environ.get("TB_LOG_BASE", "job_run_scripts/logs")
    tb_log_dir = f"{tb_log_base}/tb-{args.run_name}"

    ppo_config = PPOConfig(
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        mini_batch_size=2,
        num_ppo_epochs=2,
        gamma=1.0,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        kl_coef=args.init_kl_coef,
        max_grad_norm=1.0,
        report_to="tensorboard",
        logging_dir=tb_log_dir,
        max_steps=args.num_steps,
    )

    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=None,
        reward_model=reward_wrapper,
        train_dataset=ds,
        value_model=value_model,
        peft_config=lora_config,
    )

    # ── 7. Training loop ──────────────────────────────────────────────────────
    print(f"\n[7/7] Starting PPO training ({args.num_steps} steps)...\n")
    mlflow.set_experiment("qwen25-rlhf-ppo")

    with mlflow.start_run(run_name=args.run_name):
        ppo_trainer.train()

    # Save LoRA adapter
    print(f"\nSaving LoRA adapter to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    policy_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("✓ Model saved successfully.")


if __name__ == "__main__":
    main()
