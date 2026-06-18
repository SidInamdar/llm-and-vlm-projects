import os, sys, time, mlflow, torch
from pathlib import Path
import torch.nn as nn
from datasets import load_from_disk
from peft import LoraConfig 
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl.experimental.ppo import PPOTrainer, PPOConfig

# ── Paths relative to parent foobar directory ─────────────────────────────────
REWARD_MODEL_PATH = "llm-and-vlm-projects/models/checkpoints/OpenAssistant--reward-model-deberta-v3-large-v2"
REWARD_MODEL_LENGTH = 512

def load_reward_model() -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    import sys
    if getattr(sys, "_cached_reward_model", None) is not None:
        return sys._cached_reward_model, sys._cached_reward_tokenizer

    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH,
        num_labels=1,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
    )

    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
    n_params = sum(p.numel() for p in reward_model.parameters()) / 1e6
    print(f"  Reward model loaded on device ({n_params:.1f}M params, frozen)")
    sys._cached_reward_model = reward_model
    sys._cached_reward_tokenizer = reward_tokenizer
    return reward_model, reward_tokenizer 

@torch.no_grad()
def compute_rewards(
    texts: list[str], 
    reward_model: AutoModelForSequenceClassification,
    reward_tokenizer: AutoTokenizer,
    device: torch.device | int = 0,
    batch_size: int = 8, 
) -> list[torch.Tensor]:
    all_rewards : list[torch.Tensor] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = reward_tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=REWARD_MODEL_LENGTH,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        outputs = reward_model(**encodings)
        logits = outputs.logits.squeeze(-1)
        for j in range(logits.shape[0]):
            all_rewards.append(logits[j].detach().float())

    return all_rewards

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reward_model, reward_tokenizer = load_reward_model()
samples = [
        "The capital of France is Paris. It is known for the Eiffel Tower and rich cultural heritage.",
        "asdkjh asdkjh asdk the the the the the the the the",
        "I appreciate your question. Let me provide a detailed and helpful response to that."
        ]
rewards = compute_rewards(samples, reward_model, reward_tokenizer, device=_device)
print(rewards)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
if torch.cuda.is_available():
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM  : {vram:.1f} GB")
    else:
        print("  ⚠ No GPU — running on CPU (will be slow)")
    print(f"  VRAM  : {vram:.1f} GB")
else:
    print("  ⚠ No GPU — running on CPU (will be slow)")

POLICY_MODEL_NAME = "Qwen--Qwen2.5-0.5B-Instruct"
POLICY_MODEL_PATH = f"llm-and-vlm-projects/models/checkpoints/{POLICY_MODEL_NAME}"

policy_tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
policy_tokenizer.padding_side = "left"
policy_tokenizer.pad_token = policy_tokenizer.eos_token

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM")

policy_model = AutoModelForCausalLM.from_pretrained(
    POLICY_MODEL_PATH,
    device_map={"": "cuda:0"},
    torch_dtype=torch.float16,)

value_model = AutoModelForSequenceClassification.from_pretrained(
    POLICY_MODEL_PATH,
    num_labels=1,
    device_map={"": "cuda:0"},
    torch_dtype=torch.float16,)
value_model.config.pad_token_id = policy_tokenizer.eos_token_id

from peft import get_peft_model
value_lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="SEQ_CLS",
)
value_model = get_peft_model(value_model, value_lora_config)
reward_model, reward_tokenizer = load_reward_model()

class RewardModelWrapper(nn.Module):

    def __init__(self,):
        super().__init__()
        self.reward_model, self.reward_tokenizer = load_reward_model()
        self.policy_tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
        self.policy_tokenizer.padding_side = "left"
        self.policy_tokenizer.pad_token = self.policy_tokenizer.eos_token
        self.device = device
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
        
        # Shape: (batch_size, sequence_length, 1)
        rewards_expanded = rewards.unsqueeze(1).expand(-1, input_ids.shape[1], -1)
        rewards_expanded = rewards_expanded.to(input_ids.device)
        
        from types import SimpleNamespace
        return SimpleNamespace(hidden_states=[rewards_expanded])

reward_wrapper = RewardModelWrapper()

NUM_STEPS = 10
BATCH_SIZE = 4
MINI_BATCH_SIZE = 2
MLFLOW_EXPERIMENT = "qwen25-rlhf-ppo-smoke-test"
TRAIN_PROMPTS_PATH = "llm-and-vlm-projects/datasets/processed/alpaca_ppo_prompts"
ds = load_from_disk(str(TRAIN_PROMPTS_PATH))
n_needed = min(BATCH_SIZE * NUM_STEPS, len(ds))
ds = ds.select(range(n_needed))
print(f"  Using {len(ds)} prompts for smoke test")
def tokenize_fn(example: dict) -> dict:
    enc = policy_tokenizer(
        example["prompt"],
        truncation=True,
        max_length=128,
        padding=False,
    )
    return {"input_ids": enc["input_ids"]}

ds = ds.map(tokenize_fn, remove_columns=["prompt"])
ds.set_format("torch", columns=["input_ids"])

ppo_config = PPOConfig(
    learning_rate=1e-5,
    per_device_train_batch_size=BATCH_SIZE,
    mini_batch_size=MINI_BATCH_SIZE,
    num_ppo_epochs=2,
    gamma=1.0,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
    kl_coef=0.2,
    max_grad_norm=1.0,
    report_to="tensorboard",
    max_steps=NUM_STEPS,
)

ppo_trainer = PPOTrainer(
    args=ppo_config,
    processing_class=policy_tokenizer,
    model=policy_model,
    ref_model=None,
    reward_model=reward_wrapper,
    train_dataset=ds,
    value_model=value_model,
    peft_config=lora_config,
)

with mlflow.start_run(run_name="smoke-test"):
    ppo_trainer.train()

sample_prompts = [
    "What is machine learning?",
    "Explain gravity in simple terms.",
    "Write a short poem about the ocean.",
]
for prompt in sample_prompts:
    inputs = policy_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = policy_model.generate(
            **inputs, max_new_tokens=64, temperature=0.7, do_sample=True,
            pad_token_id=policy_tokenizer.pad_token_id,
        )
    text = policy_tokenizer.decode(out[0], skip_special_tokens=True)
    response = text[len(prompt):].strip()
    print(f"  Q: {prompt}")
    print(f"  A: {response[:150]}...")
    print()
