import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ = 32
MAX_LENGTH = 10
BATCH_SIZE = 8


baseline_model = T5ForConditionalGeneration.from_pretrained("./finetune_action/flan_t5_action_predictor").to(DEVICE)
baseline_tokenizer = T5Tokenizer.from_pretrained("t5-small")

ppo_model = T5ForConditionalGeneration.from_pretrained("./rl_action_ppo/ppo_outputs/policy_model").to(DEVICE)
ppo_tokenizer = T5Tokenizer.from_pretrained("t5-small")

reward_model = AutoModelForSequenceClassification.from_pretrained("./rl_training/models/reward_model").to(DEVICE)
reward_tokenizer = AutoTokenizer.from_pretrained("./rl_training/models/reward_model")

dataset = load_dataset(
    "json",
    data_files="./data/generated_rewards_llama3.jsonl"
)

queries = list(dataset["train"]["query"])
def generate_actions(model, tokenizer, input_texts):
    enc = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ).to(DEVICE)
    generated = model.generate(**enc, max_length=MAX_LENGTH)
    decoded_actions = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated]
    return decoded_actions

def compute_reward(queries, actions):
    texts = [q + " [SEP] " + a for q, a in zip(queries, actions)]
    enc = reward_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ).to(DEVICE)
    outputs = reward_model(**enc)
    rewards = outputs.logits.squeeze(-1) 
    return rewards.detach().cpu().numpy()


print("baseline predictions:")
baseline_actions = generate_actions(baseline_model, baseline_tokenizer, queries)
baseline_rewards = compute_reward(queries, baseline_actions)

print("PPO predictions:")
ppo_actions = generate_actions(ppo_model, ppo_tokenizer, queries)
ppo_rewards = compute_reward(queries, ppo_actions)

FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)
plt.figure(figsize=(8,5))
plt.hist(baseline_rewards, bins=20, alpha=0.6, label="Baseline")
plt.hist(ppo_rewards, bins=20, alpha=0.6, label="PPO")
plt.xlabel("Reward score")
plt.ylabel("Count")
plt.title("Reward Distribution: Baseline vs PPO")
hist_path = os.path.join(FIGURE_DIR, "reward_distribution.png")
plt.legend()
plt.savefig(hist_path)
plt.show()

means = [baseline_rewards.mean(), ppo_rewards.mean()]
stds = [baseline_rewards.std(), ppo_rewards.std()]
labels = ["Baseline", "PPO"]

plt.figure(figsize=(6,4))
plt.bar(labels, means, yerr=stds, capsize=5)
plt.ylabel("Average reward")
plt.title("Mean reward comparison")
bar_path = os.path.join(FIGURE_DIR, "reward_comparison.png")
plt.savefig(bar_path)10
plt.show()

# df = pd.DataFrame({
#     "Query": queries[:20],
#     "Baseline_Action": baseline_actions[:20],
#     "PPO_Action": ppo_actions[:20],
#     "Baseline_Reward": baseline_rewards[:20],
#     "PPO_Reward": ppo_rewards[:20],
# })
# df.to_csv("sample_predictions.csv", index=False)

# improvement = ppo_rewards.mean() - baseline_rewards.mean()
# print(f"avg reward improvement: {improvement:.4f}")
