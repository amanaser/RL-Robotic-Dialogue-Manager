import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ = 32
BATCH_SIZE = 8
MAX_LENGTH = 10
LR = 5e-5
EPOCHS = 10


policy_model = T5ForConditionalGeneration.from_pretrained("../finetune_action/flan_t5_action_predictor").to(DEVICE)
policy_tokenizer = T5Tokenizer.from_pretrained("t5-small")

value_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)
value_head = nn.Linear(value_model.config.d_model, 1).to(DEVICE)  
reward_model = AutoModelForSequenceClassification.from_pretrained("../rl_training/models/reward_model").to(DEVICE)
reward_tokenizer = AutoTokenizer.from_pretrained("../rl_training/models/reward_model")

policy_optimizer = optim.Adam(policy_model.parameters(), lr=LR)
value_optimizer = optim.Adam(list(value_model.parameters()) + list(value_head.parameters()), lr=LR)

dataset = load_dataset("json", data_files="./data/generated_rewards_llama3.jsonl")
dataloader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)

def t5_generate_and_logprobs(policy_model, tokenizer, input_texts, max_length=MAX_LENGTH):
    enc = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ).to(DEVICE)
    generated = policy_model.generate(**enc, max_length=max_length)
    sequences = generated  
    decoder_input_ids = sequences[:, :-1]
    labels = sequences[:, 1:]
    logits = policy_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], decoder_input_ids=decoder_input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    seq_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1).sum(dim=1)
    decoded_actions = [tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences]
    return decoded_actions, seq_log_probs

def compute_reward(queries, actions):
    texts = [q + " [SEP] " + a for q, a in zip(queries, actions)]
    enc = reward_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ).to(DEVICE)
    outputs = reward_model(**enc)
    rewards = outputs.logits.squeeze(-1)  
    return rewards


for epoch in range(EPOCHS):
    for batch in dataloader:
        queries = batch["query"]

        actions, old_log_probs = t5_generate_and_logprobs(policy_model, policy_tokenizer, queries)
        rewards = compute_reward(queries, actions).detach()
        enc = policy_tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ).to(DEVICE)
        value_pred = value_head(value_model.encoder(enc["input_ids"]).last_hidden_state.mean(dim=1)).squeeze(-1)
        advantages = rewards - value_pred.detach()


        policy_optimizer.zero_grad()
        policy_loss = -(old_log_probs * advantages).mean() 
        policy_loss.backward()
        policy_optimizer.step()


        value_optimizer.zero_grad()
        value_loss = nn.MSELoss()(value_pred, rewards)
        value_loss.backward()
        value_optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS} finished.")

policy_model.save_pretrained("./ppo_outputs/policy_model")
value_model.save_pretrained("./ppo_outputs/value_model")
