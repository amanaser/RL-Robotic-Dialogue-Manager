from config import Config
from utils import load_and_split_data
from preprocess import preprocess_function, tokenizer
from transformers import AutoModelForSeq2SeqLM
from metrics import compute_metrics
from transformers import Trainer

dataset = load_and_split_data()
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["question", "action"])
model = AutoModelForSeq2SeqLM.from_pretrained(Config.OUTPUT_DIR)

trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)

test_results = trainer.evaluate(tokenized_datasets["test"])
print("test results:", test_results)
