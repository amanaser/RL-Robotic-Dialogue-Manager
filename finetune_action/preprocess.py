from transformers import AutoTokenizer
from config import Config

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

def preprocess_function(examples):
    inputs = ["Question: " + q for q in examples["question"]]
    targets = examples["action"]

    model_inputs = tokenizer(
        inputs,
        max_length=Config.MAX_INPUT_LENGTH,
        padding="max_length",     
        truncation=True,
    )
    labels = tokenizer(
        targets,
        max_length=Config.MAX_OUTPUT_LENGTH,
        padding="max_length",
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs