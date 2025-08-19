from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from config import Config
from utils import set_seed, load_and_split_data
from preprocess import preprocess_function, tokenizer
from metrics import compute_metrics


set_seed(Config.SEED)
dataset = load_and_split_data()
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["question", "action"])

model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = TrainingArguments(
    output_dir=Config.OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=Config.LR,
    per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=Config.EVAL_BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=Config.EPOCHS,
    logging_dir="./logs",
    logging_steps=Config.LOGGING_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

trainer.save_model(Config.OUTPUT_DIR)
tokenizer.save_pretrained(Config.OUTPUT_DIR)

print("Training complete")
