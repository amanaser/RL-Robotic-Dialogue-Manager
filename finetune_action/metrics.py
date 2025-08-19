import evaluate
import numpy as np
from preprocess import tokenizer

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if isinstance(predictions, np.ndarray) and predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)
        predictions = predictions.tolist()

    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()

    labels = [
        [token if token != -100 else tokenizer.pad_token_id for token in label_seq]
        for label_seq in labels
    ]

    pred_ids = [pred[0] if isinstance(pred, list) else pred for pred in predictions]
    label_ids = [label[0] if isinstance(label, list) else label for label in labels]

    accuracy = accuracy_metric.compute(predictions=pred_ids, references=label_ids)
    f1 = f1_metric.compute(predictions=pred_ids, references=label_ids, average="macro")

    decoded_preds = tokenizer.batch_decode([predictions[0]], skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode([labels[0]], skip_special_tokens=True)
    print(f"[compute_metrics] decoded prediction example: {decoded_preds[0]}")
    print(f"[compute_metrics] decoded label example: {decoded_labels[0]}")
    print(f"[compute_metrics] accuracy: {accuracy['accuracy']}, macro F1: {f1['f1']}")

    return {"accuracy": accuracy["accuracy"], "macro_f1": f1["f1"]}