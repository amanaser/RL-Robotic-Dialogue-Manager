from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_DIR = "./flan_t5_action_predictor"  

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

def predict_action(question):
    input_text = "question: " + question
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    
    outputs = model.generate(**inputs, max_length=10)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return pred

if __name__ == "__main__":
    while True:
        q = input("Enter a question: ")
        if q.lower() == "exit":
            break
        prediction = predict_action(q)
        print(f"predicted action: {prediction}")
