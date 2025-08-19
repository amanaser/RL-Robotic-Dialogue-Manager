from typing import Dict, Any, List
from llm_judge import LLMJudge
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class RewardGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.judge = LLMJudge(config)
        self.actions = config['actions']
        self.action_name_to_id = {name: idx for idx, name in self.actions.items()}
        model_path = config['fine_tuned_model']['name']
        try:
            self.ft_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.ft_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
            self.device = "cpu"

            self.ft_model.to(self.device)
            print(f"Fine-tuned model loaded into: {self.device}")
        except Exception as e:
            print(f"ERROR: Could not load fine-tuned model from path '{model_path}'. Check config.yaml. Error: {e}")
            raise

    def _get_prediction_from_finetuned_model(self, query: str) -> int:
        print(f"get prediction for query: '{query}'")
        
        inputs = self.ft_tokenizer(query, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.ft_model.generate(**inputs, max_new_tokens=10)
        
        predicted_text = self.ft_tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        print(f"generated text: '{predicted_text}'")

        try:
            predicted_action_id = int(predicted_text)
            if predicted_action_id not in self.actions:
                print(f"Model generated an unknown action ID: {predicted_action_id}")
                return -1 
            return predicted_action_id
        except (ValueError, TypeError):
            if predicted_text in self.action_name_to_id:
                return self.action_name_to_id[predicted_text]
            else:
                print(f"Model generated INVALID text: '{predicted_text}'")
                return -1 

    def generate_reward_for_query(self, query: str) -> Dict[str, Any]:
        predicted_action_id = self._get_prediction_from_finetuned_model(query)
        
        if predicted_action_id == -1:
            return {
                "query": query,
                "predicted_action_id": -1,
                "predicted_action_name": "INVALID_PREDICTION",
                "reward_score": -1.0, 
                "reward_reasoning": "Fine-tuned model failed to produce a valid action ID."
            }
            
        print(f"get judgment for the action ID: {predicted_action_id}")
        reward_data = self.judge.get_reward_score(query, predicted_action_id)
        return reward_data

    def process_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        results = []
        for i, query in enumerate(queries):
            print(f"\nProcessing query {i+1}/{len(queries)}: '{query}'")
            result = self.generate_reward_for_query(query)
            results.append(result)
            print(f"-> Result: {result}")
        return results
