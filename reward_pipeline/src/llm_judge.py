import openai
import json
from typing import Dict, Any

class LLMJudge:
    """
    A class to interact with a LLM to judge the output of a fine-tuned model.
    """
    def __init__(self, config: Dict[str, Any]):
        """
            config: A dictionary containing judge_llm settings.
        """
        self.config = config['judge_llm']
        self.actions = config['actions']
        
        try:
            self.client = openai.OpenAI(
                base_url=self.config['base_url'],
                api_key=self.config['api_key'], 
            )
        except Exception as e:
            print(f"Error configuring OpenAI client for local server: {e}")
            raise

    def _create_judge_prompt(self, query: str, predicted_action: str) -> str:
        descriptions = {
            "speak": "Respond with words or information.",
            "detect_objects": "Identify objects in the environment.",
            "move": "Change physical location or orientation.",
            "pick_up": "Grab an object.",
            "express_emotion": "Show an emotion (e.g., happy, sad)."
        }

        action_descriptions_list = []
        for idx, name in self.actions.items():
            desc = descriptions.get(name, "No description available.")
            action_descriptions_list.append(f"- {name} ({idx}): {desc}")
        
        action_descriptions = "\n".join(action_descriptions_list)

        return f"""
        You are an expert AI evaluator for a robotics assistant. Your task is to rate the appropriateness of a predicted action based on a user's query.

        The user's query is: "{query}"
        The assistant's predicted action is: "{predicted_action}"

        Here are the available actions and their meanings:
        {action_descriptions}

        Please evaluate the action on a continuous scale from -1.0 to 1.0, where:
        1.0 = The perfect, most direct action to fulfill the user's intent.
        0.5 = A reasonable but sub-optimal action. It might be a good first step, but not the final goal.
        0.0 = An irrelevant but harmless action.
        -1.0 = A completely incorrect, nonsensical, or counter-productive action.

        Provide your response ONLY in the following JSON format:
        {{
          "score": <float>,
          "reasoning": "<string: a brief explanation for your score>"
        }}
        """

    def get_reward_score(self, query: str, predicted_action_id: int) -> Dict[str, Any]:
        predicted_action_name = self.actions.get(predicted_action_id, "UNKNOWN_ACTION")        
        prompt_content = self._create_judge_prompt(query, predicted_action_name)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config['model_name'],
                messages=[
                    {"role": "system", "content": "You are an expert AI evaluator. Respond only in JSON format."},
                    {"role": "user", "content": prompt_content}
                ],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_output_tokens'],
                response_format={"type": "json_object"}, 
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
                
            return {
                "query": query,
                "predicted_action_id": predicted_action_id,
                "predicted_action_name": predicted_action_name,
                "reward_score": float(result['score']),
                "reward_reasoning": result['reasoning']
            }

        except json.JSONDecodeError:
            return {"error": "Failed to decode JSON from judge LLM response", "response": result_text}
        except Exception as e:
            return {"error": f"An error occurred while calling the judge LLM: {e}"}
