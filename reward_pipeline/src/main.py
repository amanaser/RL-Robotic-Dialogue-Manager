import json
import logging
import sys
from config_loader import load_config
from reward_generate import RewardGenerator

def main():
    try:
        config = load_config("config.yaml")
        if config is None:
            logging.error("'config.yaml' is empty or invalid.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return

    logging.basicConfig(level=config['reward_generation']['log_level'])
    logging.info("config loaded")

    try:
        with open("data/test_queries.json", 'r') as f:
            query_data = json.load(f)
        
        all_queries = []
        for query_set in query_data.get("query_sets", []):
            all_queries.extend(query_set.get("queries", []))
        
        logging.info(f"Loaded {len(all_queries)} queries from data/test_queries.json")
    except FileNotFoundError:
        logging.error("Error: test_queries.json not found.")
        return
    except json.JSONDecodeError:
        logging.error("Error: Could not decode JSON from test_queries.json.")
        return

    reward_generator = RewardGenerator(config)
    logging.info("RewardGenerator initialized.")

    output_file = config['reward_generation']['output_file']
    
    try:
        with open(output_file, 'w') as f:
            pass # an empty file for erasing old content
        logging.info(f"Cleared old content in {output_file}")
    except IOError as e:
        logging.error(f"Could not clear output file: {e}")

    num_saved = 0
    for i, query in enumerate(all_queries):
        logging.info(f"\n--- Processing query {i+1}/{len(all_queries)} ---")
        
        result = reward_generator.generate_reward_for_query(query)
        try:
            with open(output_file, 'a') as f:
                f.write(json.dumps(result) + '\n')
            num_saved += 1
        except IOError as e:
            logging.error(f"Failed to write result for query '{query}' to {output_file}: {e}")

    logging.info(f"Successfully generated and saved {num_saved} rewards to {output_file}")

if __name__ == "__main__":
    main()
