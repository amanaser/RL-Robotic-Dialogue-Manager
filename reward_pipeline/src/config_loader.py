import yaml
import os
from typing import Dict, Any
import json


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
    return config

if __name__ == '__main__':
    config = load_config("config.yaml")
    print("config loaded")
    print(json.dumps(config, indent=2))
