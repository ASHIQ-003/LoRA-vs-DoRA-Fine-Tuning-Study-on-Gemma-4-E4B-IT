import yaml
import os

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_base_config() -> dict:
    base_path = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")
    return load_config(base_path)
