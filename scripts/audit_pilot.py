import os
import json
import torch
import inspect
import sys
from safetensors.torch import load_file
from src.model import load_model_and_tokenizer
from src.config import get_base_config

def audit_kaggle():
    print("==================================================")
    print("1. EVALUATION METRICS")
    print("==================================================")
    res_file = "results/raw/lora_r8_seed42.json"
    if os.path.exists(res_file):
        with open(res_file, "r") as f:
            data = json.load(f)
        total = len(data)
        correct = sum(1 for d in data if d.get("is_correct", False))
        print(f"Total evaluated: {total}")
        print(f"Exact match accuracy: {correct / total * 100:.2f}%")
    else:
        print(f"File not found: {res_file}")
        
    print("\n==================================================")
    print("2. ADAPTER UPDATE NORM VERIFICATION")
    print("==================================================")
    adapter_file = "adapters/lora_r8_seed42/adapter_model.safetensors"
    if os.path.exists(adapter_file):
        weights = load_file(adapter_file)
        lora_b_norm = sum(torch.norm(v.float()).item() for k, v in weights.items() if 'lora_B' in k)
        print(f"LoRA B total update norm: {lora_b_norm:.6f}")
        if lora_b_norm == 0.0:
            print("WARNING: Adapter weights DID NOT CHANGE from 0! Training silently failed.")
    else:
        print(f"File not found: {adapter_file}")

    print("\n==================================================")
    print("3. GEMMA 4 MODULE SOURCE INSPECTION")
    print("==================================================")
    config = get_base_config()
    model, _ = load_model_and_tokenizer(config['model']['id'], use_4bit=config['model']['use_4bit'])
    
    for module in model.modules():
        if module.__class__.__name__ == "Gemma4ClippableLinear":
            cls = module.__class__
            print("--- ORIGINAL SOURCE CODE OF Gemma4ClippableLinear.forward ---")
            try:
                print(inspect.getsource(cls.forward))
            except Exception as e:
                print(f"Could not extract source: {e}")
            break
            
if __name__ == "__main__":
    audit_kaggle()
