import os
import sys
import torch
import logging
from src.config import get_base_config
from src.model import load_model_and_tokenizer
from src.adapters import create_adapter_config, inject_adapter
from src.data import load_gsm8k_data
from src.prompting import build_gsm8k_prompt

logging.basicConfig(level=logging.INFO)

def main():
    config = get_base_config()
    model, tokenizer = load_model_and_tokenizer(config['model']['id'], use_4bit=True)
    
    adapter_config = create_adapter_config("lora", 8, 16, config['adapter']['target_modules'], 0.05)
    model, _, _ = inject_adapter(model, adapter_config)
    
    # Enable gradients explicitly
    model.train()
    
    # Create a dummy batch
    prompt = "Question: What is 2+2? Answer: 4"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # We need labels for loss
    inputs["labels"] = inputs["input_ids"].clone()
    
    print("\n--- Forward Pass ---")
    outputs = model(**inputs)
    loss = outputs.loss
    print(f"Loss: {loss.item()}")
    
    print("\n--- Backward Pass ---")
    loss.backward()
    
    grad_norm = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None:
                norm = p.grad.norm().item()
                grad_norm += norm
                print(f"Parameter: {n}, Grad Norm: {norm}")
            else:
                print(f"Parameter: {n}, Grad is NONE!")
                
    print(f"\nTotal Grad Norm: {grad_norm}")

if __name__ == "__main__":
    main()
