import os
import sys
import logging
import torch
from src.config import get_base_config
from src.model import load_model_and_tokenizer
from src.data import load_gsm8k_data, prepare_dataset_for_sft
from src.adapters import create_adapter_config, inject_adapter
from src.seed import set_seed
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import bitsandbytes as bnb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_forward_backward(method: str):
    logger.info(f"\\n{'='*50}\\nSTARTING MANUAL FORWARD/BACKWARD TEST FOR {method.upper()}\\n{'='*50}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    config = get_base_config()
    set_seed(42)
    
    model, tokenizer = load_model_and_tokenizer(config['model']['id'], use_4bit=config['model']['use_4bit'])
    
    if hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer is not None:
        tokenizer = tokenizer.tokenizer
        
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
        
    # --- 1. COMPREHENSIVE MODULE TREE INSPECTION ---
    logger.info("--- MODEL INSTANCE INSPECTION ---")
    logger.info(f"type(model): {type(model)}")
    if hasattr(model, 'model'): logger.info(f"type(model.model): {type(model.model)}")
    if hasattr(model, 'language_model'): logger.info(f"type(model.language_model): {type(model.language_model)}")
    if hasattr(model, 'text_model'): logger.info(f"type(model.text_model): {type(model.text_model)}")
    
    logger.info("\\n--- FIRST 50 MATCHING MODULES ---")
    count = 0
    all_q_v = []
    
    q_proj_count = 0
    v_proj_count = 0
    q_proj_linear_count = 0
    v_proj_linear_count = 0
    
    for name, module in model.named_modules():
        name_lower = name.lower()
        
        if name.endswith(".q_proj"): q_proj_count += 1
        if name.endswith(".v_proj"): v_proj_count += 1
        if name.endswith(".q_proj.linear"): q_proj_linear_count += 1
        if name.endswith(".v_proj.linear"): v_proj_linear_count += 1
        
        if "q_proj" in name_lower or "v_proj" in name_lower:
            all_q_v.append((name, type(module)))
            
        if any(k in name_lower for k in ["self_attn", "q_proj", "v_proj", "layers", "language_model", "model"]):
            if count < 50:
                logger.info(f"{name} | {type(module).__name__}")
                count += 1
                
    logger.info(f"\\n--- COUNT BY ENDING ---")
    logger.info(f".q_proj: {q_proj_count}")
    logger.info(f".v_proj: {v_proj_count}")
    logger.info(f".q_proj.linear: {q_proj_linear_count}")
    logger.info(f".v_proj.linear: {v_proj_linear_count}")
    
    logger.info(f"\\n--- Q_PROJ / V_PROJ CANDIDATES BY BRANCH ---")
    text_targets = []
    for name, mtype in all_q_v:
        is_linear = isinstance(dict(model.named_modules())[name], (torch.nn.Linear, bnb.nn.Linear4bit))
        
        in_vision = "vision" in name.lower()
        in_audio = "audio" in name.lower()
        in_language = "language" in name.lower() or "text" in name.lower() or name.startswith("model.layers")
        
        parent_name = ".".join(name.split(".")[:-1])
        parent_class = type(dict(model.named_modules()).get(parent_name, None)).__name__ if parent_name else "None"
        
        logger.info(f"{name} | {mtype.__name__} | Parent: {parent_class} | vis:{in_vision} aud:{in_audio} text:{in_language} | is_linear={is_linear}")
        
        # Only select actual linear modules in the text path
        if is_linear and in_language and not in_vision and not in_audio:
            text_targets.append(name)
            
    if not text_targets:
        raise ValueError("Could not find any text-specific linear modules! See the dump above to identify the true name.")
        
    logger.info(f"\\nFINAL SELECTED PEFT TARGETS (First 5): {text_targets[:5]}")
    logger.info(f"Total targets selected: {len(text_targets)}")
    
    adapter_config = create_adapter_config(method, 8, 16, text_targets, 0.05)
    model, trainable, _ = inject_adapter(model, adapter_config)
    
    dataset = load_gsm8k_data(split=config['dataset']['train_split'], max_samples=2, seed=42)
    dataset = prepare_dataset_for_sft(dataset, tokenizer, max_length=128)
    
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    batch = next(iter(dataloader))
    batch = {k: v.to(model.device) for k, v in batch.items()}
    
    model.train()
    
    weights_before = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            weights_before[name] = p.detach().clone()
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    
    logger.info("Executing Forward Pass...")
    outputs = model(**batch)
    loss = outputs.loss
    logger.info(f"Loss: {loss.item():.4f}")
    
    logger.info("Executing Backward Pass...")
    loss.backward()
    
    total_norm = 0.0
    grad_count = 0
    zero_grad_count = 0
    max_grad = 0.0
    
    q_proj_grad_found = False
    v_proj_grad_found = False
    
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                zero_grad_count += 1
            else:
                grad_count += 1
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_grad = max(max_grad, p.grad.data.abs().max().item())
                
                if "q_proj" in name and param_norm > 0: q_proj_grad_found = True
                if "v_proj" in name and param_norm > 0: v_proj_grad_found = True
                    
    total_norm = total_norm ** 0.5
    
    logger.info(f"\\nActual Trainable Grad Norm : {total_norm:.6f}")
    logger.info(f"Max Absolute Grad Value    : {max_grad:.6f}")
    logger.info(f"Parameters with gradients  : {grad_count}")
    logger.info(f"Parameters missing grads   : {zero_grad_count}\\n")
    
    assert total_norm > 0.0, "GRADIENT NORM IS EXACTLY ZERO!"
    assert q_proj_grad_found, "q_proj did not receive a nonzero gradient!"
    assert v_proj_grad_found, "v_proj did not receive a nonzero gradient!"
    
    logger.info("Executing Optimizer Step...")
    optimizer.step()
    
    total_weight_diff = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad:
            total_weight_diff += (p.detach() - weights_before[name]).norm(2).item()
            
    logger.info(f"Total weight change norm after step: {total_weight_diff:.6f}")
    assert total_weight_diff > 0.0, "Weights did not change after optimizer.step()!"
    
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"PEAK VRAM USED: {peak_gb:.2f} GB")
        
    logger.info(f"\\n{method.upper()} SMOKE TEST FULLY PASSED.\\n")
    
    del model
    del optimizer
    del dataloader
    del dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_forward_backward("lora")
    run_forward_backward("dora")
