import os
import sys
import logging
import torch
import time
from src.config import get_base_config
from src.model import load_model_and_tokenizer
from src.data import load_gsm8k_data, prepare_dataset_for_sft
from src.adapters import create_adapter_config, inject_adapter, discover_text_peft_targets
from src.training import create_trainer
from src.seed import set_seed
from src.evaluation import evaluate_model, save_results
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training(method: str, rank: int, seed: int, injection_only: bool = False):
    config = get_base_config()
    set_seed(seed)
    
    config['adapter']['r'] = rank
    config['experiment'] = {'seed': seed}
    
    model_id = config['model']['id']
    use_4bit = config['model']['use_4bit']
    
    run_name = f"{method}_r{rank}_seed{seed}"
    logger.info(f"\\n{'='*60}\\nSTARTING {'INJECTION TEST' if injection_only else 'PILOT RUN'}: {run_name}\\n{'='*60}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    model, tokenizer_or_processor = load_model_and_tokenizer(model_id, use_4bit=use_4bit)
    
    if hasattr(tokenizer_or_processor, "tokenizer") and tokenizer_or_processor.tokenizer is not None:
        tokenizer = tokenizer_or_processor.tokenizer
    else:
        tokenizer = tokenizer_or_processor
        
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    
    text_targets = discover_text_peft_targets(model)
    
    adapter_config = create_adapter_config(
        method=method,
        r=rank,
        alpha=config['adapter']['alpha'],
        target_modules=text_targets,
        dropout=config['adapter']['dropout']
    )
    
    model, trainable, total = inject_adapter(model, adapter_config)
    
    if injection_only:
        logger.info("Injection only flag set. Running manual forward/backward check...")
        
        dataset = load_gsm8k_data(split=config['dataset']['train_split'], max_samples=2, seed=seed)
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
        
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        total_norm = 0.0
        grad_count = 0
        zero_grad_count = 0
        max_grad = 0.0
        
        for name, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is None:
                    zero_grad_count += 1
                else:
                    grad_count += 1
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    max_grad = max(max_grad, p.grad.data.abs().max().item())
                        
        total_norm = total_norm ** 0.5
        
        logger.info(f"\\nActual Trainable Grad Norm : {total_norm:.6f}")
        logger.info(f"Max Absolute Grad Value    : {max_grad:.6f}")
        logger.info(f"Parameters with gradients  : {grad_count}")
        logger.info(f"Parameters missing grads   : {zero_grad_count}\\n")
        
        assert total_norm > 0.0, "GRADIENT NORM IS EXACTLY ZERO!"
        
        optimizer.step()
        
        total_weight_diff = 0.0
        for name, p in model.named_parameters():
            if p.requires_grad:
                total_weight_diff += (p.detach() - weights_before[name]).norm(2).item()
                
        logger.info(f"Total weight change norm after step: {total_weight_diff:.6f}")
        assert total_weight_diff > 0.0, "Weights did not change!"
        
        logger.info(f"INJECTION TEST PASSED FOR {method.upper()}")
        return

    # 200 train / 20 eval
    train_dataset_raw = load_gsm8k_data(split="train", max_samples=200, seed=seed)
    trainer_eval_dataset_raw = load_gsm8k_data(split="test", max_samples=20, seed=seed)
    
    train_dataset = prepare_dataset_for_sft(train_dataset_raw, tokenizer, max_length=config['training']['max_length'])
    trainer_eval_dataset = prepare_dataset_for_sft(trainer_eval_dataset_raw, tokenizer, max_length=config['training']['max_length'])
    
    output_dir = os.path.join("adapters", run_name)
    
    trainer = create_trainer(
        model=model,
        tokenizer_or_processor=tokenizer_or_processor,
        train_dataset=train_dataset,
        eval_dataset=trainer_eval_dataset,
        config=config,
        output_dir=output_dir,
        run_name=run_name
    )
    
    logger.info("Starting training...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds.")
    
    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"PEAK GPU VRAM: {peak_vram_gb:.2f} GB")
    
    logger.info(f"Saving final adapter to {output_dir}")
    trainer.save_model(output_dir)
    
    # Evaluate generation on the held-out 200 example test set
    logger.info(f"Starting GSM8K final-answer evaluation on 200 held-out examples...")
    eval_dataset = load_gsm8k_data(split="test", max_samples=200, seed=seed)
    
    model.eval()
    
    eval_output_file = os.path.join(output_dir, "predictions.json")
    accuracy, results = evaluate_model(model, tokenizer, eval_dataset, max_new_tokens=256, batch_size=4)
    save_results(results, accuracy, eval_output_file, metadata={"run_name": run_name})
    
    logger.info(f"\\n{'='*50}\\nRESULTS FOR {run_name}\\n{'='*50}")
    logger.info(f"Exact Match Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"Training Time: {train_time:.2f} s")
    logger.info(f"Peak VRAM: {peak_vram_gb:.2f} GB")
    logger.info(f"Trainable Params: {trainable}")
    logger.info(f"==================================================\\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["lora", "dora"], required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--injection_only", action="store_true")
    args = parser.parse_args()
    
    run_training(args.method, args.rank, args.seed, injection_only=args.injection_only)
