import os
import sys
import logging
from src.config import get_base_config
from src.model import load_model_and_tokenizer
from src.data import load_gsm8k_data, prepare_dataset_for_sft
from src.adapters import create_adapter_config, inject_adapter
from src.training import create_trainer
from src.seed import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training(method: str, rank: int, seed: int):
    config = get_base_config()
    set_seed(seed)
    
    # Update config with specific method and rank
    config['adapter']['r'] = rank
    config['experiment'] = {'seed': seed}
    
    model_id = config['model']['id']
    use_4bit = config['model']['use_4bit']
    
    run_name = f"{method}_r{rank}_seed{seed}"
    logger.info(f"Starting training run: {run_name}")
    
    model, tokenizer = load_model_and_tokenizer(model_id, use_4bit=use_4bit)
    
    adapter_config = create_adapter_config(
        method=method,
        r=rank,
        alpha=config['adapter']['alpha'],
        target_modules=config['adapter']['target_modules'],
        dropout=config['adapter']['dropout']
    )
    
    model, trainable, total = inject_adapter(model, adapter_config)
    
    # Load dataset
    train_dataset = load_gsm8k_data(
        split=config['dataset']['train_split'],
        max_samples=config['dataset']['max_samples_pilot'], # Remove or change for full run
        seed=seed
    )
    
    eval_dataset = load_gsm8k_data(
        split=config['dataset']['train_split'], # In a real scenario, use a validation split
        max_samples=20, # Just a tiny evaluation set to track loss during training
        seed=seed
    )
    
    train_dataset = prepare_dataset_for_sft(train_dataset, tokenizer, max_length=config['training']['max_length'])
    eval_dataset = prepare_dataset_for_sft(eval_dataset, tokenizer, max_length=config['training']['max_length'])
    
    output_dir = os.path.join("adapters", run_name)
    
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        output_dir=output_dir,
        run_name=run_name
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving final adapter to {output_dir}")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["lora", "dora"], required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_training(args.method, args.rank, args.seed)
