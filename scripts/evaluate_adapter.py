import os
import sys
import logging
import argparse
from src.config import get_base_config
from src.model import load_model_and_tokenizer
from src.data import load_gsm8k_data
from src.evaluation import evaluate_model, save_results
from src.seed import set_seed
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_evaluation(method: str, rank: int, seed: int):
    config = get_base_config()
    set_seed(seed)
    
    model_id = config['model']['id']
    use_4bit = config['model']['use_4bit']
    run_name = f"{method}_r{rank}_seed{seed}"
    adapter_path = os.path.join("adapters", run_name)
    
    logger.info(f"Running evaluation for {run_name}")
    
    model, tokenizer = load_model_and_tokenizer(model_id, use_4bit=use_4bit)
    
    # Load adapter
    logger.info(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    test_dataset = load_gsm8k_data(
        split=config['dataset']['test_split'], 
        max_samples=config['dataset']['max_samples_pilot'], 
        seed=seed
    )
    
    accuracy, results = evaluate_model(
        model, 
        tokenizer, 
        test_dataset, 
        max_new_tokens=config['evaluation']['max_new_tokens'],
        batch_size=4
    )
    
    output_path = os.path.join("results", "raw", f"{run_name}.json")
    metadata = {
        "model_id": model_id,
        "method": method,
        "rank": rank,
        "seed": seed,
        "use_4bit": use_4bit,
        "samples": len(test_dataset)
    }
    
    save_results(results, accuracy, output_path, metadata)
    logger.info(f"Accuracy for {run_name}: {accuracy*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["lora", "dora"], required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_evaluation(args.method, args.rank, args.seed)
