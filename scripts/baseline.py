import os
import sys
import logging
from src.config import get_base_config
from src.model import load_model_and_tokenizer
from src.data import load_gsm8k_data
from src.evaluation import evaluate_model, save_results
from src.seed import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_baseline():
    config = get_base_config()
    set_seed(42)
    
    model_id = config['model']['id']
    use_4bit = config['model']['use_4bit']
    
    logger.info(f"Running baseline for {model_id}")
    
    model, tokenizer = load_model_and_tokenizer(model_id, use_4bit=use_4bit)
    
    test_dataset = load_gsm8k_data(
        split=config['dataset']['test_split'], 
        max_samples=config['dataset']['max_samples_pilot'], # Use a subset for baseline pilot testing initially
        seed=42
    )
    
    accuracy, results = evaluate_model(
        model, 
        tokenizer, 
        test_dataset, 
        max_new_tokens=config['evaluation']['max_new_tokens'],
        batch_size=4
    )
    
    output_path = os.path.join("results", "raw", "baseline.json")
    metadata = {
        "model_id": model_id,
        "method": "baseline",
        "use_4bit": use_4bit,
        "samples": len(test_dataset)
    }
    
    save_results(results, accuracy, output_path, metadata)
    logger.info(f"Baseline accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    run_baseline()
