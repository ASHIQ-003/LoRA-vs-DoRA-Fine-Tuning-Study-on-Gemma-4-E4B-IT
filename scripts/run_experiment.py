import os
import sys
import subprocess
import logging
import time
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pilot():
    methods = ["lora", "dora"]
    rank = 8
    seed = 42
    
    for method in methods:
        logger.info(f"==================================================")
        logger.info(f"STARTING {method.upper()} PILOT (r={rank}, seed={seed})")
        logger.info(f"==================================================")
        
        start_time = time.time()
        
        # Training
        logger.info(f"--- Training {method.upper()} ---")
        train_cmd = [sys.executable, "scripts/train.py", "--method", method, "--rank", str(rank), "--seed", str(seed)]
        subprocess.run(train_cmd, check=True)
        
        train_time = time.time() - start_time
        logger.info(f"Training time for {method.upper()}: {train_time / 60:.2f} minutes")
        
        # Evaluation
        logger.info(f"--- Evaluating {method.upper()} ---")
        eval_cmd = [sys.executable, "scripts/evaluate_adapter.py", "--method", method, "--rank", str(rank), "--seed", str(seed)]
        subprocess.run(eval_cmd, check=True)
        
        logger.info(f"{method.upper()} pilot completed.\\n")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=["pilot", "sweep", "evaluate_only"], default="pilot")
    args = parser.parse_args()
    
    if args.stage == "pilot":
        run_pilot()
    else:
        logger.error(f"Stage {args.stage} not authorized yet. Doing pilot only.")
        
if __name__ == "__main__":
    main()
