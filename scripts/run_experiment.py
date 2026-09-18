import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "baseline", "pilot", "full"], required=True)
    args = parser.parse_args()
    
    if args.stage == "smoke":
        subprocess.run([sys.executable, "scripts/smoke_test.py"], check=True)
        
    elif args.stage == "baseline":
        subprocess.run([sys.executable, "scripts/baseline.py"], check=True)
        
    elif args.stage == "pilot":
        subprocess.run([sys.executable, "scripts/train.py", "--method", "lora", "--rank", "8"], check=True)
        subprocess.run([sys.executable, "scripts/train.py", "--method", "dora", "--rank", "8"], check=True)
        
    elif args.stage == "full":
        for method in ["lora", "dora"]:
            for rank in [4, 8, 16, 32]:
                for seed in [42, 123, 456]:
                    print(f"Running {method} rank={rank} seed={seed}")
                    subprocess.run([
                        sys.executable, "scripts/train.py", 
                        "--method", method, 
                        "--rank", str(rank),
                        "--seed", str(seed)
                    ], check=True)
                    
        # Aggregate
        subprocess.run([sys.executable, "scripts/aggregate_results.py"], check=True)

if __name__ == "__main__":
    main()
