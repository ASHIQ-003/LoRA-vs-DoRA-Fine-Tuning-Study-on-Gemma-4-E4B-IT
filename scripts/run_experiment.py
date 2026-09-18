import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["kaggle_smoke", "baseline", "pilot", "full"], required=True)
    args = parser.parse_args()
    
    if args.stage == "kaggle_smoke":
        # Run Kaggle smoke test
        print("Running Kaggle Smoke Test...")
        res = subprocess.run([sys.executable, "scripts/kaggle_smoke_test.py"])
        if res.returncode != 0:
            print("Smoke test failed. Halting pipeline.")
            sys.exit(1)
            
    elif args.stage == "baseline":
        # Baseline
        print("Running Baseline...")
        subprocess.run([sys.executable, "scripts/baseline.py"], check=True)
        
    elif args.stage == "pilot":
        # Pilot
        print("Running LoRA Pilot (r=8, seed=42)...")
        subprocess.run([sys.executable, "scripts/train.py", "--method", "lora", "--rank", "8", "--seed", "42"], check=True)
        
        print("Running DoRA Pilot (r=8, seed=42)...")
        subprocess.run([sys.executable, "scripts/train.py", "--method", "dora", "--rank", "8", "--seed", "42"], check=True)
        
        print("\nPilot runs complete. Check the results and report back before full sweep!")
        
    elif args.stage == "full":
        # Full Sweep
        print("Starting full sweep...")
        for method in ["lora", "dora"]:
            for rank in [4, 8, 16, 32]:
                for seed in [42, 123, 456]:
                    print(f"\nRunning {method} rank={rank} seed={seed}")
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
