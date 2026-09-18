import os
import json
import pandas as pd
import glob

def aggregate():
    raw_files = glob.glob("results/raw/*.json")
    all_data = []
    
    for fpath in raw_files:
        if 'baseline' in fpath:
            continue
        with open(fpath, "r") as f:
            data = json.load(f)
            meta = data.get("metadata", {})
            
            all_data.append({
                "Method": meta.get("method"),
                "Rank": meta.get("rank"),
                "Seed": meta.get("seed"),
                "Accuracy": data.get("accuracy", 0),
                "Trainable_Params": meta.get("trainable_params", 0),
                "Total_Params": meta.get("total_params", 0)
            })
            
    if not all_data:
        print("No results to aggregate.")
        return
        
    df = pd.DataFrame(all_data)
    
    # Save raw aggregated
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/aggregated_results.csv", index=False)
    
    # Mean and std by Method and Rank
    grouped = df.groupby(["Method", "Rank"]).agg({
        "Accuracy": ["mean", "std"]
    }).reset_index()
    
    grouped.to_csv("results/tables/summary_statistics.csv", index=False)
    print(grouped)

if __name__ == "__main__":
    aggregate()
