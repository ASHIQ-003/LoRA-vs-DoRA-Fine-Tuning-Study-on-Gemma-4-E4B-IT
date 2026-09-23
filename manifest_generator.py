# manifest_generator.py
# Manually encodes ONLY verified evaluation results from the previous sweep logs.
# NO result is inferred, estimated, or fabricated.
# Rules:
#   result_confidence = "VERIFIED_FROM_LOG"  -> accuracy is explicitly recorded
#   result_confidence = "NONE"               -> no confirmed evaluation result found
#   rerun_required = True                    -> NONE configs only
#   artifact_available = False               -> all, because /kaggle/working was wiped

import json
import csv
import os

# Works on both Kaggle (/kaggle/working) and local
KAGGLE_WORKING = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
JSON_OUT = os.path.join(KAGGLE_WORKING, "pilot_recovery_manifest.json")
CSV_OUT  = os.path.join(KAGGLE_WORKING, "pilot_recovery_manifest.csv")

# VERIFIED RESULTS - manually encoded from the sweep logs.
# Any key NOT in this dict has NO confirmed accuracy.
VERIFIED = {
    # LoRA - all seeds evaluated except r=32 seed=456 (eval was interrupted)
    ("lora",  4,  42): 0.40,
    ("lora",  4, 123): 0.55,
    ("lora",  4, 456): 0.45,
    ("lora",  8,  42): 0.50,
    ("lora",  8, 123): 0.50,
    ("lora",  8, 456): 0.45,
    ("lora", 16,  42): 0.40,
    ("lora", 16, 123): 0.45,
    ("lora", 16, 456): 0.45,
    ("lora", 32,  42): 0.45,
    ("lora", 32, 123): 0.50,
    # lora r=32 seed=456: training completed, evaluation INTERRUPTED - NOT included
    # DoRA - partial results
    ("dora",  4,  42): 0.40,
    ("dora",  8,  42): 0.40,
    ("dora",  8, 123): 0.50,
    ("dora", 16,  42): 0.45,
    ("dora", 16, 123): 0.45,
    ("dora", 16, 456): 0.45,
    ("dora", 32,  42): 0.45,
}

# lora r=32 seed=456: training artifact exists, only evaluation needs to run
EVAL_ONLY = {("lora", 32, 456)}

# The 5 that need full retrain + eval (no verified accuracy at all)
RERUN_REQUIRED = {
    ("dora",  4, 123),
    ("dora",  4, 456),
    ("dora",  8, 456),
    ("dora", 32, 123),
    ("dora", 32, 456),
}

METHODS = ["lora", "dora"]
RANKS   = [4, 8, 16, 32]
SEEDS   = [42, 123, 456]

json_data = []
csv_data  = []

print("=" * 110)
print("{:<8} | {:<4} | {:<4} | {:<10} | {:<10} | {:<10} | {:<10} | {:<20}".format(
    "Method", "Rank", "Seed", "Accuracy", "Artifact?", "Rerun?", "EvalOnly?", "Confidence"))
print("-" * 110)

verified_count = 0
rerun_count    = 0

for m in METHODS:
    for r in RANKS:
        for s in SEEDS:
            key = (m, r, s)
            acc = VERIFIED.get(key)
            artifact_available = False

            if acc is not None:
                verified_count   += 1
                rerun_required    = False
                eval_only         = False
                result_confidence = "VERIFIED_FROM_LOG"
                logged_src        = "manually_encoded_from_sweep_logs"
                acc_str           = "{:.2f}%".format(acc * 100)
            elif key in EVAL_ONLY:
                rerun_required    = True
                eval_only         = True
                result_confidence = "NONE"
                logged_src        = None
                acc_str           = "N/A"
                rerun_count      += 1
            else:
                rerun_required    = True
                eval_only         = False
                result_confidence = "NONE"
                logged_src        = None
                acc_str           = "N/A"
                rerun_count      += 1

            entry = {
                "method":               m,
                "rank":                 r,
                "seed":                 s,
                "logged_accuracy":      acc,
                "logged_result_source": logged_src,
                "artifact_available":   artifact_available,
                "rerun_required":       rerun_required,
                "eval_only":            eval_only,
                "result_confidence":    result_confidence,
            }
            json_data.append(entry)
            csv_data.append([
                m, r, s, acc, logged_src,
                artifact_available, rerun_required, eval_only, result_confidence
            ])

            print("{:<8} | {:<4} | {:<4} | {:<10} | {:<10} | {:<10} | {:<10} | {:<20}".format(
                m.upper(), r, s, acc_str,
                str(artifact_available), str(rerun_required),
                str(eval_only), result_confidence))

print("-" * 110)
print("VERIFIED RESULTS : {}/24".format(verified_count))
print("RERUN REQUIRED   : {}/24  (5 full reruns + 1 eval-only)".format(rerun_count))
print("=" * 110)

# Save manifest
with open(JSON_OUT, "w") as f:
    json.dump(json_data, f, indent=2)

with open(CSV_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "method", "rank", "seed", "logged_accuracy",
        "logged_result_source", "artifact_available",
        "rerun_required", "eval_only", "result_confidence"
    ])
    writer.writerows(csv_data)

print("\nJSON manifest -> {}".format(JSON_OUT))
print("CSV  manifest -> {}".format(CSV_OUT))

print("\n--- Configurations requiring compute ---")
for entry in json_data:
    if entry["rerun_required"]:
        tag = " [EVAL ONLY]" if entry["eval_only"] else " [FULL RERUN]"
        print("  {} r={} seed={}{}".format(
            entry["method"].upper(), entry["rank"], entry["seed"], tag))
