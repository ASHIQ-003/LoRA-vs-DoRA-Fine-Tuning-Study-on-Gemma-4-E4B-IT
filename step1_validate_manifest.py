# step1_validate_manifest.py
# STEP 1 - MANIFEST VALIDATION ONLY
# Reads pilot_recovery_manifest.json and validates it BEFORE any computation.
#
# Expected counts:
#   VERIFIED_FROM_LOG   : 18
#   EVAL_ONLY           :  1  (LoRA r=32 seed=456 - training done, eval interrupted)
#   FULL_RERUN_REQUIRED :  5  (DoRA r=4/123, r=4/456, r=8/456, r=32/123, r=32/456)
#
# DO NOT proceed to Step 2 unless this cell ends with:
#   [OK] MANIFEST VALIDATED - safe to proceed

import json
import os
import sys

KAGGLE_WORKING = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
MANIFEST_PATH  = os.path.join(KAGGLE_WORKING, "pilot_recovery_manifest.json")

# 1. Load manifest
if not os.path.exists(MANIFEST_PATH):
    print("[ERROR] Manifest not found at " + MANIFEST_PATH)
    print("        Run manifest_generator.py first.")
    sys.exit(1)

with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)

if len(manifest) != 24:
    print("[ERROR] Expected 24 entries, found {}".format(len(manifest)))
    sys.exit(1)

# 2. Classify each entry
VERIFIED   = []
EVAL_ONLY  = []
FULL_RERUN = []

for e in manifest:
    conf   = e.get("result_confidence", "NONE")
    evonly = e.get("eval_only", False)
    rerun  = e.get("rerun_required", True)
    acc    = e.get("logged_accuracy")

    if conf == "VERIFIED_FROM_LOG" and acc is not None and not rerun:
        e["_category"] = "VERIFIED_FROM_LOG"
        VERIFIED.append(e)
    elif rerun and evonly:
        e["_category"] = "EVAL_ONLY"
        EVAL_ONLY.append(e)
    elif rerun and not evonly:
        e["_category"] = "FULL_RERUN_REQUIRED"
        FULL_RERUN.append(e)
    else:
        e["_category"] = "UNKNOWN(conf={}, rerun={}, evonly={})".format(conf, rerun, evonly)

# 3. Print full 24-row table
print("=" * 80)
print("  PILOT RECOVERY MANIFEST - FULL 24-RUN STATUS")
print("=" * 80)
print("{:<3} | {:<6} | {:<4} | {:<4} | {:<9} | {:<22} | {:<9}".format(
    "#", "Method", "Rank", "Seed", "Accuracy", "Category", "Artifact?"))
print("-" * 80)

ordered = {(e["method"], e["rank"], e["seed"]): e for e in manifest}
row = 0
for m in ["lora", "dora"]:
    for r in [4, 8, 16, 32]:
        for s in [42, 123, 456]:
            row += 1
            e = ordered.get((m, r, s))
            if e is None:
                print("{:<3} | {:<6} | {:<4} | {:<4} | {:<9} | {:<22} | {:<9}".format(
                    row, m.upper(), r, s, "MISSING", "NOT IN MANIFEST", "?"))
                continue
            acc = e.get("logged_accuracy")
            acc_str = "{:.2f}%".format(acc * 100).rjust(7) if acc is not None else "   N/A "
            cat = e.get("_category", "UNKNOWN")
            art = str(e.get("artifact_available", False))
            print("{:<3} | {:<6} | {:<4} | {:<4} | {:<9} | {:<22} | {:<9}".format(
                row, m.upper(), r, s, acc_str, cat, art))

print("-" * 80)

# 4. Count summary
n_verified  = len(VERIFIED)
n_eval_only = len(EVAL_ONLY)
n_rerun     = len(FULL_RERUN)
n_unknown   = 24 - n_verified - n_eval_only - n_rerun

print("")
print("  VERIFIED_FROM_LOG   : {}/24".format(n_verified))
print("  EVAL_ONLY           : {}/24".format(n_eval_only))
print("  FULL_RERUN_REQUIRED : {}/24".format(n_rerun))
if n_unknown:
    print("  UNKNOWN/MISCODED    : {}/24  <-- PROBLEM".format(n_unknown))
print("  TOTAL               : {}/24".format(n_verified + n_eval_only + n_rerun + n_unknown))

# 5. Detail the 6 compute tasks
print("\n--- Compute tasks required (DO NOT launch yet) ---")
if EVAL_ONLY:
    print("\n  EVAL ONLY (training artifact already exists):")
    for e in EVAL_ONLY:
        print("    {} r={} seed={}".format(e["method"].upper(), e["rank"], e["seed"]))
if FULL_RERUN:
    print("\n  FULL RERUN (train + evaluate from scratch):")
    for e in FULL_RERUN:
        print("    {} r={} seed={}".format(e["method"].upper(), e["rank"], e["seed"]))

# 6. Validation gate
print("\n" + "=" * 80)
PASS = (n_verified == 18) and (n_eval_only == 1) and (n_rerun == 5) and (n_unknown == 0)
if PASS:
    print("[OK] MANIFEST VALIDATED - safe to proceed to Step 2")
    print("     18 VERIFIED + 1 EVAL_ONLY + 5 FULL_RERUN = 24 total")
else:
    print("[FAIL] MANIFEST VALIDATION FAILED - DO NOT launch any runs")
    if n_verified  != 18: print("  Expected 18 VERIFIED_FROM_LOG,   got {}".format(n_verified))
    if n_eval_only !=  1: print("  Expected  1 EVAL_ONLY,           got {}".format(n_eval_only))
    if n_rerun     !=  5: print("  Expected  5 FULL_RERUN_REQUIRED, got {}".format(n_rerun))
    if n_unknown   !=  0: print("  Expected  0 UNKNOWN,             got {}".format(n_unknown))
    sys.exit(1)
print("=" * 80)
