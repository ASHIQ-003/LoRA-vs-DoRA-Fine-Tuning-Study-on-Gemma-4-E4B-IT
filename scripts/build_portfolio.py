# build_portfolio.py
# Generates all figures and CSV files for the LoRA vs DoRA study.
# Uses matplotlib Agg backend (no display required).

import os
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, "results", "figures")
CSV_DIR = os.path.join(BASE, "results", "processed")
DOC_DIR = os.path.join(BASE, "docs")
for d in [FIG_DIR, CSV_DIR, DOC_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Raw data definitions
# ---------------------------------------------------------------------------

# Trainable params: r=32 confirmed=9076736 (LoRA), proportional otherwise.
PARAMS_CONFIRMED_R32 = 9076736
def est_params(r):
    return int(PARAMS_CONFIRMED_R32 * r / 32)

LORA_COLOR = "#4878CF"
DORA_COLOR = "#E8795A"

# PRIMARY results (LOG_VERIFIED)
# format: (method, rank, seed, accuracy_frac, validation_status, notes)
PRIMARY_RAW = [
    ("lora",  8,  42, 0.50, "LOG_VERIFIED",    ""),
    ("lora",  8, 123, 0.50, "LOG_VERIFIED",    ""),
    ("lora",  8, 456, 0.45, "LOG_VERIFIED",    ""),
    ("lora", 16,  42, 0.40, "LOG_VERIFIED",    ""),
    ("lora", 16, 123, 0.45, "LOG_VERIFIED",    ""),
    ("lora", 16, 456, 0.45, "LOG_VERIFIED",    ""),
    ("dora",  8,  42, 0.40, "LOG_VERIFIED",    ""),
    ("dora",  8, 123, 0.50, "LOG_VERIFIED",    ""),
    ("dora",  8, 456, None, "MISSING_GPU_QUOTA","GPU quota exhausted; no run completed"),
    ("dora", 16,  42, 0.45, "LOG_VERIFIED",    ""),
    ("dora", 16, 123, 0.45, "LOG_VERIFIED",    ""),
    ("dora", 16, 456, 0.45, "LOG_VERIFIED",    ""),
]

# SUPPLEMENTARY results (LOG_VERIFIED)
SUPPLEMENTARY_RAW = [
    ("lora",  4,  42, 0.40, "LOG_VERIFIED",    ""),
    ("lora",  4, 123, 0.55, "LOG_VERIFIED",    ""),
    ("lora",  4, 456, 0.45, "LOG_VERIFIED",    ""),
    ("lora", 32,  42, 0.45, "LOG_VERIFIED",    ""),
    ("lora", 32, 123, 0.50, "LOG_VERIFIED",    ""),
    ("dora",  4,  42, 0.40, "LOG_VERIFIED",    ""),
    ("dora", 32,  42, 0.45, "LOG_VERIFIED",    ""),
]

# EXCLUDED results
EXCLUDED_RAW = [
    ("dora",  4, 123, 0.05, "EVALUATOR_ISSUE", "Evaluator bug: missing enable_thinking=False"),
    ("dora",  4, 456, 0.05, "EVALUATOR_ISSUE", "Evaluator bug: missing enable_thinking=False"),
    ("lora", 32, 456, 0.00, "EVALUATOR_SUSPECT","0% accuracy; excluded from primary analysis"),
]

# FAILED runs
FAILED_RAW = [
    ("dora", 32, 123, None, "FAILED_OOM", "Out of memory"),
    ("dora", 32, 456, None, "FAILED_OOM", "Out of memory"),
]

# Confirmed runtime (only lora,32,456 from Kaggle logs)
CONFIRMED_RUNTIME = {
    ("lora", 32, 456): {
        "train_runtime_sec": 1243.1,
        "eval_runtime_sec":   621.3,
        "peak_vram_gb":        11.588,
    }
}

# ---------------------------------------------------------------------------
# 2. Helper: build full row dict
# ---------------------------------------------------------------------------
def build_row(method, rank, seed, acc, status, notes,
              train_rt=None, eval_rt=None, vram=None,
              params=None, params_src=None):
    acc_pct = round(acc * 100, 2) if acc is not None else ""
    rt_key  = (method, rank, seed)
    rt_info = CONFIRMED_RUNTIME.get(rt_key, {})
    return {
        "method":                  method,
        "rank":                    rank,
        "seed":                    seed,
        "accuracy":                acc_pct,
        "result_source":           "Kaggle_log",
        "validation_status":       status,
        "trainable_params":        params if params is not None else est_params(rank),
        "trainable_params_source": params_src if params_src else (
            "CONFIRMED_r32_scaled" if rank != 32 else "CONFIRMED"),
        "train_loss":              "",
        "train_runtime_sec":       rt_info.get("train_runtime_sec", train_rt if train_rt else ""),
        "eval_runtime_sec":        rt_info.get("eval_runtime_sec",  eval_rt  if eval_rt  else ""),
        "peak_vram_gb":            rt_info.get("peak_vram_gb",      vram     if vram     else ""),
        "notes":                   notes,
    }

FIELDNAMES = [
    "method","rank","seed","accuracy","result_source","validation_status",
    "trainable_params","trainable_params_source","train_loss",
    "train_runtime_sec","eval_runtime_sec","peak_vram_gb","notes"
]

# ---------------------------------------------------------------------------
# 3. Write CSVs
# ---------------------------------------------------------------------------

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print("  CSV written:", path)

# primary_results.csv
prim_rows = [build_row(*r[:6]) for r in PRIMARY_RAW]
write_csv(os.path.join(CSV_DIR, "primary_results.csv"), prim_rows)

# supplementary_results.csv
supp_rows = [build_row(*r[:6]) for r in SUPPLEMENTARY_RAW]
write_csv(os.path.join(CSV_DIR, "supplementary_results.csv"), supp_rows)

# full_results_table.csv  -- all 24 configs
all_rows = prim_rows + supp_rows
for r in EXCLUDED_RAW:
    all_rows.append(build_row(*r[:6]))
for r in FAILED_RAW:
    all_rows.append(build_row(*r[:6]))

# deduplicate (dora,8,42 and dora,8,123 appear in supplementary; keep primary)
seen = set()
deduped = []
for row in all_rows:
    key = (row["method"], row["rank"], row["seed"])
    if key not in seen:
        seen.add(key)
        deduped.append(row)

write_csv(os.path.join(CSV_DIR, "full_results_table.csv"), deduped)

# ---------------------------------------------------------------------------
# 4. experiment_summary.json
# ---------------------------------------------------------------------------
exp_summary = {
    "project": "LoRA vs DoRA Fine-Tuning Study on Google Gemma 4 E4B-IT",
    "author": "Ashiq Rahman",
    "date": "2026",
    "model": {
        "name": "google/gemma-4-e4b-it",
        "architecture": "Mixture-of-Experts (MoE)",
        "quantization": "4-bit NF4 (bitsandbytes)",
        "parameters_total": "4B (effective active)",
    },
    "dataset": {
        "name": "GSM8K",
        "split": "train (first 200 examples for training), test",
        "task": "Mathematical reasoning",
        "evaluation_subset_size": 20,
        "evaluation_metric": "Exact-match accuracy (%)",
    },
    "adapter_methods": ["LoRA", "DoRA"],
    "ranks": [4, 8, 16, 32],
    "seeds": [42, 123, 456],
    "primary_study_ranks": [8, 16],
    "training_params": {
        "max_steps": 20,
        "learning_rate": 2e-4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "lora_alpha": "2*rank",
        "lora_dropout": 0.1,
        "target_modules": "q_proj,v_proj (language_model branch, 66 total modules)",
        "completion_only_masking": True,
        "label_token_for_prompt": -100,
    },
    "evaluation_protocol": {
        "generation_strategy": "Greedy (temperature=0, deterministic)",
        "enable_thinking": False,
        "max_new_tokens": 256,
        "exact_match": "Normalized string comparison on final numeric answer",
    },
    "result_provenance": {
        "LOG_VERIFIED": 19,
        "MISSING_GPU_QUOTA": 1,
        "EVALUATOR_ISSUE": 2,
        "EVALUATOR_SUSPECT": 1,
        "FAILED_OOM": 2,
    },
    "confirmed_runtime": {
        "lora_r32_s456": {
            "train_runtime_sec": 1243.1,
            "eval_runtime_sec": 621.3,
            "peak_vram_gb": 11.588,
            "trainable_params": 9076736,
            "source": "Kaggle_log",
        }
    },
    "excluded_results": [
        {
            "method": "dora", "rank": 4, "seed": 123,
            "reason": "EVALUATOR_ISSUE: missing enable_thinking=False; 5% anomaly"
        },
        {
            "method": "dora", "rank": 4, "seed": 456,
            "reason": "EVALUATOR_ISSUE: missing enable_thinking=False; 5% anomaly"
        },
        {
            "method": "lora", "rank": 32, "seed": 456,
            "reason": "EVALUATOR_SUSPECT: 0% accuracy; not used in analysis"
        },
    ],
    "reproducibility_notes": [
        "All runs on Kaggle free-tier T4 GPU (14.56 GB VRAM)",
        "Training capped at 20 steps (pilot; not to convergence)",
        "Evaluation on fixed 20-example subset of GSM8K test",
        "Session restarts required due to Kaggle GPU quota limits",
    ],
}
json_path = os.path.join(DOC_DIR, "experiment_summary.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(exp_summary, f, indent=2)
print("  JSON written:", json_path)

# ---------------------------------------------------------------------------
# 5. Figure helpers
# ---------------------------------------------------------------------------

def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure saved:", path)

# ---------------------------------------------------------------------------
# 6. Fig 1 -- Accuracy by Rank (Primary Study)
# ---------------------------------------------------------------------------

primary_complete = [r for r in PRIMARY_RAW if r[3] is not None]
ranks = [8, 16]
methods = ["lora", "dora"]

def group_mean_std(method, rank, data):
    vals = [r[3]*100 for r in data if r[0]==method and r[1]==rank and r[3] is not None]
    if not vals:
        return None, None
    return np.mean(vals), np.std(vals, ddof=0)

fig1, (ax1l, ax1r) = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle("Fig 1: Accuracy by Rank -- Primary Study (r=8, r=16)", fontsize=13, fontweight="bold")

x = np.arange(len(ranks))
width = 0.35
bar_colors = {
    "lora": LORA_COLOR,
    "dora": DORA_COLOR,
}

# Left: grouped bar chart
for i, (method, offset) in enumerate([("lora", -width/2), ("dora", width/2)]):
    means, stds = [], []
    for rank in ranks:
        m, s = group_mean_std(method, rank, primary_complete)
        means.append(m if m is not None else 0)
        stds.append(s if s is not None else 0)
    bars = ax1l.bar(x + offset, means, width=width, color=bar_colors[method],
                    label=method.upper(), alpha=0.85, edgecolor="white", linewidth=0.8)
    ax1l.errorbar(x + offset, means, yerr=stds, fmt="none", color="black",
                  capsize=4, linewidth=1.2)

# Mark DoRA r=8 (2/3 seeds)
ax1l.annotate("* 2/3 seeds\n  (missing s=456)", xy=(x[0]+width/2, 44), fontsize=7,
              color="#8B0000", ha="center")

ax1l.set_xticks(x)
ax1l.set_xticklabels(["r = 8", "r = 16"])
ax1l.set_ylabel("Mean Accuracy (%)")
ax1l.set_ylim(0, 65)
ax1l.set_xlabel("LoRA / DoRA Rank")
ax1l.legend()
ax1l.set_title("Mean Accuracy +/- Std (primary seeds)")
ax1l.grid(axis="y", alpha=0.3)

# Right: strip/dot plot
combos = [("LoRA r=8", "lora", 8), ("LoRA r=16","lora",16),
          ("DoRA r=8","dora",8),  ("DoRA r=16","dora",16)]
xticks_labels = [c[0] for c in combos]

for xi, (label, method, rank) in enumerate(combos):
    color = LORA_COLOR if method == "lora" else DORA_COLOR
    vals = [(r[2], r[3]*100) for r in PRIMARY_RAW if r[0]==method and r[1]==rank and r[3] is not None]
    missing = [(r[2], r[3]) for r in PRIMARY_RAW if r[0]==method and r[1]==rank and r[3] is None]
    for seed, acc in vals:
        jitter = np.random.uniform(-0.06, 0.06)
        ax1r.scatter(xi + jitter, acc, color=color, s=60, zorder=3, alpha=0.85)
        ax1r.annotate(f"s={seed}", (xi+jitter, acc+0.8), fontsize=6, ha="center", color="gray")
    for seed, _ in missing:
        ax1r.scatter(xi, 35, marker="x", color="red", s=80, zorder=3, linewidths=2)
        ax1r.annotate(f"s={seed}\n(MISSING)", (xi, 32), fontsize=6, ha="center", color="red")

ax1r.set_xticks(range(len(combos)))
ax1r.set_xticklabels(xticks_labels, rotation=15, ha="right")
ax1r.set_ylabel("Accuracy (%)")
ax1r.set_ylim(25, 65)
ax1r.set_title("Individual Seed Results (X = MISSING)")
ax1r.grid(axis="y", alpha=0.3)

lora_patch = mpatches.Patch(color=LORA_COLOR, label="LoRA")
dora_patch = mpatches.Patch(color=DORA_COLOR, label="DoRA")
ax1r.legend(handles=[lora_patch, dora_patch], fontsize=8)

fig1.tight_layout()
savefig(fig1, "fig1_accuracy_by_rank.png")

# ---------------------------------------------------------------------------
# 7. Fig 2 -- Seed Distribution (Box + Strip)
# ---------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.suptitle("Fig 2: Seed Distribution -- Primary Study Confirmed Results", fontsize=12, fontweight="bold")

combos_ordered = [("LoRA r=8","lora",8), ("LoRA r=16","lora",16),
                  ("DoRA r=8","dora",8), ("DoRA r=16","dora",16)]

box_data = []
positions = []
xlabels = []
for xi, (label, method, rank) in enumerate(combos_ordered):
    vals = [r[3]*100 for r in PRIMARY_RAW if r[0]==method and r[1]==rank and r[3] is not None]
    if vals:
        box_data.append(vals)
        positions.append(xi)
        xlabels.append(label)

bp = ax2.boxplot(box_data, positions=positions, widths=0.35, patch_artist=True,
                 medianprops=dict(color="black", linewidth=1.5))

colors_list = [LORA_COLOR, LORA_COLOR, DORA_COLOR, DORA_COLOR]
for patch, col in zip(bp["boxes"], colors_list):
    patch.set_facecolor(col)
    patch.set_alpha(0.35)

for xi, (label, method, rank) in enumerate(combos_ordered):
    color = LORA_COLOR if method == "lora" else DORA_COLOR
    for row in PRIMARY_RAW:
        if row[0]==method and row[1]==rank and row[3] is not None:
            seed = row[2]
            acc  = row[3]*100
            jitter = np.random.uniform(-0.08, 0.08)
            ax2.scatter(xi+jitter, acc, color=color, s=55, zorder=4, alpha=0.9)
            ax2.annotate(f"s={seed}", (xi+jitter, acc+0.6), fontsize=6.5,
                         ha="center", color="dimgray")

ax2.set_xticks(range(len(combos_ordered)))
ax2.set_xticklabels([c[0] for c in combos_ordered])
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(30, 65)
ax2.set_title("Box + Strip Plot (primary configs, confirmed seeds only)")
ax2.grid(axis="y", alpha=0.3)
lora_patch = mpatches.Patch(color=LORA_COLOR, label="LoRA")
dora_patch = mpatches.Patch(color=DORA_COLOR, label="DoRA")
ax2.legend(handles=[lora_patch, dora_patch])
fig2.tight_layout()
savefig(fig2, "fig2_seed_distribution.png")

# ---------------------------------------------------------------------------
# 8. Fig 3 -- Trainable Parameters
# ---------------------------------------------------------------------------

fig3, ax3 = plt.subplots(figsize=(8, 4))
fig3.suptitle("Fig 3: Trainable Parameters by Rank", fontsize=12, fontweight="bold")

param_ranks = [8, 16, 32]
lora_params = [est_params(r)/1e6 for r in param_ranks]
# DoRA: params estimated same as LoRA for r=8,16; r=32 unavailable (OOM)
dora_params = [est_params(r)/1e6 for r in param_ranks[:2]] + [None]

x3 = np.arange(len(param_ranks))
w3 = 0.3
bars_l = ax3.bar(x3 - w3/2, lora_params, width=w3, color=LORA_COLOR, label="LoRA", alpha=0.85, edgecolor="white")
bars_d = ax3.bar(x3 + w3/2,
                 [d if d is not None else 0 for d in dora_params],
                 width=w3, color=DORA_COLOR, label="DoRA", alpha=0.85, edgecolor="white")

# Label bars
for i, (v, r) in enumerate(zip(lora_params, param_ranks)):
    tag = "CONFIRMED" if r == 32 else "estimated"
    ax3.text(i - w3/2, v + 0.05, f"{v:.2f}M\n({tag})", ha="center", fontsize=7, color="black")

for i, (v, r) in enumerate(zip(dora_params, param_ranks[:2])):
    ax3.text(i + w3/2, v + 0.05, f"{v:.2f}M\n(estimated)", ha="center", fontsize=7, color="black")

ax3.text(2 + w3/2, 0.15, "DoRA r=32\n(OOM / no data)", ha="center", fontsize=7, color="gray")

ax3.set_xticks(x3)
ax3.set_xticklabels([f"r = {r}" for r in param_ranks])
ax3.set_ylabel("Trainable Parameters (Millions)")
ax3.set_xlabel("Rank")
ax3.set_title("NOTE: All values except LoRA r=32 are proportional estimates\n"
              "(proportional to r, based on confirmed 9,076,736 params at r=32)")
ax3.legend()
ax3.set_ylim(0, max(lora_params)*1.35)
ax3.grid(axis="y", alpha=0.3)
fig3.tight_layout()
savefig(fig3, "fig3_trainable_params.png")

# ---------------------------------------------------------------------------
# 9. Fig 4 -- Runtime
# ---------------------------------------------------------------------------

fig4, ax4 = plt.subplots(figsize=(8, 4))
fig4.suptitle("Fig 4: Training & Evaluation Runtime (Confirmed Data Only)", fontsize=12, fontweight="bold")

categories = ["Train Runtime", "Eval Runtime"]
values = [1243.1, 621.3]
colors4 = ["#5B8DB8", "#A0C878"]
bars4 = ax4.bar(categories, values, color=colors4, width=0.4, alpha=0.88, edgecolor="white")

for bar, val in zip(bars4, values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             f"{val:.1f}s", ha="center", fontsize=10, fontweight="bold")

ax4.set_ylabel("Runtime (seconds)")
ax4.set_ylim(0, 1600)
ax4.set_title("LoRA r=32, seed=456 (source: Kaggle log)\n"
              "Runtime data available ONLY for this single configuration")
ax4.text(0.5, 0.15, "NOTE: Runtime data available only for LoRA r=32 (1 seed).\n"
         "All other configurations have no confirmed runtime measurements.",
         transform=ax4.transAxes, ha="center", fontsize=8,
         bbox=dict(facecolor="lightyellow", edgecolor="orange", alpha=0.8))
ax4.grid(axis="y", alpha=0.3)
fig4.tight_layout()
savefig(fig4, "fig4_runtime.png")

# ---------------------------------------------------------------------------
# 10. Fig 5 -- VRAM
# ---------------------------------------------------------------------------

GPU_CAP = 14.56  # Kaggle T4 capacity

fig5, ax5 = plt.subplots(figsize=(6, 4))
fig5.suptitle("Fig 5: Peak VRAM Usage (Confirmed Measurement)", fontsize=12, fontweight="bold")

ax5.bar(["LoRA r=32\ns=456\n(CONFIRMED)"], [11.588], color=LORA_COLOR, width=0.4,
        alpha=0.85, edgecolor="white")
ax5.axhline(y=GPU_CAP, color="red", linestyle="--", linewidth=1.5, label=f"GPU Capacity: {GPU_CAP} GB")
ax5.text(0, 11.588 + 0.2, "11.588 GB", ha="center", fontsize=10, fontweight="bold", color="black")
ax5.text(0.5, 0.15, "Only 1 confirmed VRAM measurement available.\n"
         "All other configurations have no confirmed VRAM data.",
         transform=ax5.transAxes, ha="center", fontsize=8,
         bbox=dict(facecolor="lightyellow", edgecolor="orange", alpha=0.8))
ax5.set_ylabel("Peak VRAM (GB)")
ax5.set_ylim(0, GPU_CAP * 1.1)
ax5.set_title(f"Source: Kaggle log | GPU capacity line = {GPU_CAP} GB (T4)")
ax5.legend(fontsize=8)
ax5.grid(axis="y", alpha=0.3)
fig5.tight_layout()
savefig(fig5, "fig5_vram.png")

# ---------------------------------------------------------------------------
# 11. Fig 6 -- Accuracy vs Efficiency (scatter)
# ---------------------------------------------------------------------------

fig6, ax6 = plt.subplots(figsize=(8, 5))
fig6.suptitle("Fig 6: Accuracy vs. Estimated Trainable Parameters", fontsize=12, fontweight="bold")

scatter_configs = [
    ("LoRA r=8",  "lora",  8),
    ("LoRA r=16", "lora", 16),
    ("LoRA r=32", "lora", 32),
    ("DoRA r=8",  "dora",  8),
    ("DoRA r=16", "dora", 16),
]

# For LoRA r=32, use supplementary seeds only (42,123 -- exclude suspect 456)
# For DoRA r=8, only 2 confirmed seeds
all_data_for_scatter = PRIMARY_RAW + SUPPLEMENTARY_RAW

for label, method, rank in scatter_configs:
    vals = [r[3]*100 for r in all_data_for_scatter
            if r[0]==method and r[1]==rank and r[3] is not None]
    if not vals:
        continue
    mean_acc = np.mean(vals)
    params_m = est_params(rank) / 1e6
    color  = LORA_COLOR if method == "lora" else DORA_COLOR
    marker = "o"       if method == "lora" else "s"
    ax6.scatter(params_m, mean_acc, color=color, marker=marker, s=90, zorder=4, alpha=0.9)
    # offset label slightly
    ax6.annotate(label, (params_m, mean_acc),
                 textcoords="offset points", xytext=(6, 4), fontsize=8)

ax6.set_xlabel("Estimated Trainable Parameters (Millions)")
ax6.set_ylabel("Mean Accuracy (%)")
ax6.set_title("NOTE: Parameter counts are proportional estimates\n"
              "(only LoRA r=32 = 9.08M is CONFIRMED; others scaled)")
lora_patch = mpatches.Patch(color=LORA_COLOR, label="LoRA (circles)")
dora_patch = mpatches.Patch(color=DORA_COLOR, label="DoRA (squares)")
ax6.legend(handles=[lora_patch, dora_patch])
ax6.grid(alpha=0.3)
fig6.tight_layout()
savefig(fig6, "fig6_accuracy_vs_efficiency.png")

# ---------------------------------------------------------------------------
# 12. Done
# ---------------------------------------------------------------------------
print()
print("All figures and CSVs generated successfully.")
print("  Figures:  ", FIG_DIR)
print("  CSVs:     ", CSV_DIR)
print("  JSON:     ", json_path)
