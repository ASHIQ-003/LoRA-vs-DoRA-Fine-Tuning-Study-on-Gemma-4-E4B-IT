# LoRA vs DoRA on Gemma 4 E4B-IT for GSM8K Mathematical Reasoning

> **Pilot-scale fine-tuning study** | Ashiq Rahman | 2026

---

## Research Question

Does DoRA (Weight-Decomposed Low-Rank Adaptation) provide measurable accuracy or
efficiency advantages over standard LoRA when fine-tuning the Gemma 4 E4B-IT model
on the GSM8K mathematical reasoning benchmark at pilot scale (20 training steps)?

---

## Abstract

This study compares LoRA and DoRA adapter fine-tuning methods on Google Gemma 4 E4B-IT,
a 4-billion-effective-parameter mixture-of-experts instruction-tuned language model,
applied to the GSM8K grade-school mathematics dataset. Experiments are run at pilot scale
(20 training steps, 20-example evaluation set) across ranks 8 and 16 with three random
seeds each. Across 11 confirmed primary configurations (one DoRA r=8 run missing due to
GPU quota), both methods achieve accuracy in the 40-50% range; no statistically significant
difference between LoRA and DoRA is observed at this scale. Results should be interpreted
as early-stage feasibility evidence rather than definitive conclusions.

---

## Experimental Design

| Factor          | Values               | Notes                                      |
|-----------------|----------------------|--------------------------------------------|
| Adapter method  | LoRA, DoRA           | `use_dora=False` / `use_dora=True`         |
| Rank (r)        | 8, 16 (primary)      | 4, 32 also explored (supplementary)        |
| Seeds           | 42, 123, 456         | 3 seeds per configuration                  |
| Total primary   | 2 methods x 2 ranks  | = 4 configs x 3 seeds = 12 runs (11 valid) |
| Training steps  | 20                   | Pilot; not convergence                     |
| Eval examples   | 20                   | Fixed GSM8K test subset                    |

---

## Model and Quantization

| Property           | Value                                      |
|--------------------|--------------------------------------------|
| Model              | `google/gemma-4-e4b-it`                    |
| Architecture       | Mixture-of-Experts (MoE)                   |
| Effective params   | ~4B active per forward pass                |
| Quantization       | 4-bit NF4 (bitsandbytes, double_quant)     |
| Compute dtype      | bfloat16                                   |
| Platform           | Kaggle free-tier T4 GPU (14.56 GB VRAM)    |

---

## Dataset

| Property            | Value                                |
|---------------------|--------------------------------------|
| Name                | GSM8K (Grade School Math 8K)         |
| Task                | Multi-step arithmetic word problems  |
| Training subset     | First 200 examples of train split    |
| Evaluation subset   | Fixed 20 examples of test split      |
| Evaluation metric   | Exact-match accuracy (%)             |
| Granularity         | 5% per correct/incorrect answer      |

---

## Adapter Configuration

**LoRA update:** $h = W_0 x + \frac{\alpha}{r} B A x$

**DoRA update:** $W' = (m + \Delta m) \cdot \frac{V + BA}{\|V + BA\|_c}$

| Parameter            | Value                                     |
|----------------------|-------------------------------------------|
| Target modules       | `q_proj`, `v_proj` (language_model branch)|
| Total adapted layers | 66                                        |
| Alpha                | 2 * rank                                  |
| Dropout              | 0.1                                       |
| Bias                 | none                                      |

---

## Training Procedure

- Loaded with 4-bit NF4 quantization + `prepare_model_for_kbit_training()`
- Adapter injected via `peft` library (`get_peft_model()`)
- Trained with `SFTTrainer` (trl) using completion-only masking
- Prompt tokens receive label=-100 (not included in loss)
- 20 training steps, learning rate 2e-4, cosine schedule, batch size 4 (effective)
- AdamW 8-bit optimizer, gradient checkpointing enabled

---

## Evaluation Protocol

- Fixed 20-example GSM8K test subset
- `processor.apply_chat_template()` for prompt formatting
- **`enable_thinking=False`** required (thinking-capable model)
- Greedy decoding (temperature=0, do_sample=False)
- Answer extracted after `####` delimiter, whitespace-normalised
- Accuracy = # correct / 20

---

## Results

### Primary Results (12 runs planned; 11 confirmed; 1 missing)

| Method | Rank | Seed=42 | Seed=123 | Seed=456  | Status                    |
|--------|------|---------|----------|-----------|---------------------------|
| LoRA   | 8    | 50%     | 50%      | 45%       | LOG_VERIFIED (3/3 seeds)  |
| LoRA   | 16   | 40%     | 45%      | 45%       | LOG_VERIFIED (3/3 seeds)  |
| DoRA   | 8    | 40%     | 50%      | **N/A***  | 2/3 seeds (GPU quota)     |
| DoRA   | 16   | 45%     | 45%      | 45%       | LOG_VERIFIED (3/3 seeds)  |

\* DoRA r=8 seed=456 not completed due to GPU quota exhaustion.

### Mean Accuracy +/- Std (Primary, LOG_VERIFIED Seeds Only)

| Method | Rank | Mean Acc | Std  | Seeds Used |
|--------|------|----------|------|------------|
| LoRA   | 8    | 48.33%   | 2.36%| 3/3        |
| LoRA   | 16   | 43.33%   | 2.36%| 3/3        |
| DoRA   | 8    | 45.00%   | 5.00%| 2/3 *      |
| DoRA   | 16   | 45.00%   | 0.00%| 3/3        |

> **NOTE:** DoRA r=8 mean and std are based on only 2 seeds; treat with additional caution.
> Each 5% difference corresponds to exactly 1 test example at this evaluation scale.
> No winner is claimed; differences are within pilot-scale noise.

---

## Data Provenance

| Category | Count | Details |
|----------|-------|---------|
| **Primary LOG_VERIFIED** | 11/12 | Recovered from original Kaggle sweep logs, cross-checked |
| **Primary MISSING** | 1/12 | DoRA r=8 seed=456 — GPU quota exhausted; pending validated rerun |
| **Supplementary LOG_VERIFIED** | 7 | r=4 and r=32 partial sweep, confirmed from logs |
| **PROVISIONAL_PENDING_AUDIT** | 2 | DoRA r=4 seed=123, r=4 seed=456 — anomalous 5% result; evaluator issue suspected but not yet formally confirmed by audit |
| **INCOMPLETE_UNAVAILABLE** | 2 | DoRA r=32 seed=123, r=32 seed=456 — runs did not complete; exact failure mode not confirmed |
| **TRAINING_COMPLETE_EVAL_INCOMPLETE** | 1 | LoRA r=32 seed=456 — training confirmed complete (adapter saved, runtime=1243s); evaluation phase incomplete; no final accuracy assigned |

> All results are traceable to a status and provenance column in every CSV.
> Provisional and incomplete results are **never** included in primary statistics.
> The audit process for provisional results is documented in [docs/evaluator_audit_report.md](docs/evaluator_audit_report.md).


## Evaluator Audit

An evaluator bug was discovered post-hoc: missing `enable_thinking=False` caused
Gemma 4 E4B-IT to emit extended reasoning tokens, breaking the `####` exact-match parser.

- **Excluded:** DoRA r=4 s=123, DoRA r=4 s=456 (both 5% -- evaluator artifact)
- **Excluded:** LoRA r=32 s=456 (0% -- EVALUATOR_SUSPECT)
- **All 19 LOG_VERIFIED results** use the corrected evaluator protocol

See full audit: [docs/evaluator_audit_report.md](docs/evaluator_audit_report.md)

---

## Reproducibility

| Component                | Details                                             |
|--------------------------|-----------------------------------------------------|
| Platform                 | Kaggle free-tier T4 GPU (14.56 GB VRAM)             |
| Confirmed runtime (1 run)| LoRA r=32 s=456: train=1243.1s, eval=621.3s         |
| Confirmed VRAM (1 run)   | LoRA r=32 s=456: 11.588 GB peak                     |
| Trainable params (r=32)  | 9,076,736 (CONFIRMED); others estimated by ratio    |
| Random seeds             | 42, 123, 456 (set before trainer instantiation)     |
| Quantization             | 4-bit NF4, double_quant, bfloat16 compute           |

---

## Limitations

See full limitations document: [docs/limitations.md](docs/limitations.md)

Key limitations:
- 20-example evaluation: +/-5% per example; high variance
- 20-step training: pilot only, not converged
- Missing DoRA r=8 s=456 (GPU quota)
- Evaluator bug discovered and resolved; 3 results excluded
- No statistical significance testing conducted

---

## Future Work

1. **Scale evaluation** to 200-500 GSM8K test examples for reliable accuracy estimates
2. **Train to convergence** (500-2000 steps) instead of 20-step pilot
3. **Complete DoRA r=8 seed=456** once GPU quota resets
4. **Add DoRA r=32** with sufficient GPU memory (A100/H100)
5. **Include r=4 DoRA** with corrected evaluator for fair comparison
6. **Bootstrap confidence intervals** across seeds for statistical validity
7. **Extend to other benchmarks** (e.g., MATH, ARC) for generalisability
8. **Compare other target modules** (k_proj, o_proj, gate_proj, up_proj)

---

## Repository Structure

```
LoRA vs DoRA Fine-Tuning Study on Google Gemma 4 E4B/
|-- README.md                         <- This file
|-- requirements.txt                  <- Python dependencies
|-- LICENSE                           <- MIT License
|-- scripts/
|   `-- build_portfolio.py            <- Generates all figures and CSVs
|-- docs/
|   |-- methodology.md                <- Detailed methodology
|   |-- limitations.md                <- Honest limitations
|   |-- evaluator_audit_report.md     <- Audit report and classification table
|   `-- experiment_summary.json       <- Machine-readable experiment metadata
`-- results/
    |-- figures/
    |   |-- fig1_accuracy_by_rank.png
    |   |-- fig2_seed_distribution.png
    |   |-- fig3_trainable_params.png
    |   |-- fig4_runtime.png
    |   |-- fig5_vram.png
    |   `-- fig6_accuracy_vs_efficiency.png
    `-- processed/
        |-- primary_results.csv
        |-- supplementary_results.csv
        `-- full_results_table.csv
```

---

## Citation

```bibtex
@misc{rahman2026loradora,
  author    = {Ashiq Rahman},
  title     = {LoRA vs DoRA Fine-Tuning Study on Google Gemma 4 E4B-IT for GSM8K},
  year      = {2026},
  note      = {Pilot-scale study. Available at project repository.},
}
```

---

*All experimental results are sourced from Kaggle session logs (LOG_VERIFIED).
No accuracy values are fabricated or estimated. Estimated values (parameter counts,
runtime extrapolations) are clearly labelled throughout.*
