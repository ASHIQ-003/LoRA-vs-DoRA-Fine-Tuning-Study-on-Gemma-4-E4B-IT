# Limitations

This document provides an honest account of the limitations of this pilot study.
Readers and future researchers should take these into account when interpreting results.

---

## 1. Extremely Small Evaluation Set (20 Examples)

All accuracy figures are computed on a fixed 20-example subset of the GSM8K test split.

**Consequence:** Each correct or incorrect answer shifts accuracy by exactly 5 percentage
points (1/20 = 5%). This means:
- Two configurations that differ by 5% may differ by only a single correct answer.
- Observed differences of 5-10% should not be treated as meaningful without further validation.
- Standard deviation estimates across seeds (3 seeds) are rough at best.
- Confidence intervals are wide; no formal statistical significance testing is reported.

**Recommendation:** A minimum of 200-500 evaluation examples would be needed to draw
reliable conclusions about 5-10% accuracy differences.

---

## 2. 20-Step Pilot Training (Not Convergence)

All fine-tuning runs are limited to **20 training steps** to fit within Kaggle free-tier
GPU session constraints.

**Consequence:**
- Models are nowhere near convergence on the training set.
- Accuracy reflects very early learning dynamics rather than fully adapted adapter weights.
- Results may not generalise to full-scale fine-tuning outcomes.
- Relative rankings between LoRA and DoRA may change substantially with more training.

This study should be understood as a **feasibility pilot** rather than a definitive
benchmark.

---

## 3. Missing DoRA r=8 Seed=456 (GPU Quota Exhausted)

The DoRA r=8 seed=456 configuration could not be completed because the Kaggle free-tier
GPU quota was exhausted before the run could execute.

**Consequence:**
- DoRA r=8 statistics are based on only 2 seeds (s=42: 40%, s=123: 50%) instead of 3.
- Mean accuracy for DoRA r=8 (45%) and standard deviation have higher uncertainty than
  3-seed estimates.
- All figures and tables note this explicitly. This configuration is marked with an
  asterisk (*) wherever it appears.

---

## 4. Evaluator Inconsistency Discovered and Resolved

An evaluator bug was discovered during the post-hoc audit: the `enable_thinking=False`
flag was missing from some evaluation calls. Gemma 4 E4B-IT is a thinking-capable model
that, without this flag, can emit extended reasoning tokens before the final answer,
breaking the `####`-based exact-match parser.

**Consequence:**
- DoRA r=4 seed=123 and DoRA r=4 seed=456 are **excluded** from all analysis
  (both returned 5%, consistent with parsing failure rather than model output).
- LoRA r=32 seed=456 returned 0%, which is flagged as EVALUATOR_SUSPECT and excluded.
- All 19 primary and supplementary results in this report have been validated with the
  corrected evaluator protocol (see `docs/evaluator_audit_report.md`).

---

## 5. GPU Memory Constraints Preventing r=32 DoRA

DoRA r=32 could not complete for seeds 123 and 456 due to out-of-memory (OOM) errors
on the Kaggle T4 GPU (14.56 GB VRAM).

**Consequence:**
- No valid DoRA r=32 comparison is available.
- The parameter-efficiency and accuracy-vs-efficiency comparisons omit DoRA r=32.
- The confirmed VRAM measurement (11.588 GB for LoRA r=32 seed=456) suggests headroom
  was limited; DoRA's additional magnitude parameters likely pushed usage over the limit.

---

## 6. Kaggle Session Restarts Wiping Artifacts

Kaggle free-tier sessions have a time limit per session and do not persist files across
new sessions without explicit manual saving (e.g., to Kaggle Datasets or Output).

**Consequence:**
- Several runs required session restarts mid-experiment.
- Some intermediate checkpoints were lost and had to be re-run from scratch.
- The DoRA r=8 seed=456 miss may be partly attributable to quota consumed by re-runs.
- Runtime figures are available for only one configuration (LoRA r=32 seed=456) because
  that was the only session where runtime was explicitly logged before session end.

---

## 7. No Statistical Significance Testing

Given the small evaluation set (n=20) and small number of seeds (3), formal
hypothesis testing (e.g., paired t-tests, bootstrap confidence intervals) is not
conducted.

**Consequence:**
- Differences between LoRA and DoRA cannot be claimed as statistically significant.
- The study documents patterns and trends, not conclusions.
- Future work should include at minimum 5 seeds and 200+ evaluation examples before
  applying any significance test.

---

## 8. Training Sampled from First 200 Examples Only

Training examples are drawn from the first 200 examples of the GSM8K training split
(sorted by dataset index, not randomly shuffled at the dataset level).

**Consequence:**
- The training sample may not be representative of the full GSM8K distribution.
- Results may be biased toward problem types that appear early in the dataset.
- Different sampling strategies may yield different relative performance patterns.

---

## 9. Single GPU Platform

All experiments were run on a single Kaggle T4 GPU.

**Consequence:**
- Results may not transfer to other hardware (A100, H100, consumer GPUs with different
  memory hierarchies).
- Multi-GPU training dynamics are not captured.
- T4 bfloat16 precision support differs from newer architectures.

---

## Summary Table

| Limitation                       | Severity | Impact on Conclusions            |
|----------------------------------|----------|----------------------------------|
| 20-example eval set              | High     | +/-5% per example; wide CI       |
| 20-step training                 | High     | Pilot only; not converged        |
| Missing DoRA r=8 s=456           | Medium   | 2/3 seeds; noted with asterisk   |
| Evaluator bug (resolved)         | Medium   | 3 results excluded from analysis |
| r=32 DoRA OOM                    | Medium   | No DoRA r=32 comparison          |
| Session restarts                 | Low-Med  | Artifact loss; partial re-runs   |
| No significance testing          | Medium   | Trends only, not conclusions     |
| 200-example training sample      | Medium   | May not represent full dataset   |
| Single GPU platform              | Low      | Limited hardware generalisability|
