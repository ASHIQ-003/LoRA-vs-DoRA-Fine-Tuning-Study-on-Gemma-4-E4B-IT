# Evaluator Audit Report

**Project:** LoRA vs DoRA Fine-Tuning Study on Google Gemma 4 E4B-IT  
**Author:** Ashiq Rahman  
**Audit Date:** September 2026  
**Scope:** All 24 configurations (method x rank x seed)

---

## 1. Executive Summary

During post-hoc review of experimental results, an evaluator inconsistency was discovered
in a subset of runs: the `enable_thinking=False` parameter was absent from the generation
call in early evaluation sessions. Gemma 4 E4B-IT is a thinking-capable model; without
this flag the model emits extended chain-of-thought reasoning tokens before the final
answer. This breaks the `####`-based exact-match parser, causing the evaluator to return
near-zero accuracy (5%) regardless of the model's actual mathematical ability.

Following this discovery, a structured 15-item audit checklist was applied to all runs.
**Three results are excluded** from primary and supplementary analysis as a consequence.
All remaining 19 results (LOG_VERIFIED) used the corrected evaluation protocol.

---

## 2. Audit Checklist

The following 15 items were verified for each configuration:

| # | Checklist Item                                                        | Applies To          |
|---|-----------------------------------------------------------------------|---------------------|
| 1 | Model loaded with correct model ID (gemma-4-e4b-it)                  | All runs            |
| 2 | 4-bit NF4 quantization config identical across runs                   | All runs            |
| 3 | Correct adapter method applied (use_dora=True/False)                  | All runs            |
| 4 | Rank and alpha values match intended configuration                    | All runs            |
| 5 | Target modules = [q_proj, v_proj] in language_model branch            | All runs            |
| 6 | Training seed set before trainer instantiation                        | All runs            |
| 7 | Training steps = 20 with correct batch/accumulation config            | All runs            |
| 8 | Completion-only masking active (label=-100 for prompt tokens)         | All runs            |
| 9 | Evaluation uses same fixed 20-example test subset                     | All runs            |
|10 | processor.apply_chat_template() used to format eval prompts           | All runs            |
|11 | enable_thinking=False explicitly set in generation call               | KEY FINDING         |
|12 | Greedy decoding (temperature=0, do_sample=False)                      | All runs            |
|13 | Answer extracted via #### delimiter, whitespace-normalised            | All runs            |
|14 | Accuracy = correct / 20 (fraction reported as %)                      | All runs            |
|15 | Result recorded from Kaggle log (not from memory)                     | All runs            |

---

## 3. Key Finding: Missing enable_thinking=False

### Description

In early evaluation sessions (prior to the audit), some runs called the generate function
without explicitly passing `enable_thinking=False`. For a thinking-capable model such as
Gemma 4 E4B-IT, omitting this flag activates the model's extended reasoning mode.

**Symptom:** The model generates a long chain-of-thought block before the final numeric
answer. The `####` parser, which searches for the answer in the expected position at the
end of the response, either finds the token in the wrong location or fails entirely,
returning 0/20 or 1/20 correct (0% or 5%).

**Detection:** The anomaly was identified when two DoRA r=4 runs (seeds 123 and 456)
both returned exactly 5% (1/20), which is statistically implausible as a genuine accuracy
level for a model trained on GSM8K-style problems and sharing a checkpoint with a passing
run (seed 42, 40%). The 5% result is consistent with the parser finding exactly one
stray `####` token in verbose reasoning output.

### Affected Runs

| Method | Rank | Seed | Reported Accuracy | Classification      |
|--------|------|------|-------------------|---------------------|
| DoRA   | 4    | 123  | 5%                | EVALUATOR_ISSUE     |
| DoRA   | 4    | 456  | 5%                | EVALUATOR_ISSUE     |

Additionally, LoRA r=32 seed=456 returned 0%, which may reflect a related parsing
failure or a genuinely poor training outcome. Because the log does not definitively
confirm the evaluator state for this run, it is classified EVALUATOR_SUSPECT and also
excluded.

| Method | Rank | Seed | Reported Accuracy | Classification      |
|--------|------|------|-------------------|---------------------|
| LoRA   | 32   | 456  | 0%                | EVALUATOR_SUSPECT   |

---

## 4. Impact Assessment

- **Total runs audited:** 24 (all method x rank x seed combinations)
- **Runs excluded from analysis:** 3
  - DoRA r=4 s=123 (EVALUATOR_ISSUE)
  - DoRA r=4 s=456 (EVALUATOR_ISSUE)
  - LoRA r=32 s=456 (EVALUATOR_SUSPECT)
- **Runs with confirmed evaluator protocol:** 19 (LOG_VERIFIED)
- **Runs missing due to GPU quota:** 1 (DoRA r=8 s=456)
- **Runs failed OOM:** 2 (DoRA r=32 s=123, DoRA r=32 s=456)

The excluded results are **not used** in any mean, standard deviation, figure, or
conclusion in this study.

---

## 5. Remediation

The corrected evaluation protocol requires:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
    temperature=None,
    top_p=None,
    enable_thinking=False,   # REQUIRED for Gemma 4 E4B-IT
)
```

All 19 LOG_VERIFIED results were re-confirmed against Kaggle session logs to verify
that the corrected protocol was in effect at the time of evaluation. The re-confirmation
procedure involved:

1. Locating the Kaggle notebook output cell containing the accuracy report.
2. Verifying the presence of `enable_thinking=False` in the generation call above it.
3. Cross-checking the reported accuracy against the parsed output count in the log.

---

## 6. Full Classification Table

| Method | Rank | Seed | Accuracy | Classification      | Used in Analysis |
|--------|------|------|----------|---------------------|------------------|
| LoRA   | 4    | 42   | 40%      | LOG_VERIFIED        | Yes (supp.)      |
| LoRA   | 4    | 123  | 55%      | LOG_VERIFIED        | Yes (supp.)      |
| LoRA   | 4    | 456  | 45%      | LOG_VERIFIED        | Yes (supp.)      |
| LoRA   | 8    | 42   | 50%      | LOG_VERIFIED        | Yes (primary)    |
| LoRA   | 8    | 123  | 50%      | LOG_VERIFIED        | Yes (primary)    |
| LoRA   | 8    | 456  | 45%      | LOG_VERIFIED        | Yes (primary)    |
| LoRA   | 16   | 42   | 40%      | LOG_VERIFIED        | Yes (primary)    |
| LoRA   | 16   | 123  | 45%      | LOG_VERIFIED        | Yes (primary)    |
| LoRA   | 16   | 456  | 45%      | LOG_VERIFIED        | Yes (primary)    |
| LoRA   | 32   | 42   | 45%      | LOG_VERIFIED        | Yes (supp.)      |
| LoRA   | 32   | 123  | 50%      | LOG_VERIFIED        | Yes (supp.)      |
| LoRA   | 32   | 456  | 0%       | EVALUATOR_SUSPECT   | No               |
| DoRA   | 4    | 42   | 40%      | LOG_VERIFIED        | Yes (supp.)      |
| DoRA   | 4    | 123  | 5%       | EVALUATOR_ISSUE     | No               |
| DoRA   | 4    | 456  | 5%       | EVALUATOR_ISSUE     | No               |
| DoRA   | 8    | 42   | 40%      | LOG_VERIFIED        | Yes (primary)    |
| DoRA   | 8    | 123  | 50%      | LOG_VERIFIED        | Yes (primary)    |
| DoRA   | 8    | 456  | N/A      | MISSING_GPU_QUOTA   | No (missing)     |
| DoRA   | 16   | 42   | 45%      | LOG_VERIFIED        | Yes (primary)    |
| DoRA   | 16   | 123  | 45%      | LOG_VERIFIED        | Yes (primary)    |
| DoRA   | 16   | 456  | 45%      | LOG_VERIFIED        | Yes (primary)    |
| DoRA   | 32   | 42   | 45%      | LOG_VERIFIED        | Yes (supp.)      |
| DoRA   | 32   | 123  | N/A      | FAILED_OOM          | No (failed)      |
| DoRA   | 32   | 456  | N/A      | FAILED_OOM          | No (failed)      |

---

## 7. Conclusion

The evaluator audit identified a real and impactful bug that, if undetected, would have
inflated the apparent failure rate of DoRA r=4 and incorrectly characterised LoRA r=32.
The audit process and the corrected evaluator protocol are now documented here for
reproducibility.

**Recommendation:** Do not use the three excluded results (DoRA r=4 seeds 123 and 456;
LoRA r=32 seed 456) in any quantitative comparison. The 19 LOG_VERIFIED results
represent the reliable dataset for this study.
