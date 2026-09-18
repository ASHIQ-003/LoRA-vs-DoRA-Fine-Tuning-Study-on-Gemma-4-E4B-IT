# LoRA vs DoRA on Gemma 4 E4B

## Research Question
How does the relative effectiveness and efficiency of LoRA and DoRA change with adaptation rank when fine-tuning Gemma 4 E4B on GSM8K?

## Why This Experiment?
DoRA (Weight-Decomposed Low-Rank Adaptation) decouples magnitude and direction in weight updates. This empirical study tests whether DoRA consistently outperforms LoRA across different ranks (4, 8, 16, 32) when fine-tuning the Gemma 4 E4B model (which uses Per-Layer Embeddings) on the GSM8K mathematical reasoning benchmark.

## Experimental Design
- **Base Model**: `google/gemma-4-E4B-it` (4-bit quantized)
- **Dataset**: `openai/gsm8k`
- **Task**: Mathematical reasoning
- **Methods**: LoRA vs DoRA
- **Ranks**: 4, 8, 16, 32
- **Metrics**: Exact match accuracy, Trainable parameters, Training time, Peak VRAM

## Reproducibility
All experiments use fixed seeds and identical hyperparameter setups (except for the `use_dora` flag). See `configs/base.yaml` for exact training parameters.

## Results
*Results pending experiment execution.*

## Accuracy vs Rank
*Pending*

## Efficiency Analysis
*Pending*

## Error Analysis
*Pending*

## Ablation
*Pending*

## Findings
*Pending*

## Limitations
*Pending*

## Reproduce the Study
1. Install requirements: `pip install -r requirements.txt`
2. Run smoke test: `python scripts/smoke_test.py`
3. Run baseline: `python scripts/baseline.py`
4. Run training sweeps: `python scripts/train.py --method [lora|dora] --rank [4|8|16|32]`

## Demo
Run the minimal UI with `python demo/app.py`
