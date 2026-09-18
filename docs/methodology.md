# Research Methodology

## Objective
The goal of this experiment is to evaluate the relative impact of DoRA (Weight-Decomposed Low-Rank Adaptation) vs standard LoRA across different adaptation ranks when fine-tuning the Gemma 4 E4B model on the GSM8K dataset.

## Experimental Controls
To ensure differences observed are strictly due to the adaptation method and rank, we strictly control all other variables:
1. **Base Model**: `google/gemma-4-E4B-it` loaded in 4-bit with BitsAndBytes (`nf4`, `bfloat16` compute dtype).
2. **Dataset**: `openai/gsm8k` (split: `train` for training, `test` for evaluation).
3. **Training Hyperparameters**:
    - Learning rate: `2.0e-4`
    - Epochs: `3`
    - Batch size: `2` (with gradient accumulation of `4`)
    - Warmup ratio: `0.1`
    - Weight decay: `0.01`
4. **Adapter Settings**:
    - Alpha policy: `alpha = 2 * r` (or fixed depending on config)
    - Dropout: `0.05`
    - Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
5. **Prompting**: Canonical conversational format using the Gemma instruct template.

## Execution Order
1. **Baseline**: Evaluate the raw instruction-tuned model without fine-tuning to establish a performance floor.
2. **Pilot**: Run a fast training and evaluation loop (Rank 8) to verify end-to-end functionality.
3. **Rank Sweep**: Train LoRA and DoRA at ranks `r=4, 8, 16, 32` using seed 42.
4. **Seed Robustness**: Rerun specific ranks with seeds 123 and 456 to establish variance.
5. **Ablation**: Compare standard all-linear targeting vs Q/V targeting only.

## Metrics Tracking
We track the following for each run:
- Evaluation Accuracy (Exact Match on GSM8K)
- Trainable Parameter Count
- Total Parameter Count
- Peak VRAM Usage
- Training Time
