# Methodology

## 1. Model Architecture

**Model:** `google/gemma-4-e4b-it`

Gemma 4 E4B-IT is a 4-billion-effective-parameter instruction-tuned language model based on a
Mixture-of-Experts (MoE) architecture. In MoE models, each token is routed to a subset of expert
feed-forward networks, allowing the total parameter count to be large while keeping per-token
computation efficient. The "E4B" designation refers to approximately 4B effective (active) parameters
per forward pass.

For all experiments the model backbone is loaded in **4-bit NF4 quantization** via `bitsandbytes`,
with `double_quant=True` and `compute_dtype=torch.bfloat16`. This reduces GPU memory requirements
sufficiently to run on a single Kaggle T4 GPU (14.56 GB VRAM) while preserving most model quality.

---

## 2. Adapter Methods

### 2.1 LoRA (Low-Rank Adaptation)

For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA freezes $W_0$ and
injects a low-rank decomposition:

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

where $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, and $r \ll \min(d, k)$
is the rank. $A$ is initialised from $\mathcal{N}(0, \sigma^2)$ and $B$ from zeros, so $\Delta W = 0$
at training start. The output is scaled by $\frac{\alpha}{r}$ (a hyperparameter ratio):

$$h = W_0 x + \frac{\alpha}{r} B A x$$

Only $A$ and $B$ are updated during training; $W_0$ remains frozen.

### 2.2 DoRA (Weight-Decomposed Low-Rank Adaptation)

DoRA decomposes the pre-trained weight $W_0$ into a **magnitude** component $m$ and a
**directional** component $V$:

$$W_0 = m \cdot \frac{V}{\|V\|_c}$$

where $\|\cdot\|_c$ denotes the column-wise norm. DoRA then adapts the directional component
using LoRA while keeping magnitude learnable:

$$W' = (m + \Delta m) \cdot \frac{V + \Delta V}{\|V + \Delta V\|_c}$$

$$\Delta V = B A \quad (LoRA\;decomposition)$$

This allows fine-grained control over both the magnitude and direction of weight updates,
which has been shown to more closely mimic full fine-tuning update patterns.

**Implementation note:** Both methods are applied via the `peft` library with
`use_dora=False` (LoRA) and `use_dora=True` (DoRA).

---

## 3. Adapter Target Modules

Adapters are injected into the **query** and **value** projection matrices of all
attention layers within the `language_model` branch of the model:

- `language_model.model.layers.*.self_attn.q_proj`
- `language_model.model.layers.*.self_attn.v_proj`

Total number of adapted modules: **66** (confirmed from `print_trainable_parameters()`).
The key-projection (`k_proj`) and output-projection (`o_proj`) are excluded to reduce
memory overhead and keep the study focused on the standard LoRA convention.

---

## 4. Hyperparameters

| Hyperparameter             | Value                             |
|----------------------------|-----------------------------------|
| Rank `r`                   | 4, 8, 16, 32                      |
| Alpha `alpha`              | `2 * r`                           |
| Dropout                    | 0.1                               |
| Bias                       | `none`                            |
| Max training steps         | 20 (pilot)                        |
| Learning rate              | 2e-4                              |
| LR scheduler               | cosine                            |
| Warmup ratio               | 0.1                               |
| Per-device batch size      | 1                                 |
| Gradient accumulation      | 4 (effective batch = 4)           |
| Optimizer                  | AdamW (8-bit via bitsandbytes)    |
| Max sequence length        | 512 tokens                        |
| Gradient checkpointing     | Enabled                           |
| Mixed precision            | bf16                              |
| Seeds                      | 42, 123, 456                      |

---

## 5. Dataset

**Dataset:** GSM8K (Grade School Math 8K)

GSM8K consists of grade-school mathematics word problems each requiring a multi-step chain
of arithmetic reasoning to arrive at a final numeric answer. The standard format includes
a problem statement and a step-by-step solution ending with `#### <answer>`.

- **Training subset:** First 200 examples from `gsm8k` train split (sampled deterministically).
- **Evaluation subset:** Fixed 20 examples from the test split (same 20 across all runs).

Sampling from only 200 training examples and evaluating on only 20 is a deliberate
**pilot-scale** design chosen to fit within Kaggle free-tier GPU session time limits.

---

## 6. Training Procedure

1. The base model is loaded with 4-bit NF4 quantization.
2. `prepare_model_for_kbit_training()` is called to enable stable gradient flow
   through quantized layers.
3. A `LoraConfig` or `DoraConfig` is applied (via `get_peft_model()`), freezing all
   base weights and injecting trainable adapter parameters.
4. Training is performed with `SFTTrainer` from the `trl` library.
5. **Completion-only masking:** Prompt tokens receive label = -100 so the loss is
   computed only on model-generated completions. This is implemented via
   `DataCollatorForCompletionOnlyLM`.
6. After 20 training steps the adapter weights are saved.

---

## 7. Evaluation Protocol

Evaluation is performed with a **deterministic greedy decoder** (temperature=0) to ensure
reproducibility across identical configurations.

Key implementation details:

- `processor.apply_chat_template()` is used to format evaluation prompts with the
  correct special tokens for the instruction-tuned model.
- **`enable_thinking=False`** is explicitly set for all evaluations. Gemma 4 E4B-IT
  is a thinking-capable model; without this flag the model may produce extended
  chain-of-thought reasoning tokens before the answer, breaking exact-match parsing.
  Failing to set this flag was the root cause of the evaluator bug discovered during
  the audit (see `docs/evaluator_audit_report.md`).
- The model output is parsed by extracting the numeric answer following `####`.
- Accuracy is computed as the fraction of 20 test examples answered correctly
  (exact-match after whitespace normalisation).
- Each percentage point therefore corresponds to 1/20 = 5% granularity.

---

## 8. Seeds and Reproducibility

Three random seeds (42, 123, 456) are used for each `(method, rank)` combination to
partially account for training stochasticity. Because training runs for only 20 steps,
variance across seeds is expected to be non-negligible. Absolute differences of +-5%
in accuracy correspond to exactly one example changing answer, so results at this
scale should be interpreted as indicative trends only, not statistically significant
differences.

All runs were executed on Kaggle free-tier T4 GPUs (14.56 GB VRAM). Reproducibility is
further limited by:

- Kaggle session restarts (can change CUDA driver state).
- Potential non-determinism in attention implementations on GPU.
- 4-bit quantization approximations.
