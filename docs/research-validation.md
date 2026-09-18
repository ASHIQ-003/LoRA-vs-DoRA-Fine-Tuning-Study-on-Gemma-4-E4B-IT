# Research Validation: Gemma 4 E4B, PEFT, and DoRA

## 1. Gemma 4 E4B Architecture
*   **Model**: `google/gemma-4-E4B-it`
*   **Architecture**: It is an "effective 4B" (E4B) parameter model. It uses a Per-Layer Embeddings (PLE) technique where each of its 42 decoder layers has its own embedding table. This allows it to achieve performance similar to larger models while keeping the active parameter count at ~4.5B.
*   **Multimodal & Context**: Supports up to 128K context window and native multimodality (text, images, audio).
*   **Reference**: [Hugging Face `google/gemma-4-E4B-it` usage](https://huggingface.co/google/gemma-4-E4B-it) and related Google release announcements.

## 2. Effective vs Total Parameter Count
*   **Active Parameters**: ~4.5B during generation.
*   **Total Parameters**: Higher than 4.5B due to Per-Layer Embeddings (each layer having its own embedding table increases the total static parameter count, but only one embedding table or a subset is active per token processing, or they are processed in a specific PLE manner). 
*   **Reference**: Google Developer announcements for Gemma 4 E4B.

## 3. Supported Transformers Loading Interface
*   Available via the standard Hugging Face `transformers` library using `AutoModelForCausalLM` and `AutoProcessor`.
*   Loading snippet:
    ```python
    from transformers import AutoProcessor, AutoModelForCausalLM
    MODEL_ID = "google/gemma-4-E4B-it"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        dtype="auto", 
        device_map="auto"
    )
    ```

## 4. Current PEFT Support & 5. DoRA Support
*   **PEFT**: Hugging Face's `peft` library supports LoRA and DoRA.
*   **DoRA**: DoRA (Weight-Decomposed Low-Rank Adaptation) is supported by setting `use_dora=True` in `LoraConfig`.
*   **Reference**: [PEFT LoraConfig Documentation](https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig.use_dora).

## 6. Recommended Quantization Workflow
*   **BitsAndBytes**: 4-bit quantization via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` is standard for QLoRA/QDoRA approaches to reduce VRAM on T4/L4 hardware.
*   *Note*: DoRA can sometimes conflict with fully quantized base layers during training depending on the specific implementation, but PEFT's integration of DoRA alongside bitsandbytes 4-bit base models is generally supported for linear layers. We will verify this during the smoke test.

## 7. Supported Target Modules
*   Standard target modules for Gemma architectures include attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP projections (`gate_proj`, `up_proj`, `down_proj`).
*   Targeting all linear layers (`"all-linear"`) is the recommended default for maximum adaptation capability.

## 8. Correct Tokenizer/Processor/Chat-Template Workflow
*   Gemma 4 uses `AutoProcessor` as it supports multimodal inputs. For text-only (GSM8K), we will use the processor's tokenizer (or `AutoTokenizer` directly if text-only is supported) and the `apply_chat_template` method to format data correctly for the `-it` (instruction-tuned) variant.
*   Format expects `user` and `model` (or `assistant`) roles.

## 9. Original DoRA Paper Evaluation
*   The original DoRA paper ("DoRA: Weight-Decomposed Low-Rank Adaptation") evaluated LLaMA and other earlier architectures on common benchmarks. It did not evaluate Gemma 4, which is a newer architecture utilizing Per-Layer Embeddings.

## 10. Existing Public Gemma 4 + DoRA Work
*   A literature search shows general availability of Gemma 4 and DoRA as a PEFT feature. A rigorous, controlled rank-sweep study of DoRA vs LoRA specifically on Gemma 4 E4B on GSM8K represents an empirical extension study, clarifying how PLE architectures respond to magnitude/direction decomposition compared to standard LoRA.
