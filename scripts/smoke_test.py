import os
import sys
import torch
import traceback

# 1. Fix Hugging Face cache configuration
project_cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "huggingface")
os.makedirs(project_cache, exist_ok=True)
os.environ["HF_HOME"] = project_cache
os.environ["HF_HUB_CACHE"] = project_cache

def test_write_cache():
    test_file = os.path.join(project_cache, "test_write.txt")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception:
        return False

def run_smoke_test():
    results = {}
    
    # ------------------------------------
    # Environment
    # ------------------------------------
    results["Environment"] = "PASS" # Assuming script runs
    
    # ------------------------------------
    # CUDA
    # ------------------------------------
    cuda_avail = torch.cuda.is_available()
    results["CUDA"] = "PASS" if cuda_avail else "FAIL"
    
    # ------------------------------------
    # Hugging Face cache
    # ------------------------------------
    cache_writable = test_write_cache()
    results["Hugging Face cache"] = "PASS" if cache_writable else "FAIL"

    # Stop early if critical dependencies fail (can't load model without CUDA for 4bit, or cache)
    # But let's try to proceed to get specific FAILs if possible
    
    model_id = "google/gemma-4-E4B-it"
    
    # ------------------------------------
    # Gemma 4 processor
    # ------------------------------------
    processor = None
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        results["Gemma 4 processor"] = "PASS"
    except Exception as e:
        results["Gemma 4 processor"] = "FAIL"
        print(f"Processor Error: {e}")

    # ------------------------------------
    # 4-bit loading & Gemma 4 model
    # ------------------------------------
    model = None
    try:
        from transformers import AutoModelForMultimodalLM, BitsAndBytesConfig
        if cuda_avail:
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype
            )
            results["4-bit loading"] = "PASS"
        else:
            # Fake config for CPU if we just want to see if it downloads
            bnb_config = None
            results["4-bit loading"] = "FAIL" # Explicitly fail if no CUDA for 4-bit
            
        # Try to load model
        if bnb_config:
            model = AutoModelForMultimodalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            model = AutoModelForMultimodalLM.from_pretrained(model_id)
            
        results["Gemma 4 model"] = "PASS"
    except Exception as e:
        results["Gemma 4 model"] = "FAIL"
        print(f"Model Error: {e}")

    # ------------------------------------
    # LoRA adapter
    # ------------------------------------
    lora_success = False
    try:
        if model is not None:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"], use_dora=False)
            model_lora = get_peft_model(model, lora_config)
            trainable_lora = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
            if trainable_lora > 0:
                results["LoRA adapter"] = "PASS"
                lora_success = True
                model = model_lora.unload() # Restore base model
            else:
                results["LoRA adapter"] = "FAIL"
        else:
            results["LoRA adapter"] = "FAIL"
    except Exception as e:
        results["LoRA adapter"] = "FAIL"
        print(f"LoRA Error: {e}")

    # ------------------------------------
    # DoRA adapter
    # ------------------------------------
    dora_success = False
    try:
        if model is not None:
            from peft import LoraConfig, get_peft_model
            dora_config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"], use_dora=True)
            model_dora = get_peft_model(model, dora_config)
            trainable_dora = sum(p.numel() for p in model_dora.parameters() if p.requires_grad)
            if trainable_dora > 0:
                results["DoRA adapter"] = "PASS"
                dora_success = True
                model = model_dora # Keep it for generation
            else:
                results["DoRA adapter"] = "FAIL"
        else:
            results["DoRA adapter"] = "FAIL"
    except Exception as e:
        results["DoRA adapter"] = "FAIL"
        print(f"DoRA Error: {e}")

    # ------------------------------------
    # Generation
    # ------------------------------------
    try:
        if model is not None and processor is not None and dora_success:
            from src.prompting import build_gsm8k_prompt
            prompt = build_gsm8k_prompt("What is 2+2?")
            # We assume text-only input for the multimodal processor
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=5)
            results["Generation"] = "PASS"
        else:
            results["Generation"] = "FAIL"
    except Exception as e:
        results["Generation"] = "FAIL"
        print(f"Generation Error: {e}")

    # ------------------------------------
    # Adapter save/reload
    # ------------------------------------
    try:
        if model is not None and dora_success:
            adapter_path = os.path.join(project_cache, "temp_adapter")
            model.save_pretrained(adapter_path)
            
            # Reload
            from peft import PeftModel
            model_base = model.unload()
            model_reloaded = PeftModel.from_pretrained(model_base, adapter_path)
            results["Adapter save/reload"] = "PASS"
        else:
            results["Adapter save/reload"] = "FAIL"
    except Exception as e:
        results["Adapter save/reload"] = "FAIL"
        print(f"Save/Reload Error: {e}")

    # ------------------------------------
    # GSM8K evaluator
    # ------------------------------------
    try:
        import subprocess
        res = subprocess.run([sys.executable, "tests/test_evaluation.py"], capture_output=True, text=True)
        if res.returncode == 0:
            results["GSM8K evaluator"] = "PASS"
        else:
            results["GSM8K evaluator"] = "FAIL"
            print(f"Evaluator Test Error:\n{res.stderr}")
    except Exception as e:
        results["GSM8K evaluator"] = "FAIL"

    print("\n--- SMOKE TEST RESULTS ---")
    for k, v in results.items():
        print(f"{k}\n\n{v}\n")
        
    if all(v == "PASS" for v in results.values()):
        print("ALL PASSED")
        return True
    return False

if __name__ == "__main__":
    run_smoke_test()
