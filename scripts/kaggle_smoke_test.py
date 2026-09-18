import os
import sys
import shutil
import subprocess
import traceback

def get_disk_usage(path):
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3) # in GB

def setup_kaggle_env():
    """Sets up the environment specifically for Kaggle Notebooks."""
    kaggle_tmp = "/kaggle/tmp"
    
    if os.path.exists("/kaggle"):
        print("Detected Kaggle environment.")
        cache_dir = os.path.join(kaggle_tmp, "huggingface")
        os.makedirs(cache_dir, exist_ok=True)
    else:
        # Fallback for non-Kaggle
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "huggingface")
        os.makedirs(cache_dir, exist_ok=True)
        
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir
    return cache_dir

def get_hf_token():
    """Retrieves HF_TOKEN from Kaggle Secrets if available, otherwise from env."""
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret("HF_TOKEN")
        if token:
            print("Retrieved HF_TOKEN from Kaggle Secrets.")
            os.environ["HF_TOKEN"] = token
            return token
    except ImportError:
        pass
    except Exception as e:
        print(f"Failed to read Kaggle Secret HF_TOKEN: {e}")
        
    return os.environ.get("HF_TOKEN")

def run_kaggle_smoke_test():
    results = {}
    
    print("\n--- KAGGLE SMOKE TEST DIAGNOSTICS ---")
    print(f"Python: {sys.version.split()[0]}")
    results["Environment"] = "PASS"
    
    cache_dir = setup_kaggle_env()
    print(f"Cache directory: {cache_dir}")
    
    free_gb = get_disk_usage(cache_dir)
    print(f"Available disk space for cache: {free_gb:.2f} GB")
    
    if free_gb < 18.0:
        print("WARNING: Insufficient disk space! Model download may fail.")
        results["Storage"] = "FAIL"
    else:
        results["Storage"] = "PASS"

    import torch
    print(f"PyTorch version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    results["CUDA"] = "PASS" if cuda_avail else "FAIL"
    
    if cuda_avail:
        print(f"CUDA version: {torch.version.cuda}")
        gpu_count = torch.cuda.device_count()
        print(f"GPU count: {gpu_count}")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"GPU {i}: {name} ({vram:.2f} GB VRAM)")
        results["GPU"] = "PASS"
    else:
        print("No GPU detected.")
        results["GPU"] = "FAIL"
        
    hf_token = get_hf_token()
    if not hf_token:
        print("WARNING: HF_TOKEN not found in Kaggle Secrets or environment variables.")
        results["HF authentication"] = "FAIL"
    else:
        results["HF authentication"] = "PASS"

    if results["CUDA"] == "FAIL" or results["Storage"] == "FAIL" or results["HF authentication"] == "FAIL":
        print("\nFoundational check failed. Stopping smoke test.")
        print_matrix(results)
        return False

    model_id = "google/gemma-4-E4B-it"
    
    processor = None
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        results["Processor"] = "PASS"
    except Exception as e:
        results["Processor"] = "FAIL"
        print(f"Processor Error: {e}")

    model = None
    try:
        from transformers import AutoModelForMultimodalLM, BitsAndBytesConfig
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype
        )
        results["4-bit loading"] = "PASS"
            
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token
        )
        results["Gemma 4 model"] = "PASS"
    except Exception as e:
        results["Gemma 4 model"] = "FAIL"
        results["4-bit loading"] = "FAIL"
        print(f"Model Error: {e}")

    try:
        if model is not None and processor is not None:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.prompting import build_gsm8k_prompt
            
            prompt = build_gsm8k_prompt("If I have 2 apples and buy 3 more, how many apples do I have?")
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10)
            results["Generation"] = "PASS"
        else:
            results["Generation"] = "FAIL"
    except Exception as e:
        results["Generation"] = "FAIL"
        print(f"Generation Error: {e}")

    try:
        if model is not None:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"], use_dora=False)
            model_lora = get_peft_model(model, lora_config)
            trainable_lora = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
            if trainable_lora > 0:
                print(f"LoRA Trainable Params: {trainable_lora}")
                results["LoRA"] = "PASS"
                model = model_lora.unload()
            else:
                results["LoRA"] = "FAIL"
        else:
            results["LoRA"] = "FAIL"
    except Exception as e:
        results["LoRA"] = "FAIL"
        print(f"LoRA Error: {e}")

    dora_success = False
    try:
        if model is not None:
            from peft import LoraConfig, get_peft_model
            dora_config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"], use_dora=True)
            model_dora = get_peft_model(model, dora_config)
            trainable_dora = sum(p.numel() for p in model_dora.parameters() if p.requires_grad)
            if trainable_dora > 0:
                print(f"DoRA Trainable Params: {trainable_dora}")
                results["DoRA"] = "PASS"
                dora_success = True
                model = model_dora
            else:
                results["DoRA"] = "FAIL"
        else:
            results["DoRA"] = "FAIL"
    except Exception as e:
        results["DoRA"] = "FAIL"
        print(f"DoRA Error: {e}")

    try:
        if model is not None and dora_success:
            inputs = processor(text="A quick test for loss.", return_tensors="pt").to(model.device)
            inputs["labels"] = inputs["input_ids"].clone()
            
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            results["Tiny training"] = "PASS"
        else:
            results["Tiny training"] = "FAIL"
    except Exception as e:
        results["Tiny training"] = "FAIL"
        print(f"Tiny training Error: {e}")

    try:
        if model is not None and dora_success:
            adapter_path = os.path.join(cache_dir, "temp_dora_adapter")
            model.save_pretrained(adapter_path)
            
            from peft import PeftModel
            model_base = model.unload()
            model_reloaded = PeftModel.from_pretrained(model_base, adapter_path)
            results["Save/reload"] = "PASS"
        else:
            results["Save/reload"] = "FAIL"
    except Exception as e:
        results["Save/reload"] = "FAIL"
        print(f"Save/Reload Error: {e}")

    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tests.test_evaluation import test_extract_final_answer, test_build_gsm8k_prompt
        test_extract_final_answer()
        test_build_gsm8k_prompt()
        results["Evaluator"] = "PASS"
    except Exception as e:
        results["Evaluator"] = "FAIL"
        print(f"Evaluator Error: {e}")

    print_matrix(results)
    
    if all(v == "PASS" for v in results.values()):
        return True
    return False

def print_matrix(results):
    print("\n--- FINAL KAGGLE VALIDATION MATRIX ---")
    keys = [
        "Environment", "CUDA", "GPU", "Storage", "HF authentication", 
        "Processor", "Gemma 4 model", "4-bit loading", "Generation", 
        "LoRA", "DoRA", "Save/reload", "Evaluator", "Tiny training"
    ]
    for k in keys:
        status = results.get(k, "NOT RUN")
        print(f"{k} {status}")

if __name__ == "__main__":
    success = run_kaggle_smoke_test()
    if not success:
        sys.exit(1)
