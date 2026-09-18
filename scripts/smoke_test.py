import os
import sys
import torch
import traceback

def run_smoke_test():
    print("Environment")
    print("------------")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"VRAM: {total_memory:.2f} GB")
    else:
        print("GPU: None")
        print("VRAM: None")
        
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: NOT INSTALLED")
        return False
        
    try:
        import peft
        print(f"PEFT: {peft.__version__}")
    except ImportError:
        print("PEFT: NOT INSTALLED")
        return False
        
    try:
        import bitsandbytes
        print(f"BitsAndBytes: INSTALLED")
    except ImportError:
        print("BitsAndBytes: NOT INSTALLED")
        return False
        
    try:
        import datasets
        print("Datasets: INSTALLED")
    except ImportError:
        print("Datasets: NOT INSTALLED")
        return False

    print("\nModel")
    print("-----")
    
    model_id = "google/gemma-2b-it" # Fallback for smoke testing if gemma-4-E4B-it is gated or too large for smoke test on small RAM. Wait, let's try 4-E4B-it first.
    model_id = "google/gemma-4-E4B-it"
    
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        
        # Load processor
        print("Loading Processor...")
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            print("Tokenizer/Processor: SUCCESS")
        except Exception as e:
            # Fallback to tokenizer if AutoProcessor fails (some versions might just use tokenizer)
            from transformers import AutoTokenizer
            processor = AutoTokenizer.from_pretrained(model_id)
            print("Tokenizer/Processor: SUCCESS (AutoTokenizer)")
            
        print("Loading Model in 4-bit...")
        # Note: Depending on authentication, we might get an error if HF token is missing.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        print("Loaded: SUCCESS")
        
        print("Applying DoRA Adapter...")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=True
        )
        model = get_peft_model(model, lora_config)
        print("Adapter injection: SUCCESS")
        
        print("Testing Generation...")
        inputs = processor("What is 2+2?", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=5)
        print("Generation: SUCCESS")
        
        print("\nSTATUS: PASS")
        return True
        
    except Exception as e:
        print("\nSTATUS: FAIL")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
