from peft import LoraConfig, get_peft_model
import torch
import logging
import bitsandbytes as bnb

logger = logging.getLogger(__name__)

def discover_text_peft_targets(model):
    text_targets = []
    for name, module in model.named_modules():
        if "vision_tower" in name or "audio_tower" in name or "vision" in name or "audio" in name:
            continue
            
        # Select ONLY modules under model.language_model (or model.layers as fallback)
        if "language_model" in name or "model.layers" in name:
            if "q_proj" in name or "v_proj" in name:
                if isinstance(module, (torch.nn.Linear, bnb.nn.Linear4bit)):
                    text_targets.append(name)
                    
    logger.info(f"Total text-branch targets selected: {len(text_targets)}")
    if text_targets:
        logger.info(f"First 10 targets:\\n" + "\\n".join(text_targets[:10]))
        logger.info(f"Last 10 targets:\\n" + "\\n".join(text_targets[-10:]))
        
    assert len(text_targets) > 0, "Discovered zero text targets! Ensure the model contains language_model or model.layers with q_proj/v_proj Linear modules."
    return text_targets

def create_adapter_config(method: str, r: int, alpha: int, target_modules: list, dropout: float):
    use_dora = (method.lower() == "dora")
    logger.info(f"Creating adapter config for {method} (r={r}, alpha={alpha}, dora={use_dora})")
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=use_dora
    )

def inject_adapter(model, config):
    logger.info("Skipping prepare_model_for_kbit_training for Gemma 4 4-bit to avoid full-model FP32 upcast.")
    
    for param in model.parameters():
        param.requires_grad_(False)
    
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        
    for name, param in model.named_parameters():
        if param.ndim == 1 and "norm" in name.lower():
            param.data = param.data.to(torch.float32)
            param.requires_grad_(False)
            
    # Verify every target resolves to a supported Linear4bit module before injection
    if config.target_modules:
        for t in config.target_modules:
            mod = dict(model.named_modules()).get(t, None)
            assert mod is not None, f"Target {t} not found in model!"
            assert isinstance(mod, (torch.nn.Linear, bnb.nn.Linear4bit)), f"Target {t} is {type(mod)}, expected Linear/Linear4bit!"
            
    model = get_peft_model(model, config)
    
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            
    logger.info(
        f"trainable params AFTER PEFT: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )
    
    return model, trainable_params, all_param
