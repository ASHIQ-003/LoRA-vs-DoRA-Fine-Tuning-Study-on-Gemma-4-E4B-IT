from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

logger = logging.getLogger(__name__)

def create_adapter_config(method: str, r: int, alpha: int, target_modules: list, dropout: float):
    """
    Creates LoraConfig with use_dora depending on the method.
    """
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
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)
    
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS or zero3, this will be different, but for basic single GPU this is fine
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )
    
    return model, trainable_params, all_param
