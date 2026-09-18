import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_id: str, use_4bit: bool = True):
    logger.info(f"Loading model: {model_id} (4-bit: {use_4bit})")
    
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        logger.warning(f"AutoProcessor failed, falling back to AutoTokenizer: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    quantization_config = None
    if use_4bit:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    return model, tokenizer
