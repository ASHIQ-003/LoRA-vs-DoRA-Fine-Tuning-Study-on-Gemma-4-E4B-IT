import logging
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_id: str, use_4bit: bool = True):
    logger.info(f"Loading processor and model from {model_id} (4-bit: {use_4bit})")
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
        )

        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            device_map="auto",
        )
        
    return model, processor
