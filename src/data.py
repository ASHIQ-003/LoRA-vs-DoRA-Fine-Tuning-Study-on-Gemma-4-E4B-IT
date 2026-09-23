from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

def load_gsm8k_data(split="train", max_samples=None, seed=42):
    logger.info(f"Loading GSM8K {split} split (max_samples={max_samples})")
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
        
    return dataset

def prepare_dataset_for_sft(dataset, tokenizer, max_length=1024):
    """
    Prepares GSM8K dataset for Supervised Fine-Tuning with explicit tokenization
    and strict completion-only masking (prompt=-100).
    """
    from src.prompting import build_gsm8k_prompt
    
    def tokenize_and_mask(example):
        prompt = build_gsm8k_prompt(example['question'])
        completion = example['answer'] + "<end_of_turn>\\n"
        
        # Tokenize separately to precisely locate the boundary
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        
        input_ids = prompt_ids + completion_ids
        # Mask the prompt tokens
        labels = [-100] * len(prompt_ids) + completion_ids
        attention_mask = [1] * len(input_ids)
        
        # Truncate if necessary (taking from the left of the prompt is usually safer for GSM8K, 
        # but here we just take the first max_length tokens to avoid crashing if it's too long)
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            attention_mask = attention_mask[:max_length]
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
    logger.info("Explicitly tokenizing and applying completion-only masking to the dataset...")
    dataset = dataset.map(tokenize_and_mask, remove_columns=dataset.column_names)
    return dataset
