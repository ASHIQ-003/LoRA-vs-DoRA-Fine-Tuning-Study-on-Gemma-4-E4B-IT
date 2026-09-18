from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

def load_gsm8k_data(split="train", max_samples=None, seed=42):
    logger.info(f"Loading GSM8K {split} split (max_samples={max_samples})")
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    
    if max_samples is not None and max_samples < len(dataset):
        # Deterministic subsampling
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
        
    return dataset

def prepare_dataset_for_sft(dataset, tokenizer, max_length=512):
    """
    Prepares GSM8K dataset for Supervised Fine-Tuning.
    """
    from src.prompting import build_gsm8k_prompt
    
    def format_function(example):
        question = example['question']
        gold_answer = example['answer']
        # The prompt up to the model's turn
        prompt = build_gsm8k_prompt(question)
        # The full text includes the gold answer and the EOS token
        full_text = prompt + gold_answer + "<end_of_turn>"
        return {"text": full_text}
        
    dataset = dataset.map(format_function, remove_columns=dataset.column_names)
    return dataset
