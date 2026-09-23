import json
import os
import torch
import logging
from tqdm import tqdm
from src.prompting import build_gsm8k_prompt, extract_final_answer

logger = logging.getLogger(__name__)

def evaluate_model(model, tokenizer, dataset, max_new_tokens=256, batch_size=4):
    """
    Evaluates the model on GSM8K dataset.
    """
    model.eval()
    results = []
    correct_count = 0
    total_count = len(dataset)
    
    tokenizer.padding_side = "left"
    
    with tqdm(total=total_count, desc="Evaluating (samples)") as pbar:
        for i in range(0, total_count, batch_size):
            batch = dataset[i:i+batch_size]
            questions = batch['question']
            gold_answers = [extract_final_answer(ans) for ans in batch['answer']]
            
            prompts = [build_gsm8k_prompt(q) for q in questions]
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
            generated_texts = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            for q, gold, gen in zip(questions, gold_answers, generated_texts):
                pred = extract_final_answer(gen)
                is_correct = (pred == gold)
                if is_correct:
                    correct_count += 1
                    
                results.append({
                    "question": q,
                    "gold_answer": gold,
                    "generated_text": gen,
                    "predicted_answer": pred,
                    "correct": is_correct
                })
                
            pbar.update(len(questions))
            
    accuracy = correct_count / total_count if total_count > 0 else 0
    logger.info(f"Evaluation Accuracy: {accuracy:.4f}")
    
    return accuracy, results

def save_results(results, accuracy, output_path, metadata=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = {
        "metadata": metadata or {},
        "accuracy": accuracy,
        "results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to {output_path}")
