from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
import os
import torch
import logging
import inspect

logger = logging.getLogger(__name__)

class CustomSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        
        if not hasattr(self, "_custom_step_count"):
            self._custom_step_count = 0
        self._custom_step_count += 1
        
        log_freq = self.args.gradient_accumulation_steps * self.args.logging_steps
        if self._custom_step_count % log_freq == 0:
            total_norm = 0.0
            max_grad = 0.0
            grad_count = 0
            zero_grad = 0
            for p in model.parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        grad_count += 1
                        total_norm += p.grad.data.norm(2).item() ** 2
                        max_grad = max(max_grad, p.grad.data.abs().max().item())
                    else:
                        zero_grad += 1
            total_norm = total_norm ** 0.5
            logger.info(f"\\n[Custom Grad Check] Norm: {total_norm:.4f} | Max Absolute Grad: {max_grad:.4f} | Tensors with grads: {grad_count} | Tensors missing grads: {zero_grad}")
            
        return loss

def create_trainer(model, tokenizer_or_processor, train_dataset, eval_dataset, config, output_dir, run_name):
    logger.info(f"Setting up trainer for run: {run_name}")
    
    if hasattr(tokenizer_or_processor, "tokenizer") and tokenizer_or_processor.tokenizer is not None:
        actual_tokenizer = tokenizer_or_processor.tokenizer
    else:
        actual_tokenizer = tokenizer_or_processor
        
    if actual_tokenizer.pad_token_id is None:
        actual_tokenizer.pad_token_id = actual_tokenizer.eos_token_id or 0
        actual_tokenizer.pad_token = actual_tokenizer.eos_token or "<pad>"
        
    desired_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config['training']['batch_size'],
        "gradient_accumulation_steps": config['training']['gradient_accumulation_steps'],
        "learning_rate": config['training']['learning_rate'],
        "num_train_epochs": config['training']['epochs'],
        "logging_steps": 5,
        "save_strategy": "epoch",
        "eval_strategy": "epoch" if eval_dataset else "no",
        "evaluation_strategy": "epoch" if eval_dataset else "no",
        "warmup_ratio": config['training']['warmup_ratio'],
        "weight_decay": config['training']['weight_decay'],
        "bf16": True,
        "fp16": False,
        "max_length": config['training']['max_length'],
        "report_to": "none",
        "run_name": run_name,
        "seed": config.get('experiment', {}).get('seed', 42),
    }
    
    valid_keys = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    config_kwargs = {k: v for k, v in desired_kwargs.items() if k in valid_keys}
    
    training_args = SFTConfig(**config_kwargs)
    collator = DataCollatorForSeq2Seq(actual_tokenizer, padding=True)
    
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": training_args,
        "data_collator": collator,
    }
    
    trainer_valid_keys = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    
    if "processing_class" in trainer_valid_keys:
        trainer_kwargs["processing_class"] = tokenizer_or_processor
    elif "tokenizer" in trainer_valid_keys:
        trainer_kwargs["tokenizer"] = actual_tokenizer
        
    trainer = CustomSFTTrainer(**trainer_kwargs)
    
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    labels = batch["labels"][0].tolist()
    if -100 in labels and any(l != -100 for l in labels):
        logger.info("REGRESSION TEST PASSED: Masking condition met (prompt is masked, completion is unmasked).")
    else:
        raise ValueError("REGRESSION TEST FAILED: Either all or none are masked. Masking condition FAILED!")
        
    return trainer
