from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling
import os
import logging
from src.adapters import create_adapter_config, inject_adapter
from src.model import load_model_and_tokenizer

logger = logging.getLogger(__name__)

def create_trainer(
    model, 
    tokenizer, 
    train_dataset, 
    eval_dataset, 
    config, 
    output_dir,
    run_name
):
    """
    Sets up the SFTTrainer with the provided configuration.
    """
    logger.info(f"Setting up trainer for run: {run_name}")
    
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        num_train_epochs=config['training']['epochs'],
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset else "no",
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        bf16=True,  # Assuming hardware supports it, usually yes for Ampere+
        fp16=False,
        dataset_text_field="text",
        max_seq_length=config['training']['max_length'],
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=run_name,
        seed=config.get('experiment', {}).get('seed', 42),
    )
    
    # Use standard language modeling collator (trains on all tokens by default if instruction template isn't applied with DataCollatorForCompletionOnlyLM)
    # To strict follow instructions, we should use DataCollatorForCompletionOnlyLM
    from trl import DataCollatorForCompletionOnlyLM
    # For Gemma 4 E4B, the assistant turn indicator is typically <start_of_turn>model\n
    response_template = "<start_of_turn>model\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    
    return trainer
