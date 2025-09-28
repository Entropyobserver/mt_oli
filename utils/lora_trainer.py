import torch
import numpy as np
import os
from typing import Dict, List
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

class LoRATrainer:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bleu = evaluate.load("bleu")
        self.chrf = evaluate.load("chrf")
        
        self.tokenizer.src_lang = "eng_Latn"
        self.tokenizer.tgt_lang = "nob_Latn"
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
    def tokenize_function(self, examples):
        self.tokenizer.src_lang = "eng_Latn"
        model_inputs = self.tokenizer(
            examples['source'], 
            max_length=128, 
            truncation=True, 
            padding=False
        )
        
        self.tokenizer.src_lang = "nob_Latn"
        labels = self.tokenizer(
            examples['target'], 
            max_length=128, 
            truncation=True, 
            padding=False
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        bleu_result = self.bleu.compute(
            predictions=decoded_preds, 
            references=[[label] for label in decoded_labels]
        )
        chrf_result = self.chrf.compute(
            predictions=decoded_preds, 
            references=decoded_labels
        )
        
        return {
            "bleu": bleu_result["bleu"],
            "chrf": chrf_result["score"]
        }
    
    def setup_model(self, r=16, alpha=32, dropout=0.1):
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        for param in base_model.parameters():
            param.requires_grad = False
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        )
        
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        
        return model
    
    def train_model(self, train_data: List[Dict], val_data: List[Dict], config: Dict):
        model = self.setup_model(
            r=config.get('r', 16),
            alpha=config.get('alpha', 32),
            dropout=config.get('dropout', 0.1)
        )
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        train_dataset = train_dataset.map(
            self.tokenize_function, 
            batched=True,
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            self.tokenize_function, 
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=model,
            padding=True
        )
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=config.get('output_dir', 'output'),
            num_train_epochs=config.get('epochs', 3),
            per_device_train_batch_size=config.get('batch_size', 8),
            per_device_eval_batch_size=config.get('batch_size', 8),
            gradient_accumulation_steps=2,
            warmup_steps=100,
            learning_rate=config.get('learning_rate', 5e-4),
            weight_decay=0.01,
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=config.get('eval_steps', 200),
            save_strategy="steps",
            save_steps=config.get('save_steps', 500),
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=128,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=["wandb"] if config.get('use_wandb', False) else []
        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()
        eval_result = trainer.evaluate()
        
        return {
            "model": model,
            "trainer": trainer,
            "bleu": eval_result.get("eval_bleu", 0.0),
            "chrf": eval_result.get("eval_chrf", 0.0),
            "loss": eval_result.get("eval_loss", 0.0)
        }