# utils/lora_trainer.py

import torch
import numpy as np
from typing import Dict, List
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from utils.evaluation import TranslationEvaluator

class LoRATrainer:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", random_seed=42):
        self.model_name = model_name
        self.random_seed = random_seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.src_lang = "eng_Latn"
        self.tgt_lang = "nob_Latn"
        
        self.evaluator = TranslationEvaluator(use_comet=False)
        
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang
        
        self.setup_seeds()
    
    def setup_seeds(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
    
    def enable_comet_evaluation(self, comet_model="Unbabel/wmt22-comet-da"):
        self.evaluator = TranslationEvaluator(use_comet=True, comet_model=comet_model)
    
    def tokenize_function(self, examples):
        sources = examples['source']
        targets = examples['target']
        
        self.tokenizer.src_lang = self.src_lang
        model_inputs = self.tokenizer(
            sources, max_length=128, truncation=True, padding=False
        )
        
        self.tokenizer.src_lang = self.tgt_lang
        labels = self.tokenizer(
            targets, max_length=128, truncation=True, padding=False
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        bleu_scores = self.evaluator.compute_bleu(decoded_preds, decoded_labels)
        chrf_scores = self.evaluator.compute_chrf(decoded_preds, decoded_labels)
        
        metrics = {}
        metrics.update(bleu_scores)
        metrics.update(chrf_scores)
        
        return metrics
    
    def setup_lora_model(self, r=16, alpha=32, dropout=0.1):
        base_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        )
        
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        return model
    
    def create_training_args(self, config: Dict):
        return Seq2SeqTrainingArguments(
            output_dir=config.get('output_dir', 'output'),
            num_train_epochs=config.get('epochs', 3),
            per_device_train_batch_size=config.get('batch_size', 16),
            per_device_eval_batch_size=config.get('batch_size', 16),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
            warmup_steps=config.get('warmup_steps', 100),
            learning_rate=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01),
            logging_steps=config.get('logging_steps', 50),
            eval_strategy="steps",
            eval_steps=config.get('eval_steps', 200),
            save_strategy="steps",
            save_steps=config.get('save_steps', 500),
            save_total_limit=config.get('save_total_limit', 2),
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=config.get('generation_max_length', 128),
            fp16=config.get('fp16', True),
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            report_to=["wandb"] if config.get('use_wandb', False) else []
        )
    
    def train_model(self, train_data: List[Dict], val_data: List[Dict], config: Dict):
        model = self.setup_lora_model(
            r=config.get('r', 16),
            alpha=config.get('alpha', 32),
            dropout=config.get('dropout', 0.1)
        )
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        train_dataset = train_dataset.map(
            self.tokenize_function, batched=True,
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            self.tokenize_function, batched=True,
            remove_columns=val_dataset.column_names
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=model, padding=True
        )
        
        training_args = self.create_training_args(config)
        
        callbacks = []
        if config.get('early_stopping_patience'):
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=config['early_stopping_patience']
            ))
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        trainer.train()
        eval_result = trainer.evaluate()
        
        return {
            "trainer": trainer,
            "model": model,
            "bleu": eval_result["eval_bleu"],
            "chrf": eval_result["eval_chrf"],
            "loss": eval_result["eval_loss"]
        }
    
    def evaluate_with_all_metrics(self, model, test_data: List[Dict], 
                                 sources: List[str] = None):
        
        test_dataset = Dataset.from_list(test_data)
        test_dataset = test_dataset.map(
            self.tokenize_function, batched=True,
            remove_columns=test_dataset.column_names
        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        predictions = trainer.predict(test_dataset)
        decoded_preds = self.tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
        
        labels = np.where(predictions.label_ids != -100, predictions.label_ids, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        if sources is None:
            sources = [item.get('source', '') for item in test_data]
        
        all_metrics = self.evaluator.evaluate_all_metrics(sources, decoded_preds, decoded_labels)
        
        return all_metrics