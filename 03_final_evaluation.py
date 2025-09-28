#!/usr/bin/env python3
import os
import json
import wandb
import torch
from utils.lora_trainer import LoRATrainer
from utils.data_loader import DataLoader

def setup_environment():
    torch.manual_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs("experiments/final_evaluation", exist_ok=True)

def load_best_configuration():
    try:
        with open("experiments/parameter_sensitivity/best_config.json", "r") as f:
            config = json.load(f)
            return {
                'r': int(config.get('r', 16)),
                'alpha': int(config.get('alpha', 32)),
                'dropout': float(config.get('dropout', 0.1))
            }
    except:
        return {'r': 16, 'alpha': 32, 'dropout': 0.1}

def load_recommended_data_size():
    try:
        with open("experiments/data_scaling/recommended_size.txt", "r") as f:
            return int(f.read().strip())
    except:
        return 2000

def evaluate_on_test_set(trainer, model, test_data):
    try:
        test_sources = [item['source'] for item in test_data]
        test_targets = [item['target'] for item in test_data]
        
        predictions = []
        batch_size = 8
        
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            batch_sources = [item['source'] for item in batch]
            
            inputs = trainer.tokenizer(
                batch_sources,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            batch_predictions = trainer.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_predictions)
        
        bleu_result = trainer.bleu.compute(
            predictions=predictions,
            references=[[target] for target in test_targets]
        )
        
        chrf_result = trainer.chrf.compute(
            predictions=predictions,
            references=test_targets
        )
        
        return {
            'bleu': bleu_result['bleu'],
            'chrf': chrf_result['score'],
            'predictions': predictions[:10],
            'references': test_targets[:10]
        }
        
    except Exception as e:
        print(f"Test evaluation failed: {e}")
        return {
            'bleu': 0.0,
            'chrf': 0.0,
            'predictions': [],
            'references': []
        }

def main():
    setup_environment()
    
    trainer = LoRATrainer()
    data_loader = DataLoader()
    
    train_data, val_data, test_data = data_loader.load_fixed_splits()
    
    best_params = load_best_configuration()
    data_size = load_recommended_data_size()
    train_subset = train_data[:data_size]
    
    print("Final Model Training and Evaluation")
    print(f"Training samples: {len(train_subset)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Configuration: {best_params}")
    
    with wandb.init(
        project="lora-final-evaluation",
        config={
            "experiment": "final_evaluation",
            "data_size": data_size,
            **best_params
        }
    ) as run:
        
        config = {
            'output_dir': 'experiments/final_model',
            'r': best_params['r'],
            'alpha': best_params['alpha'],
            'dropout': best_params['dropout'],
            'epochs': 5,
            'batch_size': 8,
            'learning_rate': 5e-4,
            'use_wandb': True,
            'eval_steps': 200,
            'save_steps': 500
        }
        
        result = trainer.train_model(train_subset, val_data, config)
        trained_model = result['model']
        
        print("Evaluating on test set")
        test_metrics = evaluate_on_test_set(trainer, trained_model, test_data)
        
        final_results = {
            'validation_bleu': result['bleu'],
            'validation_chrf': result['chrf'],
            'test_bleu': test_metrics['bleu'],
            'test_chrf': test_metrics['chrf'],
            'configuration': best_params,
            'data_size': data_size,
            'sample_predictions': test_metrics['predictions'],
            'sample_references': test_metrics['references']
        }
        
        print(f"Final Test Results:")
        print(f"BLEU: {final_results['test_bleu']:.4f}")
        print(f"chrF: {final_results['test_chrf']:.4f}")
        
        run.log({
            "final_test_bleu": final_results['test_bleu'],
            "final_test_chrf": final_results['test_chrf'],
            "validation_bleu": final_results['validation_bleu'],
            "validation_chrf": final_results['validation_chrf']
        })
        
        with open("experiments/final_evaluation/final_results.json", "w") as f:
            json.dump(final_results, f, indent=2)
        
        print("Final evaluation completed")

if __name__ == "__main__":
    main()