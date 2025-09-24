# 03_final_evaluation.py

from lora_trainer import LoRATrainer
from datasets import Dataset
import wandb

def main():
    trainer = LoRATrainer()
    train_data, val_data, test_data = trainer.load_fixed_splits()
    
    data_size = 10000
    train_subset = trainer.create_subset(train_data, data_size)
    
    best_params = {'r': 16, 'alpha': 32, 'dropout': 0.1}
    
    wandb.init(project="lora-final-evaluation", config={
        "experiment": "final_evaluation",
        "data_size": data_size,
        "best_params": best_params
    })
    
    print("Final Model Training and Evaluation")
    print(f"Training samples: {len(train_subset)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Configuration: {best_params}")
    
    config = {
        'output_dir': 'experiments/final_model',
        'r': best_params['r'],
        'alpha': best_params['alpha'],
        'dropout': best_params['dropout'],
        'epochs': 5,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'use_wandb': True
    }
    
    model = trainer.setup_lora_model(
        r=config['r'],
        alpha=config['alpha'],
        dropout=config['dropout']
    )
    
    train_dataset = Dataset.from_list(train_subset)
    test_dataset = Dataset.from_list(test_data)
    
    train_dataset = train_dataset.map(
        trainer.tokenize_function, batched=True,
        remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        trainer.tokenize_function, batched=True,
        remove_columns=test_dataset.column_names
    )
    
    result = trainer.train_model(train_subset, val_data, config)
    
    print("\nEvaluating with COMET scores...")
    test_sources = [item['source'] for item in test_data]
    final_metrics = trainer.evaluate_with_comet(model, test_dataset, test_sources)
    
    print(f"\nFinal Test Results:")
    print(f"BLEU: {final_metrics['bleu']:.4f}")
    print(f"chrF: {final_metrics['chrf']:.4f}")
    print(f"COMET: {final_metrics['comet']:.4f}")
    
    wandb.log({
        "final_test_bleu": final_metrics['bleu'],
        "final_test_chrf": final_metrics['chrf'],
        "final_test_comet": final_metrics['comet']
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()