from lora.lora_trainer import LoRATrainer
from datasets import Dataset
import wandb
import random
import numpy as np
import torch
import os
import json

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    setup_seed(42)
    os.makedirs("experiments/03_final_evaluation", exist_ok=True)
    
    trainer = LoRATrainer()
    train_data, val_data, test_data = trainer.load_fixed_splits()

    data_size = 10000
    train_subset = trainer.create_subset(train_data, data_size)

    best_params = {'r': 16, 'alpha': 32, 'dropout': 0.1}

    with wandb.init(project="lora-final-evaluation", config={
        "experiment": "final_evaluation",
        "data_size": data_size,
        **best_params
    }) as run:

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

        result = trainer.train_model(train_subset, val_data, config)
        trained_model = result.get('model') or trainer.setup_lora_model(
            r=config['r'],
            alpha=config['alpha'],
            dropout=config['dropout']
        )

        test_dataset = Dataset.from_list(test_data)
        test_dataset = test_dataset.map(
            trainer.tokenize_function, batched=True,
            remove_columns=test_dataset.column_names
        )

        print("\nEvaluating with COMET scores...")
        test_sources = [item['source'] for item in test_data]
        final_metrics = trainer.evaluate_with_comet(trained_model, test_dataset, test_sources)

        print(f"\nFinal Test Results:")
        print(f"BLEU: {final_metrics['bleu']:.4f}")
        print(f"chrF: {final_metrics['chrf']:.4f}")
        print(f"COMET: {final_metrics['comet']:.4f}")

        run.log({
            "final_test_bleu": final_metrics['bleu'],
            "final_test_chrf": final_metrics['chrf'],
            "final_test_comet": final_metrics['comet']
        })

        with open("experiments/03_final_evaluation/final_results.json", "w") as f:
            json.dump(final_metrics, f, indent=2)

if __name__ == "__main__":
    main()