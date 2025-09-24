from lora.lora_trainer import LoRATrainer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    os.makedirs("experiments/02_parameter_sensitivity", exist_ok=True)
    
    trainer = LoRATrainer()
    train_data, val_data, test_data = trainer.load_fixed_splits()
    
    data_size = 10000
    train_subset = trainer.create_subset(train_data, data_size)
    
    r_values = [8, 16, 32]
    alpha_values = [16, 32, 64]
    dropout_values = [0.0, 0.1, 0.2]
    
    results = []
    experiment_count = 0
    total = len(r_values) * len(alpha_values) * len(dropout_values)
    
    for r in r_values:
        for alpha in alpha_values:
            for dropout in dropout_values:
                experiment_count += 1
                print(f"Experiment {experiment_count}/{total}: r={r}, alpha={alpha}, dropout={dropout}")
                
                try:
                    with wandb.init(project="lora-parameter-sensitivity", reinit=True, config={
                        "experiment": "parameter_sensitivity",
                        "data_size": data_size,
                        "r": r,
                        "alpha": alpha,
                        "dropout": dropout
                    }) as run:
                        
                        config = {
                            'output_dir': f'experiments/lora_r{r}_a{alpha}_d{dropout}',
                            'r': r, 'alpha': alpha, 'dropout': dropout,
                            'epochs': 3, 'batch_size': 16,
                            'learning_rate': 1e-4, 'use_wandb': True
                        }
                        
                        result = trainer.train_model(train_subset, val_data, config)
                        result.update({'r': r, 'alpha': alpha, 'dropout': dropout})
                        results.append(result)
                        
                        run.log({
                            "bleu": result['bleu'], 
                            "chrf": result['chrf']
                        })
                        
                        print(f"BLEU: {result['bleu']:.4f}")
                        
                except Exception as e:
                    print(f"Experiment r={r}, alpha={alpha}, dropout={dropout} failed: {e}")
                    results.append({
                        'r': r, 'alpha': alpha, 'dropout': dropout,
                        'bleu': 0.0, 'chrf': 0.0, 'failed': True
                    })
                    continue
    
    df = pd.DataFrame(results)
    df.to_csv("experiments/02_parameter_sensitivity/results.csv", index=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    r_impact = df.groupby('r')['bleu'].mean()
    axes[0,0].bar(r_impact.index, r_impact.values)
    axes[0,0].set_title('BLEU vs LoRA Rank')
    axes[0,0].set_xlabel('r')
    
    alpha_impact = df.groupby('alpha')['bleu'].mean()
    axes[0,1].bar(alpha_impact.index, alpha_impact.values)
    axes[0,1].set_title('BLEU vs LoRA Alpha')
    axes[0,1].set_xlabel('alpha')
    
    dropout_impact = df.groupby('dropout')['bleu'].mean()
    axes[1,0].bar(dropout_impact.index, dropout_impact.values)
    axes[1,0].set_title('BLEU vs Dropout')
    axes[1,0].set_xlabel('dropout')
    
    pivot_df = df.pivot_table(values='bleu', index='r', columns='alpha', aggfunc='mean')
    sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1,1])
    axes[1,1].set_title('BLEU Heatmap: r vs alpha')
    
    plt.tight_layout()
    plt.savefig('experiments/02_parameter_sensitivity/sensitivity.png', dpi=300)
    plt.show()
    
    best_idx = df['bleu'].idxmax()
    best_config = df.loc[best_idx]
    
    print(f"Best Configuration:")
    print(f"r={best_config['r']}, alpha={best_config['alpha']}, dropout={best_config['dropout']}")
    print(f"BLEU: {best_config['bleu']:.4f}")
    
    with open("experiments/02_parameter_sensitivity/best_config.json", "w") as f:
        json.dump(best_config.to_dict(), f, indent=2)

if __name__ == "__main__":
    main()