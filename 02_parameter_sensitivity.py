#!/usr/bin/env python3
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import torch
from utils.lora_trainer import LoRATrainer
from utils.data_loader import DataLoader

def setup_environment():
    torch.manual_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs("experiments/parameter_sensitivity", exist_ok=True)

def run_parameter_grid(train_data, val_data, trainer):
    r_values = [8, 16, 32]
    alpha_values = [16, 32, 64]
    dropout_values = [0.0, 0.1, 0.2]
    
    results = []
    experiment_count = 0
    total_experiments = len(r_values) * len(alpha_values) * len(dropout_values)
    
    for r in r_values:
        for alpha in alpha_values:
            for dropout in dropout_values:
                experiment_count += 1
                print(f"Experiment {experiment_count}/{total_experiments}: r={r}, alpha={alpha}, dropout={dropout}")
                
                try:
                    with wandb.init(
                        project="lora-parameter-sensitivity",
                        reinit=True,
                        config={"r": r, "alpha": alpha, "dropout": dropout}
                    ) as run:
                        
                        config = {
                            'output_dir': f'experiments/param_r{r}_a{alpha}_d{dropout}',
                            'r': r,
                            'alpha': alpha,
                            'dropout': dropout,
                            'epochs': 3,
                            'batch_size': 8,
                            'learning_rate': 5e-4,
                            'use_wandb': True,
                            'eval_steps': 200,
                            'save_steps': 500
                        }
                        
                        result = trainer.train_model(train_data, val_data, config)
                        
                        metrics = {
                            'r': r,
                            'alpha': alpha,
                            'dropout': dropout,
                            'bleu': result['bleu'],
                            'chrf': result['chrf'],
                            'loss': result['loss']
                        }
                        
                        results.append(metrics)
                        
                        run.log({
                            "bleu": metrics['bleu'],
                            "chrf": metrics['chrf']
                        })
                        
                        print(f"BLEU: {metrics['bleu']:.4f}")
                        
                except Exception as e:
                    print(f"Experiment r={r}, alpha={alpha}, dropout={dropout} failed: {e}")
                    results.append({
                        'r': r,
                        'alpha': alpha,
                        'dropout': dropout,
                        'bleu': 0.0,
                        'chrf': 0.0,
                        'loss': 999.0,
                        'failed': True
                    })
                    
    return results

def create_visualizations(df):
    if df.empty or df['bleu'].max() == 0:
        print("No valid results for visualization")
        return
    
    valid_df = df[~df.get('failed', False)]
    
    if valid_df.empty:
        print("No valid results for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    r_impact = valid_df.groupby('r')['bleu'].mean()
    axes[0,0].bar(r_impact.index.astype(str), r_impact.values)
    axes[0,0].set_title('BLEU vs LoRA Rank')
    axes[0,0].set_xlabel('r')
    axes[0,0].set_ylabel('Average BLEU')
    
    alpha_impact = valid_df.groupby('alpha')['bleu'].mean()
    axes[0,1].bar(alpha_impact.index.astype(str), alpha_impact.values)
    axes[0,1].set_title('BLEU vs LoRA Alpha')
    axes[0,1].set_xlabel('alpha')
    axes[0,1].set_ylabel('Average BLEU')
    
    dropout_impact = valid_df.groupby('dropout')['bleu'].mean()
    axes[1,0].bar(dropout_impact.index.astype(str), dropout_impact.values)
    axes[1,0].set_title('BLEU vs Dropout')
    axes[1,0].set_xlabel('dropout')
    axes[1,0].set_ylabel('Average BLEU')
    
    pivot_df = valid_df.pivot_table(values='bleu', index='r', columns='alpha', aggfunc='mean')
    if not pivot_df.empty:
        sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1,1])
        axes[1,1].set_title('BLEU Heatmap: r vs alpha')
    
    plt.tight_layout()
    plt.savefig('experiments/parameter_sensitivity/sensitivity.png', dpi=300)
    plt.close()

def find_best_config(df):
    valid_df = df[~df.get('failed', False)]
    
    if valid_df.empty or valid_df['bleu'].max() == 0:
        return {'r': 16, 'alpha': 32, 'dropout': 0.1, 'bleu': 0.0}
    
    best_idx = valid_df['bleu'].idxmax()
    best_config = valid_df.loc[best_idx].to_dict()
    
    return best_config

def load_recommended_data_size():
    try:
        with open("experiments/data_scaling/recommended_size.txt", "r") as f:
            return int(f.read().strip())
    except:
        return 2000

def main():
    setup_environment()
    
    trainer = LoRATrainer()
    data_loader = DataLoader()
    
    train_data, val_data, test_data = data_loader.load_fixed_splits()
    
    recommended_size = load_recommended_data_size()
    train_subset = train_data[:recommended_size]
    
    print(f"Running parameter sensitivity with {len(train_subset)} training samples")
    
    results = run_parameter_grid(train_subset, val_data, trainer)
    
    df = pd.DataFrame(results)
    df.to_csv("experiments/parameter_sensitivity/results.csv", index=False)
    
    create_visualizations(df)
    
    best_config = find_best_config(df)
    
    print(f"Best Configuration:")
    print(f"r={best_config['r']}, alpha={best_config['alpha']}, dropout={best_config['dropout']}")
    print(f"BLEU: {best_config['bleu']:.4f}")
    
    with open("experiments/parameter_sensitivity/best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

if __name__ == "__main__":
    main()