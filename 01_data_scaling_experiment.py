#!/usr/bin/env python3
from lora.lora_trainer import LoRATrainer
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import os
import random
import numpy as np
import torch
import json

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    setup_seed(42)
    os.makedirs("experiments/data_scaling", exist_ok=True)
    
    trainer = LoRATrainer()
    
    if not os.path.exists("data/train_fixed.json"):
        train_data, val_data, test_data = trainer.load_and_split_data("data/npd_training_mt.json")
    else:
        train_data, val_data, test_data = trainer.load_fixed_splits()
    
    data_sizes = [1000, 5000, 10000, len(train_data)]
    results = []
    
    for size in data_sizes:
        if size > len(train_data):
            continue
            
        try:
            print(f"Training with {size} samples")
            
            with wandb.init(project="lora-data-scaling", reinit=True, config={
                "experiment": "data_scaling",
                "model": trainer.model_name,
                "data_size": size
            }) as run:
                
                train_subset = trainer.create_subset(train_data, size)
                config = {
                    'output_dir': f'experiments/data_size_{size}',
                    'r': 16,
                    'alpha': 32,
                    'dropout': 0.1,
                    'epochs': 3,
                    'batch_size': 16,
                    'learning_rate': 1e-4,
                    'use_wandb': True
                }
                run.config.update(config)
                
                result = trainer.train_model(train_subset, val_data, config)
                result['data_size'] = size
                results.append(result)
                
                run.log({
                    "bleu": result['bleu'],
                    "chrf": result['chrf']
                })
                print(f"BLEU: {result['bleu']:.4f}, chrF: {result['chrf']:.4f}")
                
        except Exception as e:
            print(f"Data size {size} failed: {e}")
            results.append({
                'data_size': size,
                'bleu': 0.0,
                'chrf': 0.0,
                'failed': True
            })
            continue
    
    df = pd.DataFrame(results)
    df.to_csv("experiments/data_scaling/results.csv", index=False)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['data_size'], df['bleu'], 'b-o')
    plt.xlabel('Training Data Size')
    plt.ylabel('BLEU Score')
    plt.title('Learning Curve - BLEU')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(df['data_size'], df['chrf'], 'r-o')
    plt.xlabel('Training Data Size')
    plt.ylabel('chrF Score')
    plt.title('Learning Curve - chrF')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('experiments/data_scaling/learning_curve.png', dpi=300)
    plt.close()
    
    max_bleu_idx = df['bleu'].idxmax()
    optimal_size = df.loc[max_bleu_idx, 'data_size']
    
    performance_ratios = df['bleu'] / df['bleu'].max()
    efficient_sizes = df[performance_ratios >= 0.95]['data_size'].tolist()
    recommended_size = min(efficient_sizes) if efficient_sizes else optimal_size
    
    with open("experiments/data_scaling/recommended_size.txt", "w") as f:
        f.write(str(recommended_size))
    
    summary_results = {
        "optimal_size": int(optimal_size),
        "recommended_size": int(recommended_size),
        "all_results": results
    }
    
    with open("experiments/data_scaling/summary.json", "w") as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"Optimal data size: {optimal_size}")
    print(f"Recommended size: {recommended_size}")
    print(f"Results saved to experiments/data_scaling/")

if __name__ == "__main__":
    main()