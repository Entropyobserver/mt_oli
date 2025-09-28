#!/usr/bin/env python3
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import torch
from utils.lora_trainer import LoRATrainer
from utils.data_loader import DataLoader

def setup_environment():
    torch.manual_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs("experiments/data_scaling", exist_ok=True)

def run_phase_experiment(data_sizes, train_data, val_data, trainer):
    results = []
    
    for size in data_sizes:
        if size > len(train_data):
            continue
            
        print(f"Training with {size} samples")
        
        try:
            with wandb.init(
                project="lora-data-scaling", 
                reinit=True,
                config={"data_size": size}
            ) as run:
                
                train_subset = train_data[:size]
                
                config = {
                    'output_dir': f'experiments/data_size_{size}',
                    'r': 16,
                    'alpha': 32,
                    'dropout': 0.1,
                    'epochs': 3,
                    'batch_size': 8,
                    'learning_rate': 5e-4,
                    'use_wandb': False,  # Disable WandB logging inside trainer
                    'eval_steps': min(200, len(train_subset) // 4),
                    'save_steps': min(500, len(train_subset) // 2)
                }
                
                result = trainer.train_model(train_subset, val_data, config)
                
                metrics = {
                    'data_size': size,
                    'bleu': result['bleu'],
                    'chrf': result['chrf'],
                    'loss': result['loss']
                }
                
                results.append(metrics)
                
                run.log({
                    "bleu": metrics['bleu'],
                    "chrf": metrics['chrf']
                })
                
                print(f"BLEU: {metrics['bleu']:.4f}, chrF: {metrics['chrf']:.4f}")
                
        except Exception as e:
            print(f"Size {size} failed: {e}")
            results.append({
                'data_size': size,
                'bleu': 0.0,
                'chrf': 0.0,
                'loss': 999.0,
                'failed': True
            })
            
    return results

def should_continue_experiment(results, phase_name):
    if not results or len(results) == 0:
        return False
        
    valid_results = [r for r in results if not r.get('failed', False)]
    
    if len(valid_results) == 0:
        print(f"Phase {phase_name}: All experiments failed, stopping")
        return False
    
    best_bleu = max(r['bleu'] for r in valid_results)
    
    if phase_name == "1" and best_bleu < 0.15:
        print(f"Phase {phase_name}: Poor performance (BLEU < 0.15), check data quality")
        return False
    
    return True

def analyze_results(all_results):
    df = pd.DataFrame(all_results)
    
    if df.empty or df['bleu'].max() == 0:
        return {"optimal_size": 500, "recommended_size": 500}
    
    valid_df = df[~df.get('failed', False)]
    
    if valid_df.empty:
        return {"optimal_size": 500, "recommended_size": 500}
    
    best_idx = valid_df['bleu'].idxmax()
    optimal_size = valid_df.loc[best_idx, 'data_size']
    
    performance_threshold = valid_df['bleu'].max() * 0.95
    efficient_candidates = valid_df[valid_df['bleu'] >= performance_threshold]
    recommended_size = efficient_candidates['data_size'].min()
    
    return {
        "optimal_size": int(optimal_size),
        "recommended_size": int(recommended_size),
        "all_results": all_results
    }

def save_results(results, analysis):
    df = pd.DataFrame(results)
    df.to_csv("experiments/data_scaling/results.csv", index=False)
    
    if not df.empty and df['bleu'].max() > 0:
        plt.figure(figsize=(12, 5))
        
        valid_df = df[~df.get('failed', False)]
        
        if not valid_df.empty:
            plt.subplot(1, 2, 1)
            plt.plot(valid_df['data_size'], valid_df['bleu'], 'b-o')
            plt.xlabel('Training Data Size')
            plt.ylabel('BLEU Score')
            plt.title('Learning Curve - BLEU')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(valid_df['data_size'], valid_df['chrf'], 'r-o')
            plt.xlabel('Training Data Size')
            plt.ylabel('chrF Score')
            plt.title('Learning Curve - chrF')
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig('experiments/data_scaling/learning_curve.png', dpi=300)
        plt.close()
    
    with open("experiments/data_scaling/summary.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    with open("experiments/data_scaling/recommended_size.txt", "w") as f:
        f.write(str(analysis["recommended_size"]))

def main():
    setup_environment()
    
    trainer = LoRATrainer()
    data_loader = DataLoader()
    
    if not os.path.exists("data/train_fixed.json"):
        print("Creating fixed data splits")
        train_data, val_data, test_data = data_loader.create_fixed_splits("data/npd_training_mt.json")
        data_loader.save_fixed_splits(train_data, val_data, test_data, 
                                    "data/train_fixed.json", 
                                    "data/val_fixed.json", 
                                    "data/test_fixed.json")
    else:
        train_data, val_data, test_data = data_loader.load_fixed_splits()
    
    all_results = []
    
    phase_1_sizes = [500, 1000, 2000]
    phase_2_sizes = [3000, 5000]
    phase_3_sizes = [10000, min(21059, len(train_data))]
    
    print("Starting Phase 1: Feasibility validation")
    phase_1_results = run_phase_experiment(phase_1_sizes, train_data, val_data, trainer)
    all_results.extend(phase_1_results)
    
    if not should_continue_experiment(phase_1_results, "1"):
        analysis = analyze_results(all_results)
        save_results(all_results, analysis)
        return
    
    print("Starting Phase 2: Performance balance")
    phase_2_results = run_phase_experiment(phase_2_sizes, train_data, val_data, trainer)
    all_results.extend(phase_2_results)
    
    if should_continue_experiment(phase_2_results, "2"):
        print("Starting Phase 3: Maximum potential")
        phase_3_results = run_phase_experiment(phase_3_sizes, train_data, val_data, trainer)
        all_results.extend(phase_3_results)
    
    analysis = analyze_results(all_results)
    save_results(all_results, analysis)
    
    print(f"Optimal data size: {analysis['optimal_size']}")
    print(f"Recommended size: {analysis['recommended_size']}")
    print("Results saved to experiments/data_scaling/")

if __name__ == "__main__":
    main()