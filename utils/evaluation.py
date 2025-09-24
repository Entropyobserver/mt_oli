# utils/evaluation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import evaluate
from comet import download_model, load_from_checkpoint

class TranslationEvaluator:
    def __init__(self, use_comet=False, comet_model="Unbabel/wmt22-comet-da"):
        self.bleu = evaluate.load("bleu")
        self.chrf = evaluate.load("chrf")
        
        self.use_comet = use_comet
        self.comet_model = None
        
        if self.use_comet:
            try:
                comet_model_path = download_model(comet_model)
                self.comet_model = load_from_checkpoint(comet_model_path)
                print("COMET model loaded successfully")
            except Exception as e:
                print(f"COMET model loading failed: {e}")
                self.use_comet = False
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> Dict:
        formatted_refs = [[ref] for ref in references]
        result = self.bleu.compute(predictions=predictions, references=formatted_refs)
        
        return {
            "bleu": result["bleu"],
            "bleu_1": result["precisions"][0] if len(result["precisions"]) > 0 else 0,
            "bleu_2": result["precisions"][1] if len(result["precisions"]) > 1 else 0,
            "bleu_3": result["precisions"][2] if len(result["precisions"]) > 2 else 0,
            "bleu_4": result["precisions"][3] if len(result["precisions"]) > 3 else 0,
        }
    
    def compute_chrf(self, predictions: List[str], references: List[str]) -> Dict:
        result = self.chrf.compute(predictions=predictions, references=references)
        
        return {
            "chrf": result["score"]
        }
    
    def compute_comet(self, sources: List[str], predictions: List[str], 
                     references: List[str]) -> Dict:
        if not self.use_comet or self.comet_model is None:
            return {"comet": 0.0, "comet_std": 0.0}
        
        try:
            comet_data = []
            for src, pred, ref in zip(sources, predictions, references):
                comet_data.append({
                    "src": str(src),
                    "mt": str(pred),
                    "ref": str(ref)
                })
            
            scores = self.comet_model.predict(comet_data, batch_size=8)
            
            return {
                "comet": float(np.mean(scores)),
                "comet_std": float(np.std(scores))
            }
        except Exception as e:
            print(f"COMET calculation error: {e}")
            return {"comet": 0.0, "comet_std": 0.0}
    
    def evaluate_all_metrics(self, sources: List[str], predictions: List[str], 
                           references: List[str]) -> Dict:
        
        predictions = [pred.strip() for pred in predictions]
        references = [ref.strip() for ref in references]
        
        metrics = {}
        
        bleu_scores = self.compute_bleu(predictions, references)
        metrics.update(bleu_scores)
        
        chrf_scores = self.compute_chrf(predictions, references)
        metrics.update(chrf_scores)
        
        if self.use_comet and sources:
            comet_scores = self.compute_comet(sources, predictions, references)
            metrics.update(comet_scores)
        
        return metrics

class ExperimentVisualizer:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        plt.style.use('default')
    
    def plot_learning_curve(self, results_df: pd.DataFrame, save_path="learning_curve.png"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].plot(results_df['data_size'], results_df['bleu'], 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Training Data Size', fontsize=12)
        axes[0].set_ylabel('BLEU Score', fontsize=12)
        axes[0].set_title('Learning Curve - BLEU', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=0)
        
        axes[1].plot(results_df['data_size'], results_df['chrf'], 'ro-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Training Data Size', fontsize=12)
        axes[1].set_ylabel('chrF Score', fontsize=12)
        axes[1].set_title('Learning Curve - chrF', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Learning curve saved to {save_path}")
    
    def plot_parameter_sensitivity(self, results_df: pd.DataFrame, save_path="parameter_sensitivity.png"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        r_impact = results_df.groupby('r')['bleu'].mean()
        axes[0,0].bar(r_impact.index.astype(str), r_impact.values, color=self.colors[0])
        axes[0,0].set_title('BLEU vs LoRA Rank (r)', fontsize=14)
        axes[0,0].set_xlabel('r', fontsize=12)
        axes[0,0].set_ylabel('Average BLEU', fontsize=12)
        
        alpha_impact = results_df.groupby('alpha')['bleu'].mean()
        axes[0,1].bar(alpha_impact.index.astype(str), alpha_impact.values, color=self.colors[1])
        axes[0,1].set_title('BLEU vs LoRA Alpha', fontsize=14)
        axes[0,1].set_xlabel('alpha', fontsize=12)
        axes[0,1].set_ylabel('Average BLEU', fontsize=12)
        
        dropout_impact = results_df.groupby('dropout')['bleu'].mean()
        axes[1,0].bar(dropout_impact.index.astype(str), dropout_impact.values, color=self.colors[2])
        axes[1,0].set_title('BLEU vs LoRA Dropout', fontsize=14)
        axes[1,0].set_xlabel('dropout', fontsize=12)
        axes[1,0].set_ylabel('Average BLEU', fontsize=12)
        
        pivot_df = results_df.pivot_table(values='bleu', index='r', columns='alpha', aggfunc='mean')
        sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1,1], 
                   cbar_kws={'label': 'BLEU Score'})
        axes[1,1].set_title('BLEU Heatmap: r vs alpha', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Parameter sensitivity plot saved to {save_path}")
    
    def plot_final_comparison(self, baseline_scores: Dict, final_scores: Dict, 
                            save_path="final_comparison.png"):
        
        metrics = ['bleu', 'chrf']
        if 'comet' in final_scores:
            metrics.append('comet')
        
        baseline_values = [baseline_scores.get(metric, 0) for metric in metrics]
        final_values = [final_scores.get(metric, 0) for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color=self.colors[0])
        bars2 = ax.bar(x + width/2, final_values, width, label='Fine-tuned', color=self.colors[1])
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Baseline vs Fine-tuned Model Performance', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Final comparison plot saved to {save_path}")
