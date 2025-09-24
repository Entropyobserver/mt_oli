FINAL_EVAL_CONFIG = {
    "model_name": "facebook/nllb-200-distilled-600M",
    "random_seed": 42,
    "data_size": 10000,
    "best_params": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.1
    },
    "training_params": {
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "warmup_steps": 100,
        "eval_steps": 200,
        "save_steps": 400
    },
    "wandb": {
        "project": "lora-final-evaluation",
        "enabled": True
    },
    "output_dir": "experiments/final_model",
    "results_file": "final_results.json"
}