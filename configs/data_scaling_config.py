DATA_SCALING_CONFIG = {
    "model_name": "facebook/nllb-200-distilled-600M",
    "random_seed": 42,
    "data_file": "data/npd_training_mt.json",
    "data_sizes": [1000, 5000, 10000, 20000],
    "fixed_lora_params": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.1
    },
    "training_params": {
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "warmup_steps": 100,
        "eval_steps": 200,
        "save_steps": 500
    },
    "wandb": {
        "project": "lora-data-scaling",
        "enabled": True
    },
    "output_dir_prefix": "experiments/data_size_"
}