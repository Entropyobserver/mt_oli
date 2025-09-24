PARAM_SENSITIVITY_CONFIG = {
    "model_name": "facebook/nllb-200-distilled-600M",
    "random_seed": 42,
    "recommended_data_size": 10000,
    "parameter_grid": {
        "r_values": [8, 16, 32],
        "alpha_values": [16, 32, 64],
        "dropout_values": [0.0, 0.1, 0.2]
    },
    "training_params": {
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "warmup_steps": 100,
        "eval_steps": 200,
        "save_steps": 500,
        "early_stopping_patience": 3
    },
    "wandb": {
        "project": "lora-parameter-sensitivity",
        "enabled": True
    },
    "output_dir_prefix": "experiments/lora_",
    "results_file": "parameter_sensitivity_results.csv"
}