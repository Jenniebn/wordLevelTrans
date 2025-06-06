import os
import yaml

CONFIG_SCHEMAS = {
    "confs/train.yaml": {
        "prefix": "Training or translation",
        "base_lr": "Starting learning rate for cosine scheduler",
        "final_lr": "Final learning rate for cosine scheduler",
        "batch_size": "Training batch size",
        "num_epochs": "Number of training epochs",
        "pos_weight": "Hyperparameter to adjust the penalty on wrong translation in the nn.BCEWithLogitsLoss",
        "save_every": "Frequency how often model is ran on the validation set",
        "save_name": "Filename of the final trained model checkpoint",
        "zhzh_model_path": "Path to zh-zh model. Required."
    },
    "confs/test.yaml": {
        "prefix": "Training or translation",
        "batch_size": "Test batch size",
        "num_trans": "Number of translations you'd like to see from model output. Default to the same number of correct translations in the golden set. Set to 'all' if you want to every model output.",
        "pos_weight": "Hyperparameter to adjust the penalty on wrong translation in the nn.BCEWithLogitsLoss",
        "zhzh_model_path": "Path to zh-zh model. Required.",
        "enzh_model_path": "Path to en-zh model. Required."
    },
}

def yamlread(path):
    with open(os.path.expanduser(path), 'r') as f:
        return yaml.safe_load(f)
    
def print_yaml_help(conf_name):
    schema = CONFIG_SCHEMAS.get(conf_name, {})
    if not schema:
        raise FileNotFoundError(f"No help schema found for '{conf_name}'. Please double check your file path.")
    print(f"\nYAML Config Help for {conf_name}:")
    for k, v in schema.items():
        print(f"  {k}: {v}")
    print()

