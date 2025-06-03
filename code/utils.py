import os
import yaml

def yamlread(path):
    with open(os.path.expanduser(path), 'r') as f:
        return yaml.safe_load(f)