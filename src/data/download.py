from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os
from pathlib import Path
import yaml
import argparse



def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_data(config_path):
    config = load_config(config_path)

    load_dotenv()
    hf_token= os.getenv("hf_token")
    login(token=hf_token)

    RAW_DATA_DIR = Path(config['paths']['local_dir'])
    GOUD_DATA_PATH = config['paths']['hf_path']
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if any(RAW_DATA_DIR.iterdir()) if RAW_DATA_DIR.exists() else False:
        print(f"data already exists")
        return

    print("downloading dataset")
    ds = load_dataset(GOUD_DATA_PATH)
    
    ds.save_to_disk(RAW_DATA_DIR)
    print(f"dataset saved {RAW_DATA_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    args = parser.parse_args()
    
    download_data(args.config)