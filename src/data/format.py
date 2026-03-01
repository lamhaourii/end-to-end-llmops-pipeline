import json
import yaml
import argparse
from pathlib import Path
from datasets import load_from_disk

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def map_goud(item):
    return {
        "instruction":  "لخص هذا النص بالدارجة المغربية:",
        "input": item.get('article', ''),
        "output": item.get('headline', '')
    }

def format_data(config_path: str):
    config = load_config(config_path)
    
    cleaned_dir = Path(config['paths']['processed_dir']) / "cleaned_dataset"
    output_dir = Path(config['paths']['processed_dir'])
    formatted_data_file= config['paths']['formatted_data']
    print(f"loading processed data")
    dataset_dict = load_from_disk(str(cleaned_dir))
    output_file = output_dir / formatted_data_file
    for split_name, dataset in dataset_dict.items():        
        
        print(f"formatting {split_name} split -> {output_file}...")
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for record in dataset:
                formatted_record = map_goud(record)
                
                f.write(json.dumps(formatted_record, ensure_ascii=False) + '\n')
                
    print("formatting complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    args = parser.parse_args()
    
    format_data(args.config)