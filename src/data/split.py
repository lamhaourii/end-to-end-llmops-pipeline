import json
import yaml
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_split(config_path: str):
    config = load_config(config_path)
    
    seed = config['split']['seed']
    train_ratio = config['split']['train_ratio']
    val_ratio = config['split']['val_ratio']
    test_ratio = config['split']['test_ratio']
    processed_dir = Path(config['paths']['processed_dir'])
    
    input_file = processed_dir / config['paths']['formatted_data']
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"total records: {len(data)}")

    train_data, temp_data = train_test_split(
        data, 
        train_size=train_ratio, 
        random_state=seed, 
        shuffle=True
    )
    relative_test_size = test_ratio / (val_ratio + test_ratio)

    val_data, test_data = train_test_split(
        temp_data, 
        test_size=relative_test_size, 
        random_state=seed, 
        shuffle=True
    )

    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    for name, subset in splits.items():
        output_path = processed_dir / f"{name}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in subset:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"created {name}.jsonl with {len(subset)} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    args = parser.parse_args()
    run_split(args.config)