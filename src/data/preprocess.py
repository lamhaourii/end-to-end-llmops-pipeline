import re
import unicodedata
from typing import List
import json
from datasets import load_from_disk
from pathlib import Path
import argparse 
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def remove_html_and_urls(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    return text

def normalize_arabic(text: str) -> str:
    text = re.sub(r'\u0640', '', text)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[إأآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[ﻻﻷﻹﻵ]', 'لا', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text) # e.g. لاااا → لا
    return text

def filter_characters(text: str) -> str:
    pattern = r'[^0-9a-zA-Z\u0600-\u06FF\u00C0-\u00FF\s]'
    text = re.sub(pattern, '', text)
    return text

def clean_whitespace(text: str) -> str:
    return " ".join(text.split())

def is_valid_length(text: str, min_length: int = 10) -> bool:
    return len(text.strip()) >= min_length

def pipeline_step(text: str) -> str:
    text = remove_html_and_urls(text)
    text = normalize_arabic(text)
    text = filter_characters(text)
    text = clean_whitespace(text)
    return text


def map_cleaning_function(batch):
    batch["article"] = [pipeline_step(t) for t in batch["article"]]
    batch["headline"] = [pipeline_step(t) for t in batch["headline"]]
    return batch

def run_preprocessing(config_path: str):
    config = load_config(config_path)
    RAW_DATA_DIR = Path(config['paths']['local_dir'])
    ds = load_from_disk(RAW_DATA_DIR)

    print("cleaning text")
    cleaned_ds = ds.map(map_cleaning_function, batched=True, num_proc=4)


    seen_titles = set()

    def filter_and_dedup(example):
        if example['headline'] in seen_titles:
            return False
        if not is_valid_length(example['headline'], min_length=15):
            return False
        seen_titles.add(example['headline'])
        return True

    final_ds = cleaned_ds.filter(filter_and_dedup)
    
    processed_path =Path(config['paths']['processed_dir'])
    processed_path.mkdir(exist_ok=True)
    output_path= Path(config['paths']['cleaned_dataset'])
    final_ds.save_to_disk(output_path)
    print(f"preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    args = parser.parse_args()
    run_preprocessing(args.config)