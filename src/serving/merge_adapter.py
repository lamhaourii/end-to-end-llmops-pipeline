
import os
import yaml
import torch
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
import os
from peft import PeftModel
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def merge_and_save(config_path: str):
    load_dotenv()
    config = load_config(config_path)
    
    hf_token = os.getenv("hf_token")
    login(token=hf_token)
    
    m = config["model"]
    h = config["huggingface"]
    
    base_model_id = m["base_model"]
    adapter_id    = h["repo_id"]
    merged_path   = str(Path(config["output"]["dir"]) / "merged_model")
    
    logger.info(f"Loading tokenizer from: {adapter_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_id,
        token=hf_token
    )
    
    logger.info(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        trust_remote_code=m["trust_remote_code"],
        torch_dtype=torch.float16,
        token=hf_token
    )
    
    logger.info(f"Loading adapter from HF Hub: {adapter_id}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_id,
        token=hf_token
    )
    
    logger.info("Merging adapter into base model weights...")
    model = model.merge_and_unload()   
    
    logger.info(f"Saving merged model to: {merged_path}")
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        merged_path,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    tokenizer.save_pretrained(merged_path)
    
    logger.info("Merge complete — ready for vLLM serving")
    return merged_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()
    merge_and_save(args.config)