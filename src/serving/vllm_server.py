import os
import yaml
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_vllm_args(config: dict) -> list:
    v = config["vllm"]
    merged_path = str(Path(config["output"]["dir"]) / "merged_model")
    
    args = [
        "--model",                   merged_path,
        "--host",                    v["host"],
        "--port",                    str(v["port"]),
        "--dtype",                   v["dtype"],
        "--max-model-len",           str(v["max_model_len"]),
        "--gpu-memory-utilization",  str(v["gpu_memory_utilization"]),
        "--max-num-seqs",            str(v["max_num_seqs"]),
        "--served-model-name",       v["served_model_name"],
        "--trust-remote-code",
    ]
    if v.get("max_num_batched_tokens"):
        args += ["--max-num-batched-tokens", str(v["max_num_batched_tokens"])]
    
    return args

def start_server(config_path: str):
    config = load_config(config_path)
    v = config["vllm"]
    merged_path = str(Path(config["output"]["dir"]) / "merged_model")
    
    if not Path(merged_path).exists():
        raise FileNotFoundError(
            f"Merged model not found at {merged_path}. "
            f"Run merge_adapter.py first."
        )
    
    logger.info(f"Starting vLLM server for model: {merged_path}")
    logger.info(f"Server will be available at: http://{v['host']}:{v['port']}")
    logger.info(f"OpenAI-compatible endpoint: http://{v['host']}:{v['port']}/v1/completions")
    logger.info(f"GPU memory utilization: {v['gpu_memory_utilization']}")
    logger.info(f"Max concurrent sequences: {v['max_num_seqs']}")
    
    import subprocess
    import sys
    
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] + build_vllm_args(config)
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/serving.yaml")
    args = parser.parse_args()
    start_server(args.config)