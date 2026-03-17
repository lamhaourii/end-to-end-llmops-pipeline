import torch
import json
import csv
import yaml
import mlflow
import logging
import argparse
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
import os
from huggingface_hub import login
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
class DarijaBERTEvaluator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_name = "SI2M-Lab/DarijaBERT"
        logger.info(f"loading DarijaBERT from {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
                    self.model_name).to(self.device)
        self.model.eval()
        
        logger.info("DarijaBERT loaded")

    def get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256, # summaries deja sghaar
            padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(
                        token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        return (sum_embeddings / sum_mask).squeeze(0)
    def semantic_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        cosine_sim = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        )
        return cosine_sim.item()

    def batch_similarity(self, predictions: list, references: list) -> list:
        scores = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            score = self.semantic_similarity(pred, ref)
            scores.append(score)
            if (i + 1) % 10 == 0:
                logger.info(f"Evaluated {i+1}/{len(predictions)} samples | avg score so far: {np.mean(scores):.4f}")
        return scores
    
def load_finetuned_model(config: dict):
    load_dotenv()
    hf=os.getenv("hf_token")  
    login(token=hf)

    d = config["data"]
    m= config["model"]
    h= config["huggingface"]
    
    test_dataset = load_dataset(
        "json",
        data_files=d["test_path"],
        split="train"
    )
    
    test_dataset = test_dataset.select(range(min(200, len(test_dataset))))


    base_model_id = m["base_model"]
    adapter_id = h["repo_id"]

    tokenizer = AutoTokenizer.from_pretrained(adapter_id)
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
    print("loading base model")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    print("base model loaded")

    model = PeftModel.from_pretrained(base_model, adapter_id) #comment to use base model 
    return model, tokenizer

def generate_summary(model, tokenizer, device, instruction: str, text: str) -> str:
    
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": text},
    ]
     
    with torch.no_grad():
        input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True
        ).to(device)
        prompt_length = input_ids["input_ids"].shape[-1]
    
        if prompt_length > 1024:
            print(f"skipping sample - ({prompt_length} tokens)")
            return False
        outputs = model.generate(
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            max_new_tokens=768,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return response
    
def evaluate(config_path: str, n_samples: int = 200):
    config = load_config(config_path)
    d = config["data"]
    test_dataset = load_dataset("json", data_files=d["test_path"], split="train")
    test_dataset = test_dataset.select(range(min(n_samples, len(test_dataset))))
    logger.info(f"evaluating on {len(test_dataset)} samples")
    model, tokenizer = load_finetuned_model(config)
    logger.info("summarizing")
    predictions = []
    references  = []
    inputs_text  = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i, sample in enumerate(test_dataset):
        pred = generate_summary(model, tokenizer,device,
            instruction=sample[d["instruction_field"]],
            text=sample[d["input_field"]])
        if not pred:
            continue
        predictions.append(pred)
        references.append(sample[d["output_field"]])
        inputs_text.append(sample[d["input_field"]])
        if i < 3:
            logger.info(f"\nsample {i+1}")
            logger.info(f"INPUT:     {sample[d['input_field']][:150]}")
            logger.info(f"REFERENCE: {sample[d['output_field']]}")
            logger.info(f"PREDICTED: {pred}")

    logger.info("freeing gpu") # lfaqr s3ib
    del model
    del tokenizer
    torch.cuda.empty_cache()  
    gc.collect() 
    evaluator = DarijaBERTEvaluator(device="cuda")
    logger.info("semantic similarity")
    scores = evaluator.batch_similarity(predictions, references)
    
    results = {
        "semantic_similarity_mean": float(np.mean(scores)),
        "semantic_similarity_std":  float(np.std(scores)),
        "semantic_similarity_min":  float(np.min(scores)),
        "semantic_similarity_max":  float(np.max(scores)),
        "n_samples": len(scores),
        "pct_above_0.8": float(np.mean([s > 0.8 for s in scores])),
        "pct_above_0.7": float(np.mean([s > 0.7 for s in scores])),
        "pct_above_0.6": float(np.mean([s > 0.6 for s in scores])),
        "pct_below_0.5": float(np.mean([s < 0.5 for s in scores])),
    }
    logger.info("\nevaluation results")
    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    output_path = Path("outputs/base_semantic_eval.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "reference", "predicted", "similarity_score"])
        writer.writeheader()
        for inp, ref, pred, score in zip(inputs_text, references, predictions, scores):
            writer.writerow({
                "input":            inp,
                "reference":        ref,
                "predicted":        pred,
                "similarity_score": round(score, 4)
            })
    logger.info(f"results saved to {output_path}")
    mlflow.set_tracking_uri(config.get("mlflow", {}).get("tracking_uri", "http://localhost:5000"))
    with mlflow.start_run(run_name="phase1-semantic-eval"):
        mlflow.log_metrics({
            "eval/darija_bert_similarity_mean": results["semantic_similarity_mean"],
            "eval/darija_bert_similarity_std":  results["semantic_similarity_std"],
            "eval/pct_above_0.8":               results["pct_above_0.8"],
            "eval/pct_above_0.7":               results["pct_above_0.7"],
            "eval/pct_below_0.5":               results["pct_below_0.5"],
        })
        mlflow.log_artifact(str(output_path))
        mlflow.log_param("evaluator", "SIEML/DarijaBERT")
        mlflow.log_param("n_samples", len(scores))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    type=str, default="configs/train.yaml")
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()
    evaluate(args.config, args.n_samples)
