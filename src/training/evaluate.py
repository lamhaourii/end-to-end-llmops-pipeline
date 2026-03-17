
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
from datasets import load_dataset
from evaluate import load
import mlflow
import yaml
from huggingface_hub import login
from dotenv import load_dotenv
import os



def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

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
        
    #prompt_length = input_ids.input_ids.shape[-1]
    response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return response
    
    
def rouge_eval(config_path: str):
    load_dotenv()
    hf=os.getenv("hf_token")  
    login(token=hf)

    config = load_config(config_path)
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

    model = PeftModel.from_pretrained(base_model, adapter_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    rouge = load("rouge")
    
    predictions = []
    references  = []
    
    print(f"Evaluating on {len(test_dataset)} samples...")
    
    for i, sample in enumerate(test_dataset):
        print(i)
        pred = generate_summary(
            model, tokenizer,device,
            instruction=sample[d["instruction_field"]],
            text=sample[d["input_field"]]
        )
        if not pred:
            continue

        predictions.append(pred)
        references.append(sample[d["output_field"]])
        
        if i < 3:
            print(f"\n--- Sample {i+1} ---")
            print(f"INPUT:     {sample[d['input_field']][:200]}...")
            print(f"REFERENCE: {sample[d['output_field']]}")
            print(f"PREDICTED: {pred}")
    
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=False   #arabic
    )
    
    print("\n=== ROUGE Scores ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    
    with mlflow.start_run(run_name="phase1-evaluation"):
        mlflow.log_metrics({
            "eval/rouge1":    results["rouge1"],
            "eval/rouge2":    results["rouge2"],
            "eval/rougeL":    results["rougeL"],
            "eval/rougeLsum": results["rougeLsum"],
        })
        
        output_path = Path("outputs/phase1_eval.csv")
        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["input", "reference", "predicted"])
            writer.writeheader()
            for inp, ref, pred in zip(
                [s[d["input_field"]] for s in test_dataset],
                references,
                predictions
            ):
                writer.writerow({"input": inp, "reference": ref, "predicted": pred})
        
        mlflow.log_artifact(str(output_path))
        print(f"\nPredictions saved to {output_path}")
    
    return results

def llm_judge(input_text: str, reference: str, prediction: str) -> dict:
    prompt = f"""
    You are evaluating a Darija summarization model.
    
    Original text: {input_text}
    Reference summary: {reference}  
    Model summary: {prediction}
    
    Score the model summary on:
    1. Faithfulness (1-5): Does it accurately reflect the original?
    2. Darija quality (1-5): Is it natural Darija, not MSA?
    3. Conciseness (1-5): Is it appropriately short?
    
    Respond in JSON only.
    """
    #LLM 
    return scores

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()
    rouge_eval(args.config)
