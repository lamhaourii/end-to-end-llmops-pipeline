from unsloth import FastLanguageModel
import sys
import logging
import yaml
import torch
import mlflow
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer
from src.training.mlflow_utils import (
    setup_tracking,
    setup_experiments,
    start_run,
    log_config,
    log_metrics,
    log_artifact,
    register_model,
)
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers.trainer_utils import get_last_checkpoint

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/train.log")
    ]
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"config loaded from {path}")
    return config

def load_model_and_tokenizer(config: dict, hf_token:str):
    m = config["model"]
    t = config["training"]
    l = config["lora"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m["base_model"],
        max_seq_length=t["max_seq_length"],
        dtype=config["unsloth"]["dtype"],               
        load_in_4bit=True,       
        trust_remote_code=m["trust_remote_code"],
        token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    logger.info(f"model loaded via unsloth: {m['base_model']}")
    return model, tokenizer

def apply_lora(model, config: dict):
    l = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=l["r"],
        lora_alpha=l["alpha"],
        lora_dropout=l["dropout"],
        bias=l["bias"],
        target_modules=l["target_modules"],
        use_gradient_checkpointing="unsloth",  # unsloth's own optimized implementation
        random_state=config["data"]["seed"],
        use_rslora=config["unsloth"]["use_rslora"],        # rank-stabilized LoRA, keep False to match your config
    )
    model.print_trainable_parameters()
    logger.info("lora adapters applied via unsloth")
    return model

"""def load_data(config: dict, tokenizer):
    d = config["data"]
    train_dataset = load_dataset("json", data_files=d["train_path"], split="train")
    val_dataset = load_dataset("json", data_files=d["val_path"], split="train")
    train_dataset = train_dataset.map(
        lambda sample: {"text": format_prompt(sample, tokenizer, config)},
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda sample: {"text": format_prompt(sample, tokenizer, config)},
        remove_columns=val_dataset.column_names
    )
    
    logger.info(f"train samples: {len(train_dataset)} | val samples: {len(val_dataset)}")
    return train_dataset, val_dataset"""
def load_data(config: dict, tokenizer):
    d = config["data"]
    max_len = config["training"]["max_seq_length"]

    def format_sample(sample):
        text = format_prompt(sample, tokenizer, config)
        token_len = len(tokenizer(text, truncation=False)["input_ids"])
        if token_len > max_len:
            return {"text": None, "length": token_len}
        return {"text": text, "length": token_len}

    train_dataset = load_dataset("json", data_files=d["train_path"], split="train")
    val_dataset   = load_dataset("json", data_files=d["val_path"], split="train")

    train_dataset = train_dataset.map(lambda s: format_sample(s), remove_columns=train_dataset.column_names)
    val_dataset   = val_dataset.map(lambda s: format_sample(s),   remove_columns=val_dataset.column_names)

    before_train = len(train_dataset)
    train_dataset = train_dataset.filter(lambda x: x["text"] is not None)
    val_dataset   = val_dataset.filter(lambda x: x["text"] is not None)

    train_dataset = train_dataset.remove_columns(["length"])
    val_dataset   = val_dataset.remove_columns(["length"])

    logger.info(f"train samples: {len(train_dataset)} (filtered {before_train - len(train_dataset)} too long)")
    logger.info(f"val samples: {len(val_dataset)}")
    return train_dataset, val_dataset
    

def format_prompt(sample: dict, tokenizer, config: dict) -> str:
    d = config["data"]
    messages = [
        {"role": "system", "content": sample[d["instruction_field"]]},
        {"role": "user", "content": sample[d["input_field"]]},
        {"role": "assistant", "content": sample[d["output_field"]]}
    ]
    return tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )

def build_training_args(config: dict) -> TrainingArguments:
    t = config["training"]
    e = config["evaluation"]
    return TrainingArguments(
        output_dir=config["output"]["dir"],
        num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        fp16=t["fp16"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        optim=t["optim"],
        eval_strategy=e["strategy"],
        eval_steps=e["eval_steps"],
        save_steps=e["save_steps"],
        save_total_limit=e["save_total_limit"],
        torch_compile=False,
        load_best_model_at_end=e["load_best_model_at_end"],
        metric_for_best_model=e["metric_for_best_model"],
        logging_steps=config["output"]["logging_steps"],
        report_to=config["output"]["report_to"],  #ana ntklf b mlflow not hf   
    )

class MLflowCallback(TrainerCallback):
    def __init__(self, config: dict):
        self.log_steps = config["output"]["logging_steps"]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        metrics = {}
        if "loss" in logs:
            metrics["train/loss"] = logs["loss"]
        if "eval_loss" in logs:
            metrics["eval/loss"] = logs["eval_loss"]
        if "learning_rate" in logs:
            metrics["train/learning_rate"] = logs["learning_rate"]
        if torch.cuda.is_available():
            metrics["system/gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024 ** 2

        if metrics:
            log_metrics(metrics, step=state.global_step)

def main(config_path: str):
    Path("outputs").mkdir(parents=True, exist_ok=True)
    load_dotenv()
    config = load_config(config_path)
    hf_token = os.getenv("hf_token")
    login(token=hf_token)
    setup_tracking()
    setup_experiments()     
    with start_run(
        run_name=config["mlflow"]["run_name"]
    ):
        try:
            logger.info("starting training")
            log_config(config)
            logger.info("loading model and tokenizer")
            model, tokenizer = load_model_and_tokenizer(config,hf_token=hf_token)
            logger.info("applying lora")
            model = apply_lora(model, config)
            logger.info("loading datasets")
            train_dataset, val_dataset = load_data(config, tokenizer)
            logger.info("building trainer")
            training_args = build_training_args(config)
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                #formatting_func=lambda batch: [
                                #format_prompt(dict(zip(batch.keys(), t)), tokenizer, config)
                                #for t in zip(*batch.values())
                                #],
                dataset_text_field="text",
                max_seq_length=config["training"]["max_seq_length"],
                callbacks=[MLflowCallback(config)],
            )
            logger.info("lets start training")
            checkpoint = get_last_checkpoint(config["output"]["dir"])
            if checkpoint:
                logger.info(f"Resuming from checkpoint: {checkpoint}")
            else:
                logger.info("No checkpoint found, starting from scratch")

            trainer.train(resume_from_checkpoint=checkpoint)
            logger.info("training completed")
            logger.info("saving adapter")
            adapter_path = Path(config["output"]["dir"]) / "final_adapter"
            trainer.model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            if config.get("huggingface", {}).get("push_to_hub"):
                logger.info(f"pushing model hf: {config['huggingface']['repo_id']}")
                
                model.push_to_hub(
                    repo_id=config["huggingface"]["repo_id"],
                    token=hf_token,
                    private=config["huggingface"].get("private", True)
                )
                tokenizer.push_to_hub(
                    repo_id=config["huggingface"]["repo_id"],
                    token=hf_token
                )
                
                mlflow.log_param("hf_hub_url", f"https://huggingface.co/{config['huggingface']['repo_id']}")
                logger.info("model and tokenizer pushed")
            log_artifact(str(adapter_path))
            logger.info("registering model")
            register_model(
                run_id=mlflow.active_run().info.run_id,
                artifact_path="final_adapter",
                model_name=config["mlflow"]["model_registry"]
            )
        except Exception as e:
            logger.error(f"training failed: {e}", exc_info=True)
            mlflow.log_param("reason_of_failure", str(e))
            raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()
    main(args.config)