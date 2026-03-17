import os
import json
import time
import csv
import mlflow
import yaml
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def load_config(path:str)->dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")
    client = genai.Client(api_key="YOUR_KEY")
    return client

def build_judge_prompt(
    input_text: str,
    reference: str,
    prediction_a: str,   # base model output
    prediction_b: str    # your finetuned model output
) -> str:
    return f"""
    أنت خبير في تقييم جودة تلخيص النصوص بالدارجة المغربية.
    مهمتك هي المقارنة بين ملخصين وتحديد أيهما أفضل.

    النص الأصلي:
    {input_text}

    الملخص المرجعي (الإجابة الصحيحة):
    {reference}

    الملخص A:
    {prediction_a}

    الملخص B:
    {prediction_b}

    قارن بين الملخصين على ثلاثة محاور:

    1. الدقة (Faithfulness): أيهما يعكس النص الأصلي بشكل أدق بدون هلوسة؟
    2. جودة الدارجة (Darija Quality): أيهما يستخدم دارجة مغربية أكثر طبيعية؟
    3. الإيجاز (Conciseness): أيهما أكثر إيجازاً وتركيزاً؟

    قواعد مهمة:
    - يجب أن تختار winner واضح في كل محور، لا يوجد تعادل
    - إذا كان الفرق طفيفاً، اختر الأفضل ولو بفارق بسيط
    - كن صارماً وموضوعياً

    أجب فقط بـ JSON بهذا الشكل بدون أي نص إضافي:
    {{
    "faithfulness": {{
        "winner": "A" or "B",
        "confidence": "low" or "medium" or "high",
        "reason": "<one sentence>"
    }},
    "darija_quality": {{
        "winner": "A" or "B",
        "confidence": "low" or "medium" or "high",
        "reason": "<one sentence>"
    }},
    "conciseness": {{
        "winner": "A" or "B",
        "confidence": "low" or "medium" or "high",
        "reason": "<one sentence>"
    }},
    "overall_winner": "A" or "B",
    "overall_confidence": "low" or "medium" or "high",
    "overall_reason": "<one sentence>"
    }}
    """

def judge_sample(
    client,
    input_text: str,
    reference: str,
    prediction: str,
    retries: int = 3
) -> dict:
    prompt = build_judge_prompt(input_text, reference, prediction)
    
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="medium")
                ),
                )
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]        
            scores = json.loads(text.strip())
            required = {"faithfulness", "darija_quality", "conciseness", "reasoning"}
            if not required.issubset(scores.keys()):
                raise ValueError(f"Missing keys in response: {scores}")
            return scores
        except json.JSONDecodeError as e:
            print(f"JSON parse error on attempt {attempt+1}: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(5)
    return {
        "faithfulness": None,
        "darija_quality": None,
        "conciseness": None,
        "reasoning": "evaluation failed"
    }


def run_llm_judge(
    predictions_csv: str, config_path: str, n_samples: int = 100):

    config = load_config(config_path)
    gemini = setup_gemini()
    
    samples = []
    with open(predictions_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    samples = samples[:n_samples]
    print(f"Evaluating {len(samples)} samples with Gemini as judge...")
    results = []
    faithfulness_scores = []
    darija_scores       = []
    conciseness_scores  = []
    for i, sample in enumerate(samples):
        print(f"Judging sample {i+1}/{len(samples)}...", end="\r")
        scores = judge_sample(
            gemini,
            input_text=sample["input"],
            reference=sample["reference"],
            prediction=sample["predicted"]
        )
        if i<10:
            print(scores)
        results.append({
            "input":         sample["input"][:100],  # truncate for readability
            "reference":     sample["reference"],
            "predicted":     sample["predicted"],
            **scores
        })
        if scores["faithfulness"] is not None:
            faithfulness_scores.append(scores["faithfulness"])
            darija_scores.append(scores["darija_quality"])
            conciseness_scores.append(scores["conciseness"])
        time.sleep(4)
    avg_scores = {
        "llm_judge/faithfulness":   sum(faithfulness_scores) / len(faithfulness_scores),
        "llm_judge/darija_quality": sum(darija_scores) / len(darija_scores),
        "llm_judge/conciseness":    sum(conciseness_scores) / len(conciseness_scores),
        "llm_judge/overall":        sum(
            faithfulness_scores + darija_scores + conciseness_scores
        ) / (len(faithfulness_scores) * 3),
    }
    print("\n=== LLM Judge Results ===")
    for k, v in avg_scores.items():
        print(f"{k}: {v:.2f} / 5.0")
    output_path = Path("outputs/llm_judge_results.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    with mlflow.start_run(run_name="llm-judge-evaluation"):
        mlflow.log_metrics(avg_scores)
        mlflow.log_artifact(str(output_path))
    
    print(f"\nDetailed results saved to {output_path}")
    return avg_scores

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, default="outputs/phase1_eval.csv")
    parser.add_argument("--config",      type=str, default="configs/train.yaml")
    parser.add_argument("--n_samples",   type=int, default=100)
    args = parser.parse_args()
    run_llm_judge(
        predictions_csv=args.predictions,
        config_path=args.config,
        n_samples=args.n_samples
    )
    
