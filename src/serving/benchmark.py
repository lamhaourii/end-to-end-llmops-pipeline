import time
import json
import statistics
import asyncio
import aiohttp
import yaml
import logging
import mlflow
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

TEST_TEXTS = [
    "أعلنت وزارة التربية الوطنية عن إطلاق برنامج جديد لتحسين جودة التعليم في المدارس العمومية المغربية، يشمل توفير أجهزة لوحية لجميع تلاميذ السنة الأولى إعدادي.",
    "سجل الاقتصاد المغربي نمواً بنسبة 3.2 بالمئة خلال الربع الثالث من السنة الجارية، مدفوعاً بالأداء الجيد لقطاعي الفلاحة والسياحة.",
    "حقق المنتخب المغربي لكرة القدم فوزاً مهماً على نظيره الغاني بهدفين مقابل هدف ضمن مباريات التصفيات الإفريقية المؤهلة لكأس العالم.",
    "أفادت مصادر طبية بأن وزارة الصحة أطلقت حملة تلقيح وطنية شاملة تستهدف الفئات الأكثر هشاشة في جميع أنحاء المملكة.",
    "أعلنت المديرية العامة للأرصاد الجوية عن موجة برد قارس ستضرب عدة مناطق من المغرب خلال الأيام القادمة مع تساقطات ثلجية على المرتفعات.",
]

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_payload(text: str) -> dict:
    return {
        "model": "darija-llm",
        "messages": [
            {"role": "system", "content": "لخص هذا النص بالدارجة المغربية:"},
            {"role": "user",   "content": text}
        ],
        "max_tokens":        50,
        "temperature":       0.1,
        "repetition_penalty": 1.3,
        "stream":            True,  
    }

async def measure_ttft(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict
) -> dict:
    start_time = time.perf_counter()
    first_token_time = None
    full_response = ""
    token_count = 0

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                logger.error(f"Request failed with status {response.status}")
                return {"error": f"HTTP {response.status}"}

            async for line in response.content:
                line = line.decode("utf-8").strip()

                if not line or line == "data: [DONE]":
                    continue

                if line.startswith("data: "):
                    data = line[6:] 
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")

                        if delta and first_token_time is None:
                            first_token_time = time.perf_counter() 

                        full_response += delta
                        if delta:
                            token_count += 1

                    except json.JSONDecodeError:
                        continue

        end_time = time.perf_counter()

        ttft_ms         = (first_token_time - start_time) * 1000 if first_token_time else None
        total_time_ms   = (end_time - start_time) * 1000
        tokens_per_sec  = token_count / (end_time - start_time) if token_count > 0 else 0

        return {
            "ttft_ms":        ttft_ms,
            "total_time_ms":  total_time_ms,
            "token_count":    token_count,
            "tokens_per_sec": tokens_per_sec,
            "response":       full_response[:100],  
            "error":          None
        }

    except Exception as e:
        return {"error": str(e), "ttft_ms": None}

async def run_sequential_benchmark(
    base_url: str,
    n_requests: int = 100
) -> list:
    url = f"{base_url}/v1/chat/completions"
    results = []

    connector = aiohttp.TCPConnector(limit=1)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(n_requests):
            text = TEST_TEXTS[i % len(TEST_TEXTS)]
            payload = build_payload(text)

            result = await measure_ttft(session, url, payload)
            results.append(result)

            if result.get("error"):
                logger.warning(f"Request {i+1} failed: {result['error']}")
            else:
                logger.info(
                    f"Request {i+1:3d}/{n_requests} | "
                    f"TTFT: {result['ttft_ms']:.1f}ms | "
                    f"Total: {result['total_time_ms']:.1f}ms | "
                    f"Tokens/s: {result['tokens_per_sec']:.1f}"
                )

    return results

async def run_concurrent_benchmark(
    base_url: str,
    n_requests: int = 20,
    concurrency: int = 4
) -> list:
    url = f"{base_url}/v1/chat/completions"
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(session, i):
        async with semaphore:
            text = TEST_TEXTS[i % len(TEST_TEXTS)]
            return await measure_ttft(session, url, build_payload(text))

    async with aiohttp.ClientSession() as session:
        tasks = [bounded_request(session, i) for i in range(n_requests)]
        results = await asyncio.gather(*tasks)

    return list(results)

def compute_stats(results: list, label: str) -> dict:
    valid = [r for r in results if r.get("ttft_ms") is not None and not r.get("error")]

    if not valid:
        logger.error("No valid results to compute stats")
        return {}

    ttfts      = [r["ttft_ms"] for r in valid]
    totals     = [r["total_time_ms"] for r in valid]
    throughput = [r["tokens_per_sec"] for r in valid]

    stats = {
        f"{label}/n_requests":          len(results),
        f"{label}/n_successful":        len(valid),
        f"{label}/n_failed":            len(results) - len(valid),
        f"{label}/ttft_mean_ms":        statistics.mean(ttfts),
        f"{label}/ttft_median_ms":      statistics.median(ttfts),
        f"{label}/ttft_p95_ms":         sorted(ttfts)[int(len(ttfts) * 0.95)],
        f"{label}/ttft_p99_ms":         sorted(ttfts)[int(len(ttfts) * 0.99)],
        f"{label}/ttft_min_ms":         min(ttfts),
        f"{label}/ttft_max_ms":         max(ttfts),
        f"{label}/total_time_mean_ms":  statistics.mean(totals),
        f"{label}/throughput_mean_tps": statistics.mean(throughput),
        f"{label}/throughput_max_tps":  max(throughput),
    }

    return stats

def print_stats(stats: dict, target_ttft_ms: float = 100.0):
    print("\n" + "=" * 60)
    print(f"  BENCHMARK RESULTS")
    print("=" * 60)

    for k, v in stats.items():
        label = k.split("/")[1]
        if isinstance(v, float):
            print(f"  {label:<30} {v:.2f}")
        else:
            print(f"  {label:<30} {v}")

    ttft_mean = stats.get(list(stats.keys())[3])  
    print("\n" + "=" * 60)
    if ttft_mean and ttft_mean < target_ttft_ms:
        print(f"  TARGET MET: TTFT {ttft_mean:.1f}ms < {target_ttft_ms}ms")
    else:
        print(f"  TARGET MISSED: TTFT {ttft_mean:.1f}ms > {target_ttft_ms}ms")
        print(f"  See below for diagnosis")
    print("=" * 60)

def diagnose_ttft(stats: dict, target_ms: float = 100.0) -> str:

    ttft_mean = stats.get("sequential/ttft_mean_ms", 999)

    if ttft_mean < target_ms:
        return "Target met. No diagnosis needed."

    diagnosis = []
    diagnosis.append(f"\nTTFT {ttft_mean:.1f}ms exceeds {target_ms}ms target.")
    diagnosis.append("\nDiagnosis for RTX 2060 (6GB VRAM):\n")

    if ttft_mean > 500:
        diagnosis.append(
            "SEVERE: TTFT > 500ms suggests GPU memory paging.\n"
            "The KV cache is likely spilling to CPU RAM.\n"
            "Fix: reduce gpu_memory_utilization in serving.yaml to 0.75\n"
            "and reduce max_num_seqs to 2."
        )
    elif ttft_mean > 200:
        diagnosis.append(
            "HIGH: TTFT 200-500ms suggests memory fragmentation.\n"
            "Fix: add PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
            "and restart the vLLM server."
        )
    elif ttft_mean > 100:
        diagnosis.append(
            "MODERATE: TTFT 100-200ms on RTX 2060 is expected.\n"
            "The RTX 2060 has 6GB VRAM and 336 GB/s memory bandwidth.\n"
            "For comparison, an A100 has 2TB/s — 6x faster.\n"
            "The <100ms target was designed for datacenter GPUs.\n"
            "On consumer hardware, 100-200ms TTFT is the realistic floor.\n"
            "This is a hardware limitation, not a software issue."
        )

    return "\n".join(diagnosis)

async def main(config_path: str, n_requests: int = 100):
    config   = load_config(config_path)
    v        = config["vllm"]
    base_url = f"http://{v['host']}:{v['port']}"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/health") as r:
                if r.status != 200:
                    raise ConnectionError("Server not healthy")
            logger.info(f"Server is up at {base_url}")
        except Exception:
            logger.error(
                f"Cannot connect to vLLM server at {base_url}\n"
                f"Start it with: PYTHONPATH=. python3 src/serving/vllm_server.py"
            )
            return

    logger.info(f"Running sequential benchmark ({n_requests} requests)...")
    seq_results = await run_sequential_benchmark(base_url, n_requests)
    seq_stats   = compute_stats(seq_results, "sequential")
    print_stats(seq_stats)

    logger.info("Running concurrent benchmark (20 requests, concurrency=4)...")
    con_results = await run_concurrent_benchmark(base_url, n_requests=20, concurrency=4)
    con_stats   = compute_stats(con_results, "concurrent")

    diagnosis = diagnose_ttft(seq_stats)
    print(diagnosis)

    all_stats = {**seq_stats, **con_stats}
    output_path = Path("outputs/latency_benchmark.txt")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Benchmark Run: {datetime.now().isoformat()}\n")
        f.write(f"Hardware: RTX 2060 6GB\n")
        f.write(f"Model: {config['vllm']['served_model_name']}\n")
        f.write(f"Requests: {n_requests}\n\n")

        f.write("=== SEQUENTIAL BENCHMARK ===\n")
        for k, v in seq_stats.items():
            f.write(f"{k}: {v:.2f}\n" if isinstance(v, float) else f"{k}: {v}\n")

        f.write("\n=== CONCURRENT BENCHMARK ===\n")
        for k, v in con_stats.items():
            f.write(f"{k}: {v:.2f}\n" if isinstance(v, float) else f"{k}: {v}\n")

        f.write(f"\n=== DIAGNOSIS ===\n{diagnosis}\n")

    logger.info(f"Results saved to {output_path}")

    with mlflow.start_run(run_name="latency-benchmark"):
        mlflow.log_metrics(all_stats)
        mlflow.log_artifact(str(output_path))
        mlflow.log_param("hardware", "RTX 2060 6GB")
        mlflow.log_param("n_requests", n_requests)

    logger.info("Benchmark complete")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="configs/serving.yaml")
    parser.add_argument("--n_requests", type=int, default=100)
    args = parser.parse_args()

    asyncio.run(main(args.config, args.n_requests))