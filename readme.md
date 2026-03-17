[![CI](https://github.com/lamhaourii/darija-llmops/actions/workflows/ci.yml/badge.svg)](https://github.com/lamhaourii/darija-llmops/actions/workflows/ci.yml)


> End-to-end MLOps pipeline for Moroccan Darija summarization.  
> Finetuned LLaMA-3.2-1B with QLoRA · vLLM serving · FastAPI · Gradio

### Quickstart
```bash
git clone https://github.com/lamhaourii/darija-llmops
cd darija-llmops

cp .env.example .env
# edit .env

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='igitsml/darija-llama-3.2-1b-summ',
    local_dir='outputs/phase1/merged_model'
)
"
docker compose up --build -d
```
### Services
| Service | URL | Description |
|---------|-----|-------------|
| Gradio Demo | http://localhost:7860 | Interactive demo |
| FastAPI | http://localhost:8080/docs | API documentation |
| MLflow | http://localhost:5000 | Experiment tracking |
---
##  Results

| Metric | Value |
|--------|-------|
| BERTScore | 0.68 |
| TTFT (median) | 80ms |
| Throughput | 46 tokens/s |
---
## Training Pipeline:
Raw Data → DVC → Preprocess → QLoRA Finetune → Merge → Push to HF Hub
---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Base Model | LLaMA-3.2-1B-Instruct |
| Finetuning | QLoRA (r=16, α=32) via Unsloth |
| Data Versioning | DVC + Google Drive |
| Experiment Tracking | MLflow (self-hosted) |
| Inference Engine | vLLM + PagedAttention |
| API | FastAPI + SSE streaming |
| Demo | Gradio |
| CI/CD | GitHub Actions |
| Containerization | Docker Compose |
---
uv pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

uv pip install "unsloth[cu121-torch251]"

uv pip install -r requirements.txt

uv pip uninstall torchao



PYTHONPATH=. python3 src/serving/merge_adapter.py --config configs/train.yaml
docker compose up --build -d


