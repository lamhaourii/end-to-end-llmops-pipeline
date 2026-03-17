[![CI](https://github.com/yourname/darija-llmops/actions/workflows/ci.yml/badge.svg)](https://github.com/yourname/darija-llmops/actions/workflows/ci.yml)


uv pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

uv pip install "unsloth[cu121-torch251]"

uv pip install -r requirements.txt

uv pip uninstall torchao



PYTHONPATH=. python3 src/serving/merge_adapter.py --config configs/train.yaml
docker compose up --build
