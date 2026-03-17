import re
from pathlib import Path

vllm_tokenizer = list(Path(".venv").rglob("vllm/transformers_utils/tokenizer.py"))[0]

content = vllm_tokenizer.read_text()

patched = content.replace(
    "tokenizer.all_special_tokens_extended",
    "getattr(tokenizer, 'all_special_tokens_extended', tokenizer.all_special_tokens)"
)

vllm_tokenizer.write_text(patched)
