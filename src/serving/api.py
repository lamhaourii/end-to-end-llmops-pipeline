import os
import time
import yaml
import logging
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import json

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

def load_config(path: str = "configs/serving.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
config = load_config()
v = config["vllm"]
VLLM_BASE_URL = f"http://{v['host']}:{v['port']}"
MODEL_NAME    = v["served_model_name"]
VERSION       = config["api"].get("version", "1.0.0")

class GenerateRequest(BaseModel):
    prompt:             str                  = Field(...,  min_length=1, max_length=4000)
    max_tokens:         Optional[int]        = Field(150,  ge=1, le=512)
    temperature:        Optional[float]      = Field(0.5,  ge=0.0, le=2.0)
    repetition_penalty: Optional[float]      = Field(1.3,  ge=1.0, le=2.0)
    system_prompt:      Optional[str]        = Field(
        "لخص هذا النص بالدارجة المغربية:",
        max_length=500
    )
    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("prompt cannot be empty or whitespace")
        return v.strip()

    @field_validator("temperature")
    @classmethod
    def temperature_range(cls, v):
        if v > 1.0:
            logger.warning(f"High temperature {v} may cause incoherent output")
        return v
    
class GenerateResponse(BaseModel):
    summary:          str
    model:            str
    prompt_tokens:    int
    completion_tokens: int
    total_tokens:     int
    latency_ms:       float

class HealthResponse(BaseModel):
    status:       str
    model:        str
    version:      str
    vllm_url:     str
    vllm_healthy: bool

class ErrorResponse(BaseModel):
    error:   str
    detail:  str

class StreamRequest(BaseModel):
    prompt:             str           = Field(..., min_length=1, max_length=4000)
    max_tokens:         Optional[int] = Field(150, ge=1, le=512)
    temperature:        Optional[float] = Field(0.5, ge=0.0, le=2.0)
    repetition_penalty: Optional[float] = Field(1.3, ge=1.0, le=2.0)
    system_prompt:      Optional[str] = Field(
        "لخص هذا النص بالدارجة المغربية:",
        max_length=500
    )

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError("prompt cannot be empty")
        return v.strip()

class ArabicJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,   
            allow_nan=False,
            indent=None,
            separators=(",", ":")
        ).encode("utf-8")
    
app = FastAPI(
    title="DarijaLLM API",
    description="Moroccan Darija summarization API powered by a finetuned LLaMA model",
    version=VERSION,
    default_response_class=ArabicJSONResponse, 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #ay website f dnia iqdr isift request
    allow_methods=["*"],  #all the http methods allowed
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    app.state.client = httpx.AsyncClient(
        base_url=VLLM_BASE_URL,
        timeout=httpx.Timeout(60.0)   
    )
    logger.info(f"API started | vLLM: {VLLM_BASE_URL} | model: {MODEL_NAME}")

@app.on_event("shutdown")
async def shutdown():
    await app.state.client.aclose()
    logger.info("API shutdown — HTTP client closed")

@app.get("/health", response_model=HealthResponse)
async def health():
    vllm_healthy = False
    try:
        response = await app.state.client.get("/health", timeout=5.0)
        vllm_healthy = response.status_code == 200
    except Exception:
        vllm_healthy = False

    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        version=VERSION,
        vllm_url=VLLM_BASE_URL,
        vllm_healthy=vllm_healthy,
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    start_time = time.perf_counter()
    payload = {
        "model":              MODEL_NAME,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user",   "content": request.prompt},
        ],
        "max_tokens":         request.max_tokens,
        "temperature":        request.temperature,
        "repetition_penalty": request.repetition_penalty,
        "stop":               ["<|eot_id|>", "<|end_of_text|>"],
    }
    try:
        logger.info(f"Generate request | prompt_len={len(request.prompt)}")
        response = await app.state.client.post(
            "/v1/chat/completions",
            json=payload
        )
        if response.status_code != 200:
            logger.error(f"vLLM error: {response.status_code} {response.text}")
            raise HTTPException(
                status_code=502,
                detail=f"vLLM backend error: {response.status_code}"
            )
        data          = response.json()
        choice        = data["choices"][0]
        summary       = choice["message"]["content"].strip()
        usage         = data["usage"]
        latency_ms    = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Generate complete | "
            f"tokens={usage['completion_tokens']} | "
            f"latency={latency_ms:.1f}ms"
        )
        return GenerateResponse(
            summary=summary,
            model=MODEL_NAME,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except httpx.TimeoutException:
        logger.error("vLLM request timed out")
        raise HTTPException(
            status_code=504,
            detail="Generation timed out. Try reducing max_tokens."
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    
@app.post("/generate/stream")
async def generate_stream(request: StreamRequest):
    """
    Stream generated tokens via Server-Sent Events.
    Each token is sent as it is generated — not buffered.

    SSE format:
        data: {"token": "وزاره", "index": 0}\n\n
        data: {"token": " التربيه", "index": 1}\n\n
        data: [DONE]\n\n
    """
    payload = {
        "model":   MODEL_NAME,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user",   "content": request.prompt},
        ],
        "max_tokens":         request.max_tokens,
        "temperature":        request.temperature,
        "repetition_penalty": request.repetition_penalty,
        "stop":               ["<|eot_id|>", "<|end_of_text|>"],
        "stream":             True,  
    }

    async def event_generator():
        token_index = 0
        start_time  = time.perf_counter()

        try:
            async with app.state.client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                timeout=httpx.Timeout(60.0)
            ) as response:

                if response.status_code != 200:
                    error_data = json.dumps(
                        {"error": f"vLLM error: {response.status_code}"},
                        ensure_ascii=False
                    )
                    yield f"data: {error_data}\n\n"
                    return

                async for line in response.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        raw = line[6:]  
                        try:
                            chunk = json.loads(raw)
                            delta = chunk["choices"][0]["delta"].get("content", "")

                            if delta:
                                event = json.dumps(
                                    {
                                        "token":      delta,
                                        "index":      token_index,
                                        "elapsed_ms": round((time.perf_counter() - start_time) * 1000, 2)
                                    },
                                    ensure_ascii=False
                                )
                                yield f"data: {event}\n\n"
                                token_index += 1

                        except json.JSONDecodeError:
                            continue

            done_event = json.dumps(
                {
                    "done":          True,
                    "total_tokens":  token_index,
                    "total_ms":      round((time.perf_counter() - start_time) * 1000, 2)
                },
                ensure_ascii=False
            )
            yield f"data: {done_event}\n\n"
            logger.info(f"Stream complete | tokens={token_index}")

        except httpx.TimeoutException:
            error = json.dumps({"error": "generation timed out"}, ensure_ascii=False)
            yield f"data: {error}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            error = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",     # disables nginx buffering if behind proxy
            "Access-Control-Allow-Origin": "*",
        }
    )
    
if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/serving.yaml")
    parser.add_argument("--host",   type=str, default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "src.serving.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
