import pytest
import httpx
import json

BASE_URL = "http://localhost:8080"


@pytest.fixture
def client():
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as c:
        yield c

@pytest.fixture
def sample_prompt():
    return "أعلنت وزارة التربية الوطنية عن إطلاق برنامج جديد لتحسين جودة التعليم في المدارس العمومية المغربية."


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200

def test_health_response_structure(client):
    response = client.get("/health")
    data = response.json()
    assert "status"       in data
    assert "model"        in data
    assert "version"      in data
    assert "vllm_healthy" in data

def test_health_status_is_ok(client):
    response = client.get("/health")
    assert response.json()["status"] == "ok"

def test_health_vllm_is_reachable(client):
    response = client.get("/health")
    assert response.json()["vllm_healthy"] is True

def test_generate_returns_200(client, sample_prompt):
    response = client.post("/generate", json={"prompt": sample_prompt})
    assert response.status_code == 200

def test_generate_returns_non_empty_summary(client, sample_prompt):
    response = client.post("/generate", json={"prompt": sample_prompt})
    data = response.json()
    assert "summary" in data
    assert len(data["summary"]) > 0

def test_generate_response_structure(client, sample_prompt):
    response = client.post("/generate", json={"prompt": sample_prompt})
    data = response.json()
    assert "summary"           in data
    assert "model"             in data
    assert "prompt_tokens"     in data
    assert "completion_tokens" in data
    assert "total_tokens"      in data
    assert "latency_ms"        in data

def test_generate_token_counts_are_positive(client, sample_prompt):
    response = client.post("/generate", json={"prompt": sample_prompt})
    data = response.json()
    assert data["prompt_tokens"]     > 0
    assert data["completion_tokens"] > 0
    assert data["total_tokens"]      == data["prompt_tokens"] + data["completion_tokens"]

def test_generate_respects_max_tokens(client, sample_prompt):
    response = client.post("/generate", json={
        "prompt":     sample_prompt,
        "max_tokens": 10
    })
    data = response.json()
    assert data["completion_tokens"] <= 10

def test_generate_model_name_matches(client, sample_prompt):
    response = client.post("/generate", json={"prompt": sample_prompt})
    assert response.json()["model"] == "darija-llm"


def test_stream_returns_200(client, sample_prompt):
    with client.stream("POST", "/generate/stream", json={"prompt": sample_prompt}) as r:
        assert r.status_code == 200

def test_stream_content_type_is_sse(client, sample_prompt):
    with client.stream("POST", "/generate/stream", json={"prompt": sample_prompt}) as r:
        assert "text/event-stream" in r.headers["content-type"]

def test_stream_returns_multiple_chunks(client, sample_prompt):
    chunks = []
    with client.stream("POST", "/generate/stream", json={"prompt": sample_prompt}) as r:
        for line in r.iter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                raw = line[6:]
                try:
                    data = json.loads(raw)
                    if "token" in data:
                        chunks.append(data["token"])
                except json.JSONDecodeError:
                    continue

    assert len(chunks) > 1

def test_stream_chunks_have_correct_structure(client, sample_prompt):
    with client.stream("POST", "/generate/stream", json={"prompt": sample_prompt}) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                raw = line[6:]
                try:
                    data = json.loads(raw)
                    if "token" in data:
                        assert "index"      in data
                        assert "elapsed_ms" in data
                        assert isinstance(data["index"], int)
                        break   
                except json.JSONDecodeError:
                    continue

def test_stream_ends_with_done_event(client, sample_prompt):
    last_event = None
    with client.stream("POST", "/generate/stream", json={"prompt": sample_prompt}) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                raw = line[6:]
                try:
                    last_event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

    assert last_event is not None
    assert last_event.get("done") is True
    assert "total_tokens" in last_event
    assert "total_ms"     in last_event

def test_stream_tokens_are_sequential(client, sample_prompt):
    indices = []
    with client.stream("POST", "/generate/stream", json={"prompt": sample_prompt}) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "index" in data and "token" in data:
                        indices.append(data["index"])
                except json.JSONDecodeError:
                    continue

    assert indices == list(range(len(indices)))   

def test_empty_prompt_returns_422(client):
    response = client.post("/generate", json={"prompt": ""})
    assert response.status_code == 422

def test_whitespace_prompt_returns_422(client):
    response = client.post("/generate", json={"prompt": "   "})
    assert response.status_code == 422

def test_missing_prompt_returns_422(client):
    response = client.post("/generate", json={})
    assert response.status_code == 422

def test_invalid_temperature_returns_422(client):
    response = client.post("/generate", json={
        "prompt":      "النص هنا",
        "temperature": 5.0   
    })
    assert response.status_code == 422

def test_invalid_max_tokens_returns_422(client):
    response = client.post("/generate", json={
        "prompt":     "النص هنا",
        "max_tokens": 0   
    })
    assert response.status_code == 422

def test_prompt_too_long_returns_422(client):
    response = client.post("/generate", json={
        "prompt": "ن" * 5000   
    })
    assert response.status_code == 422

def test_invalid_repetition_penalty_returns_422(client):
    response = client.post("/generate", json={
        "prompt":             "النص هنا",
        "repetition_penalty": 0.5   
    })
    assert response.status_code == 422


def test_stream_empty_prompt_returns_422(client):
    response = client.post("/generate/stream", json={"prompt": ""})
    assert response.status_code == 422

def test_stream_missing_prompt_returns_422(client):
    response = client.post("/generate/stream", json={})
    assert response.status_code == 422