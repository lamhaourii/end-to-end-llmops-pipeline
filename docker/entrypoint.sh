set -e
echo "starting DarijaLLM stack"
echo "model path: $MODEL_PATH"
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model not found at $MODEL_PATH"
    echo "mount the merged model to $MODEL_PATH"
    exit 1
fi
echo "starting vLLM server"
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --dtype float16 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 4 \
    --served-model-name darija-llm \
    --trust-remote-code \
    --enforce-eager &
VLLM_PID=$!
echo "vLLM started with PID $VLLM_PID"
echo "waiting for vLLM to be ready"
until curl -sf http://localhost:$VLLM_PORT/health > /dev/null 2>&1; do
    echo "  vLLM not ready yet, waiting 5s..."
    sleep 5
done
echo "vLLM is ready"
echo "starting FastAPI..."
python -m uvicorn src.serving.api:app \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --log-level info &
API_PID=$!
echo "FastAPI started with PID $API_PID"
trap "kill $VLLM_PID $API_PID" SIGTERM SIGINT
wait $VLLM_PID $API_PID