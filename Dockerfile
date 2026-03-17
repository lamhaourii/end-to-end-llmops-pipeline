FROM vllm/vllm-openai:v0.6.5
WORKDIR /app
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src/ ./src/
COPY configs/ ./configs/
RUN mkdir -p outputs
ENV MODEL_PATH=/models/merged_model
ENV VLLM_HOST=0.0.0.0
ENV VLLM_PORT=8000
ENV API_HOST=0.0.0.0
ENV API_PORT=8080
EXPOSE 8000 8080
COPY docker/entrypoint.sh .
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]