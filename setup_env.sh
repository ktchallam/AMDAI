#!/usr/bin/env bash
set -eo pipefail

echo "=== AMD AI Hackathon Multi-Agent Workflow Environment Setup ==="

# 1. Install Dependencies
echo "Installing dependencies..."
pip install --break-system-packages \
    vllm \
    sglang \
    sentence-transformers \
    chromadb \
    bertopic \
    transformers \
    pdfplumber \
    python-pptx \
    pandas \
    numpy \
    httpx \
    openai \
    pydantic \
    pydantic-settings \
    loguru \
    tenacity \
    youtube-transcript-api \
    feedparser \
    faster-whisper \
    finbert-embedding

# 2. Verify ROCm Availability
echo "Verifying ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    echo "ROCm is available:"
    rocm-smi
else
    echo "WARNING: rocm-smi not found. This might fail on non-ROCm systems, but continuing setup..."
fi

# 3. Start vLLM Server in background
echo "Starting vLLM server serving meta-llama/Llama-3-70B-Instruct..."
# Start server and redirect outputs to vllm.log
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --dtype float16 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --port 8000 > vllm.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# 4. Wait for server health endpoint to be ready
echo "Waiting for vLLM server to be ready at http://localhost:8000/v1..."
MAX_ATTEMPTS=60
ATTEMPT=1
while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:8000/v1/health &> /dev/null; then
        echo "vLLM server is healthy and ready!"
        break
    fi
    echo "Waiting... (Attempt $ATTEMPT/$MAX_ATTEMPTS)"
    sleep 5
    let ATTEMPT=ATTEMPT+1
done

if [ $ATTEMPT -gt $MAX_ATTEMPTS ]; then
    echo "Error: vLLM server failed to start within the timeout period."
    echo "Check vllm.log for details:"
    tail -n 20 vllm.log
    exit 1
fi

echo "Environment setup and server verification complete."
