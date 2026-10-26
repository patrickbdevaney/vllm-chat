#!/usr/bin/env bash
# Start vLLM's OpenAI-compatible server for app/stream_ui.py.
#   VLLM_MODEL      HF id or local path (default: a public 7B AWQ DeepSeek-R1-Distill)
#   MAX_MODEL_LEN   8192 by default; 4096 prefills faster
# --enable-reasoning was required by vLLM 0.7.x and later removed; set
# ENABLE_REASONING_FLAG= (empty) on newer vLLM, where --reasoning-parser alone suffices.
set -euo pipefail
VLLM_MODEL="${VLLM_MODEL:-casperhansen/deepseek-r1-distill-qwen-7b-awq}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
ENABLE_REASONING_FLAG="${ENABLE_REASONING_FLAG---enable-reasoning}"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --served-model-name "${SERVED_MODEL_NAME:-deepseek-r1-awq}" \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_MODEL_LEN" \
    $ENABLE_REASONING_FLAG \
    --reasoning-parser deepseek_r1
