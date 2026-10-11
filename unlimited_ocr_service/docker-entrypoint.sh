#!/usr/bin/env bash
#
# Launches the vLLM OpenAI-compatible server for Unlimited-OCR.
#
# The flags below are the ones Unlimited-OCR requires to produce usable
# markdown; they are not interchangeable with a generic vLLM deployment:
#   * --logits-processors   the model has no chat template and relies on the
#                           no-repeat-n-gram processor, otherwise long
#                           documents loop on <|det|> coordinate tokens
#   * --trust-remote-code   the repo ships custom modeling code in its auto_map
#   * --no-enable-prefix-caching / --mm-processor-cache-gb 0
#                           OCR requests never reuse an image, so both caches
#                           only cost memory
#
set -euo pipefail

MODEL_ID="${MODEL_ID:-baidu/Unlimited-OCR}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Unlimited-OCR}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_LOGITS_PROCESSORS="${VLLM_LOGITS_PROCESSORS:-vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor}"

args=(
    serve "$MODEL_ID"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$VLLM_HOST"
    --port "$VLLM_PORT"
    --dtype "$VLLM_DTYPE"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --logits-processors "$VLLM_LOGITS_PROCESSORS"
    --trust-remote-code
    --no-enable-prefix-caching
    --mm-processor-cache-gb 0
)

# Multi-page OCR is already unbounded by default (vLLM defaults every modality
# to 999 items per prompt). Set this to cap it, e.g. '{"image": 16}'.
if [ -n "${VLLM_LIMIT_MM_PER_PROMPT:-}" ]; then
    args+=(--limit-mm-per-prompt "$VLLM_LIMIT_MM_PER_PROMPT")
fi

if [ -n "${VLLM_API_KEY:-}" ]; then
    args+=(--api-key "$VLLM_API_KEY")
fi

# Appended last so anything set here is overridable from the command line.
args+=("$@")

echo "[unlimited-ocr] launching: vllm ${args[*]}" >&2
exec vllm "${args[@]}"
