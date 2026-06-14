#!/usr/bin/env bash
# MiniMax-M3 (MXFP4) launcher for 4x RTX PRO 6000 Blackwell (SM120).
#
# Builds the patched vLLM image if missing, then serves MiniMax-M3 at the
# largest BF16-KV context that fits on 4x 96GB cards.
#
# Configure via env vars (see .env.example) or pass them inline:
#   MODEL_DIR=/data/models/MiniMax-M3-MXFP4 MAX_MODEL_LEN=250000 bash scripts/serve.sh
set -euo pipefail

# --- config (override via env / .env) ---------------------------------------
IMAGE=${M3_IMAGE:-minimax-m3-vllm:latest}
MODEL_DIR=${MODEL_DIR:?MODEL_DIR must point at the MiniMax-M3-MXFP4 weights}
MODELS_ROOT=${MODELS_ROOT:-$(dirname "$MODEL_DIR")}
SWIGLU_PATCH=${SWIGLU_PATCH:-$MODEL_DIR/vllm_patch/compressed_tensors_moe_w4a4_mxfp4.py}
CONTAINER=${CONTAINER:-minimax-m3-vllm}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-250000}
GPU_UTIL=${GPU_UTIL:-0.95}
TP=${TP:-4}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}

# load .env if present
[ -f "$(dirname "$0")/../.env" ] && set -a && . "$(dirname "$0")/../.env" && set +a || true

# --- build if missing -------------------------------------------------------
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[build] ${IMAGE} not found — building patched image..."
  docker build -t "${IMAGE}" -f Dockerfile .
fi

# --- launch -----------------------------------------------------------------
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

PATCH_MOUNT=""
if [ -f "${SWIGLU_PATCH}" ]; then
  PATCH_TARGET=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py
  PATCH_MOUNT="-v ${SWIGLU_PATCH}:${PATCH_TARGET}:ro"
else
  echo "[warn] swiglu patch not found at ${SWIGLU_PATCH}; MoE quality will degrade." >&2
  exit 1
fi

echo "[serve] ${CONTAINER}: model=${MODEL_DIR} ctx=${MAX_MODEL_LEN} tp=${TP} util=${GPU_UTIL}"

exec docker run \
  --name "${CONTAINER}" \
  --runtime=nvidia --gpus all \
  --network host --ipc host --shm-size 16g \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_MXFP4_USE_MARLIN=1 \
  -e VLLM_LOGGING_LEVEL=INFO \
  -v "${MODELS_ROOT}":"${MODELS_ROOT}":ro \
  ${PATCH_MOUNT} \
  "${IMAGE}" \
  --model "${MODEL_DIR}" \
  --served-model-name minimax-m3 \
  --host 0.0.0.0 --port "${PORT}" \
  --trust-remote-code \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --block-size 128 \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --load-format fastsafetensors \
  --linear-backend marlin \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice
