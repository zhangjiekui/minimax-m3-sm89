# syntax=docker/dockerfile:1.7
#
# MiniMax-M3 serving image for NVIDIA Blackwell consumer GPUs (SM120).
#
# Layers the M3 model code, reasoning/tool parsers, MSA warmup, and the
# clamped-swiglu MoE patch onto a pinned upstream vLLM CUDA-13 image.
#
# The patched files live in ./patches and mirror the in-image paths under
# /usr/local/lib/python3.12/dist-packages/vllm/. scripts/register_minimax_m3.py
# adds the registry + parser entries idempotently at build time.
#
# Base pin: upstream vllm/vllm-openai built from commit g454b47db8
# (0.1.dev17492). If a later upstream base is used, re-run the registration
# script — it is idempotent and safe across registry dict reorders.

ARG BASE_IMAGE=vllm/vllm-openai:latest

FROM ${BASE_IMAGE}

# CUDA 13 + Python 3.12 layout in the upstream image.
ARG VLLM_SITE=/usr/local/lib/python3.12/dist-packages/vllm

# 1. M3 model package (nvidia + amd + common paths, MSA sparse attention, MTP).
COPY patches/vllm/models/minimax_m3 ${VLLM_SITE}/models/minimax_m3

# 2. Reasoning + tool parsers (minimax_m3 grammar, <mm:think> blocks).
COPY patches/vllm/reasoning/minimax_m3_reasoning_parser.py ${VLLM_SITE}/reasoning/minimax_m3_reasoning_parser.py
COPY patches/vllm/tool_parsers/minimax_m3_tool_parser.py ${VLLM_SITE}/tool_parsers/minimax_m3_tool_parser.py

# 3. MSA kernel warmup hook.
COPY patches/vllm/model_executor/warmup/minimax_m3_msa_warmup.py ${VLLM_SITE}/model_executor/warmup/minimax_m3_msa_warmup.py

# 4. Clamped-swiglu MoE patch (GPT-OSS-style: alpha=1.702, limit=7.0).
#    Without this, MXFP4 MoE falls back to vanilla SwiGLU and degrades quality.
COPY patches/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py \
     ${VLLM_SITE}/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py

# 5. Register architecture + parsers into the registry dicts (idempotent).
COPY scripts/register_minimax_m3.py /tmp/register_minimax_m3.py
RUN python3 /tmp/register_minimax_m3.py && rm /tmp/register_minimax_m3.py

# Entrypoint is inherited from the base image: ["vllm", "serve"].
# Pass vllm serve args as CMD (see scripts/launch_minimax_m3.sh / docker-compose).
