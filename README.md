# MiniMax-M3 on 4× RTX PRO 6000 Blackwell (SM120)

Serve **[olka-fi/MiniMax-M3-MXFP4](https://huggingface.co/olka-fi/MiniMax-M3-MXFP4)** (MiniMax-M3, 428B / 23B-active MoE with **MiniMax Sparse Attention**) at up to **250K context** on consumer Blackwell GPUs, using the MXFP4 quantization.

This repo rebuilds the patched vLLM image from source so anyone with 4× RTX PRO 6000 (96 GB) can reproduce a correct, verified serving endpoint.

## What this fixes

Upstream `vllm` does not yet ship the pieces M3 needs, all of which are bundled here:

- **MSA model** — `MiniMaxM3SparseForConditionalGeneration` (per-GQA-group top-16 block selection, max-pool scoring, forced local block)
- **`minimax_m3` parsers** — reasoning (`<mm:think>`) + tool-call (namespaced XML)
- **clamped-swiglu MoE** — `(clamp(up,±7)+1)·clamp(gate,max=7)·σ(1.702·gate)`. Without it MXFP4 MoE silently drops to vanilla SwiGLU and **degrades all 57 expert layers**.

## Requirements

| | |
|---|---|
| GPUs | 4× RTX PRO 6000 Blackwell (96 GB), compute 12.0 |
| Driver | ≥ 580 (CUDA 13.0) |
| Docker | + NVIDIA Container Toolkit (`--runtime=nvidia`) |
| Disk | ~256 GB for MXFP4 weights |

## Setup

**1. Download the weights** (MXFP4, ~256 GB):

```bash
huggingface-cli download olka-fi/MiniMax-M3-MXFP4 --local-dir /data/models/MiniMax-M3-MXFP4
```

**2. Clone + configure:**

```bash
git clone <this-repo> minimax-m3-sm120 && cd minimax-m3-sm120
cp .env.example .env
# edit .env: set MODEL_DIR to your weights path
```

**3. Build + serve:**

```bash
bash scripts/serve.sh        # builds the image on first run, then serves
# or: docker compose up -d --build
```

First boot JIT-compiles the MSA kernels (~3–4 min). Watch progress:

```bash
docker logs -f minimax-m3-vllm
```

When you see `Application startup complete`, the endpoint is live:

```bash
curl http://localhost:8000/v1/models    # -> max_model_len: 250000
```

## Verify

```bash
bash scripts/validate.sh     # needle retrieval + throughput
```

## Measured throughput (4× RTX PRO 6000, MXFP4, thinking off, temp 0)

| Metric | Value |
|---|---|
| Decode (single) | ~113 tok/s, TTFT 0.13 s |
| Decode (4× concurrent) | ~184 tok/s aggregate |
| Prefill (cold) | ~2,800 tok/s, flat 8 K→131 K |

## Configuration

All via `.env` (see `.env.example`):

| Var | Default | Notes |
|---|---|---|
| `MODEL_DIR` | *(required)* | Path to the downloaded `olka-fi/MiniMax-M3-MXFP4` weights |
| `MAX_MODEL_LEN` | `250000` | Lower if you hit KV OOM |
| `GPU_UTIL` | `0.95` | Lower to 0.92 if you hit OOM; caps context at ~190K |
| `TP` | `4` | Tensor parallelism |
| `PORT` | `8000` | |
| `MAX_NUM_SEQS` | `4` | Concurrent requests |

## Context ceiling

The MXFP4 weights occupy ~60 GB/card. At `gpu_memory_utilization=0.92` only ~8 GiB/card remains for KV cache (ceiling **~190K**); at `0.95` ~10.9 GiB frees up (ceiling **250K**). One 262K-token request needs ~11.06 GiB, just over the `0.95` ceiling — the last ~0.17 GiB can't be reclaimed.

Full 256K+ via FP8 KV cache is **not achievable correctly on SM120**: FlashAttention has no FP8 KV, FlashInfer needs the gated `trtllm-gen` backend, and Triton produces gibberish past ~2K (the broken sparse path). Use BF16 KV (the default) at 250K / `0.95`.

## Alternative: NVFP4 (SGLang) — image + video, up to ~643K context

This repo serves the **MXFP4** checkpoint on **vLLM**. There is a second, fully
working option on the same 4× RTX PRO 6000 (SM120) host that serves the
**NVFP4** checkpoint on **SGLang** and additionally models the **vision tower**
(true image + video), which the vLLM `minimax-m3` image does not.

- **Model:** [lukealonso/MiniMax-M3-NVFP4](https://huggingface.co/lukealonso/MiniMax-M3-NVFP4) (~243 GB)
- **Harness:** [0xSero/minimax-m3-nvfp4-sglang](https://github.com/0xSero/minimax-m3-nvfp4-sglang)
- **Image:** `minimax-m3-sglang:dev-cu13-minimax-m3-patched` (base `lmsysorg/sglang:dev-cu13-minimax-m3`, loader + modelopt + config patches baked in)

Verified working config (4× RTX PRO 6000, TP4):

```bash
python3 scripts/patch_model_config.py "$MODEL_DIR/config.json"   # idempotent
docker run -d --name minimax-m3-sglang --gpus all --ipc=host --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /mnt/llm_models:/mnt/llm_models:ro \
  minimax-m3-sglang:dev-cu13-minimax-m3-patched \
    --model-path "$MODEL_DIR" --tokenizer-path "$MODEL_DIR" --trust-remote-code \
    --host 0.0.0.0 --port 8000 --served-model-name minimax-m3 \
    --tp-size 4 --context-length 640000 \
    --kv-cache-dtype bfloat16 --quantization modelopt_fp4 \
    --reasoning-parser minimax-m3 --tool-call-parser minimax-m3 \
    --moe-runner-backend flashinfer_cutlass --disable-shared-experts-fusion \
    --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
    --max-running-requests 4 --chunked-prefill-size 4096 --max-prefill-tokens 16384 \
    --mem-fraction-static 0.95 \
    --enable-multimodal --limit-mm-data-per-request '{"image": 4, "video": 1}'
```

Measured (NVFP4 / SGLang): **~87 tok/s** single-stream decode; **12K-token**
outputs stay coherent; **155K-token** needle retrieval works; **4 concurrent
sessions** sharing a **643,659-token** KV pool (≈4×157K simultaneously) all
retrieve correctly. Architectural max is `max_position_embeddings=1,048,576`;
VRAM caps the practical ceiling at **~643K** (weights ~62 GB/card, KV auto-sized
into the rest at `mem-fraction-static=0.95`).

Getting NVFP4 to produce correct output required the **same clamped-SwiGLU fix**
as MXFP4 (the lukealonso export shipped `hidden_act="silu"`; it must be
`"swigluoai"` so the dense MLP + shared experts use `clamp(up,±7)+1)·clamp(gate,max=7)·σ(1.702·gate)`),
plus: disabling the lightning-indexer value/output projection per sparse layer
(`sparse_disable_index_value`), rewriting the quant `ignore` list from
`block_sparse_moe.shared_experts` → `mlp.shared_experts` (shared experts ship
BF16), deriving `moe_layer_freq` (dense layers 0–2), expert-name remap
(`gate/up/down_proj`→`w1/w3/w2`), and a vision-tower remap
(`vision_tower.layers`→`vision_tower.vision_model.encoder.layers`,
`embeddings.proj`→`patch_embedding`, `multi_modal_projector.merge_linear`→`patch_merge_mlp.linear`).

## MXFP4 vs NVFP4

Both are 4-bit (E2M1) element formats; they differ in **how the scale is stored**:

| | **MXFP4** (this repo, vLLM) | **NVFP4** (SGLang option) |
|---|---|---|
| Checkpoint | [olka-fi/MiniMax-M3-MXFP4](https://huggingface.co/olka-fi/MiniMax-M3-MXFP4) (~256 GB) | [lukealonso/MiniMax-M3-NVFP4](https://huggingface.co/lukealonso/MiniMax-M3-NVFP4) (~243 GB) |
| Element | FP4 E2M1 | FP4 E2M1 |
| Block size | 32 elements | 16 elements (finer) |
| Block scale | **E8M0** — a shared power-of-two exponent (no mantissa) | **FP8 E4M3** — has a mantissa (finer per-block scaling) |
| Global scale | none | per-tensor **FP32** (`weight_scale_2`), two-level dequant |
| Fidelity | coarser (pow-2 scales) | higher (FP8 block scale + FP32 global) |
| Serving here | vLLM (Marlin MXFP4 path) | SGLang (FlashInfer **cutlass** fp4 MoE on SM120) |
| Vision | text-only in the vLLM image | **image + video** (vision tower modeled) |
| Practical context | 250K (BF16 KV) | ~643K (BF16 KV, auto-sized) |
| Decode (single) | ~113 tok/s | ~87 tok/s |

In short: **MXFP4** uses one coarse power-of-two scale per 32-value block;
**NVFP4** uses a finer FP8 scale per 16-value block plus a per-tensor FP32
global scale, so NVFP4 generally preserves accuracy better at the cost of
needing the two-level dequant the FlashInfer cutlass kernels provide. On this
host the NVFP4/SGLang path also unlocks real multimodal and much longer context;
the MXFP4/vLLM path is simpler to run and decodes faster.

## Repo layout

```
Dockerfile              # layers M3 patches onto upstream vllm/vllm-openai
docker-compose.yml      # managed launch
.env.example            # copy to .env, set MODEL_DIR
scripts/
  serve.sh              # build + run (reads .env)
  validate.sh           # needle + throughput checks
  register_minimax_m3.py# idempotent registry registration (runs in build)
  bench_m3.py           # throughput benchmark
patches/vllm/           # the M3 code (mirrors vLLM in-image layout)
```

## License

Patched vLLM files are Apache-2.0 (upstream). Glue/launch code here is MIT. The MiniMax-M3 checkpoint is under its own license — see the [`olka-fi/MiniMax-M3-MXFP4` model card](https://huggingface.co/olka-fi/MiniMax-M3-MXFP4).
