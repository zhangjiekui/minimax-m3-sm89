# MiniMax-M3 on 4× RTX PRO 6000 Blackwell (SM120)

Serve **[MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3)** (428B / 23B-active MoE with **MiniMax Sparse Attention**) at up to **250K context** on consumer Blackwell GPUs, using the MXFP4 quantization.

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
huggingface-cli download MiniMaxAI/MiniMax-M3-MXFP4 --local-dir /data/models/MiniMax-M3-MXFP4
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
| `MODEL_DIR` | *(required)* | Path to MXFP4 weights |
| `MAX_MODEL_LEN` | `250000` | Lower if you hit KV OOM |
| `GPU_UTIL` | `0.95` | Lower to 0.92 if you hit OOM; caps context at ~190K |
| `TP` | `4` | Tensor parallelism |
| `PORT` | `8000` | |
| `MAX_NUM_SEQS` | `4` | Concurrent requests |

## Context ceiling

The MXFP4 weights occupy ~60 GB/card. At `gpu_memory_utilization=0.92` only ~8 GiB/card remains for KV cache (ceiling **~190K**); at `0.95` ~10.9 GiB frees up (ceiling **250K**). One 262K-token request needs ~11.06 GiB, just over the `0.95` ceiling — the last ~0.17 GiB can't be reclaimed.

Full 256K+ via FP8 KV cache is **not achievable correctly on SM120**: FlashAttention has no FP8 KV, FlashInfer needs the gated `trtllm-gen` backend, and Triton produces gibberish past ~2K (the broken sparse path). Use BF16 KV (the default) at 250K / `0.95`.

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

Patched vLLM files are Apache-2.0 (upstream). Glue/launch code here is MIT. The MiniMax-M3 model is under its own license — see the model card.
