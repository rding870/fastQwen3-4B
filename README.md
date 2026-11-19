# fastQwen3-4B

Fast inference and benchmarking for [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) with [vLLM](https://github.com/vllm-project/vllm).

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU whose driver supports CUDA 13.2 or newer (check the "CUDA Version" shown by `nvidia-smi`).

```bash
./setup.sh
```

This does two things:

1. `uv sync --frozen` creates `.venv` with the exact package versions in `uv.lock` (Python 3.12, vLLM 0.34.0, PyTorch 2.14.0 with CUDA 13.2).
2. `hf download Qwen/Qwen3-4B --revision 1cfa9a7…` downloads the model weights (about 7.6 GB) into the Hugging Face cache.

Neither the model weights nor `.venv` are stored in git. Both can be rebuilt from the steps above.

### Notes on the dependency setup

- The original install used `uv pip install vllm --torch-backend auto`, which picked the CUDA 13.2 build of PyTorch. A uv project can't use that flag, so `pyproject.toml` points `torch`, `torchvision` and `torchcodec` at PyTorch's `cu132` package index instead.
- `torchaudio` comes from PyTorch's CPU-only index on purpose. The PyPI build is compiled for CUDA 13.0, and with it vLLM fails at import with `PyTorch and TorchAudio were compiled with different CUDA versions`. A text model never uses torchaudio.
- uv applies index pins only to direct dependencies, so `torchaudio` and `torchcodec` are listed in `dependencies` even though only vLLM uses them.

## Benchmark

```bash
.venv/bin/python bench.py                                      # defaults
.venv/bin/python bench.py --input-len 1024 --output-len 256 --batch-sizes 1 16 64
.venv/bin/python bench.py --json results/my_run.json           # save results
.venv/bin/python bench.py --no-profile                         # skip the kernel-time pass
```

The benchmark runs vLLM's async engine. For each batch size N, it submits N requests at the same time and times each one as its tokens stream back. Prompts are random fixed-length token IDs and generation ignores end-of-sequence tokens, so every run does the same amount of work. It reports:

| Metric | What it measures |
|---|---|
| **TTFT** (time to first token) | Time from submitting a request to receiving its first token, p50/p95/p99. With many requests at once, prompts are processed in chunks, so later requests wait longer. |
| **TPOT** (time per output token) | (end-to-end time − TTFT) / (output tokens − 1), per request. 1000 / TPOT is the tokens/s each request sees. |
| **Throughput** | Generated tokens/s (`out tok/s`) and prompt + generated tokens/s (`total tok/s`) across all requests. |
| **p95 latency** | 95th percentile end-to-end request latency, plus p50 and p99. Percentiles are taken over all requests in all iterations. |
| **GPU memory** | Device memory from NVML: before the engine starts, after it starts, and the peak during each batch. vLLM pre-allocates memory up to `--gpu-memory-utilization` (default 0.9) at start-up, so the peak stays flat. Most of it is reserved for the KV cache. |
| **KV-cache utilization** | Peak fraction of vLLM's KV-cache blocks in use, recorded every engine step. The total KV-cache capacity in tokens is also printed. |
| **GPU kernel time** | A separate pass at the largest batch size with the PyTorch profiler on: total GPU kernel time, GPU busy %, kernel time per output token and the top kernels. The profiler adds some overhead, so this pass is kept out of the latency and throughput numbers. |

It doesn't measure real text or a live HTTP server.

## Results

### Baseline: NVIDIA L40S (46 GB), vLLM defaults, bf16

Environment: driver 595.112.07 (CUDA 13.2), vLLM 0.34.0, PyTorch 2.14.0+cu132, Python 3.12.3.
Run: 512 prompt tokens, 256 output tokens per request, 3 iterations per batch size (throughput is the median).
Raw data: [`results/baseline_l40s.json`](results/baseline_l40s.json), full log in [`results/baseline_l40s.log`](results/baseline_l40s.log).

| Requests at once | Output tok/s | Total tok/s | TTFT p50 / p95 (ms) | TPOT p50 / p95 (ms) | E2E p50 / p95 / p99 (s) | KV cache peak |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **81.7** | 245.0 | 36.4 / 41.6 | 12.15 / 12.15 | 3.13 / 3.14 / 3.14 | 0.3% |
| 8 | 563.7 | 1,691.2 | 157.7 / 193.5 | 13.60 / 13.93 | 3.63 / 3.63 / 3.63 | 2.7% |
| 32 | 1,585.3 | 4,755.8 | 445.2 / 749.6 | 18.32 / 19.35 | 5.12 / 5.17 / 5.17 | 10.8% |
| 128 | **2,939.3** | 8,817.8 | 1,593.9 / 3,270.7 | 36.23 / 39.97 | 10.84 / 11.13 / 11.15 | 43.2% |

**GPU memory:** 0.59 GiB used before vLLM starts and 40.76 GiB after (out of 45.0 GiB), staying flat during every run. The KV cache gets 31.3 GiB, enough for 227,840 tokens at 144 KiB per token.

**GPU kernel time** (profiled pass, 128 requests): 11,146 ms of kernel time over 165,090 kernel launches. The GPU was busy 99.4% of the time, with about 340 µs of kernel time per generated token.

| Share (top 10 kernels) | Kernel |
|---:|---|
| 42.9% | FlashAttention split-KV attention (two variants) |
| 51.1% | bf16 matrix multiplies (cuBLAS / CUTLASS GEMMs) |
| ~3% | fused Triton elementwise kernels (SiLU, RMSNorm) |

**What this shows:**

- With one request at a time, TPOT is 12.15 ms (81.7 tok/s). Each token requires reading all ~8 GB of weights, and the L40S's 864 GB/s memory bandwidth caps that at about 108 tok/s, so this is about 75% of the limit. Quantization (FP8/INT4) and speculative decoding are the main ways to go faster here.
- With 128 requests at once, total throughput is 36× higher, but each request's TPOT triples to 36 ms. TTFT p95 jumps to 3.3 s because all 128 prompts (65k tokens) arrive at once and are processed in chunks.
- At batch 128, attention takes 43% of GPU kernel time. That makes attention, not just the weight matrices, a target for optimization at high concurrency (for example an FP8 KV cache).
- The KV cache was only 43% full at 128 requests, so there's room for more concurrency or longer contexts.

### Cross-check with `vllm bench serve`

[`bench_serve.sh`](bench_serve.sh) runs vLLM's own serving benchmark against a real HTTP server with the same settings (512 in, 256 out, EOS ignored, 3 requests per concurrency slot). Raw results are in [`results/vllm_bench_serve_l40s/`](results/vllm_bench_serve_l40s/).

| Concurrency | Output tok/s (bench.py / vllm) | TTFT p50 ms | TTFT p95 ms | TPOT p50 ms | E2E p95 s |
|---:|---:|---:|---:|---:|---:|
| 1 | 81.7 / 80.6 | 36.4 / 40.4 | 41.6 / 43.2 | 12.15 / 12.29 | 3.14 / 3.18 |
| 8 | 563.7 / 558.2 | 157.7 / 132.4 | 193.5 / 200.4 | 13.60 / 13.83 | 3.63 / 3.70 |
| 32 | 1,585.3 / 1,579.5 | 445.2 / 352.7 | 749.6 / 717.9 | 18.32 / 18.80 | 5.17 / 5.49 |
| 128 | 2,939.3 / 2,934.5 | 1,593.9 / 475.5 | 3,270.7 / 2,887.7 | 36.23 / 41.32 | 11.13 / 13.49 |

Throughput agrees to within about 1.5% at every level. The HTTP server adds about 1–4 ms to each request.

The latency differences at high concurrency come from how requests arrive:

- **`bench.py`** sends all N requests at once, waits for them all to finish, then sends the next batch.
- **`vllm bench serve`** keeps N requests in flight and starts a new one as soon as one finishes.

With a steady stream of new requests, a newcomer usually starts within a few hundred ms (lower median TTFT). But its prompt processing interrupts everyone else's generation (higher TPOT and p95 latency). Burst numbers fit batch jobs, and the steady-stream numbers are closer to a live service.
