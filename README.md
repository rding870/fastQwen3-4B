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
```

The benchmark uses vLLM's offline engine with random fixed-length prompts and ignores end-of-sequence tokens, so every run does the same amount of work. It reports:

- **Prefill latency:** time to process the prompt and produce 1 token, roughly the time to first token.
- **tok/s/req:** decode speed seen by each request.
- **out tok/s:** total generated tokens per second across all requests in the batch.

It doesn't measure real text or a live server under load.

## Results

### Baseline: NVIDIA L40S (46 GB), vLLM defaults, bf16

Environment: driver 595.112.07 (CUDA 13.2), vLLM 0.34.0, PyTorch 2.14.0+cu132, Python 3.12.3.
Run: 512 prompt tokens, 256 output tokens per request, median of 3 runs.
Raw data: [`results/baseline_l40s.json`](results/baseline_l40s.json), full log in [`results/baseline_l40s.log`](results/baseline_l40s.log).

Prefill latency (batch 1, 512 tokens): **31.4 ms**

| Requests at once | Time (s) | Total output tok/s | tok/s per request |
|---:|---:|---:|---:|
| 1 | 3.13 | **81.7** | 81.7 |
| 8 | 3.63 | 563.7 | 70.5 |
| 32 | 5.16 | 1,587.7 | 49.6 |
| 128 | 11.02 | **2,973.8** | 23.2 |

At batch 1, each generated token requires reading all ~8 GB of weights. With the L40S's 864 GB/s memory bandwidth, that caps out around 108 tok/s, so 81.7 tok/s is about 75% of the limit. Quantization (FP8/INT4) and speculative decoding are the main ways to go faster at batch 1. Total throughput was still rising at 128 concurrent requests.
