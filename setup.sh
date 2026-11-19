#!/usr/bin/env bash
# Recreate the environment and download the model.
# Requires: uv (https://docs.astral.sh/uv/), an NVIDIA GPU with a driver that supports CUDA 13.2+.
set -euo pipefail
cd "$(dirname "$0")"

# Exact model snapshot the benchmarks were run against.
MODEL_ID="Qwen/Qwen3-4B"
MODEL_REVISION="1cfa9a7208912126459214e8b04321603b3df60c"

# 1) Python environment (.venv) from the lockfile - exact same package versions.
uv sync --frozen

# 2) Model weights (~7.6 GB) into the Hugging Face cache ($HF_HOME, default ~/.cache/huggingface).
.venv/bin/hf download "$MODEL_ID" --revision "$MODEL_REVISION"

echo
echo "Done. Run the benchmark with:  .venv/bin/python bench.py"
