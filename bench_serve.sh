#!/usr/bin/env bash
# Cross-check bench.py with vLLM's own serving benchmark (`vllm bench serve`).
# Starts an OpenAI-compatible vLLM server, runs the benchmark at several
# concurrency levels against it, then stops the server.
#
# Usage: ./bench_serve.sh [result_dir]     (default: results/vllm_bench_serve)
set -euo pipefail
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-4B"
PORT=8000
INPUT_LEN=512
OUTPUT_LEN=256
CONCURRENCIES=(1 8 32 128)
RESULT_DIR="${1:-results/vllm_bench_serve}"
mkdir -p "$RESULT_DIR"

# Same engine settings as bench.py.
.venv/bin/vllm serve "$MODEL" --port "$PORT" --max-model-len 4096 \
    --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
    > "$RESULT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true' EXIT

echo "Waiting for server (log: $RESULT_DIR/server.log)..."
for _ in $(seq 1 600); do
    curl -sf "localhost:$PORT/health" >/dev/null && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "server exited; see log"; exit 1; }
    sleep 1
done

for c in "${CONCURRENCIES[@]}"; do
    n=$(( c * 3 > 10 ? c * 3 : 10 ))   # 3 requests per concurrency slot, at least 10
    echo "=== concurrency $c, $n prompts"
    .venv/bin/vllm bench serve --model "$MODEL" --port "$PORT" \
        --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$n" --max-concurrency "$c" --ignore-eos --seed 0 \
        --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99 \
        --save-result --result-dir "$RESULT_DIR" --result-filename "c$c.json"
done
