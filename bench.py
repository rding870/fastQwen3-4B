"""Benchmark Qwen3-4B inference with vLLM (offline engine).

Measures:
  * prefill latency  - time to process the prompt and emit 1 token (~TTFT)
  * decode speed     - tokens/s for a single request (batch 1)
  * throughput       - total generated tokens/s across a batch of requests

Prompts are random token IDs of a fixed length and generation ignores EOS,
so every run does exactly the same amount of work and results are comparable.

Usage:
  python bench.py                                  # defaults
  python bench.py --input-len 1024 --output-len 256 --batch-sizes 1 16 64
  python bench.py --json results.json              # also save results
"""

import argparse
import json
import random
import statistics
import time

from vllm import LLM, SamplingParams


def make_prompts(n, input_len, vocab_size, seed):
    rng = random.Random(seed)
    # Stay well below the vocab size to avoid special tokens.
    hi = min(vocab_size, 150_000) - 1
    return [
        {"prompt_token_ids": [rng.randint(100, hi) for _ in range(input_len)]}
        for _ in range(n)
    ]


def timed_generate(llm, prompts, params):
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return elapsed, gen_tokens


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    p.add_argument("--iters", type=int, default=3, help="timed repetitions per measurement")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--enforce-eager", action="store_true", help="disable CUDA graphs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="write results to this JSON file")
    args = p.parse_args()

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,  # random prompts + repeats: don't let caching skew results
        seed=args.seed,
    )
    vocab_size = llm.get_tokenizer().vocab_size

    prefill_params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    decode_params = SamplingParams(max_tokens=args.output_len, temperature=0.0, ignore_eos=True)

    # Warmup (compilation, CUDA graph capture, allocator).
    warm = make_prompts(max(args.batch_sizes), args.input_len, vocab_size, seed=-1)
    llm.generate(warm[:1], decode_params, use_tqdm=False)
    llm.generate(warm, decode_params, use_tqdm=False)

    results = {"model": args.model, "input_len": args.input_len, "output_len": args.output_len, "runs": []}

    # 1) Prefill latency, batch 1.
    times = []
    for i in range(args.iters):
        t, _ = timed_generate(llm, make_prompts(1, args.input_len, vocab_size, args.seed + i), prefill_params)
        times.append(t)
    prefill_ms = statistics.median(times) * 1000
    results["prefill_latency_ms"] = prefill_ms

    # 2) Decode / throughput across batch sizes.
    for bs in args.batch_sizes:
        times, toks = [], 0
        for i in range(args.iters):
            prompts = make_prompts(bs, args.input_len, vocab_size, args.seed + 1000 * bs + i)
            t, toks = timed_generate(llm, prompts, decode_params)
            times.append(t)
        t = statistics.median(times)
        results["runs"].append({
            "batch_size": bs,
            "latency_s": t,
            "output_tok_per_s": toks / t,
            "per_request_tok_per_s": args.output_len / t,
            "total_tok_per_s": bs * (args.input_len + args.output_len) / t,
        })

    # Report.
    print()
    print(f"Model: {args.model}   input_len={args.input_len}   output_len={args.output_len}   iters={args.iters} (median)")
    print(f"Prefill latency (batch 1, {args.input_len} tokens -> 1 token): {prefill_ms:.1f} ms")
    print()
    print(f"{'batch':>6} {'latency (s)':>12} {'out tok/s':>11} {'tok/s/req':>10} {'total tok/s':>12}")
    for r in results["runs"]:
        print(f"{r['batch_size']:>6} {r['latency_s']:>12.2f} {r['output_tok_per_s']:>11.1f} "
              f"{r['per_request_tok_per_s']:>10.1f} {r['total_tok_per_s']:>12.1f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.json}")


if __name__ == "__main__":
    main()
