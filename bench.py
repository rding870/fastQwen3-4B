"""Benchmark Qwen3-4B inference with vLLM.

For each concurrency level (N requests submitted at the same time) it reports:

  * TTFT        - time to first token, per request (mean / p50 / p95 / p99)
  * TPOT        - time per output token after the first, per request
  * E2E latency - full request latency (mean / p50 / p95 / p99)
  * Throughput  - generated tokens/s and total (prompt + generated) tokens/s
  * GPU memory  - NVML device memory: idle, after engine start, peak during the run
  * KV cache    - peak / mean fraction of vLLM's KV-cache blocks in use
  * Kernel time - separate profiled pass (torch profiler): total GPU kernel time,
                  GPU busy %, and the top kernels

Prompts are random token IDs of a fixed length and generation ignores EOS, so
every run does the same amount of work. Profiling adds overhead, so kernel time
is measured in its own pass and does not affect the latency/throughput numbers.

Usage:
  python bench.py
  python bench.py --input-len 1024 --output-len 256 --batch-sizes 1 16 64
  python bench.py --json results/run.json
  python bench.py --no-profile            # skip the kernel-time pass
"""

import argparse
import asyncio
import glob
import gzip
import json
import math
import os
import random
import shutil
import statistics
import tempfile
import threading
import time
import uuid
from collections import defaultdict

import pynvml
from vllm import SamplingParams
from vllm.config import ProfilerConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase

GIB = 1024**3


# --------------------------------------------------------------------------- helpers

def pct(values, p):
    """Percentile with linear interpolation (same as numpy's default)."""
    xs = sorted(v for v in values if not math.isnan(v))
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(values):
    xs = [v for v in values if not math.isnan(v)]
    if not xs:
        return {k: float("nan") for k in ("mean", "p50", "p95", "p99", "max")}
    return {"mean": statistics.fmean(xs), "p50": pct(xs, 50), "p95": pct(xs, 95),
            "p99": pct(xs, 99), "max": max(xs)}


def make_prompts(n, input_len, vocab_size, rng):
    hi = min(vocab_size, 150_000) - 1  # stay clear of special tokens
    return [{"prompt_token_ids": [rng.randint(100, hi) for _ in range(input_len)]} for _ in range(n)]


class KVCacheTracker(StatLoggerBase):
    """Receives vLLM scheduler stats every engine step; keeps KV-cache usage samples."""

    samples = []  # (time, kv_cache_usage 0..1, running, waiting)

    def __init__(self, vllm_config, engine_index=0):
        pass

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        if scheduler_stats is not None:
            KVCacheTracker.samples.append((time.perf_counter(), scheduler_stats.kv_cache_usage,
                                           scheduler_stats.num_running_reqs,
                                           scheduler_stats.num_waiting_reqs))

    def log_engine_initialized(self):
        pass


class GpuMemorySampler:
    """Polls NVML device memory in a background thread to catch the peak."""

    def __init__(self, device_index, interval=0.02):
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def used(self):
        return pynvml.nvmlDeviceGetMemoryInfo(self.handle).used

    def total(self):
        return pynvml.nvmlDeviceGetMemoryInfo(self.handle).total

    def __enter__(self):
        self.peak = self.used()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.used())
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.used())


# --------------------------------------------------------------------------- requests

async def run_request(engine, prompt, params, t0):
    ttft, n_tokens = None, 0
    async for out in engine.generate(prompt, params, request_id=uuid.uuid4().hex):
        k = len(out.outputs[0].token_ids)  # DELTA mode: only new tokens
        if k:
            if ttft is None:
                ttft = time.perf_counter() - t0
            n_tokens += k
    e2e = time.perf_counter() - t0
    tpot = (e2e - ttft) / (n_tokens - 1) if n_tokens > 1 else float("nan")
    return {"ttft": ttft, "tpot": tpot, "e2e": e2e, "output_tokens": n_tokens}


async def run_batch(engine, prompts, params):
    t0 = time.perf_counter()
    reqs = await asyncio.gather(*(run_request(engine, p, params, t0) for p in prompts))
    return time.perf_counter() - t0, reqs


# --------------------------------------------------------------------------- profiling

def parse_kernel_trace(trace_dir, top_n):
    """Sum GPU kernel time from the worker's torch profiler trace (Chrome trace JSON)."""
    best = None
    for path in glob.glob(os.path.join(trace_dir, "**", "*.json*"), recursive=True):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            events = json.load(f).get("traceEvents", [])
        kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
        if kernels and (best is None or len(kernels) > len(best[1])):
            best = (events, kernels)
    if best is None:
        return None
    events, kernels = best
    memcpy_us = sum(e["dur"] for e in events if e.get("cat") in ("gpu_memcpy", "gpu_memset")
                    and e.get("ph") == "X")

    total_us = sum(e["dur"] for e in kernels)
    # GPU busy time = union of kernel intervals (kernels on different streams can overlap).
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    busy_us, cur_s, cur_e = 0.0, *intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            busy_us += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy_us += cur_e - cur_s
    span_us = intervals[-1][1] - intervals[0][0]

    by_name = defaultdict(lambda: [0.0, 0])
    for e in kernels:
        by_name[e["name"]][0] += e["dur"]
        by_name[e["name"]][1] += 1
    top = sorted(by_name.items(), key=lambda kv: -kv[1][0])[:top_n]
    return {
        "kernel_time_ms": total_us / 1000,
        "kernel_launches": len(kernels),
        "memcpy_memset_time_ms": memcpy_us / 1000,
        "gpu_busy_ms": busy_us / 1000,
        "trace_span_ms": span_us / 1000,
        "gpu_busy_pct": 100 * busy_us / span_us if span_us else float("nan"),
        "top_kernels": [{"name": n, "time_ms": t / 1000, "calls": c, "share_pct": 100 * t / total_us}
                        for n, (t, c) in top],
    }


# --------------------------------------------------------------------------- main

async def main_async(args):
    pynvml.nvmlInit()
    gpu = GpuMemorySampler(args.device)
    mem_idle = gpu.used()

    trace_dir = None if args.no_profile else tempfile.mkdtemp(prefix="vllm_trace_")
    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,  # random prompts; don't let caching skew results
        seed=args.seed,
        profiler_config=ProfilerConfig(
            profiler="torch", torch_profiler_dir=trace_dir,
            torch_profiler_with_stack=False, torch_profiler_dump_cuda_time_total=False,
            torch_profiler_use_gzip=True,
        ) if trace_dir else ProfilerConfig(),
    )
    engine = AsyncLLM.from_engine_args(engine_args, stat_loggers=[KVCacheTracker])
    try:
        return await benchmark(engine, args, gpu, mem_idle, trace_dir)
    finally:
        engine.shutdown()
        if trace_dir and not args.keep_trace:
            shutil.rmtree(trace_dir, ignore_errors=True)


async def benchmark(engine, args, gpu, mem_idle, trace_dir):
    cfg = engine.vllm_config
    hf = cfg.model_config.hf_config
    tokenizer = engine.get_tokenizer()
    vocab_size = tokenizer.vocab_size
    rng = random.Random(args.seed)

    # KV-cache capacity.
    num_blocks = cfg.cache_config.num_gpu_blocks or 0
    block_size = cfg.cache_config.block_size
    head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
    dtype_bytes = cfg.model_config.dtype.itemsize
    kv_bytes_per_token = 2 * hf.num_hidden_layers * hf.num_key_value_heads * head_dim * dtype_bytes
    kv_capacity_tokens = num_blocks * block_size

    params = SamplingParams(max_tokens=args.output_len, temperature=0.0, ignore_eos=True,
                            output_kind=RequestOutputKind.DELTA)

    # Warmup: compilation / CUDA graphs / allocator.
    await run_batch(engine, make_prompts(1, args.input_len, vocab_size, rng), params)
    await run_batch(engine, make_prompts(max(args.batch_sizes), args.input_len, vocab_size, rng), params)
    mem_engine = gpu.used()

    results = {
        "model": args.model, "input_len": args.input_len, "output_len": args.output_len,
        "iters": args.iters,
        "gpu": pynvml.nvmlDeviceGetName(gpu.handle),
        "memory": {
            "gpu_total_gib": gpu.total() / GIB,
            "used_before_engine_gib": mem_idle / GIB,
            "used_after_engine_start_gib": mem_engine / GIB,
            "engine_footprint_gib": (mem_engine - mem_idle) / GIB,
            "gpu_memory_utilization_setting": args.gpu_memory_utilization,
        },
        "kv_cache": {
            "num_blocks": num_blocks, "block_size": block_size,
            "capacity_tokens": kv_capacity_tokens,
            "bytes_per_token": kv_bytes_per_token,
            "capacity_gib": kv_capacity_tokens * kv_bytes_per_token / GIB,
        },
        "runs": [],
    }

    for bs in args.batch_sizes:
        walls, reqs_all, out_tokens = [], [], []
        KVCacheTracker.samples.clear()
        with gpu:
            for _ in range(args.iters):
                wall, reqs = await run_batch(engine, make_prompts(bs, args.input_len, vocab_size, rng), params)
                walls.append(wall)
                reqs_all.extend(reqs)
                out_tokens.append(sum(r["output_tokens"] for r in reqs))
        kv = [s[1] for s in KVCacheTracker.samples]
        wall = statistics.median(walls)
        out_tok_s = statistics.median(o / w for o, w in zip(out_tokens, walls))
        results["runs"].append({
            "batch_size": bs,
            "wall_time_s_median": wall,
            "output_tok_per_s": out_tok_s,
            "total_tok_per_s": statistics.median(
                (o + bs * args.input_len) / w for o, w in zip(out_tokens, walls)),
            "ttft_ms": {k: v * 1000 for k, v in summarize(r["ttft"] for r in reqs_all).items()},
            "tpot_ms": {k: v * 1000 for k, v in summarize(r["tpot"] for r in reqs_all).items()},
            "e2e_latency_s": summarize(r["e2e"] for r in reqs_all),
            "kv_cache_usage_pct": {"peak": 100 * max(kv, default=float("nan")),
                                   "mean_while_busy": 100 * statistics.fmean(
                                       [x for x in kv if x > 0] or [float("nan")])},
            "gpu_memory_peak_gib": gpu.peak / GIB,
            "num_requests_measured": len(reqs_all),
        })
        r = results["runs"][-1]
        print(f"[bench] batch {bs}: {r['output_tok_per_s']:.1f} out tok/s, "
              f"TTFT p95 {r['ttft_ms']['p95']:.1f} ms, TPOT p95 {r['tpot_ms']['p95']:.2f} ms", flush=True)

    # Kernel-time pass.
    if trace_dir:
        bs = args.profile_batch_size or max(args.batch_sizes)
        prompts = make_prompts(bs, args.input_len, vocab_size, rng)
        await engine.start_profile()
        wall, reqs = await run_batch(engine, prompts, params)
        await engine.stop_profile()
        prof = None
        for _ in range(60):  # trace is written by the engine process; wait for it
            prof = parse_kernel_trace(trace_dir, args.top_kernels)
            if prof:
                break
            await asyncio.sleep(1)
        if prof:
            n_out = sum(r["output_tokens"] for r in reqs)
            prof.update({"batch_size": bs, "wall_time_ms_profiled": wall * 1000,
                         "kernel_time_per_output_token_ms": prof["kernel_time_ms"] / n_out})
            if args.keep_trace:
                prof["trace_dir"] = trace_dir
        results["kernel_profile"] = prof

    return results


def report(res):
    m, kv = res["memory"], res["kv_cache"]
    print()
    print(f"Model: {res['model']}   GPU: {res['gpu']}   input_len={res['input_len']}   "
          f"output_len={res['output_len']}   iters={res['iters']}")
    print()
    print("GPU memory")
    print(f"  total {m['gpu_total_gib']:.1f} GiB | before engine {m['used_before_engine_gib']:.2f} GiB | "
          f"after engine start {m['used_after_engine_start_gib']:.2f} GiB "
          f"(vLLM pre-allocates up to gpu_memory_utilization={m['gpu_memory_utilization_setting']})")
    print(f"  KV cache: {kv['capacity_tokens']:,} tokens capacity = {kv['capacity_gib']:.2f} GiB "
          f"({kv['num_blocks']:,} blocks x {kv['block_size']} tokens, {kv['bytes_per_token'] / 1024:.0f} KiB/token)")
    print()
    hdr = (f"{'batch':>5} | {'out tok/s':>9} {'total tok/s':>11} | {'TTFT ms p50':>11} {'p95':>7} | "
           f"{'TPOT ms p50':>11} {'p95':>6} | {'E2E s p50':>9} {'p95':>6} {'p99':>6} | "
           f"{'KV peak%':>8} | {'GPU mem peak':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in res["runs"]:
        print(f"{r['batch_size']:>5} | {r['output_tok_per_s']:>9.1f} {r['total_tok_per_s']:>11.1f} | "
              f"{r['ttft_ms']['p50']:>11.1f} {r['ttft_ms']['p95']:>7.1f} | "
              f"{r['tpot_ms']['p50']:>11.2f} {r['tpot_ms']['p95']:>6.2f} | "
              f"{r['e2e_latency_s']['p50']:>9.2f} {r['e2e_latency_s']['p95']:>6.2f} {r['e2e_latency_s']['p99']:>6.2f} | "
              f"{r['kv_cache_usage_pct']['peak']:>8.1f} | {r['gpu_memory_peak_gib']:>8.2f} GiB")

    prof = res.get("kernel_profile")
    if "kernel_profile" in res:
        print()
        if not prof:
            print("GPU kernel time: profiler trace not found")
        else:
            print(f"GPU kernel time (profiled pass, batch {prof['batch_size']}; profiler adds overhead)")
            print(f"  kernel time {prof['kernel_time_ms']:.1f} ms over {prof['kernel_launches']:,} launches | "
                  f"memcpy/memset {prof['memcpy_memset_time_ms']:.1f} ms")
            print(f"  GPU busy {prof['gpu_busy_pct']:.1f}% of {prof['trace_span_ms']:.1f} ms trace span | "
                  f"{prof['kernel_time_per_output_token_ms'] * 1000:.1f} us kernel time per output token")
            print("  top kernels:")
            for k in prof["top_kernels"]:
                name = k["name"] if len(k["name"]) <= 90 else k["name"][:87] + "..."
                print(f"    {k['share_pct']:5.1f}%  {k['time_ms']:9.1f} ms  {k['calls']:>7,}x  {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    p.add_argument("--iters", type=int, default=3, help="timed repetitions per batch size")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--enforce-eager", action="store_true", help="disable CUDA graphs")
    p.add_argument("--device", type=int, default=0, help="GPU index for memory readings")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-profile", action="store_true", help="skip the GPU kernel-time pass")
    p.add_argument("--profile-batch-size", type=int, help="batch size for the profiled pass (default: largest)")
    p.add_argument("--top-kernels", type=int, default=10)
    p.add_argument("--keep-trace", action="store_true", help="keep the torch profiler trace files")
    p.add_argument("--json", help="write results to this JSON file")
    args = p.parse_args()

    res = asyncio.run(main_async(args))
    report(res)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved results to {args.json}")


if __name__ == "__main__":
    main()
