# 03 — NVFP4 KV cache port: 4.83× the KV pool, where it pays and where it doesn't

**Date:** 2026-08-28 · **Rig:** cb98 + 3b24 (TP2, 200G RoCEv2) · **Image:** `sglang-qwen38fn:sm121-nvfp4kv`

## What we tested

MiaAI-Lab published a packed-FP4 KV cache stack for this model (FP4 values +
per-block FP8 scales, custom Triton pack/dequant kernels) claiming ~4.7× the
token pool. We ported their kernel design onto our SM121 image — surgical
patches, not a wholesale fork — and measured whether the capacity buys
anything real on 2× DGX Spark.

## The port

Seven patches on top of `sglang-qwen38fn:sm121-qsa`, all inert unless
`--kv-cache-dtype nvfp4` is set:

1. `qwen_sparse_attn_backend.py` — QSA gather paths (chunked prefill, decode
   verify) read packed FP4 + scale buffers and dequantize gathered rows;
   **trtllm-gen sparse decode gated off for FP4 pools** (it reads BF16 pool
   views directly) — FP4 routes through the FA2/Triton varlen path instead
2. `fp4_kv_cache_quant_method.py` — route `nvfp4` to a plain-dequant method
   (no FP8 dequant workspace, no native-FP4 decode path)
3. `server_args.py` — allow nvfp4 KV for QSA hybrid models
4. `pool_configurator.py` — don't reserve the FP8 workspace share of the FP4
   cell size (the QSA method allocates none)
5. `memory_pool.py` — NVFP4 store uses on-device `k_scales_gpu` (host→CUDA
   `torch.tensor` is illegal during decode CUDA-graph capture)
6. `sparse_attn.py` — SM121 Triton can't `tl.dot` fp8e4nv; upcast K/V/Q to
   fp32 before the dots
7. New modules: `qsa_nvfp4_kv.py` (packed-view helpers, compact+dequant,
   history gather), `qsa_fa_fallback.py` (Triton varlen attention for SM121)

Two boot failures found and fixed along the way: a heredoc terminator line
leaked into `qsa_nvfp4_kv.py` during staging (import-time `NameError`), and
the trtllm dispatch above (`k_buffer.shape` on a `None` BF16 view — the first
prefill crashed until the gate went in).

## Boot result

| Metric | bf16 (winner, report 02) | NVFP4 KV |
|---|---|---|
| KV dtype | bf16 | `float4_e2m1fn_x2` (packed) |
| Pool | 600,000 tokens (pinned) | **2,895,680 tokens** |
| Pool ratio | 1.0× | **4.83×** |
| Full-attn layers K+V | ~29 GB | 4.66 + 4.66 GB |
| GDN layers K+V | ~4 GB | 0.39 + 0.39 GB |
| Residual GPU mem after pool | ~12 GB | 21.3 GB |
| Max context / running reqs | 262K / 8 | 262K / 6 |

Sanity verified (arithmetic probe correct), no NVRM or retraction events
through every test below.

## Short-form battery (512-token outputs): NVFP4 KV loses

| Cell | bf16 agg | FP4-KV agg | Δ |
|---|---|---|---|
| C1 prose | 48.1 | 47.3 | −1.7% |
| C4 prose | 119.7 | 111.4 | −7.0% |
| C8 prose | 126.1 | 104.3 | −17.3% |
| C1 code | 63.8 | 58.1 | −9.0% |
| C4 code | 163.0 | 167.4 | +2.7% |
| C8 code | 159.9 | 146.8 | −8.2% |

Why: per-gather dequant tax (2–9%), and the FP4 config's
`max_running_requests=6` caps C8 (bf16 ran all 8) — that's the entire C8
regression. At short lengths the pool is never the bottleneck, so the tax is
all you see.

## Long-context ladder: the capacity win is real

bf16 arithmetic: 600K ÷ 260K = **2 concurrent requests max**. FP4 pool holds
11; the engine cap allows 6. Ladder (256 output tokens, medians):

| Rung | Load | TTFT med | TTFT max | Per-stream decode | Result |
|---|---|---|---|---|---|
| R1 | C4 @ 66K | 78 s | 115 s | (contaminated¹) | 4/4 served |
| R2 | C4 @ 131K | 141 s | 217 s | (contaminated¹) | 4/4 served |
| R3 | C2 @ 260K | 177 s | 236 s | (contaminated¹) | 2/2 served |
| R4 | **C6 @ 260K** | 411 s | 731 s | (contaminated¹) | **6/6 served, 1.56M tokens resident** |

¹ With chunked prefill, early-finishing streams decode while later streams
still prefill, so ladder decode numbers understate true decode speed. Clean
C1 numbers below.

**Prefill throughput is the long-context wall:** ~2.1–2.3K tok/s aggregate
(QSA indexer + FP4 pack). A 260K-token prompt costs ~2 min solo, ~12 min
when six queue up. Plan batch workloads around it.

## Clean single-stream decode at long context

C1, 512-token essay outputs, solo requests (no prefill interleaving), identical probe on both configs:

| Context | FP4-KV decode | bf16 decode | FP4-KV TTFT | bf16 TTFT |
|---|---:|---:|---:|---:|
| 66K | 13.7 tok/s | 16.9 tok/s | 24.6 s | 35.4 s |
| 131K | 13.1 tok/s | 17.1 tok/s | 53.8 s | 49.0 s |
| 260K | 15.0 tok/s | 16.9 tok/s | 145.4 s | 104.7 s |

**The long-context decode collapse is stack-wide, not FP4-specific.** bf16
drops from 48–64 tok/s (short context) to ~17 tok/s at long context too;
FP4-KV adds only ~18% more tax (~14 vs ~17). Both are flat across 66K→260K —
the sparse top-k gather does not scale with history length in either path.
The ~3.5× short→long slowdown is the QSA indexer/verify cost at long context
on GB10, common to both KV dtypes.

Prefill (single runs, noisy): ~1.8–2.7K tok/s on both configs — no clear
FP4 penalty, no clear win.

## Verdict

**Capacity win, speed loss, domain-specific.** The port works end-to-end and
delivers exactly the claimed pool ratio (4.83×):

- **Use bf16 (the report-02 winner) for interactive/short-form serving.**
  NVFP4 KV loses 2–17% there — dequant tax plus the `max_running_requests=6`
  cap — and the pool is never the constraint at short lengths.
- **Use NVFP4 KV when the working set exceeds 600K tokens:** many concurrent
  long-context requests (6×260K served vs a hard ceiling of 2 on bf16), or
  single contexts that must coexist with real concurrency. Per-stream decode
  at long context costs ~18% more than bf16 (~14 vs ~17 tok/s) — but that
  regime is already 3.5× slower than short-form on either config, so the FP4
  tax is not the thing that hurts.
- **Budget prefill, not decode, for long-context work:** ~2.1–2.5K tok/s
  aggregate means a 260K-token prompt costs ~2 min solo and ~12 min when six
  queue up.

Cross-model note: with this port, Qwen3.8-Flash-Next now fields a 2.89M-token
pool — closing most of GLM-5.3-Flash's fp8 pool advantage (5.75M at TP4)
while keeping the throughput lead from report 02.

## Reproduce

- Image patches: `scripts/nvfp4-kv-port/` (Dockerfile + patch chain)
- Launch: `QWEN_KVDTYPE=nvfp4 QWEN_KVTOK=auto bash launch_qwen38_param.sh <rank>`
  (kv token cap must be `auto` — the pool sizes from memfrac 0.80)
- Ladder: `scripts/qwen_nvfp4kv_ladder.py` (stdlib only, runs on-box)
- Raw data: `results/battery_qwen_nvfp4kv.json`, `results/qwen_nvfp4kv_ladder.jsonl`,
  `results/c1probe_*.jsonl`
