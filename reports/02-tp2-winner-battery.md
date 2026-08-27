# Report 02 — TP2 Winner Battery (recipe-locked config)

**Date:** 2026-08-27 · **Topology:** TP2 (cb98 head + 3b24) · **Config:** q0 — mem-fraction-static 0.80, NEXTN steps 3 / draft 4, 600K-token KV pin, 262K context, CUDA graphs to batch 8

## Question

How does the winner hold up across concurrency (C1/C4/C8) and content type (prose vs code)?

## Method

Two-pass protocol (warmup discarded, then measure), median of 3, 512 output tokens, temp 0, thinking off, on-box localhost.

## Results

| Cell | Aggregate tok/s | Per-stream tok/s | TTFT s |
|---|---:|---:|---:|
| Prose C1 | 48.09 | 48.76 | 0.166 |
| Prose C4 | 119.71 | 31.11 | 0.191 |
| Prose C8 | 126.09 | 29.71 | 0.196 |
| Code C1 | 63.82 | 65.10 | 0.172 |
| Code C4 | **163.03** | 44.45 | 0.197 |
| Code C8 | 159.93 | 34.09 | 0.211 |

## Findings

1. **Code decodes ~33% faster than prose** (C1: 63.8 vs 48.1; C8 aggregate 159.9 vs 126.1). NEXTN draft acceptance is content-driven — structured/low-entropy tokens verify better. Code C1 lands inside the source's "53–64 structured / 70 peak" claim band; prose C1 (48.1) matches their "47 typical".
2. **TTFT is flat and fast** — 0.17–0.21 s across every cell including C8, the best TTFT profile measured in this campaign (vs GLM TP4's 0.22–0.35 s).
3. **Concurrency scaling:** 2.5× aggregate at C4, 2.6× at C8 (max-running-requests 6 queues 2 at C8).
4. Harness consistency: prose C1 48.1 vs sweep probe 46.7 — warm-state uplift, same class.

## Verdict

Validated as the TP2 reference. Cross-model note: at this concurrency-free single-stream level, Qwen3.8-Flash-Next on 2 nodes out-throughputs GLM-5.3-Flash NVFP4 even at TP4 in every cell (48.1 vs 40.0 prose C1; 159.9 vs 110.9 code C8) — GLM counters with its much larger fp8 KV pool (see the GLM repo's KV ladder report).
