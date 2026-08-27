# Report 01 — TP2 Autoresearch Sweep

**Date:** 2026-08-27 · **Topology:** TP2 (aitopatom-cb98 head + edgexpert-3b24) · **Engine:** SGLang day-0 `qwen38flashnext` image + SM121 patches (`sglang-qwen38fn:sm121-qsa`)

## Question

The source recipe ships a fully locked serve config (claims 47 typical / 70 peak tok/s). Do any of the main serving levers — speculative depth, static memory fraction — improve on it for our fleet?

## Method

- On-box probes against `http://127.0.0.1:8000` on the head (no tailnet hop in timing)
- Prose prompt, 512 output tokens, temperature 0, thinking off, median of 3 runs
- Full teardown + `drop_caches` + cache-flusher before every arm; worker-first launch (rank 1, head ~25 s later)
- Constant across arms: NVFP4 + flashinfer_cutlass GEMM, page-size 64, mamba extra-buffer scheduler, NEXTN speculative head, 600K-token KV pin, 262K context, agent-safety stack (thinking off + radix off + pytorch sampling)

## Config matrix & results

| Arm | mem-fraction-static | NEXTN steps / draft tokens | Decode tok/s | TTFT s |
|---|---:|---|---:|---:|
| **q0 — recipe-locked baseline** | 0.80 | 3 / 4 | **46.74** | 0.165 |
| q1 — lighter speculation | 0.80 | 2 / 3 | 46.14 | 0.158 |
| q2 — more static memory | 0.82 | 3 / 4 | 46.31 | 0.167 |

Per-arm runs (decode tok/s): q0 [45.83, 48.27, 46.74] · q1 [46.14, 46.46, 39.77] · q2 [46.60, 41.22, 46.31]. KV pool identical across arms: 600,000 tokens bf16 (K 3.43 GB + V 3.43 GB per rank).

## Findings

1. **The recipe-locked config is the winner.** Unlike GLM-5.3 (where sweeps found MTP3 > MTP4), Qwen's NEXTN defaults — 3 steps / 4 draft tokens — are already optimal: q1's lighter speculation loses 1.3%.
2. **mem-fraction 0.82 buys nothing** at a fixed 600K pin (46.31 vs 46.74, within noise) — and the source recipe's KV-ladder study reports the 1.05M-token 0.82 config OOMs under load, so 0.80 stays.
3. **46.74 tok/s matches the source's "47 typical" claim** within 0.6% — the recipe transfers to our fleet intact (after the two SM121 image patches: QSA guard + NCCL 2.30.7).

## Verdict

**q0 adopted unchanged as the production config** — a confirmed negative result on both sweep levers is a valid sweep outcome. Proceeds to the winner battery → [Report 02](02-tp2-winner-battery.md).
