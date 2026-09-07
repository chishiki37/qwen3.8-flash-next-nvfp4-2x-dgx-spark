# Qwen3.8-Flash-Next NVFP4 on 2× NVIDIA DGX Spark

Measured deployment recipe for **Qwen3.8-Flash-Next** — `qwen4_exp` hybrid linear-attn/mamba + sparse-attn MoE with a NEXTN speculative head, 51B-parameter PLE embedding table, vision-capable, 262K context — quantized to NVFP4 ([RadixArk quant](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), 135.3 GB), served with SGLang on a 2-node DGX Spark (GB10, SM121) pair over a 200G RoCEv2 fabric.

Measured 2026-08-27/28: day-0 image patched for SM121, a 3-arm autoresearch sweep on the serving levers, a full winner battery, and a ported packed-FP4 KV cache that multiplies the token pool 4.83×. Start with the [runbook](runbook-qwen38-flash-next-nvfp4.md) to reproduce; each optimization has its own report below.

---

## Headline numbers

**Single-stream decode:** **48.1 tok/s prose · 63.8 tok/s code** (code decodes ~33% faster — NEXTN acceptance is content-driven)

**Aggregate throughput, 8 concurrent streams:** **126.1 tok/s prose · 159.9 tok/s code**, TTFT ≤ 0.21 s in every cell

**KV capacity:** 600,000 tokens pinned (bf16) at mem-fraction-static 0.80 — the production config from the source recipe's KV-ladder study (1.05M reachable at 0.82 but OOMs under load). With the [NVFP4 KV port](reports/03-nvfp4-kv-port.md): **2,895,680 tokens (4.83×)** — six concurrent 260K-context requests where bf16 caps at two

All numbers: on-box, 512 output tokens, temperature 0, thinking off, median of 3.

---

## The optimization reports

| # | Report | What it tested | Outcome |
|---|---|---|---|
| 01 | [TP2 autoresearch sweep](reports/01-tp2-autoresearch-sweep.md) | NEXTN depth × mem-fraction-static, 3 arms | Recipe-locked config wins — both alternative levers lose or tie |
| 02 | [TP2 winner battery](reports/02-tp2-winner-battery.md) | C1/C4/C8 × prose/code | 159.9 tok/s aggregate (code C8); flat 0.17–0.21 s TTFT |
| 03 | [NVFP4 KV cache port](reports/03-nvfp4-kv-port.md) | Packed-FP4 KV (MiaAI-Lab kernel design) on SM121 | 4.83× pool (2.89M tokens), 6×260K concurrent; loses 2–17% short-form, ~18% long-context decode tax — capacity-bound workloads only |

## Key findings, in one list

1. **The locked recipe is already optimal** — lighter NEXTN speculation loses 1.3%, higher mem-fraction ties. A confirmed negative result on both levers
2. **Matches the source claims:** "47 typical" → 48.1 prose C1; "53–64 structured / 70 peak" → 63.8 code C1
3. **Two SM121 image patches are mandatory** (QSA sparse-attn guard + NCCL 2.30.7 pin) — stock day-0 image dies in warmup on GB10
4. **Agent-safety stack matters:** thinking-off + radix-off + pytorch sampling prevents the token-0 loop; keep agent temps ≤ 0.7
5. **Cross-model:** on this pair, Qwen3.8-Flash-Next (TP2) out-throughputs GLM-5.3-Flash NVFP4 even at TP4 in every throughput cell — GLM counters with a 5.75M-token fp8 pool
6. **NVFP4 KV is a capacity lever, not a speed lever:** 4.83× the pool and 6×260K concurrency, but −2…−17% on short-form and ~18% long-context decode tax. Long-context decode collapses to ~14–17 tok/s on BOTH KV dtypes (QSA indexer cost on GB10) — the port doesn't cause it

## Repo layout

```
runbook-qwen38-flash-next-nvfp4.md   procedure: prereqs → launch → flags → troubleshooting
reports/                              one report per optimization (above)
scripts/                              TP2 launcher, sweep + battery drivers, stdlib-only
                                      harness, cache-flusher sidecar
results/                              raw JSONL/JSON for every completed arm
```

## Credits

- Model: Qwen/Qwen3.8-Flash-Next · Quant: [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
- Day-0 recipe: [tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark)
- Same model on a **single** Spark (MiaAI-Lab vLLM recipe, verified 2026-09-06/07): [qwen3.8-flash-next-single-spark-miaai-verification](https://github.com/chishiki37/qwen3.8-flash-next-single-spark-miaai-verification) — ~35/101/142 tok/s C1/C4/C8; their 48.7 C1 claim lands at 70–76% here
- Campaign: Vikas Sridhar's CRS812 DGX Spark cluster, measured 2026-08-27
