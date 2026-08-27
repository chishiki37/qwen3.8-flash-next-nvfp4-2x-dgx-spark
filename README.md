# Qwen3.8-Flash-Next NVFP4 on 2× NVIDIA DGX Spark (TP2, SGLang)

Measured deployment recipe for **Qwen3.8-Flash-Next** (`qwen4_exp` hybrid linear-attn/mamba + sparse-attn MoE with NEXTN speculative head, 262K context, vision-capable) quantized to **NVFP4** ([RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), 135.3 GB) on a CRS812-fabric pair of NVIDIA DGX Spark (GB10, SM121) nodes.

Built from tonyd2wild's day-0 recipe with a 2-patch SM121 image fix and an autoresearch serving sweep. Full procedure: [runbook-qwen38-flash-next-nvfp4.md](runbook-qwen38-flash-next-nvfp4.md).

## Headline numbers (measured on-box, 512 tok out, temp 0, thinking off, median of 3)

Winner config: recipe-locked — mem-fraction-static 0.80, NEXTN steps 3 / draft 4, 600K-token KV pin.

| Cell | Aggregate tok/s | TTFT s |
|---|---:|---:|
| Prose C1 | 48.1 | 0.166 |
| Prose C4 | 119.7 | 0.191 |
| Prose C8 | 126.1 | 0.196 |
| Code C1 | 63.8 | 0.172 |
| Code C4 | 163.0 | 0.197 |
| Code C8 | 159.9 | 0.211 |

- Matches the source recipe's "47 typical / 53–64 structured / 70 peak" claims (prose C1 48.1; code C1 63.8)
- Code decodes ~33% faster than prose — NEXTN acceptance is content-driven
- Sweep confirmed the locked config: lighter spec (2/3) −1.3%, memfrac 0.82 no gain
- On this pair, Qwen3.8-Flash-Next (TP2) out-throughputs GLM-5.3-Flash NVFP4 even at TP4 in every cell (see runbook comparison)

## Repo layout

- `runbook-qwen38-flash-next-nvfp4.md` — prerequisites, step-by-step, flag rationale, sweep + battery reference, troubleshooting (QSA guard, NCCL pin, NFS head trap)
- `scripts/` — parameterized TP2 launcher, sweep + battery drivers, stdlib-only harness, cache-flusher sidecar
- `results/` — sweep JSONL (3 arms, all completed) + winner battery JSON

## Credits

- Model: Qwen/Qwen3.8-Flash-Next · Quant: [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
- Day-0 recipe: tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark
- Campaign: Vikas Sridhar's CRS812 DGX Spark cluster, measured 2026-08-27
