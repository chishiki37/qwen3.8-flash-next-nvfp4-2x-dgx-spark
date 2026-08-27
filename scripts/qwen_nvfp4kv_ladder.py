#!/usr/bin/env python3
"""Qwen3.8-Flash-Next NVFP4-KV long-context ladder (stdlib only, streaming).

Proves the capacity win of the packed-FP4 KV pool (2.9M tokens) where the
bf16 pool (600K) arithmetic caps out: bf16 can hold at most 2x262K-context
requests; FP4 can hold 11 (capped here by max_running_requests=6).

Rungs: C4@64K, C4@128K, C2@250K, C6@250K. Prompt targets are conservative
(under context_length=262144) to avoid allow_auto_truncate; actual
prompt_tokens are recorded from usage.
"""
import json, sys, threading, time, urllib.request
from statistics import median

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-flash-next"
OUT = "/home/vikassridhar/qwen_nvfp4kv_ladder.jsonl"

PARA = ("The history of distributed computing is a history of memory walls. "
        "Every generation of hardware moved the bottleneck: from registers to caches, "
        "from caches to main memory, from main memory to the network, and from the network "
        "back again to the unified fabric that joins processor and storage. Inference on "
        "small clusters repeats this arc at desk scale, where the page cache, the driver "
        "allocator, and the KV slab negotiate a border that no specification records. ")
TOKENS_PER_COPY = 83  # calibrated in longprefill_gate.py

RUNGS = [
    # (name, concurrency, target_prompt_tokens, max_output_tokens)
    ("R1_C4_64K", 4, 63000, 256),
    ("R2_C4_128K", 4, 125000, 256),
    ("R3_C2_250K", 2, 248000, 256),
    ("R4_C6_250K", 6, 248000, 256),
]

def build_prompt(target_tokens):
    copies = max(1, target_tokens // TOKENS_PER_COPY)
    return ("Context study. Read the passage below once, then answer the question at the end.\n\n"
            + PARA * copies
            + "\n\nQuestion: In one short paragraph, what does the passage say moves the "
              "bottleneck in each hardware generation?")

def one_request(prompt, max_tokens, timeout=2400):
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; toks = 0; usage = None; err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                ch = chunk.get("choices") or []
                if ch and ch[0].get("delta", {}).get("content"):
                    if first is None:
                        first = time.time() - t0
                    toks += 1
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    total = time.time() - t0
    decode = (toks - 1) / (total - first) if (toks > 1 and first and total > first) else None
    return {"err": err, "tokens": toks, "wall_s": round(total, 2),
            "ttft_s": round(first, 3) if first is not None else None,
            "decode_tps": round(decode, 2) if decode else None,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens")}

def run_rung(name, conc, target, max_out):
    prompt = build_prompt(target)
    print(f"[{time.strftime('%H:%M:%S')}] {name}: conc={conc} target~{target}tok "
          f"prompt_chars={len(prompt)}", flush=True)
    results = [None] * conc
    def worker(i):
        results[i] = one_request(prompt, max_out)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = round(time.time() - t0, 2)
    ok = [r for r in results if not r["err"] and r["decode_tps"]]
    bad = [r["err"] for r in results if r["err"]]
    agg = round(sum(r["completion_tokens"] or 0 for r in ok) / wall, 2) if (ok and wall) else None
    rung = {"rung": name, "concurrency": conc, "target_prompt_tokens": target,
            "wall_s": wall, "ok": len(ok), "errors": bad,
            "agg_out_tps": agg,
            "per_stream_decode_med": round(median(r["decode_tps"] for r in ok), 2) if ok else None,
            "ttft_med_s": round(median(r["ttft_s"] for r in ok), 3) if ok else None,
            "ttft_max_s": round(max(r["ttft_s"] for r in ok), 3) if ok else None,
            "prompt_tokens_med": int(median(r["prompt_tokens"] for r in ok)) if ok and all(r["prompt_tokens"] for r in ok) else None,
            "runs": results}
    print(f"  -> ok={len(ok)}/{conc} wall={wall}s agg_out={agg} tok/s "
          f"per_stream_med={rung['per_stream_decode_med']} "
          f"ttft_med={rung['ttft_med_s']}s ttft_max={rung['ttft_max_s']}s "
          f"prompt_tokens={rung['prompt_tokens_med']}", flush=True)
    if bad:
        print(f"  ERRORS: {bad}", flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rung) + "\n")
    return rung

def main():
    open(OUT, "w").close()
    for name, conc, target, max_out in RUNGS:
        try:
            run_rung(name, conc, target, max_out)
        except Exception as e:
            print(f"  RUNG CRASH: {type(e).__name__}: {e}", flush=True)
    print("LADDER DONE", flush=True)

if __name__ == "__main__":
    main()
