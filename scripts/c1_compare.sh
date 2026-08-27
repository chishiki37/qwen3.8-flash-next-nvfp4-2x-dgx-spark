#!/bin/bash
# Long-context C1 decode comparison: NVFP4 KV vs bf16 KV (same probe, same prompt).
set -uo pipefail
WORKER=edgexpert-3b24
HEAD=aitopatom-cb98

echo "[$(date +%H:%M)] === FP4 probe (live server) ==="
python3 /home/vikassridhar/qwen_nvfp4kv_c1probe.py
cp /home/vikassridhar/qwen_nvfp4kv_c1probe.jsonl /home/vikassridhar/c1probe_fp4.jsonl

echo "[$(date +%H:%M)] === swap to bf16 ==="
for n in "$WORKER" "$HEAD"; do ssh "$n" 'docker rm -f sglang_qwen38 2>/dev/null' || true; done
ssh "$WORKER" 'bash ~/launch_qwen38_param.sh 1' || exit 1
sleep 25
ssh "$HEAD" 'bash ~/launch_qwen38_param.sh 0' || exit 1
echo "[$(date +%H:%M)] health wait (bf16)"
for i in $(seq 1 240); do
  curl -s -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen3.8-flash-next && { echo HEALTHY; break; }
  docker ps --format "{{.Names}}" | grep -q sglang_qwen38 || { echo "CONTAINER GONE"; exit 4; }
  sleep 10
done

echo "[$(date +%H:%M)] === bf16 probe ==="
python3 /home/vikassridhar/qwen_nvfp4kv_c1probe.py
cp /home/vikassridhar/qwen_nvfp4kv_c1probe.jsonl /home/vikassridhar/c1probe_bf16.jsonl
echo "[$(date +%H:%M)] COMPARE DONE"
