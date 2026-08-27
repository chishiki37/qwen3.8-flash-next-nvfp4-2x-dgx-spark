#!/bin/bash
# Launch Qwen3.8-Flash-Next TP2 winner (q0: memfrac 0.80, NEXTN steps3/draft4, 600K KV)
# and run the full battery on-box.
set -uo pipefail
WORKER=edgexpert-3b24
HEAD=aitopatom-cb98
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[$(date +%H:%M)] pre-launch drop_caches + flushers"
for n in "$WORKER" "$HEAD"; do
  timeout 30 ssh "$n" 'sync; cat /tmp/.spw | sudo -S bash -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; pkill -f cache_flusher 2>/dev/null; nohup /tmp/cache_flusher_spw.sh > /tmp/flusher.log 2>&1 &' || true
done

echo "[$(date +%H:%M)] launch worker (rank 1, q0)"
timeout 120 ssh "$WORKER" 'bash ~/launch_qwen38_param.sh 1' || exit 1
sleep 25
echo "[$(date +%H:%M)] launch head (rank 0, q0)"
timeout 120 ssh "$HEAD" 'bash ~/launch_qwen38_param.sh 0' || exit 1

echo "[$(date +%H:%M)] health wait"
timeout 2400 ssh "$HEAD" 'for i in $(seq 1 240); do
  curl -s -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen3.8-flash-next && { echo HEALTHY; exit 0; }
  docker ps --format "{{.Names}}" | grep -q sglang_qwen38 || { echo "CONTAINER GONE"; exit 4; }
  sleep 10
done; echo TIMEOUT; exit 3'
[ $? -eq 0 ] || { echo "engine not healthy"; exit 1; }

echo "[$(date +%H:%M)] running full battery on-box"
timeout 3600 ssh "$HEAD" 'BENCH_MODEL=qwen3.8-flash-next bash ~/battery_onbox.sh full /home/vikassridhar/battery_qwen_q0.json'
RC=$?
echo "[$(date +%H:%M)] battery rc=$RC"
timeout 60 scp "$HEAD:/home/vikassridhar/battery_qwen_q0.json" "$DIR/battery_qwen_q0.json" && echo "banked battery_qwen_q0.json"
timeout 30 ssh "$HEAD" 'docker logs sglang_qwen38 2>&1 | grep -iE "accept length|spec" | tail -3' > "$DIR/battery_qwen_q0_meta.txt" 2>/dev/null
for n in "$WORKER" "$HEAD"; do timeout 60 ssh "$n" 'docker rm -f sglang_qwen38 2>/dev/null' || true; done
echo DONE
