#!/bin/bash
# Qwen3.8-Flash-Next NVFP4 TP2 mini-sweep driver (gateway-side; benches on-box on cb98).
# Arms: q0 baseline (recipe-locked), q1 lighter spec (steps2/draft3), q2 memfrac 0.82.
set -uo pipefail
HEAD=aitopatom-cb98
WORKER=edgexpert-3b24
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/sweep_qwen_results.jsonl"

health_wait() { # on-box health wait, 40 min cap
  timeout 2400 ssh "$HEAD" 'for i in $(seq 1 240); do
    curl -s -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen3.8-flash-next && { echo HEALTHY; exit 0; }
    docker ps --format "{{.Names}}" | grep -q sglang_qwen38 || { echo "CONTAINER GONE"; exit 4; }
    sleep 10
  done; echo TIMEOUT; exit 3'
}

teardown() {
  for n in "$WORKER" "$HEAD"; do
    timeout 60 ssh "$n" 'docker rm -f sglang_qwen38 2>/dev/null || true' || true
  done
}

run_arm() { # name memfrac steps draft kvtok
  local name="$1" mf="$2" st="$3" dr="$4" kv="$5"
  echo "[$(date +%H:%M)] === arm $name (memfrac=$mf steps=$st draft=$dr kvtok=$kv)"
  teardown
  for n in "$WORKER" "$HEAD"; do
    timeout 30 ssh "$n" 'sync; cat /tmp/.spw | sudo -S bash -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; pkill -f cache_flusher 2>/dev/null; nohup /tmp/cache_flusher_spw.sh > /tmp/flusher.log 2>&1 &' || true
  done
  echo "[$(date +%H:%M)] launch worker (rank 1)"
  timeout 120 ssh "$WORKER" "QWEN_MEMFRAC=$mf QWEN_STEPS=$st QWEN_DRAFT=$dr QWEN_KVTOK=$kv bash ~/launch_qwen38_param.sh 1" || { echo "arm $name: worker launch fail"; return 1; }
  sleep 25
  echo "[$(date +%H:%M)] launch head (rank 0)"
  timeout 120 ssh "$HEAD" "QWEN_MEMFRAC=$mf QWEN_STEPS=$st QWEN_DRAFT=$dr QWEN_KVTOK=$kv bash ~/launch_qwen38_param.sh 0" || { echo "arm $name: head launch fail"; return 1; }
  echo "[$(date +%H:%M)] health wait"
  local hs; hs=$(health_wait)
  echo "health: $hs"
  [ "$hs" = "HEALTHY" ] || { echo "arm $name: engine not healthy ($hs)"; teardown; return 1; }
  local probe; probe=$(timeout 300 ssh "$HEAD" 'BENCH_MODEL=qwen3.8-flash-next bash ~/probe_onbox.sh' 2>/dev/null)
  echo "probe: $probe"
  local kvline; kvline=$(timeout 30 ssh "$HEAD" 'docker logs sglang_qwen38 2>&1 | grep -m1 -E "max_total_num_tokens|KV Cache is allocated"' 2>/dev/null)
  local spec; spec=$(timeout 30 ssh "$HEAD" 'docker logs sglang_qwen38 2>&1 | grep -m1 -iE "accept length|spec.*metric" ' 2>/dev/null)
  python3 - "$OUT" "$name" "$mf" "$st" "$dr" "$kv" "$probe" "$kvline" <<'EOF'
import json, sys
out, name, mf, st, dr, kv, probe, kvline = sys.argv[1:9]
rec = {"config": name, "memfrac": mf, "steps": st, "draft": dr, "kvtok": kv,
       "probe_raw": probe.strip(), "kv_raw": kvline.strip(),
       "ts": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds")}
try:
    rec["probe"] = json.loads(probe)
except Exception:
    rec["probe"] = None
with open(out, "a") as f:
    f.write(json.dumps(rec) + "\n")
EOF
  echo "[$(date +%H:%M)] arm $name banked"
}

run_arm q0_baseline 0.80 3 4 600000
run_arm q1_spec23   0.80 2 3 600000
run_arm q2_mf082    0.82 3 4 600000
teardown
echo "SWEEP DONE"
