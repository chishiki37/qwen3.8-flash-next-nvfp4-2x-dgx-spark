#!/usr/bin/env bash
# Parameterized Qwen3.8-Flash-Next NVFP4 TP2 launcher for SGLang (fleet port of
# tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark; GID_INDEX omitted per fleet rule).
# Usage: launch_qwen38_param.sh <0|1>
# Env levers (must match across ranks):
#   QWEN_MEMFRAC  mem-fraction-static (default 0.80)
#   QWEN_STEPS    speculative-num-steps (default 3)
#   QWEN_DRAFT    speculative-num-draft-tokens (default 4)
#   QWEN_KVTOK    max-total-tokens (default 600000)
set -euo pipefail
NODE_RANK="${1:?usage: launch_qwen38_param.sh <0|1>}"
QWEN_MEMFRAC="${QWEN_MEMFRAC:-0.80}"
QWEN_STEPS="${QWEN_STEPS:-3}"
QWEN_DRAFT="${QWEN_DRAFT:-4}"
QWEN_KVTOK="${QWEN_KVTOK:-600000}"

IMAGE="sglang-qwen38fn:sm121-qsa"
NAME="sglang_qwen38"
MODEL_PATH="/models/qwen3.8-flash-next-nvfp4"
CACHE_HOST_PATH="/var/tmp/qwen38-sglang-cache"
DIST_ADDR="10.10.10.14:29531"
PORT="8000"

case "$NODE_RANK" in
  0) MODEL_HOST_PATH="/home/vikassridhar/models/qwen3.8-flash-next-nvfp4" ;;
  1) MODEL_HOST_PATH="/var/tmp/models-cb98/qwen3.8-flash-next-nvfp4" ;;
  *) echo "rank must be 0-1" >&2; exit 2 ;;
esac

test -f "$MODEL_HOST_PATH/config.json" || { echo "missing $MODEL_HOST_PATH/config.json" >&2; exit 3; }
mkdir -p "$CACHE_HOST_PATH"
docker rm -f "$NAME" 2>/dev/null || true

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --memory 110g --memory-swap 110g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=10.10.10.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=1 \
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
    python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tp-size 2 --nnodes 2 --node-rank "$NODE_RANK" \
    --dist-init-addr "$DIST_ADDR" \
    --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass \
    --page-size 64 \
    --mamba-scheduler-strategy extra_buffer --mamba-track-interval 64 \
    --speculative-algorithm NEXTN --speculative-num-steps "$QWEN_STEPS" \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens "$QWEN_DRAFT" \
    --enable-linear-replayssm-spec \
    --chunked-prefill-size 4096 --max-running-requests 6 \
    --context-length 262144 --max-total-tokens "$QWEN_KVTOK" \
    --mem-fraction-static "$QWEN_MEMFRAC" \
    --allow-auto-truncate --ple-offload-embedding \
    --cuda-graph-max-bs 8 --disable-cuda-graph-padding \
    --disable-radix-cache --sampling-backend pytorch \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser auto --tool-call-parser qwen3_coder

echo "launched $NAME rank=$NODE_RANK memfrac=$QWEN_MEMFRAC steps=$QWEN_STEPS draft=$QWEN_DRAFT kvtok=$QWEN_KVTOK tp2"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited; docker logs $NAME" >&2; exit 1; }
