#!/bin/bash
# Qwen3.8-Flash-Next-FP4-Local on ONE DGX Spark, PLE n-gram table on NVMe.
#
# This is NVIDIA's own shape for the checkpoint, and sojufx's pinned recipe: TP=1, the B12X image,
# VLLM_PLE_TABLE_MEMORY=disk. Measured on this box by sojufx (sparkDash structured counting, thinking
# off): C1 66.17, C2 106.27, C4 166.36, follow-up TTFT 0.22 s.
#
# Why not TP=2 across both Sparks: the checkpoint's NVFP4-CSF scale format refuses expert parallel
# ("NotImplementedError: NVFP4-CSF supports TP without EP/DP"), so a second box only adds a
# cross-box all-reduce on every layer. Measured here, TP=2 decoded at ~30 tok/s against ~66 on one
# box. The model does not need two boxes anyway: 95.12 GiB becomes ~62.5 GiB resident once the
# 32.6 GiB PLE table lives on NVMe.
#
# --served-model-name carries both Qwen3.8-Flash-Local (what the AxonHub provider, the DSH profile
# and every client used against the previous TabbyAPI/TensorFold service) and
# qwen3.8-flash-next-fp4 (this deployment's own name), so swapping the engine under those clients
# needs no client change. Port 8899 and bind 10.100.65.1 also match that service.
#
# No --default-chat-template-kwargs: sojufx's recipe pinned enable_thinking=false, which made every
# reply skip reasoning. Dropping the override leaves the template's own default, and Qwen3.8's
# template treats an undefined enable_thinking as ON ("if enable_thinking is undefined or
# enable_thinking is true"), which is what the previous TensorFold service did. Clients can still
# turn thinking off per request with chat_template_kwargs. --reasoning-parser qwen3 turns the
# think block into reasoning_content.
#
# Expert parallel is absent for the same reason it fails: this format does not support it.
#
# NCCL settings follow NVIDIA's DGX Spark block rather than a hand-rolled RoCE setup; the first
# attempt here used NCCL_IB_HCA/NCCL_PROTO=Simple and lost concurrency badly.
set -uo pipefail

IMAGE=eugr/spark-vllm-b12x:latest
NAME=qwen38-fp4-tp1
MODEL_DIR=/home/xujie/models/Qwen3.8-Flash-Next-FP4-Local
CACHE=$HOME/.cache/nvidia-fp4-local

BIND_HOST="${BIND_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8001}"
# Deliberately unquoted at the call site: vLLM takes the names as separate argv entries, so
# "$SERVED" would arrive as one name containing a space and 404 every request.
SERVED="${SERVED:-Qwen3.8-Flash-Local qwen3.8-flash-next-fp4}"
GMU="${GMU:-0.80}"

IFACE=enP2p1s0f0np0

echo "[1/3] preflight"
[[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || { echo "  missing model: $MODEL_DIR"; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "  missing image: $IMAGE"; exit 1; }
echo "  model $(du -sh "$MODEL_DIR" | cut -f1), image ok"
python3 - "$MODEL_DIR" <<'PY'
import json, os, sys
d = json.load(open(os.path.join(sys.argv[1], "model.safetensors.index.json")))
missing = [v for v in set(d["weight_map"].values()) if not os.path.exists(os.path.join(sys.argv[1], v))]
print("  index references %d shards, %d missing" % (len(set(d["weight_map"].values())), len(missing)))
if missing:
    print("  REFUSING: first missing = %s" % missing[0]); raise SystemExit(1)
PY
free -g | awk '/^Mem/{printf "  mem total %s available %s GiB\n",$2,$7}'

echo "[2/3] reset container"
docker rm -f "$NAME" >/dev/null 2>&1 && echo "  removed old $NAME" || echo "  none"

mkdir -p "$CACHE"/{vllm,flashinfer,triton,b12x}

echo "[3/3] launch TP=1 on $BIND_HOST:$API_PORT (GMU $GMU)"
docker run -d --name "$NAME" --init --restart no \
    --gpus all --ipc host --network host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -v "$MODEL_DIR:/model:ro" \
    -v "$CACHE/vllm:/root/.cache/vllm" \
    -v "$CACHE/flashinfer:/root/.cache/flashinfer" \
    -v "$CACHE/triton:/root/.triton" \
    -v "$CACHE/b12x:/root/.cache/b12x" \
    -e HF_HUB_OFFLINE=1 \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_NET_PLUGIN=spcx \
    -e NCCL_PROTO=LL,LL128,Simple \
    -e NCCL_BUFFSIZE=2097152 \
    -e NCCL_CUMEM_ENABLE=0 \
    -e NCCL_MAX_NCHANNELS=16 \
    -e NCCL_MIN_NCHANNELS=16 \
    -e NCCL_P2P_LEVEL=SYS \
    -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
    -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
    -e TORCH_NCCL_USE_COMM_NONBLOCKING=0 \
    -e VLLM_PLE_TABLE_MEMORY=disk \
    -e B12X_DISK_BACKEND=io_uring \
    -e CUTE_DSL_ARCH=sm_121a \
    -e SAFETENSORS_FAST_GPU=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
    -e VLLM_USE_AOT_COMPILE=1 \
    -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e B12X_POLICY_MODE=auto \
    -e VLLM_MXFP8_LM_HEAD=1 \
    -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 \
    -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
    -e B12X_W4A16_A4_PREFILL=1 \
    --entrypoint /usr/local/bin/vllm "$IMAGE" \
    serve /model \
    --host "$BIND_HOST" --port "$API_PORT" \
    --served-model-name $SERVED \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --quantization modelopt_mixed \
    --block-size 16 \
    --load-format b12x \
    --max-model-len 262144 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"standard"}' \
    --gdn-decode-kernel b12x \
    --linear-backend b12x \
    --moe-backend b12x \
    --no-enable-flashinfer-autotune \
    --mm-encoder-tp-mode data \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice \
    --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
    --gpu-memory-utilization "$GMU"

sleep 4
docker ps --format "  {{.Names}}  {{.Status}}" | grep "$NAME" || echo "  container did not stay up"
echo "  logs: docker logs -f $NAME"
