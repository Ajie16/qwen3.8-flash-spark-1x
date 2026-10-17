#!/bin/bash
# Qwen3.8-Flash-Next-FP4-Local on TWO DGX Sparks, TensorFold-style TP=2 over CX-7, PLE on NVMe.
#
# Matches NVIDIA's own single-Spark command and sojufx's pinned recipe, with the two changes the
# operator asked for: tensor-parallel across both boxes, and VLLM_PLE_TABLE_MEMORY=disk on each so
# the 32.6 GiB n-gram table stays on NVMe instead of competing with weights and KV cache.
#
# Why TP=2 is the only shape that fits: the checkpoint is 95.12 GiB, of which ~62.5 GiB is model
# weights and ~32.6 GiB is the PLE table (shards 15-21). One GB10 shows 121.69 GiB, and a measured
# TP=1 attempt with the table resident died on OOM with 97 GiB budgeted against 95.12 GiB of
# weights. Split two ways the weights are ~47.6 GiB a node and everything fits with room for KV.
#
# --gpu-memory-utilization defaults to 0.70 rather than NVIDIA's 0.80: spark-0d97 (rank 1) runs the
# CLM encoder on 8091 plus the CLM service, holding ~17.9 GiB of unified memory, and vLLM refused
# 0.80 there with "Free memory on device cuda:0 (86.74/121.69 GiB)". At TP=2 each rank holds ~47.6
# GiB of weights, so 0.70 still leaves ~35 GiB for KV cache. Override with GMU=... if the box is free.
#
# Role 1 must be started first and goes headless; role 0 then serves the API.
#   ./serve-fp4-tp2.sh 1     # on spark-0d97 (10.100.65.2)
#   ./serve-fp4-tp2.sh 0     # on spark-ac8f (10.100.65.1)
set -uo pipefail

ROLE="${1:?usage: $0 0|1   (0=head serving API, 1=worker headless)}"

IMAGE=eugr/spark-vllm-b12x:latest
NAME="${NAME:-qwen38-fp4-tp2}"
# Overridable so the same launcher serves the offline/abliterated variants, which are the same
# checkpoint with a handful of tensors replaced -- see docs/fp4-b12x-deployment.md. Their remaining
# shards are symlinked to this directory, so only the differing ones cost disk.
MODEL_DIR="${MODEL_DIR:-/home/xujie/models/Qwen3.8-Flash-Next-FP4-Local}"
CACHE=$HOME/.cache/nvidia-fp4-local

HEAD_IP=10.100.65.1
WORKER_IP=10.100.65.2
MASTER_PORT=50000
API_PORT=8899
# Unquoted at the call site: vLLM takes each name as its own argv entry, so "$SERVED" would
# arrive as one name with a space in it and 404 every request.
SERVED="${SERVED:-Qwen3.8-Flash-Local qwen3.8-flash-next-fp4}"

# Expert parallel is deliberately absent: this checkpoint's NVFP4-CSF format refuses it outright
# ("NotImplementedError: NVFP4-CSF supports TP without EP/DP"), so the MoE path is tensor-split like
# everything else and every layer still needs a cross-box all-reduce. That is why TP=2 decodes at
# ~30 tok/s where one Spark gets ~66: the model already fits on one node once the PLE table is on
# NVMe, so the second box buys KV capacity, not speed.
#
# Parameters and env follow eugr/spark-vllm-docker's recipes/qwen3.8-flash-next-nvfp4-cluster.yaml
# (the dual-DGX-Spark vLLM project, 2.4k stars), which is the only published cluster recipe for this
# exact checkpoint. Two things there were missing here and matter:
#   VLLM_ENABLE_ROCE_ALLREDUCE=1 + VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB -- vLLM's own RoCE all-reduce
#     path instead of generic NCCL collectives. This is the per-layer cost that kept TP=2 at +23%.
#   draft_tensor_parallel_size=2 -- the MTP drafter also spans both ranks. Without it the drafter's
#     work is not split, which undercuts the point of TP for speculative decoding.
# It also sets PLE to "shared" rather than "disk", uses GMU 0.7 for both ranks, max-num-seqs 16 and
# num_speculative_tokens 4. --enable-expert-parallel is absent there too, which confirms the CSF
# format's refusal of EP is real and not a local mistake.
#
# --kv-cache-memory caps the KV pool at 24 GiB (25769803776 B) instead of letting
# --gpu-memory-utilization hand out whatever is left. At 0.70 vLLM allocated 39.52 GiB / 3.45M tokens
# (13.17x concurrency) when six streams need only 6 x 262,144 x ~12.0 KiB = 18.0 GiB, and rank1 was
# left with 3.0 GiB of system memory. 24 GiB gives 2.45M tokens = 9.34x, and returned ~10 GiB per node.
# Override with KVCACHE=<bytes>; vLLM logs the value to use under "Replace gpu_memory_utilization
# config with --kv-cache-memory=..." each start.
#
# --max-num-batched-tokens must stay 4096. 8192 and 16384 abort in B12X's cross-rank preparation
# with "preparation ranks reached incompatible tuning boundaries"; clearing the compile cache does not
# change it. Patching that assertion (apply_tuning_patch.py keeps the patched file in b12x-patch/)
# only moves the failure to flashinfer/experimental/b12x/preparation/session.py's "tuning
# consolidation does not match pending races" -- it is a chain of cross-rank invariants, not one
# hardcoded size, so the size is not safely raisable by patching.
#
# NCCL uses the workspace recipe's proven block (.env IB_HCA="=roceP2p1s0f0"), not NVIDIA's spcx
# one. Two earlier attempts were both wrong and both fell back to TCP:
#   attempt 1: NCCL_IB_DISABLE=0 but IB_HCA without the leading '=' (prefix match, wrong device),
#              NCCL_PROTO=Simple (worst latency for the small per-layer messages)  -> 34 tok/s
#   attempt 2: NVIDIA README's NCCL_IB_DISABLE=1 + NCCL_NET_PLUGIN=spcx, and no NCCL_SOCKET_IFNAME
#              -> the spcx plugin does not exist in this image at all (no *spcx*, no libnccl-net*,
#              no nets/ directory), so it was ignored and NCCL picked an interface itself -> 30 tok/s
# 205 us per all-reduce is TCP-over-the-wrong-NIC latency. RoCE should be 10-20 us, which is the
# difference between 30 tok/s and the ~104 tok/s the compute split can support.
#
# The container also needs --device /dev/infiniband and --cap-add SYS_NICE. Without the device node
# the container cannot see uverbs/rdma_cm at all and NCCL falls back to TCP no matter how the HCA
# variables are set: measured here as all four roceP2p1s0f0 port_* counters staying at 0 while TCP
# sessions filled 10.100.65.1:10.100.65.2. Both hosts do expose /dev/infiniband (uverbs0-3, umad0-3,
# rdma_cm); the passthrough was simply missing.
#
# The leading '=' in NCCL_IB_HCA means exact-match only; without it NCCL does prefix matching and
# can bind a different device. NCCL_SOCKET_IFNAME pins the bootstrap/socket path to the same CX-7
# port. LL128 is the low-latency protocol for small messages. NCCL_NET_PLUGIN is deliberately gone:
# this image ships no net plugin.
IFACE=enP2p1s0f0np0
IB_HCA=roceP2p1s0f0
IB_GID_INDEX=3

echo "[1/3] preflight (role $ROLE)"
[[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || { echo "  missing model: $MODEL_DIR"; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "  missing image: $IMAGE"; exit 1; }
echo "  model $(du -sh "$MODEL_DIR" | cut -f1), image ok"
free -g | awk '/^Mem/{printf "  mem total %s available %s GiB\n",$2,$7}'

echo "[2/3] reset container"
docker rm -f "$NAME" >/dev/null 2>&1 && echo "  removed old $NAME" || echo "  none"

mkdir -p "$CACHE"/{vllm,flashinfer,triton,b12x}

if [[ "$ROLE" == "0" ]]; then
    BIND_HOST=$HEAD_IP
    HEADLESS=""
else
    BIND_HOST=127.0.0.1
    HEADLESS="--headless"
fi

# KV is sized for six full-length streams rather than the 40 GiB vLLM hands out at 0.70. Six streams
# of 262,144 tokens is 1,572,864 tokens; at the ~10.3 KiB/token this model reports that is ~15.5 GiB,
# so these leave ~18 GiB each. rank1 also carries the CLM service (17.9 GiB on 8091) and at 0.70 the
# box went 5 GiB into swap and had paged out 36 GB across start-up, so it gets the smaller budget.
# rank1 shares the box with the CLM service (17.9 GiB on 8091), so it gets a smaller budget: at
# 0.70 vLLM wanted 85.18 GiB there and the start-up aborted with "Free memory on device
# (80.46/121.69 GiB) on startup is less than the desired GPU memory utilization". Dropping the page
# cache before start helps (77 -> 84 GiB) but the checkpoint route then refills it, so the check sees
# ~80 GiB again; 0.64 (77.9 GiB) is the value that fits reliably.
if [[ "$ROLE" == "0" ]]; then GMU="${GMU:-0.70}"; else GMU="${GMU:-0.64}"; fi

echo "[3/3] launch rank $ROLE (bind $BIND_HOST${HEADLESS:+, headless})"
docker run -d --name "$NAME" --init --restart no \
    --gpus all --ipc host --network host --shm-size=16g \
    --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \
    --device /dev/infiniband:/dev/infiniband \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -v "$MODEL_DIR:/model:ro" \
    -v "$CACHE/vllm:/root/.cache/vllm" \
    -v "$CACHE/flashinfer:/root/.cache/flashinfer" \
    -v "$CACHE/triton:/root/.triton" \
    -v "$CACHE/b12x:/root/.cache/b12x" \
    -e HF_HUB_OFFLINE=1 \
    -e NCCL_DEBUG=INFO \
    -e NCCL_DEBUG_SUBSYS=INIT,NET \
    -e NCCL_IB_DISABLE=0 \
    -e NCCL_IB_HCA==roceP2p1s0f0 \
    -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_SOCKET_IFNAME=enP2p1s0f0np0 \
    -e NCCL_PROTO=LL128 \
    -e NCCL_NET_GDR_LEVEL=SYS \
    -e GLOO_SOCKET_IFNAME=enP2p1s0f0np0 \
    -e TP_SOCKET_IFNAME=enP2p1s0f0np0 \
    -e VLLM_ENABLE_ROCE_ALLREDUCE=1 \
    -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB \
    -e NCCL_BUFFSIZE=1048576 \
    -e NCCL_MAX_CTAS=8 \
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
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --distributed-executor-backend mp \
    --nnodes 2 \
    --node-rank "$ROLE" \
    --master-addr "$HEAD_IP" \
    --master-port "$MASTER_PORT" \
    $HEADLESS \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --quantization modelopt_mixed \
    --block-size 16 \
    --load-format b12x \
    --max-model-len 262144 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 4096 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_tensor_parallel_size":2}' \
    --gdn-decode-kernel b12x \
    --linear-backend b12x \
    --moe-backend b12x \
    --no-enable-flashinfer-autotune \
    --mm-encoder-tp-mode data \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
    --gpu-memory-utilization "$GMU" \
    --kv-cache-memory "${KVCACHE:-25769803776}"

sleep 4
docker ps --format "  {{.Names}}  {{.Status}}" | grep "$NAME" || echo "  container did not stay up"
echo "  logs: docker logs -f $NAME"
