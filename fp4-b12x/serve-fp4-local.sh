#!/bin/bash
# Serve nvidia/Qwen3.8-Flash-Next-FP4-Local on one DGX Spark, with the PLE n-gram table on NVMe.
#
# NVIDIA's reference run for this checkpoint is TP=1 on a single Spark in their own
# eugr/spark-vllm-b12x image, with VLLM_PLE_TABLE_MEMORY=disk. This uses the stock
# vllm/vllm-openai:qwen38-flash-next image instead, because Docker Hub is unreachable from
# spark-ac8f and the two mirrors that list eugr/spark-vllm-b12x serve the manifest but never
# deliver the blobs. The stock image does have --quantization modelopt_mixed, which is the part
# that decides whether these weights load at all; what it cannot give is the B12X kernels, so
# --load-format b12x / --gdn-decode-kernel b12x / --linear-backend b12x / --moe-backend b12x and
# B12X_W4A16_A4_PREFILL are all absent here. Speed will not be comparable to NVIDIA's numbers.
#
# --kv-cache-dtype is bfloat16, not the fp8 NVIDIA's command uses: this build's QSA sparse
# attention refuses a non-BF16 main KV cache ("Qwen3.8-Flash-Next QSA requires a BF16 main KV
# cache"). The recipe repo's files/patch_qsa_fp8_kv.py patches that, at the cost of bf16 KV being
# about twice the memory of fp8 for the same token count. Measured below, so compare against
# NVIDIA's "1M tokens of FP8 KV" knowing this is the bf16 number.
#
# NVIDIA warns the NVFP4 scale factors in this checkpoint use a new lossless compression format
# that upstream vLLM does not support yet. That is the open question this run answers.
#
# Everything else follows NVIDIA's command. The weights are bind-mounted read-only from
# ~/models rather than the HF cache, so nothing is copied or re-downloaded.
set -uo pipefail

MODEL_DIR=/home/xujie/models/Qwen3.8-Flash-Next-FP4-Local
IMAGE=vllm/vllm-openai:qwen38-flash-next
NAME=qwen38-fp4
HOST_IP=10.100.65.1
PORT=8899

echo "[1/3] 前置检查"
[[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || { echo "  ✗ 权重缺失: $MODEL_DIR"; exit 1; }
echo "  权重索引 ✓  $(du -sh "$MODEL_DIR" | cut -f1)"
free -g | awk '/^Mem/{printf "  内存 total %s used %s available %s\n",$2,$3,$7}'

echo "[2/3] 清理旧容器"
docker rm -f "$NAME" >/dev/null 2>&1 && echo "  移除旧 $NAME" || echo "  无旧容器"

echo "[3/3] 启动"
docker run -d --name "$NAME" \
  --gpus all --ipc=host --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864:67108864 \
  -e VLLM_PLE_TABLE_MEMORY=disk \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_MXFP8_LM_HEAD=1 \
  -v "$MODEL_DIR:/model:ro" \
  --entrypoint /bin/bash "$IMAGE" -c '
exec vllm serve /model \
  --host '"$HOST_IP"' --port '"$PORT"' \
  --served-model-name qwen3.8-flash-next \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --quantization modelopt_mixed \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --no-enable-flashinfer-autotune \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice
'

sleep 3
docker ps --format "  {{.Names}}  {{.Status}}" | grep "$NAME" || echo "  ✗ 容器没起来"
echo "  日志: docker logs -f $NAME"
