#!/bin/bash
# EXL3 uncensored pack on its OWN, UNPATCHED TensorFold engine.
#
# Why a second engine: the patches in ~/tensorfold-venv were written for the MLX 4-bit path
# (MiaAI-Lab's 0002-flash-next-v061 for copy drafts / SSD read-ahead / video, plus four of ours
# for the thinking-budget edge cases). This pack is served by stock v0.6.1 instead, so nothing
# in the MLX work can perturb it. Verified pristine: _MIN_ANSWER, ping_stop, scheduler guard,
# indexed_prefill_rows and TENSORFOLD_MTP_COPY all absent from this tree.
#
#   engine  ~/tensorfold-venv-exl3   stock v0.6.1 (4137074)   -> this launcher
#   engine  ~/tensorfold-venv        v0.6.1 + patches          -> tf-serve-mlx*.sh
#
# The one non-stock thing here is data, not code: draft_vocab.txt is the 102,089-id table
# (upstream 79,591 + 22,088 Chinese + 410 corpus backfill). Measured on this box: Chinese decode
# +25-40% and MTP acceptance 32% -> 57%, English unchanged. Revert by copying
# ~/tensorfold-patches/draft_vocab.txt.upstream-0.6.1 over it.
#
# DO NOT add --thinking-budget here. This engine has stock _think_budget (cuda/server.py:510),
# which returns None only when the budget is <= 0; a budget at or above --max-tokens truncates
# the forced </think> to nothing and every reply arrives as content: null. Ours is not set, so
# reasoning length is the model's own choice, bounded only by --max-tokens 32768.
#
# Vision: the EXL3 pack's own tower is EXL3-quantised with split q/k/v, which 0.6.1 rejects, so
# TENSORFOLD_VISION_WEIGHTS must point at the separate BF16 tower. The MLX pack bundles its tower
# and must NOT be given this variable.
#
# NO --ple-on-ssd, unlike the MLX launcher. TensorFold refuses it here: "--ple-on-ssd reads the MLX
# checkpoint's n-gram tables; an EXL3 pack maps its own table from its file, so drop --ple-on-ssd".
# The two packs reach the same place by different routes, and the flag is MLX-only. An earlier
# version of this file carried it over from the MLX launcher and the server refused to start.
#
# No nvidia-smi -lgc: it needs root and fails silently, so it never capped anything.
#
# MLX rollback: /home/xujie/tf-serve-mlx.sh (patched engine).
cd /home/xujie
export TENSORFOLD_MEMORY_RESERVE_GIB=4
export TENSORFOLD_VISION_WEIGHTS=/home/xujie/models/qwen38-vision-tower/vision-tower-bf16.safetensors
export TENSORFOLD_VISION_WORKSPACE_MIB=0
export TENSORFOLD_IMAGE_TOKENS=16384
exec /home/xujie/tensorfold-venv-exl3/bin/tensorfold serve /home/xujie/models/Qwen3.8-Flash-Next-Uncensored-EXL3-3bpw \
  --host 0.0.0.0 --port 8899 --name Qwen3.8-Flash-Local \
  --context 262144 --parallel 4 --kv-dtype int8 \
  --vision --vision-max-images 16 \
  --max-tokens 32768 --mtp-confidence 0.6