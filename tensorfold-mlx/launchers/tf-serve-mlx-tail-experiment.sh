#!/bin/bash
# TensorFold serving for the OFFICIAL MLX 4-bit (group 32) pack on spark-2 :8899.
# Engine: ~/tensorfold-venv (TensorFold 0.6.1 + ~/patch_ttsilence.py + ~/patch_tt_reserve.py).
#
# Why this pack: Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP is what MiaAI-Lab's recipe serves. Same
# engine, same GPU, same box: EXL3 measured 805 tok/s prefill at 32k, this measures 2,425 - a 3.0x
# difference that is the quantisation format, not the server. It is also the only MLX family
# TensorFold accepts: {group_size: 32, bits: 4, mode: affine} + Qwen4ExpForConditionalGeneration.
# Every uncensored MLX pack is group 64, which the qwen4_exp CUDA family rejects outright.
#
# --ple-on-ssd is the single biggest serving knob here, and it is worth +20% decode / +2% prefill:
# the 29.8 GiB of n-gram tables do not fit in RAM beside the weights, and PLE is looked up every
# decode step (forward.py:268 runs on both paths), so leaving them mapped pages them from disk
# mid-token. Measured without it: decode 40.4/34.7, prefill 2,389. With it: 49.0/41.7, 2,430.
#
# --parallel 5 matches MiaAI-Lab's default. At 4, a fifth concurrent request queues and C=5
# aggregate falls below C=4 (84.6 vs 89.8); at 5 it rises to 92.1. Drop to 4 if four lanes is the
# ceiling you want - single-stream decode is unaffected.
#
# Vision: bundled in this checkpoint (BF16 tower, attn.qkv fused), so there is deliberately NO
# TENSORFOLD_VISION_WEIGHTS here. The EXL3 pack needs one; this one must not be given it.
#
# Chinese MTP drafting needs no flag either. draft_vocab.txt inside the package is already the
# 102,089-id table (default 79,591 + Chinese extension 22,088 + corpus backfill 410), and
# weights.py:420 reads it via draft_token_ids("default"), which is the engine default - the same
# code path EXL3 used.
#
# NOT set, because they do nothing on 0.6.1: nvidia-smi -lgc (needs root, silently fails - the
# clock was never capped) and TENSORFOLD_MTP_COPY (no such variable exists anywhere in the package).
#
# EXL3 rollback: /home/xujie/tf-serve.sh, untouched by this file.
nvidia-smi -rgc >/dev/null 2>&1 || true   # undo a cap some earlier run may have left; needs root
cd /home/xujie
export TENSORFOLD_MEMORY_RESERVE_GIB=4
exec /home/xujie/tensorfold-venv/bin/tensorfold serve /home/xujie/models/Vontra-Qwen3.8-Flash-Next-MLX-4bit-MTP-TAIL \
  --host 0.0.0.0 --port 8899 --name Qwen3.8-Flash-Local \
  --context 262144 --parallel 5 --kv-dtype int8 \
  --vision --vision-max-images 16 --ple-on-ssd \
  --thinking-budget 4096 --max-tokens 32768 --mtp-confidence 0.6