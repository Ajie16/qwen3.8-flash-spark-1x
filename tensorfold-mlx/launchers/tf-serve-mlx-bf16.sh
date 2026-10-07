#!/bin/bash
# TensorFold serving for the OFFICIAL MLX 4-bit (group 32) pack on spark-2 :8899.
# Engine: ~/tensorfold-venv (TensorFold 0.6.1 + ~/patch_ttsilence.py + ~/patch_tt_reserve.py
#         + MiaAI-Lab's patches/0002-flash-next-v061.patch).
#
# Why this pack: Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP is what MiaAI-Lab's recipe serves. Same
# engine, same GPU, same box: EXL3 measured 805 tok/s prefill at 32k, this measures 2,425 - a 3.0x
# difference that is the quantisation format, not the server. It is also the only MLX family
# TensorFold accepts: {group_size: 32, bits: 4, mode: affine} + Qwen4ExpForConditionalGeneration.
# Every uncensored MLX pack is group 64, which the qwen4_exp CUDA family rejects outright.
#
# --- SSD / NVMe, aligned with MiaAI-Lab's scripts/config.sh -----------------------------
# --ple-on-ssd (his PLE_ON_SSD=1): the 29.8 GiB of n-gram tables stay on NVMe instead of being
#   mapped into a RAM that cannot hold them beside 75 GiB of weights. PLE is looked up every
#   decode step (forward.py:268 runs on both paths), so leaving them mapped made the kernel page
#   mid-token. Measured here: decode 40.4/34.7 -> 49.0/41.7, prefill 2,389 -> 2,430.
# TENSORFOLD_PREFILL_ROWS=2048 (Mia, and only when PLE_ON_SSD=1): his measurement is that
#   4,096-row prompt pieces run 10-30% slower from 5k to 16k tokens and ~3% slower at 31k with
#   the tables on NVMe. The variable itself comes from his patch (upstream 0.6.1 has no such
#   knob); unset, TensorFold picks 2,048 with vision anyway, so this only pins the value.
# TENSORFOLD_VISION_WORKSPACE_MIB=0 (Mia's value): the vision scratch comes from the system
#   reserve only while an image or video is being encoded, then goes back. Leaving it at the
#   12,288 this file used to set held ~12 GiB that the KV pool could not touch.
# TENSORFOLD_IMAGE_TOKENS / VIDEO_TOKENS=16384: the tokens all of a request's images (or video
#   frames) share, each single image capped at 4,096. Matches his VISION_MAX_IMAGES=50 intent.
# TENSORFOLD_SSD_NATIVE=1 is the default already (ssd_read.py:23): a batch of preads on C++
#   threads with the GIL released. Set explicitly so it cannot be inherited away.
#
# TENSORFOLD_MTP_COPY=1: Mia's default. The variable is ADDED BY HIS PATCH (multi.py:58) and
#   defaults to 0, so it is off unless set. It turns on copy drafts with compact MTP rows.
#   NOTE: an earlier version of this file removed this line after grepping upstream 0.6.1 and
#   finding no such variable - correct before the patch was applied, stale afterwards.
#
# --- other knobs -----------------------------------------------------------------------
# --thinking-budget: deliberately ABSENT. Upstream default is 0 = no limit (cli_args.py:58),
#   which is also Mia's configuration (he never passes the flag). The 4,096 this file used to
#   set was ours, not the model's: it pinned every xhigh reply to exactly 4,097 tokens and made
#   xhigh indistinguishable from "whatever fits". --max-tokens 32768 is the only ceiling now.
#   Dropping it also disables our own _think_budget clamp, which returns None when budget <= 0.
# --parallel 5 matches Mia. At 4, a fifth concurrent request queues and C=5 aggregate falls below
#   C=4 (84.6 vs 89.8); at 5 it rises to 92.1. Single-stream decode is unaffected.
# TENSORFOLD_MEMORY_RESERVE_GIB=4 vs his 2: the extra 2 GiB is headroom because this box also
#   runs other things. Spark unified memory freezes the machine instead of failing an allocation.
#
# Vision: bundled in this checkpoint (BF16 tower, attn.qkv fused), so there is deliberately NO
# TENSORFOLD_VISION_WEIGHTS here. The EXL3 pack needs one; this one must not be given it.
#
# Chinese MTP drafting needs no flag either. draft_vocab.txt inside the package is already the
# 102,089-id table (default 79,591 + Chinese extension 22,088 + corpus backfill 410), and
# weights.py:420 reads it via draft_token_ids("default"), which is the engine default.
#
# nvidia-smi -lgc is NOT used: it needs root and fails silently, so the clock was never capped.
#
# EXL3 rollback: /home/xujie/tf-serve.sh, untouched by this file.
nvidia-smi -rgc >/dev/null 2>&1 || true   # undo a cap some earlier run may have left; needs root
cd /home/xujie
export TENSORFOLD_MEMORY_RESERVE_GIB=4
export TENSORFOLD_PREFILL_ROWS=2048
export TENSORFOLD_VISION_WORKSPACE_MIB=0
export TENSORFOLD_IMAGE_TOKENS=16384
export TENSORFOLD_VIDEO_TOKENS=16384
export TENSORFOLD_SSD_NATIVE=1
export TENSORFOLD_MTP_COPY=1
exec /home/xujie/tensorfold-venv/bin/tensorfold serve /home/xujie/models/Vontra-Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --host 0.0.0.0 --port 8899 --name Qwen3.8-Flash-Local \
  --context 262144 --parallel 4 --kv-dtype bf16 \
  --vision --vision-max-images 16 --ple-on-ssd \
  --max-tokens 32768 --mtp-confidence 0.6