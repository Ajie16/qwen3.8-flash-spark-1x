#!/bin/bash
# TensorFold serving for the UNCENSORED MLX 4-bit (group 32) pack on spark-2 :8899.
#
# Why this pack: it is the only uncensored MLX checkpoint at group_size 32. Every other
# uncensored MLX pack found on the Hub is group 64, and the qwen4_exp CUDA family rejects
# those outright. yikoudazhong's conversion mirrors the official TensorFold layout exactly
# (same tensor names, shards, shapes, dtypes) from orcarouter's abliterated FP8 release,
# which is the same abliteration our EXL3 pack derives from.
#
# ENGINE: TensorFold python-0.6 branch HEAD (ed78d6f), reported as 0.6.6. NOT the v0.6.6 tag.
#   Measured against 0.6.1 on identical warm runs: 74.6 vs 70.7 tok/s median, 49.0 vs 51.7
#   ms/round, so the upgrade is worth keeping.
#   The tag is 12 commits behind and lacks the upstreamed copy-drafts feature
#   (families/qwen4_exp/cuda/copy_drafts.py, commit 6bf4caf) that Mia's 0.6.1 patch used to
#   provide us as TENSORFOLD_MTP_COPY. Installing the tag would have been a regression.
#   Source is built from a tarball fetched through spark-0d97's proxy, because this box cannot
#   reach github.com; see the upgrade note at the bottom.
#
# ACCESS CONTROL -- the one deliberate difference from tf-serve-mlx.sh:
#   The template binds --host 0.0.0.0. This box deliberately serves the model on the CX-7
#   point-to-point link only (10.100.65.1), with AxonHub as the sole route for everything
#   else, so the model service is unreachable from the LAN and the tailnet. Binding 0.0.0.0
#   would silently reopen that. Do not "fix" this back to 0.0.0.0.
#
# --- SSD / NVMe, aligned with MiaAI-Lab's scripts/config.sh -----------------------------
# --ple-on-ssd: the 29.8 GiB of n-gram tables stay on NVMe instead of being mapped into a RAM
#   that cannot hold them beside 75 GiB of weights. PLE is looked up every decode step, so
#   leaving them mapped made the kernel page mid-token. Measured on this box with the official
#   pack: decode 40.4/34.7 -> 49.0/41.7, prefill 2,389 -> 2,430.
# TENSORFOLD_PREFILL_ROWS=2048: only meaningful with PLE_ON_SSD. 4,096-row pieces run 10-30%
#   slower from 5k to 16k tokens and ~3% slower at 31k with the tables on NVMe. Native in
#   upstream 0.6.x (was Mia's patch on 0.6.1); unset, TensorFold picks 2,048 with vision anyway.
# TENSORFOLD_VISION_WORKSPACE_MIB=0: the vision scratch comes from the system reserve only
#   while an image or video is being encoded, then goes back. Leaving it at 12,288 held ~12 GiB
#   the KV pool could not touch.
# TENSORFOLD_IMAGE_TOKENS / VIDEO_TOKENS=16384: tokens all of a request's images (or video
#   frames) share, each single image capped at 4,096.
# TENSORFOLD_SSD_NATIVE=1: a batch of preads on C++ threads with the GIL released (default).
#
# COPY DRAFTS: on by default in this build -- copy_drafts.py reads TENSORFOLD_COPY_DRAFTS and
#   0/off/false/no turns it off, unset leaves it on. The old TENSORFOLD_MTP_COPY=1 line is
#   GONE: Mia's 0.6.1 patch read that name, upstream renamed it, and the old line would now be
#   a silent no-op. Do not reintroduce it.
#   Measured: copy drafts on 74.6 tok/s median vs off 74.5 over 8 warm repeats of the same
#   deterministic case, so they cost nothing when the reply does not repeat earlier text. Leave
#   the default alone.
#
# COLD START -- READ BEFORE BENCHMARKING THIS BOX:
#   The first request after a restart runs at roughly 61 ms/round against a warm 49 ms/round on
#   the same deterministic case (164 rounds either way), i.e. ~20% slow. A whole-suite pass puts
#   that penalty on whichever case happens to go first, which is how an earlier A/B of 0.6.1
#   against 0.6.6 produced a confident but completely wrong "0.6.6 is 17% slower" reading.
#   Discard a warmup pass, then repeat one deterministic case and compare medians -- warm runs
#   hold within 0.5% of each other (48.8-49.1 ms/round over 7 consecutive runs).
#
#
# SAMPLING IS PINNED to the checkpoint's recommendation, by
# patches/kernel/server_pin_sampling.py.patch (apply with patches/apply_pin_sampling.py).
#   Without it App.sampling_for reads the client's numbers first, so a client sending temperature 0
#   or 0.7 silently overrides the pack's generation_config.json (temperature 1.0 / top_k 20 /
#   top_p 0.95) and one conversation answers differently per client. The patch makes the checkpoint
#   win; `seed` is still the request's, because it controls reproducibility rather than the
#   distribution. Set TENSORFOLD_PIN_SAMPLING=0 to restore the per-request behaviour -- a
#   server-side variable on purpose, so benchmarks that need greedy have it and clients do not.
#   The patch lives in the venv, so a TensorFold upgrade wipes it; re-run the apply script.
# --- other knobs -----------------------------------------------------------------------
# --thinking-budget deliberately ABSENT: upstream default is 0 = no limit, which is also Mia's
#   configuration. A 4,096 budget pinned every xhigh reply to exactly 4,097 tokens.
# --parallel 5 matches Mia. At 4, a fifth concurrent request queues and C=5 aggregate falls
#   below C=4 (84.6 vs 89.8); at 5 it rises to 92.1. Single-stream decode is unaffected.
# TENSORFOLD_MEMORY_RESERVE_GIB=4 vs Mia's 2: the extra 2 GiB is headroom because this box runs
#   other things. Spark unified memory freezes the machine instead of failing an allocation.
# --mtp-confidence 0.6 is the optimum for THIS box, measured by sweeping 0.3/0.45/0.6/0.75:
#   decode means 45.6 / 49.2 / 50.2 / 50.0 tok/s (that sweep predates the cold-start correction
#   above, so its absolute values are low, but all four arms carried the same penalty and the
#   ordering holds). Lower thresholds do deepen the chain (prose drafts per round 498 -> 1146)
#   but halve acceptance (49.2% -> 23.6%), so the product falls. See ~/conf-sweep.txt.
#
# Vision: bundled in this checkpoint (BF16 tower, attn.qkv fused), so there is deliberately NO
# TENSORFOLD_VISION_WEIGHTS here. The EXL3 pack needs one; this one must not be given it.
#
# DRAFT VOCAB: the checkpoint ships none, so the engine's own table is used. That file was
#   REPLACED with our 102,089-id Chinese-aware table after the upgrade; upstream's is 79,591
#   English/code-only. Any future reinstall silently reverts it, which would cost the Chinese
#   acceptance this box was tuned for. The saved copy is ~/draft_vocab.engine-0.6.1.txt.
#
# Rollback: the previous engine is at ~/tensorfold-venv-0.6.1.bak (whole venv, patches included).
#   Swap it back over ~/tensorfold-venv and restart. The EXL3 service rollback is
#   ~/tf-serve-exl3.sh plus the TabbyAPI launcher in the recipe repo.
nvidia-smi -rgc >/dev/null 2>&1 || true   # undo a cap some earlier run may have left; needs root
cd /home/xujie
export TENSORFOLD_MEMORY_RESERVE_GIB=4
export TENSORFOLD_PREFILL_ROWS=2048
export TENSORFOLD_VISION_WORKSPACE_MIB=0
export TENSORFOLD_IMAGE_TOKENS=16384
export TENSORFOLD_VIDEO_TOKENS=16384
export TENSORFOLD_SSD_NATIVE=1
exec /home/xujie/tensorfold-venv/bin/tensorfold serve \
  /home/xujie/models/yikoudazhong-TF-4bit-MTP \
  --host 10.100.65.1 --port 8899 --name Qwen3.8-Flash-Local \
  --context 262144 --parallel 5 --kv-dtype int8 \
  --vision --vision-max-images 16 --ple-on-ssd \
  --max-tokens 32768 --mtp-confidence 0.6
