#!/bin/bash
# Production launcher for the Qwen3.8-Flash-Next uncensored EXL3 pack on one DGX Spark.
#
# Thin wrapper over serve.sh that pins the deployment choices which are NOT upstream defaults:
#
#   1. The uncensored pack. serve.sh's MODEL_DIR default is the turboderp/censored pack; the state
#      directory's symlink gets rewritten from MODEL_DIR on every launch, so it must be set here or
#      the running model silently changes.
#   2. The feat/hybrid-draft worktree (EXL3_SPEC_SRC). serve.sh defaults to master, whose
#      exllamav3_ext*.so does not contain the fused GR kernel, so EXL3_GR_TUNED=1 would be a no-op.
#   3. PROFILE=concurrent + NGRAM_RAM=true. The concurrent profile streams the n-gram table from
#      NVMe (about 2.2 s of fixed cost per cold request, measured by fitting intercept + slope over
#      8k..240k); forcing RAM trades 18 GiB for that back. 88 GiB estimate, fits.
#   4. VISION=true. The recipe defaults it off. The EXL3 pack's own tower is EXL3-quantised with
#      split q/k/v, which exllamav3 rejects, so the fork needs the separate BF16 tower.
#   5. HOST=0.0.0.0 + DISABLE_AUTH=true. Note serve.sh flips DISABLE_AUTH to false automatically
#      when HOST is not loopback; this is a LAN-trusted box, matching how it was already running.
#
# Everything else (the GB10 kernel knobs, the pruned-head width, the fused GR kernel) comes from
# env.sh and is documented there with its measurements.
#
# Usage:  bash exllamav3-tabby/serve-local.sh
# Stop:   kill $(pgrep -f "tabbyAPI/main.py" | head -1)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-Uncensored-EXL3-3bpw}"
export EXL3_SPEC_SRC="${EXL3_SPEC_SRC:-$HOME/qwen38-exl3/exllamav3-spec}"

if [[ ! -d "$EXL3_SPEC_SRC/exllamav3" ]]; then
  echo "warning: \$EXL3_SPEC_SRC ($EXL3_SPEC_SRC) has no exllamav3/ package." >&2
  echo "         EXL3_GR_TUNED and the hybrid-draft switches will not exist. Run with EXL3_SPEC_SRC='' to use master." >&2
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "error: no pack at \$MODEL_DIR ($MODEL_DIR). Set it to an EXL3 pack directory." >&2
  exit 1
fi

export SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Local}"
export PROFILE="${PROFILE:-concurrent}"
export NGRAM_RAM="${NGRAM_RAM:-true}"
export HOST="${HOST:-0.0.0.0}"
export DISABLE_AUTH="${DISABLE_AUTH:-true}"
export VISION="${VISION:-true}"

exec bash "$HERE/serve.sh"
