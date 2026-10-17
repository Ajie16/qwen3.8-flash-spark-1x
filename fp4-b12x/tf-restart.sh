#!/bin/bash
# Restart the TP=2 pair, wait for the API, and print what it came up with.
#
# Every vLLM tuning knob here is start-up only, and a warm start costs ~370 s. Measured breakdown:
#   134 s  b12x ROUTE -- reads 95 GiB with O_DIRECT and routes it through Python. Irreducible: a
#          page-cache prewarm changed it from 126 s to 134 s, so the reads bypass the cache entirely.
#    95 s  CUDA graph capture (PIECEWISE 9 + FULL 3, decode and prefill, on both ranks). --fast
#          cuts this by capturing only sizes 1,2,4,6.
#   107 s  KV cache allocation, engine bring-up, API start.
#    30 s  container start and imports.
#     3 s  kernel tuning, once the b12x compile cache is warm. Clearing that cache costs ~170 s on
#          the next start (3309 gdn_prefill compilations), so do not clear it to "fix" anything.
#
# This wraps a restart into one command that reports what the new config actually produced, so a
# 6-minute wait does not need to be watched.
#
# Usage:
#   ./tf-restart.sh                          # current settings
#   ./tf-restart.sh --ple ram                # PLE table in RAM instead of NVMe
#   ./tf-restart.sh --gmu0 0.62 --gmu1 0.58  # explicit per-rank GPU memory fractions
#   ./tf-restart.sh --seqs 6                 # max concurrent sequences
#   ./tf-restart.sh --nbt 4096               # max-num-batched-tokens (B12X accepts 4096 only on TP=2;
#                                            # 8192 and 16384 both fail with "preparation ranks
#                                            # reached incompatible tuning boundaries", even with the
#                                            # compile cache cleared)
#   ./tf-restart.sh --model /path/to/checkpoint   # serve a different checkpoint tree (same 21-shard
#                                                 # layout; the abliterated variants share 19 shards
#                                                 # with the official one via symlinks)
#   ./tf-restart.sh --fast                    # cap CUDA graph capture sizes, ~60-80 s faster start
# The sudo password, when needed for drop_caches and /dev/shm cleanup, is read from
# ${SUDO_PASS_FILE:-$HOME/sudo-pass} on each host. That file is deliberately not in this repo: it is
# host-local, mode 600, and referenced only by path. If it is absent both steps are skipped with a
# printed note and the script still works -- it just stops reclaiming the ~14 GiB that /dev/shm and
# the page cache would otherwise hold.
set -uo pipefail

# Resolve against this script, not a hardcoded path: a stale absolute path here silently ran a
# different copy of serve-fp4-tp2.sh, so edits to GMU looked like they had no effect.
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_SCRIPT=/home/xujie/serve-fp4-tp2.sh
REMOTE_HOST=spark-2

PLE=""
GMU0=""
GMU1=""
SEQS=""
NBT=""
FAST=""
MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ple)  PLE="$2"; shift 2 ;;
        --gmu0) GMU0="$2"; shift 2 ;;
        --gmu1) GMU1="$2"; shift 2 ;;
        --seqs) SEQS="$2"; shift 2 ;;
        --nbt)  NBT="$2"; shift 2 ;;
        --fast) FAST=1; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
done

# Push this directory's launcher to the head node before doing anything else, so the two ranks always
# run the same file. Rank 1 runs $SCRIPTS_DIR/serve-fp4-tp2.sh and rank 0 runs $REMOTE_SCRIPT; when
# only one of them was edited the ranks silently diverged (different PLE modes both start fine, and
# the only evidence is the weights-size line in each log: 38 GB for disk, 56 GB for shared).
echo "[cfg] syncing launcher to $REMOTE_HOST:$REMOTE_SCRIPT"
scp -q "$SCRIPTS_DIR/serve-fp4-tp2.sh" "$REMOTE_HOST:$REMOTE_SCRIPT" || {
    echo "  FAILED to copy $SCRIPTS_DIR/serve-fp4-tp2.sh to $REMOTE_HOST:$REMOTE_SCRIPT"
    exit 1
}
echo "  synced"

if [[ -n "$PLE" || -n "$GMU0" || -n "$GMU1" || -n "$SEQS" || -n "$NBT" || -n "$FAST" ]]; then
    echo "[cfg] applying overrides on $REMOTE_HOST"
    ssh "$REMOTE_HOST" "python3 - <<'PY'
import re
from pathlib import Path
p = Path('$REMOTE_SCRIPT')
s = p.read_text()
if '$PLE':
    s = re.sub(r'-e VLLM_PLE_TABLE_MEMORY=\w+', '-e VLLM_PLE_TABLE_MEMORY=$PLE', s)
if '$GMU0':
    s = re.sub(r'(ROLE\" == \"0\" \]; then GMU=\"\\$\{GMU:-)[0-9.]+', r'\\g<1>$GMU0', s)
if '$GMU1':
    s = re.sub(r'(else GMU=\"\\$\{GMU:-)[0-9.]+', r'\\g<1>$GMU1', s)
if '$SEQS':
    s = re.sub(r'--max-num-seqs \d+', '--max-num-seqs $SEQS', s)
if '$NBT':
    s = re.sub(r'--max-num-batched-tokens \d+', '--max-num-batched-tokens $NBT', s)
if '$FAST':
    # Capture only the batch sizes actually used instead of vLLM's default 9-decode/9-prefill grid.
    # This trades a little decode headroom for ~60-80 s off start-up, which matters when iterating.
    s = s.replace(
        '''--compilation-config '{"pass_config":{"fuse_act_quant":true}}' ''',
        '''--compilation-config '{"pass_config":{"fuse_act_quant":true},"cudagraph_capture_sizes":[1,2,4,6],"cudagraph_num_of_warmups":0}' ''')
p.write_text(s)
print('  ple=%s gmu0=%s gmu1=%s seqs=%s' % (
    re.search(r'VLLM_PLE_TABLE_MEMORY=(\w+)', s).group(1),
    re.search(r'ROLE\" == \"0\" \]; then GMU=\"\\$\{GMU:-([0-9.]+)', s).group(1),
    re.search(r'else GMU=\"\\$\{GMU:-([0-9.]+)', s).group(1),
    re.search(r'--max-num-seqs (\d+)', s).group(1)) + ' nbt=%s' % (
    re.search(r'--max-num-batched-tokens (\d+)', s).group(1),))
PY"
    # Apply the same override to the LOCAL copy too. rank 1 runs $SCRIPTS_DIR/serve-fp4-tp2.sh and
    # rank 0 runs the remote one; patching only the remote copy left the two ranks on different PLE
    # modes, which is silent -- both start, and only the weights-size line in each log differs.
    python3 - "$SCRIPTS_DIR/serve-fp4-tp2.sh" "$PLE" "$GMU0" "$GMU1" "$SEQS" "$NBT" <<'PYPATCH'
import re, sys
from pathlib import Path
path, ple, gmu0, gmu1, seqs, nbt = sys.argv[1:7]
p = Path(path)
s = p.read_text()
if ple:  s = re.sub(r'-e VLLM_PLE_TABLE_MEMORY=\w+', '-e VLLM_PLE_TABLE_MEMORY=' + ple, s)
if gmu0: s = re.sub(r'(ROLE" == "0" \]; then GMU="\$\{GMU:-)[0-9.]+', r'\g<1>' + gmu0, s)
if gmu1: s = re.sub(r'(else GMU="\$\{GMU:-)[0-9.]+', r'\g<1>' + gmu1, s)
if seqs: s = re.sub(r'--max-num-seqs \d+', '--max-num-seqs ' + seqs, s)
if nbt:  s = re.sub(r'--max-num-batched-tokens \d+', '--max-num-batched-tokens ' + nbt, s)
p.write_text(s)
print("  local: ple=%s gmu0=%s gmu1=%s seqs=%s nbt=%s" % (
    re.search(r'VLLM_PLE_TABLE_MEMORY=(\w+)', s).group(1),
    re.search(r'ROLE" == "0" \]; then GMU="\$\{GMU:-([0-9.]+)', s).group(1),
    re.search(r'else GMU="\$\{GMU:-([0-9.]+)', s).group(1),
    re.search(r'--max-num-seqs (\d+)', s).group(1),
    re.search(r'--max-num-batched-tokens (\d+)', s).group(1)))
PYPATCH
    bash -n "$REMOTE_SCRIPT" && echo "  remote syntax ok"
    bash -n "$SCRIPTS_DIR/serve-fp4-tp2.sh" && echo "  local syntax ok"
fi

T0=$(date +%s)
echo "[1/4] stopping both ranks"
docker rm -f qwen38-fp4-tp2 >/dev/null 2>&1 && echo "  rank1 stopped"
ssh "$REMOTE_HOST" 'docker rm -f qwen38-fp4-tp2 >/dev/null 2>&1' && echo "  rank0 stopped"

# No page-cache prewarm: B12X loads with O_DIRECT, so warming the cache changes nothing. Measured
# 126 s ROUTE before the prewarm and 134 s after it, with "49.59 GB physical reads" either way.
#
# Drop the page cache on both nodes before starting.
#
# vLLM's start-up check compares --gpu-memory-utilization against MemFree, NOT MemAvailable, so the
# page cache that the previous run's 95 GiB checkpoint reads left behind counts as unavailable. After
# a few restarts that cache had grown to 41 GiB on spark-ac8f (MemFree 77 GiB, Cached 41 GiB), and
# start-ups began aborting with "Free memory on device (80.46/121.69 GiB) on startup is less than
# desired". Dropping it took spark-ac8f from 77 to 105 GiB MemFree in under a second. The workspace
# AGENTS.md calls this mandatory on GB10 unified memory; its start.sh never actually ran it.
echo "[1.7/4] dropping page cache (vLLM checks MemFree, not MemAvailable)"
for target in local remote; do
    if [[ "$target" == local ]]; then
        if [[ -f "${SUDO_PASS_FILE:-$HOME/sudo-pass}" ]]; then
            P=$(cat "${SUDO_PASS_FILE:-$HOME/sudo-pass}")
            echo "$P" | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
            printf "  0d97 MemFree now %s GiB\n" "$(awk '/MemFree/{printf "%.1f", $2/1048576}' /proc/meminfo)"
        else
            echo "  0d97 skipped (no sudo password file)"
        fi
    else
        ssh "$REMOTE_HOST" 'if [ -f ${SUDO_PASS_FILE:-$HOME/sudo-pass} ]; then P=$(cat ${SUDO_PASS_FILE:-$HOME/sudo-pass}); echo "$P" | sudo -S sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null; fi; printf "  ac8f MemFree now %s GiB\n" "$(awk "/MemFree/{printf \"%0.1f\", \$2/1048576}" /proc/meminfo)"'
    fi
done

# Clear /dev/shm between stop and start.
#
# The container runs with --ipc host, so --shm-size is ignored and it shares the host's 60.8 GiB
# /dev/shm tmpfs. Every run leaves psm_* files behind (34 accumulated over one afternoon of restarts)
# and /proc/meminfo's Shmem sat at 14 GiB on both nodes -- 11% of total RAM counted as used, with no
# visible files to account for it (du reports 7.4 MB; the psm_* entries are sparse). Clearing the
# directory while nothing is running reclaims it, and the previous run's files are garbage regardless.
echo "[1.6/4] clearing /dev/shm (leftover psm_* from earlier runs, ~14 GiB of Shmem)"
if [[ -f "${SUDO_PASS_FILE:-$HOME/sudo-pass}" ]]; then
    P=$(cat "${SUDO_PASS_FILE:-$HOME/sudo-pass}")
    B=$(awk '/^Shmem/{printf "%.1f", $2/1048576}' /proc/meminfo)
    echo "$P" | sudo -S sh -c 'rm -rf /dev/shm/* 2>/dev/null' 2>/dev/null
    A=$(awk '/^Shmem/{printf "%.1f", $2/1048576}' /proc/meminfo)
    printf "  0d97 Shmem %s -> %s GiB\n" "$B" "$A"
fi
ssh "$REMOTE_HOST" 'if [ -f ${SUDO_PASS_FILE:-$HOME/sudo-pass} ]; then P=$(cat ${SUDO_PASS_FILE:-$HOME/sudo-pass}); B=$(awk "/^Shmem/{printf \"%0.1f\", \$2/1048576}" /proc/meminfo); echo "$P" | sudo -S sh -c "rm -rf /dev/shm/* 2>/dev/null" 2>/dev/null; A=$(awk "/^Shmem/{printf \"%0.1f\", \$2/1048576}" /proc/meminfo); printf "  ac8f Shmem %s -> %s GiB\n" "$B" "$A"; fi'

echo "[2/4] starting rank1 (worker, headless)"
MODEL_DIR="$MODEL" bash "$SCRIPTS_DIR/serve-fp4-tp2.sh" 1 >/dev/null 2>&1

echo "[3/4] starting rank0 (head, API)"
ssh "$REMOTE_HOST" "MODEL_DIR='$MODEL' bash $REMOTE_SCRIPT 0 >/dev/null 2>&1"

echo "[4/4] waiting for the API (a start-up routes the whole checkpoint, ~4 min)"
for i in $(seq 1 40); do
    sleep 15
    if ssh "$REMOTE_HOST" 'ss -ltn 2>/dev/null | grep -q ":8899"' 2>/dev/null; then
        ELAPSED=$(( $(date +%s) - T0 ))
        echo "  up in ${ELAPSED}s"
        echo
        echo "=== what it came up with ==="
        ssh "$REMOTE_HOST" 'docker logs qwen38-fp4-tp2 2>&1' \
            | grep -vE "NCCL INFO" \
            | grep -E "Available KV cache|GPU KV cache size|weights loaded|Free memory on device" \
            | tail -4 | sed 's/^/  /'
        echo
        echo "=== memory now ==="
        printf "  0d97 available %s GiB, swap %s GiB\n" \
            "$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)" \
            "$(free -g | awk '/^交换/{print $3}')"
        ssh "$REMOTE_HOST" 'printf "  ac8f available %.1f GiB, swap %s GiB\n" $(awk "/MemAvailable/{printf \"%.1f\", \$2/1048576}" /proc/meminfo) $(free -g | awk "/^交换/{print \$3}")'
        exit 0
    fi
    if ssh "$REMOTE_HOST" 'docker logs qwen38-fp4-tp2 2>&1' 2>/dev/null \
            | grep -qE "ValueError|RuntimeError|Engine core initialization failed"; then
        echo "  FAILED after $(( $(date +%s) - T0 ))s -- root cause:"
        ssh "$REMOTE_HOST" 'docker logs qwen38-fp4-tp2 2>&1' \
            | grep -vE "NCCL INFO" \
            | grep -E "ValueError|NotImplementedError|RuntimeError:|out of memory" \
            | grep -v "raise " | tail -3 | sed 's/^/  /'
        exit 1
    fi
    printf "  ... %ss\n" "$(( $(date +%s) - T0 ))"
done
echo "  timed out after $(( $(date +%s) - T0 ))s"
exit 1
