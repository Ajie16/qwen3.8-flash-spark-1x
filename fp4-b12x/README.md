# fp4-b12x — Qwen3.8-Flash-Next-FP4-Local on two DGX Sparks

Serving config, restart tooling, and benchmarks for the stack that replaced TensorFold on
2026-10-17. Everything here assumes both Sparks are reachable and the checkpoint is already on both.

The parameters are not invented: they come from
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)'s
`recipes/qwen3.8-flash-next-nvfp4-cluster.yaml` (2.4k stars, the dual-DGX-Spark vLLM project), which
is the only published cluster recipe for this exact checkpoint. A copy is kept here as
`qwen3.8-flash-next-nvfp4-cluster.yaml` so the reference does not drift silently.

## The three things that decide whether it works

**1. `--device /dev/infiniband:/dev/infiniband`**

Without it the container cannot see `uverbs`/`rdma_cm`, NCCL silently falls back to TCP, and TP=2
drops from 432 to ~281 tok/s. Tell: `roceP2p1s0f0`'s `port_xmit_data` counter stays flat. `ss` is
misleading here — the out-of-band channel is TCP even when RDMA works. Confirm with
`NCCL_DEBUG=INFO` and look for `Using network IB`.

**2. `VLLM_ENABLE_ROCE_ALLREDUCE=1` + `VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB`**, plus
`draft_tensor_parallel_size: 2` in `--speculative-config`.

Generic NCCL collectives cost ~40 µs each and there are ~122 per token, which is why TP=2 was only
+23% over one box. vLLM's own RoCE path takes that to +75%. The drafter needs to span both ranks too;
without `draft_tensor_parallel_size` the MTP work is not split.

**3. `--max-num-batched-tokens` must stay 4096**

8192 and 16384 abort with `preparation ranks reached incompatible tuning boundaries`, and clearing
the compile cache does not change it. It is not a hardcoded size list but a chain of cross-rank
invariants in `b12x_startup.py`; `apply_tuning_patch.py` gets past the first and then fails on a
second one in `flashinfer/.../preparation/session.py`, so it is archived as a dead end rather than
shipped as a fix.

## Files

| File | What it is |
|---|---|
| `serve-fp4-tp2.sh` | launcher. `0` = head (serves the API), `1` = worker (headless). Start worker first. |
| `tf-restart.sh` | one command: stop → drop caches → clear /dev/shm → start → wait → report KV and memory. Prints the root cause on failure. Use this rather than the launcher directly. |
| `bench.py` | `decode` / `prefill` / `accept` modes. Replaces five overlapping scripts. |
| `apply_tuning_patch.py` | **archived dead end.** Attempt at making nbt>4096 work; it does not. |
| `b12x_startup.py.patch` | the diff that script produces, for the record. |
| `qwen3.8-flash-next-nvfp4-cluster.yaml` | the upstream reference recipe. |

## Measured

Decode, sparkDash counting prompt, thinking off, warm-up discarded:

| concurrency | tok/s | per stream |
|---|---|---|
| 1 | 108 | 108 |
| 4 | 283 | 71 |
| 8 | 432 | 54 |

Prefill 2,300–2,400 tok/s from 5k to 179k tokens, flat. MTP acceptance varies widely by content:
98% counting, ~70% code and JSON, ~52% English prose, ~32% Chinese prose — which is why Chinese
creative writing is the slowest thing this box does.

## Leaks that eat memory silently

Both are handled by `tf-restart.sh`; they had together consumed up to 55 GiB.

**`/dev/shm`.** The container runs `--ipc host`, so `--shm-size` is ignored and it shares the host's
60.8 GiB tmpfs. Each run leaves `psm_*` entries behind; they are sparse, so `du` reports ~7 MB while
tmpfs charges 14 GiB to `Shmem`.

**Page cache.** vLLM compares `--gpu-memory-utilization` against `MemFree`, not `MemAvailable`, so
the cache left by the previous run's 95 GiB of checkpoint reads counts as unavailable — 41 GiB of it
at worst, which is enough to abort a start with `Free memory on device (80.46/121.69 GiB)`.

## Start-up cost

~5 minutes warm, of which 134 s is B12X routing the 95 GiB checkpoint with `O_DIRECT`. A page-cache
prewarm does not help (measured 126 s before, 134 s after) because the reads bypass the cache. Do not
clear `~/.cache/nvidia-fp4-local/b12x` to "fix" anything: it costs ~170 s of `gdn_prefill`
recompilation on the next start and solves nothing.

## Secrets

Nothing here holds credentials. `tf-restart.sh` reads a sudo password from `~/sudo-pass` **on the
host** with `sudo -S`; the file itself is not in this repo and must not be added. Set
`SUDO_PASS_FILE=/path` to point elsewhere, or leave the file absent and the script skips the steps
that need root (it prints which).

Internal addresses (10.100.65.x, 192.168.1.x) and hostnames appear throughout because they are the
subject matter. They are RFC1918 / local only.
