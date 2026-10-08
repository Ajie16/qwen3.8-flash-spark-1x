#!/usr/bin/env bash
# One knob-sweep arm: restart the service with extra env and measure prefill.
#
# Usage: knob_arm.sh LABEL [VAR=VALUE ...]
#
# Restart discipline, learned the hard way earlier in this project:
#   * kill by PID read from /proc, never by pattern (a pattern matches this script itself)
#   * wait for the process to actually exit AND for port 8899 to free before starting
#   * confirm the new process started *after* the edit and log to a fresh file each arm
#   * print the resulting affinity/env so an arm can never silently measure the wrong build
set -uo pipefail

LABEL="$1"; shift
BENCH=/home/xujie/prefill_bench.py
LOGDIR=/home/xujie/knoblogs
RECIPE=/home/xujie/workspace/qwen3.8-flash-spark-1x
mkdir -p "$LOGDIR"

kill_service() {
  for p in $(pgrep -f 'tabbyAPI/main[.]py' 2>/dev/null); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)
    case "$cmd" in *tabbyAPI/main.py*) kill "$p" 2>/dev/null;; esac
  done
  for _ in $(seq 1 40); do
    pgrep -f 'tabbyAPI/main[.]py' >/dev/null 2>&1 || break
    sleep 1
  done
  sleep 6
}

echo "===== $LABEL ====="
echo "  额外环境: ${*:-（无）}"
kill_service
if pgrep -f 'tabbyAPI/main[.]py' >/dev/null 2>&1; then
  echo "  !! 旧服务未退出，跳过本臂"; exit 1
fi
if ss -ltn 2>/dev/null | grep -q ':8899'; then
  echo "  !! 8899 仍被占用，跳过本臂"; exit 1
fi

LOG="$LOGDIR/$LABEL.log"
cd "$RECIPE" || exit 1
setsid nohup env "$@" bash exllamav3-tabby/serve-local.sh > "$LOG" 2>&1 < /dev/null &

ok=0
for i in $(seq 1 100); do
  c=$(curl -s -m 4 -o /dev/null -w "%{http_code}" http://10.100.65.1:8899/health 2>/dev/null || true)
  [ "$c" = "200" ] && { ok=1; echo "  [$((i*8))s] health=200"; break; }
  sleep 8
done
if [ "$ok" != 1 ]; then
  echo "  !! 启动失败，日志尾部:"; tail -6 "$LOG" | sed 's/^/     /'; exit 1
fi
sleep 4

P=$(pgrep -f 'tabbyAPI/main[.]py' | head -1)
echo "  pid=$P  启动 $(ps -o lstart= -p "$P" 2>/dev/null | xargs)"
echo "  亲和: $(taskset -cp "$P" 2>/dev/null | sed 's/.*: //')"
for kv in "$@"; do
  k="${kv%%=*}"
  printf "  %-26s %s\n" "$k" "$(tr '\0' '\n' < "/proc/$P/environ" | grep "^$k=" | cut -d= -f2-)"
done

timeout 900 /home/xujie/qwen38-exl3/venv/bin/python "$BENCH" 40000 3 2>&1 | tail -5
