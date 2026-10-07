#!/usr/bin/env python3
"""Patch TensorFold 0.6.1 on spark-2 for "干活没有下文".

Four fixes. Each is applied exactly once and the module is byte-compiled after.

 1. cuda/server.py      _think_budget   clamp the budget so the forced </think> close always fits
                                        inside max_tokens (budget >= max_tokens => content: null).
 2. engine/call_gate.py take()           never apply a fix that cannot land whole (half a close
                                        = no </think> = the whole reply stays reasoning).
 3. cuda/http.py        streaming        SSE ": ping" comment every 10s of silence, so a client
                                        with a short idle-read timeout is not severed mid-reply.
 4. cuda/scheduler.py   _loop            one bad round must not kill the worker: a dead worker
                                        leaves /health green and every later request hanging.
"""
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/home/xujie/tensorfold-venv/lib/python3.12/site-packages/tensorfold")
STAMP = time.strftime("%Y%m%d-%H%M%S")

EDITS: list[tuple[str, str, str, str, str]] = []        # rel, label, old, new, probe
SPLICES: list[tuple[str, str, str, str, str, str]] = []  # rel, label, start, end, new, probe


def edit(rel: str, label: str, old: str, new: str, probe: str = "") -> None:
    EDITS.append((rel, label, old, new, probe))


def splice(rel: str, label: str, start: str, end: str, new: str, probe: str = "") -> None:
    SPLICES.append((rel, label, start, end, new, probe))


# ---------------------------------------------------------------- 1. budget clamp
edit(
    "cuda/server.py",
    "1. clamp the thinking budget to leave room for its own close",
    """        if prepared.think_budget <= 0 or think_end is None:
            return None
        close = [*self.tok.encode("\\n", add_special_tokens=False).ids, think_end]
        if prepared.grammar is None:
            close += self.tok.encode("\\n\\n", add_special_tokens=False).ids
        return ThinkBudget(prepared.think_budget, close, think_end)""",
    """        if prepared.think_budget <= 0 or think_end is None:
            return None
        close = [*self.tok.encode("\\n", add_special_tokens=False).ids, think_end]
        if prepared.grammar is None:
            close += self.tok.encode("\\n\\n", add_special_tokens=False).ids
        # The cut replaces a reply token with ``close``, so all of ``close`` has to fit under
        # max_tokens. A budget that reaches the cap first gets cut to nothing: </think> never
        # lands, split_thinking keeps the whole reply as reasoning, and the client is handed
        # content: null with finish_reason "length" - a silent, empty answer. Clamp instead.
        budget = min(prepared.think_budget, max(0, prepared.max_tokens - len(close)))
        if budget <= 0:
            return None
        return ThinkBudget(budget, close, think_end)""",
    "budget = min(prepared.think_budget, max(0, prepared.max_tokens - len(close)))",
)

# ------------------------------------------------------- 2. no half-applied close
edit(
    "engine/call_gate.py",
    "2. apply a gate fix only when it fits whole",
    """            (at, fix), replay = min(hits, key=lambda h: (h[0][0], h[1]))   # the earliest; a fix before a replay
            new, state["cut"], state["replay"] = [*new[:at], *fix][:max_tokens - len(reply)], True, replay""",
    """            (at, fix), replay = min(hits, key=lambda h: (h[0][0], h[1]))   # the earliest; a fix before a replay
            # A fix landing half is worse than no fix: half a think-block close leaves the block
            # open. Skip the cut unless all of it fits, and let max_tokens stop the run instead.
            if max_tokens - len(reply) - at >= len(fix):
                new, state["cut"], state["replay"] = [*new[:at], *fix], True, replay""",
    "max_tokens - len(reply) - at >= len(fix)",
)

# ------------------------------------------------------------ 3. SSE keepalive
edit(
    "cuda/http.py",
    "3a. import threading",
    """import json
import time
import traceback
import uuid""",
    """import json
import threading
import time
import traceback
import uuid""",
    "import threading\nimport time",
)
splice(
    "cuda/http.py",
    "3b. serialized writes + a ping while the stream is silent",
    "                def emit(delta: dict[str, Any]) -> bool:",
    '                    return self._stream_error({"message": _error_message(exc), "type": "server_error"})',
    '''                write_lock = threading.Lock()         # the ping thread shares this socket with the reply
                wrote = [time.monotonic()]

                def _write(text: str) -> bool:
                    try:
                        with write_lock:
                            self.wfile.write(text.encode())
                            self.wfile.flush()
                    except OSError:     # reset, broken pipe, timed out, host unreachable: the client has gone
                        return False
                    wrote[0] = time.monotonic()
                    return True

                def emit(delta: dict[str, Any]) -> bool:
                    return _write(f"data: {json.dumps(chunk(delta))}\\n\\n")

                if chat:
                    emit({"role": "assistant"})

                # Queueing, prefilling and thinking all pass without a byte, which is minutes at
                # this context length: an idle-read timeout in the client, a proxy or a load
                # balancer severs the stream and the caller sees no answer at all. An SSE comment
                # is ignored by every client and costs one line every 10 seconds of silence.
                ping_stop = threading.Event()

                def keepalive() -> None:
                    while not ping_stop.wait(2.0):
                        if time.monotonic() - wrote[0] >= 10.0:
                            _write(": ping\\n\\n")

                threading.Thread(target=keepalive, daemon=True).start()
                try:
                    result = app.run(body, chat, emit, prepared=prepared, cancelled=cancelled)
                except RequestCancelled:
                    self.close_connection = True
                    return
                except RequestError as exc:
                    return self._stream_error(error_body(exc))
                except Exception as exc:
                    _log_error(exc)
                    return self._stream_error({"message": _error_message(exc), "type": "server_error"})
                finally:
                    ping_stop.set()''',
    "ping_stop = threading.Event()",
)
splice(
    "cuda/http.py",
    "3c. route the terminating write through the same lock",
    "                try:\n                    self.wfile.write(f\"data: {json.dumps(end)}",
    "                self.close_connection = True\n                return",
    '''                _write(f"data: {json.dumps(end)}\\n\\ndata: [DONE]\\n\\n")
                self.close_connection = True
                return''',
)

# ------------------------------------------------------- 4. the worker must live
edit(
    "cuda/scheduler.py",
    "4a. import time and traceback",
    """import itertools
import queue
import threading""",
    """import itertools
import queue
import threading
import time
import traceback""",
    "import time\nimport traceback",
)
splice(
    "cuda/scheduler.py",
    "4b. one bad round may not kill the worker",
    "    def _loop(self) -> None:\n        while True:",
    '''self._reply(s, *(("error", s.error) if s.error is not None else ("done", s.stats())))''',
    '''    def _loop(self) -> None:
        while True:
            # Every call in a round but decoder.round() was unguarded: one raise ended this daemon
            # thread for good, and with it the whole endpoint - /health stayed green while every
            # later request hung forever on an open stream. A round that fails is a bad round.
            try:
                self._yield()
                idle = not self.decoder.live() and self.held is None
                first = self.waiting.get() if idle else None                          # idle: wait for a request
                if idle and first is None:
                    return                                                            # close()
                done = self._admit(first)
                try:
                    done += self.decoder.round()
                except Exception as exc:                 # noqa: BLE001  (the live requests fail)
                    for s in self.decoder.drop():
                        self._reply(s, "error", exc)
                self.decoder.finish(done)
                for s in done:
                    self._reply(s, *(("error", s.error) if s.error is not None else ("done", s.stats())))
            except Exception as exc:                     # noqa: BLE001  (the worker outlives the round)
                print(f"[tensorfold] scheduler round failed, continuing: {exc!r}", flush=True)
                traceback.print_exc()
                time.sleep(0.05)''',
    "scheduler round failed, continuing",
)


def backup_dir() -> Path:
    d = Path(f"/home/xujie/tensorfold-patches/pre-061-silencefix-{STAMP}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> int:
    dst = backup_dir()
    touched = sorted({e[0] for e in EDITS} | {s[0] for s in SPLICES})
    for rel in touched:
        shutil.copy2(ROOT / rel, dst / rel.replace("/", "__"))
    print(f"backup: {dst}")

    failed = False
    for rel, label, old, new, probe in EDITS:
        path = ROOT / rel
        text = path.read_text()
        if probe and probe in text:
            print(f"  SKIP  {label}  (already applied)")
            continue
        if text.count(old) != 1:
            print(f"  FAIL  {label}: anchor found {text.count(old)} times in {rel}")
            failed = True
            continue
        path.write_text(text.replace(old, new, 1))
        print(f"  ok    {label}")

    for rel, label, start, end, new, probe in SPLICES:
        path = ROOT / rel
        text = path.read_text()
        if probe and probe in text:
            print(f"  SKIP  {label}  (already applied)")
            continue
        i, j = text.find(start), text.find(end)
        if i < 0 or j < 0 or j < i:
            print(f"  FAIL  {label}: markers not found in {rel} (start={i} end={j})")
            failed = True
            continue
        j += len(end)
        path.write_text(text[:i] + new + text[j:])
        print(f"  ok    {label}")

    import py_compile
    for rel in touched:
        py_compile.compile(str(ROOT / rel), doraise=True)
        print(f"  compiles  {rel}")
    if failed:
        print("INCOMPLETE - do not restart until this succeeds")
        return 1
    print("all patches applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())