import subprocess
import sys
import time
from pathlib import Path

import pytest


CHATTY_CHILD = (
    "import sys\n"
    "sys.stdout.write('{\"type\":\"start\"}\\n'); sys.stdout.flush()\n"
    "sys.stderr.write('W' * 200_000); sys.stderr.flush()\n"
    "sys.stdout.write('{\"type\":\"result\"}\\n'); sys.stdout.flush()\n"
)


def _fake_pi_script(tmp_path: Path, body: str, name: str = "fake_pi.py") -> Path:
    """A stand-in `pi` binary: ignores every argv it is given and just runs
    `body`. agent_pi.run() builds a long, specific argv (`--provider`,
    `--model`, `--session-id`, ...); the fake ignores all of it because this
    suite is exercising subprocess IO, not pi's CLI surface."""
    script = tmp_path / name
    script.write_text("#!/usr/bin/env python3\n" + body)
    script.chmod(0o755)
    return script


def _patch_pi_resolution(monkeypatch, agent_pi, pi_path: Path) -> None:
    """Point agent_pi.run() at the fake binary, and bypass `pi --list-models`
    / `~/.pi/agent/models.json` — those require a real `pi` install and are
    orthogonal to what this suite tests (subprocess IO safety)."""
    monkeypatch.setattr(agent_pi, "PI_PATH", str(pi_path))
    monkeypatch.setattr(agent_pi, "resolve_model", lambda pattern: ("test", "test-model"))
    monkeypatch.setattr(agent_pi, "context_window", lambda provider, model_id: 0)


def _pi_request(tmp_path: Path, **overrides):
    from adw_modules.data_types import PiRequest
    defaults = dict(
        prompt="hi", system_prompt="sys", model="unused",
        session_id="test-session", session_dir=str(tmp_path / "pi_sessions"),
        raw_output_path=str(tmp_path / "agent" / "raw_output.jsonl"),
        cwd=str(tmp_path),
    )
    defaults.update(overrides)
    return PiRequest(**defaults)


def test_stderr_to_file_does_not_deadlock(tmp_path, monkeypatch):
    """Regression test for the deadlock agent_pi.py used to have: stderr=PIPE
    plus a blocking stdout read wedges once the child fills the ~64KB stderr
    pipe buffer. This drives the PRODUCTION `agent_pi.run()` and its real
    `Popen` — a test that builds its own separate `Popen` call (as this one
    originally did) stays green even after `agent_pi.py`'s `stderr=err` is
    reverted to `stderr=subprocess.PIPE`, so it verifies nothing about the
    code it is supposed to guard.

    A deadlock also cannot be caught by an assertion placed inside the
    blocking `for line in process.stdout` loop — the thread is stuck THERE,
    so the assertion is simply never reached, and the test would hang the
    whole suite forever instead of failing it. The watchdog below runs on a
    separate thread and can actually preempt the blocked read by killing the
    child, so a real regression produces a fast, visible test failure instead
    of a hang.
    """
    import os
    import signal
    import threading
    from adw_modules import agent_pi

    fake_pi = _fake_pi_script(tmp_path, CHATTY_CHILD)
    _patch_pi_resolution(monkeypatch, agent_pi, fake_pi)
    request = _pi_request(tmp_path)

    timed_out = threading.Event()
    watchdogs: list[threading.Timer] = []

    def on_spawn(pid: int) -> None:
        def _fire():
            timed_out.set()
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        timer = threading.Timer(10.0, _fire)
        timer.daemon = True
        timer.start()
        watchdogs.append(timer)

    result = agent_pi.run(request, on_spawn=on_spawn)

    for timer in watchdogs:
        timer.cancel()
    assert not timed_out.is_set(), "deadlocked: the watchdog had to kill the child"
    assert result.returncode == 0
    stderr_path = Path(request.raw_output_path).with_name("stderr.log")
    assert stderr_path.stat().st_size == 200_000


def test_run_reports_both_warnings_and_the_crash_tail(tmp_path, monkeypatch):
    """A benign `Warning:` line must not hide a fatal traceback that matches
    none of stderr_warnings' four prefixes — `stderr_warnings(...) or tail`
    would let it, since any warning line makes the `or`'s left side truthy."""
    from adw_modules import agent_pi

    fake_pi = _fake_pi_script(tmp_path, (
        "import sys\n"
        "sys.stderr.write(\"Warning: Unknown --effort value 'off'\\n\")\n"
        "sys.stderr.write('Traceback (most recent call last):\\n')\n"
        "sys.stderr.write('RuntimeError: boom\\n')\n"
        "sys.exit(1)\n"
    ))
    _patch_pi_resolution(monkeypatch, agent_pi, fake_pi)
    request = _pi_request(tmp_path)

    with pytest.raises(RuntimeError) as exc_info:
        agent_pi.run(request)

    message = str(exc_info.value)
    assert "Unknown --effort value" in message, "warnings must still be surfaced"
    assert "RuntimeError: boom" in message, \
        "the actual crash cause must not be hidden by an unrelated warning"


def test_run_scans_only_this_attempts_stderr_on_retry(tmp_path, monkeypatch):
    """agents.py deliberately calls agent_pi.run() again against the SAME
    raw_output_path (parse-fix and gate-correction retries re-enter the SAME
    pi session — agents.py:106-108), which appends to the SAME stderr.log.
    A later attempt must not report an EARLIER attempt's stderr as its own."""
    from adw_modules import agent_pi

    request = _pi_request(tmp_path)

    first_pi = _fake_pi_script(tmp_path, (
        "import sys\n"
        "sys.stderr.write('Warning: first attempt problem\\n')\n"
        "sys.stdout.write('{\"type\":\"result\"}\\n')\n"
    ), name="fake_pi_1.py")
    _patch_pi_resolution(monkeypatch, agent_pi, first_pi)
    agent_pi.run(request)   # attempt 1: succeeds (exit 0), leaves a warning behind

    second_pi = _fake_pi_script(tmp_path, (
        "import sys\n"
        "sys.stdout.write('{\"type\":\"result\"}\\n')\n"
        "sys.exit(1)\n"
    ), name="fake_pi_2.py")
    _patch_pi_resolution(monkeypatch, agent_pi, second_pi)
    with pytest.raises(RuntimeError) as exc_info:
        agent_pi.run(request)   # attempt 2: fails, writes NOTHING new to stderr

    assert "first attempt problem" not in str(exc_info.value), \
        "a retry must not surface a PREVIOUS attempt's stderr as its own cause"


def test_run_kills_and_reaps_the_child_when_on_event_raises(tmp_path, monkeypatch):
    """Regression for the merge-blocking Pi orphan: `agent_pi.py:249` gained
    `start_new_session=True` (needed so `kill_tree` can reach the child via its
    own process group) but the read loop had no `try/finally` guard around it —
    unlike `agent_cc._run_once`, which has exactly this guard at
    `agent_cc.py:603-610`. A realistic fault (a locked-sqlite write inside
    `tracer.event()`, called from `on_event`) then unwinds `run()` leaving the
    `pi` child alive and DETACHED — Ctrl-C on the ADW no longer reaches it,
    `on_exit` never fires so the pid is stuck in `run.live_children` forever,
    and `process.wait()` is never called so it is never reaped.

    This test must fail against unfixed `agent_pi.py` (the child survives and
    on_exit never fires) and pass once `run()` gets the same
    `except BaseException: kill_tree(...); raise` / `finally: wait(); on_exit()`
    shape `agent_cc._run_once` already has.
    """
    import os
    from adw_modules import agent_pi

    # Emits one real event (so `on_event` has something to fire on), then
    # sleeps well past this test's assertions — long enough that, if the
    # child is NOT killed, it is still observably alive when we check.
    fake_pi = _fake_pi_script(tmp_path, (
        "import sys, time\n"
        "sys.stdout.write('{\"type\":\"message_end\",\"message\":"
        "{\"role\":\"assistant\",\"content\":[],\"usage\":{}}}\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(60)\n"
    ))
    _patch_pi_resolution(monkeypatch, agent_pi, fake_pi)
    request = _pi_request(tmp_path)

    spawned: list[int] = []
    exited: list[int] = []

    def _boom(event: dict) -> None:
        raise RuntimeError("simulated tracer/sqlite failure inside on_event")

    try:
        with pytest.raises(RuntimeError, match="simulated tracer/sqlite failure"):
            agent_pi.run(request, on_event=_boom,
                        on_spawn=spawned.append, on_exit=exited.append)

        assert spawned, "the child must have been spawned before the raise"
        pid = spawned[0]

        still_alive = True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            still_alive = False

        assert not still_alive, (
            "the pi child must not survive on_event raising — it must be "
            "killed the same way agent_cc._run_once kills its child on any "
            "BaseException from the read loop")
        assert exited == [pid], (
            "on_exit must fire so run.live_children/the processes row can "
            "close — otherwise the pid is stuck 'alive' in the trace forever")
    finally:
        # Best-effort cleanup so a failing (pre-fix) run of this test does not
        # leak a detached sleeping process into the rest of the suite.
        if spawned:
            try:
                os.kill(spawned[0], 9)
            except ProcessLookupError:
                pass


def test_stderr_warnings_extracts_warning_lines(fixture_path):
    from adw_modules.utils import stderr_warnings
    lines = stderr_warnings(fixture_path("effort_off.err"))
    assert any("Unknown --effort value" in ln for ln in lines)


def test_stderr_warnings_on_missing_file_is_empty(tmp_path):
    from adw_modules.utils import stderr_warnings
    assert stderr_warnings(tmp_path / "nope.log") == []


def test_stderr_warnings_offset_skips_earlier_content(tmp_path):
    """The offset param is what lets a retried agent_pi.run() scan only its
    OWN attempt's stderr out of a log shared with earlier attempts."""
    from adw_modules.utils import stderr_warnings
    path = tmp_path / "stderr.log"
    path.write_text("Warning: attempt one\n")
    offset = path.stat().st_size
    with path.open("a") as f:
        f.write("Warning: attempt two\n")
    assert stderr_warnings(path) == ["Warning: attempt one", "Warning: attempt two"]
    assert stderr_warnings(path, offset=offset) == ["Warning: attempt two"]


def test_kill_tree_rejects_pid_zero_and_negative(monkeypatch):
    """os.getpgid(0) returns the CALLER's own pgid — never 0 — so pid=0 would
    otherwise fall through to os.kill(0, SIGTERM), signalling this ADW's own
    process group. kill_tree must refuse pid<=0 before it ever calls getpgid."""
    import os
    from adw_modules import utils
    calls = []
    monkeypatch.setattr(os, "getpgid", lambda pid: calls.append(("getpgid", pid)) or pid)
    monkeypatch.setattr(os, "killpg", lambda *a: calls.append(("killpg", *a)))
    monkeypatch.setattr(os, "kill", lambda *a: calls.append(("kill", *a)))
    utils.kill_tree(0, grace=0.1)
    utils.kill_tree(-4242, grace=0.1)
    assert calls == [], "pid<=0 must never reach getpgid/killpg/kill at all"


def test_kill_tree_falls_back_when_the_pid_is_not_a_group_leader(monkeypatch):
    """os.killpg takes a process GROUP id. Handing it a non-leader pid could
    signal an unrelated group that happens to share the number — and the
    non-leader path must still actually reach the pid via os.kill, not
    silently do nothing. (An os.kill stub that always raises ProcessLookupError
    makes "signalled, then found dead" indistinguishable from "never signalled
    at all" — this one only goes unreachable once SIGKILL has actually gone
    out, so it also pins the SIGTERM -> SIGKILL escalation order.)"""
    import os
    import signal
    from adw_modules import utils
    killpg_calls = []
    kill_calls = []

    def fake_kill(pid, sig):
        kill_calls.append((pid, sig))
        if sig == 0 and any(s == signal.SIGKILL for _, s in kill_calls):
            raise ProcessLookupError

    monkeypatch.setattr(os, "getpgid", lambda pid: pid + 1)      # not a leader
    monkeypatch.setattr(os, "killpg", lambda *a: killpg_calls.append(a))
    monkeypatch.setattr(os, "kill", fake_kill)
    utils.kill_tree(4242, grace=0.05)
    assert not killpg_calls, "must not signal a group we do not own"
    assert (4242, signal.SIGTERM) in kill_calls, "SIGTERM must actually reach the pid via os.kill"
    assert (4242, signal.SIGKILL) in kill_calls, "escalation to SIGKILL must actually reach the pid"


def test_kill_tree_kills_grandchildren(tmp_path):
    """A bare kill(pid) orphans grandchildren. kill_tree must not."""
    from adw_modules.utils import kill_tree
    marker = tmp_path / "alive"
    parent_src = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable,'-c',\"import time;open({str(marker)!r},'w').write('x');time.sleep(60)\"])\n"
        "time.sleep(60)\n"
    )
    p = subprocess.Popen([sys.executable, "-c", parent_src], start_new_session=True)
    for _ in range(100):
        if marker.exists():
            break
        time.sleep(0.05)
    assert marker.exists(), "grandchild never started"
    kill_tree(p.pid, grace=1.0)
    p.wait(timeout=10)
    out = subprocess.run(["pgrep", "-g", str(p.pid)], capture_output=True, text=True)
    assert out.stdout.strip() == "", f"survivors in process group: {out.stdout!r}"


def test_claude_code_env_strips_key_and_nested_session(monkeypatch):
    from adw_modules.utils import claude_code_env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-be-stripped")
    monkeypatch.setenv("CLAUDE_EFFORT", "max")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "abc")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = claude_code_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "CLAUDE_EFFORT" not in env
    assert "CLAUDECODE" not in env
    assert "CLAUDE_CODE_SESSION_ID" not in env
    assert env["PATH"] == "/usr/bin"


def test_claude_code_env_can_inherit_the_key_when_asked(monkeypatch):
    from adw_modules.utils import claude_code_env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-kept")
    monkeypatch.setenv("CLAUDECODE", "1")
    env = claude_code_env(inherit_api_key=True)
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-kept"
    # nested-session contamination is stripped regardless — never configurable
    assert "CLAUDECODE" not in env


def test_claude_code_env_sets_disable_auto_memory(monkeypatch):
    """Spec §7b: the child env must always disable Claude Code's own
    auto-memory loading, for every claude_code agent except the writes: []
    ones --restricted already covers for free."""
    from adw_modules.utils import claude_code_env
    monkeypatch.delenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", raising=False)
    env = claude_code_env()
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"


def test_claude_code_env_disable_auto_memory_survives_an_operator_override(monkeypatch):
    """The blanket CLAUDE_CODE_* strip (a few lines above the fix) would
    otherwise delete an operator's own CLAUDE_CODE_DISABLE_AUTO_MEMORY before
    the child ever saw it — the setter must run AFTER that strip."""
    from adw_modules.utils import claude_code_env
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "0")
    env = claude_code_env()
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
