import subprocess
import sys
import time
from pathlib import Path


CHATTY_CHILD = (
    "import sys\n"
    "sys.stdout.write('{\"type\":\"start\"}\\n'); sys.stdout.flush()\n"
    "sys.stderr.write('W' * 200_000); sys.stderr.flush()\n"
    "sys.stdout.write('{\"type\":\"result\"}\\n'); sys.stdout.flush()\n"
)


def test_stderr_to_file_does_not_deadlock(tmp_path):
    """stderr=PIPE + a blocking stdout read deadlocks once the child fills the
    ~64KB stderr pipe buffer. A file has no such buffer."""
    err_path = tmp_path / "stderr.log"
    started = time.monotonic()
    lines = []
    with err_path.open("a") as err:
        p = subprocess.Popen([sys.executable, "-c", CHATTY_CHILD],
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=err, text=True, bufsize=1)
        for line in p.stdout:
            lines.append(line)
            assert time.monotonic() - started < 15, "deadlocked"
        p.wait()
    assert len(lines) == 2
    assert err_path.stat().st_size == 200_000


def test_stderr_warnings_extracts_warning_lines(fixture_path):
    from adw_modules.utils import stderr_warnings
    lines = stderr_warnings(fixture_path("effort_off.err"))
    assert any("Unknown --effort value" in ln for ln in lines)


def test_stderr_warnings_on_missing_file_is_empty(tmp_path):
    from adw_modules.utils import stderr_warnings
    assert stderr_warnings(tmp_path / "nope.log") == []


def test_kill_tree_falls_back_when_the_pid_is_not_a_group_leader(monkeypatch):
    """os.killpg takes a process GROUP id. Handing it a non-leader pid could
    signal an unrelated group that happens to share the number."""
    import os
    from adw_modules import utils
    signalled = []
    monkeypatch.setattr(os, "getpgid", lambda pid: pid + 1)      # not a leader
    monkeypatch.setattr(os, "killpg", lambda *a: signalled.append(("pg", *a)))
    monkeypatch.setattr(os, "kill", lambda *a: (_ for _ in ()).throw(ProcessLookupError))
    utils.kill_tree(4242, grace=0.1)
    assert not any(s[0] == "pg" for s in signalled), "must not signal a group we do not own"


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
