"""Small shared helpers. Anything bigger belongs in its own module."""

from __future__ import annotations

import os
import secrets
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def operator_env() -> dict[str, str]:
    """The engineer's own environment, as their shell would hand it over.

    Agents and quality blocks are meant to see exactly what the operator sees:
    their PATH, their toolchains, their globally installed packages. Copying
    os.environ gets almost all the way there — but ADWs launch under `uv run`,
    which prepends its ephemeral venv's bin to PATH and sets VIRTUAL_ENV. That
    venv holds the ADW's OWN dependencies (pydantic, pyyaml), not the
    operator's, so anything a subprocess resolves through it — `python3`,
    `pip`, every globally pip-installed CLI — silently becomes the wrong one.

    Stripping the venv restores parity: `python3` in an agent's bash is the
    same `python3` the engineer gets in their terminal. The ADW's own imports
    are unaffected; this env is only ever handed to child processes.
    """
    env = os.environ.copy()
    venv = env.pop("VIRTUAL_ENV", "")
    if not venv:
        return env
    venv_bin = str(Path(venv) / "bin")
    parts = [p for p in env.get("PATH", "").split(os.pathsep) if p and p != venv_bin]
    env["PATH"] = os.pathsep.join(parts)
    return env


def new_id(length: int = 8) -> str:
    return secrets.token_hex(length // 2)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_prompt(arg: str) -> str:
    """CLI prompt arg: a file path resolves to its contents, else inline text."""
    try:
        p = Path(arg)
        if p.is_file():
            return p.read_text()
    except OSError:
        pass
    return arg


def engineer_name() -> str:
    name = os.environ.get("ENGINEER_NAME", "").strip()
    if name:
        return name
    try:
        out = subprocess.run(["git", "config", "user.name"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except OSError:
        pass
    return os.environ.get("USER", "engineer")


# Credentials and provider switches that would move a claude_code agent OFF the
# subscription and onto per-token billing. Stripped by default; see spec §7a.
CC_STRIPPED_ENV = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_EFFORT",
)


def claude_code_env(inherit_api_key: bool = False) -> dict[str, str]:
    """The operator's environment, cleaned for a Claude Code child.

    Two removals, only one of them optional:

    * Credentials (`CC_STRIPPED_ENV`). With ANTHROPIC_API_KEY set the child
      reports `apiKeySource: "ANTHROPIC_API_KEY"` and bills the API instead of
      the subscription — the exact outcome this adapter exists to avoid. A bad
      key is worse than a wrong bill: the child hangs past 120s with no `init`
      event rather than failing fast.
    * Nested-session identity. An ADW launched from inside a Claude Code
      terminal inherits CLAUDECODE, CLAUDE_CODE_SESSION_ID, CLAUDE_CODE_
      MESSAGING_TOKEN and friends, which point the child at its parent's live
      session. That is never wanted and is not configurable.
    """
    env = operator_env()
    if not inherit_api_key:
        for key in CC_STRIPPED_ENV:
            env.pop(key, None)
    for key in [k for k in env
                if k.startswith("CLAUDE_CODE_")
                or k in ("CLAUDECODE", "CLAUDE_PID", "CLAUDE_TRANSCRIPT_PATH")]:
        env.pop(key, None)
    return env


def kill_tree(pid: int, grace: float = 5.0) -> None:
    """SIGTERM a process GROUP, then SIGKILL whatever is left.

    Coding agents spawn real grandchildren — bash tool commands, MCP servers.
    `kill(pid)` orphans them, leaving work running after the run is recorded
    dead. Children are only reachable as a group, which is why they are spawned
    with `start_new_session=True` (each becomes its own group leader, so its pid
    IS the pgid).

    The leadership check is not paranoia. `os.killpg` takes a process GROUP id,
    so handing it a pid that leads no group either fails or — worse — signals
    an unrelated group that happens to share the number. Taking out someone
    else's whole process group is a far bigger mistake than failing to reap
    one child, so a non-leader gets a single-process kill instead.
    """
    if pid <= 0:
        # os.getpgid(0) returns the CALLER's own pgid — it is never 0, so a
        # stray pid=0 would pass the leadership check, fall through to
        # os.kill, and os.kill(0, SIGTERM) signals every process in this
        # ADW's own group, including itself. A negative pid is `killpg`'s own
        # spelling for a group id, which this function must never accept as
        # a bare pid either.
        return
    try:
        group = os.getpgid(pid) == pid
    except (ProcessLookupError, PermissionError):
        return                          # already gone
    send = os.killpg if group else os.kill
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            send(pid, sig)
        except (ProcessLookupError, PermissionError):
            return                      # already gone, or not ours to kill
        deadline = time.monotonic() + (grace if sig == signal.SIGTERM else 1.0)
        while time.monotonic() < deadline:
            try:
                send(pid, 0)            # signal 0 = liveness probe
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.05)


def stderr_warnings(path: str | Path, limit: int = 20, offset: int = 0) -> list[str]:
    """Warning/error lines from a child's stderr log, for the trace.

    The CLI reports real problems here that nothing in this system reads —
    `Warning: Unknown --effort value 'off' …` exits 0 and never reaches the
    trace, so a misconfigured roster looks fine. Returns [] when there is no
    log, which is the common case.

    `offset` (bytes) skips content written before it. The log is one file per
    agent-phase that every retried `send()` appends to (parse-fix and gate
    corrections re-enter the SAME pi session by design), so without an offset
    attempt 3's warnings would include attempts 1 and 2's — burying the
    current attempt's own lines past `limit`, or blaming it for an old one's
    warning entirely. Default 0 keeps every other caller's behavior unchanged.
    """
    try:
        data = Path(path).read_bytes()[offset:]
    except OSError:
        return []
    text = data.decode(errors="replace")
    hits = [ln.strip() for ln in text.splitlines()
            if ln.strip().startswith(("Warning:", "Error:", "warning:", "error:"))]
    return hits[:limit]


RESULT_SNIPPET_CHARS = 20_000   # tool output rides along whole; clip only guards pathological cases
ARG_VALUE_CHARS = 20_000        # args too — the UI scrolls, it must not be handed cut-off data
LABEL_CHARS = 80                # "bash: <command>" shown as the event name

# The arg that identifies a call at a glance, in the order tools tend to use.
# Covers both harnesses: Pi's `command`/`path`, Claude Code's `file_path`,
# `pattern` (Grep), `query` (WebSearch), `url` (WebFetch).
PRIMARY_ARGS = ("command", "path", "file_path", "pattern", "query", "url")


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def tool_label(tool: str, args: dict) -> str:
    """One-line human name for a tool call: `Bash: ls -la src`.

    Shared by both adapters on purpose. The label is the event NAME in the
    trace, so two harnesses computing it differently would make the same tool
    call look like two different things in the UI.
    """
    value = next((args[key] for key in PRIMARY_ARGS
                  if isinstance(args.get(key), str) and args[key].strip()), "")
    if not value:
        value = next((v for v in args.values() if isinstance(v, str) and v.strip()), "")
    value = " ".join(str(value).split())
    return f"{tool}: {clip(value, LABEL_CHARS)}" if value else tool
