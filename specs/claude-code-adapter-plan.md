# Claude Code Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `coding_agent: claude_code` real, so SSSF roster agents run through headless Claude Code (`claude -p`) on a Max subscription instead of per-token providers.

**Architecture:** A new `agent_cc.py` adapter mirrors `agent_pi.py`'s exact call contract — `run(request, on_event, on_spawn, on_exit) -> CodingAgentResult` — so `agents.execute()`, the JSON-correction loop, the gate loop, and `permissions.enforce()` all stay harness-blind. `agents.py` gains a two-entry dispatch table. Claude Code's create-only `--session-id` is bridged to Pi's create-or-continue semantics by threading a `resume` flag through `execute()`'s send sequence. Two latent defects in `agent_pi.py` (a stdout/stderr pipe deadlock, and orphaned child processes on kill) are fixed in the shared layer because the new adapter would otherwise inherit them.

**Tech Stack:** Python 3.14 · pydantic · pytest (via `uv run --with`) · sqlite3 · Claude Code CLI 2.1.278 · Vue 3 (visualizer)

**Spec:** [`specs/claude-code-adapter.md`](./claude-code-adapter.md) — 1146 lines, 24 `[probed]` findings, 11 recorded decisions. **Read it before starting.** Every task below cites the spec section it implements; where this plan and the spec disagree, the spec wins and the plan is wrong.

> **Plan location note:** the `writing-plans` skill defaults to `docs/superpowers/plans/`. This repo's established convention (set by the user for the design doc) is `specs/`, so the plan lives beside the spec it implements.

## Global Constraints

- **Edit templates ONLY** — `.claude/skills/sssf/templates/adws/...`. Never edit a stamped copy under a target repo's `adws/`. `install.py` stamps from templates.
- **Claude Code floor: 2.1.278.** `--permission-prompts` is the lowest required flag.
- **`--output-format stream-json` REQUIRES `--verbose`.** Without it: `Error: When using --print, --output-format=stream-json requires --verbose`.
- **`--session-id` must be a valid UUID and is create-only.** Reuse exits 1 with `Session ID <id> is already in use.` Continue with `--resume`. The two flags are mutually exclusive without `--fork-session`.
- **`--tools` is the availability filter; `--allowedTools` is permission pre-approval.** They are not interchangeable. An unknown name passed to `--tools` is **silently dropped**.
- **The output-contract triad is FROZEN.** No new `EnvelopeBase` subclass, no `user.md` `## Report` edit, no `output_type=` change. Swapping an agent's harness must not change what it is asked to emit.
- **Hard rule 1 (`SKILL.md`):** everything checkable fails in `agents.validate()` before any phase opens, never mid-chain.
- **Hard rule 4 (four-param rule):** any function over 4 params takes one `data_types` object instead.
- **No bare `print()` in modules** — report through `run.console`, which also traces a `log` event.
- **Rate-limit discipline:** this account was at `seven_day: 0.88` during design. Tasks 1–13 make **zero model calls** — the unit suite replays captured fixtures. Only Task 14 spends quota.
- **Test command (proven working):** `./tests/run_tests.sh` (see Task 1). `just` is NOT installed on this machine; use the `uv run` forms.

---

## File Structure

**New — fork repo root (NOT stamped by `install.py`, which only copies specific template paths):**

| File | Responsibility |
|---|---|
| `tests/conftest.py` | put `templates/adws` on `sys.path` so `from adw_modules import …` resolves; shared fixtures |
| `tests/run_tests.sh` | the one proven `uv run --with …` pytest invocation |
| `tests/fixtures/*.jsonl` | **real captured** Claude Code stream-json; makes the suite hermetic |
| `tests/fixtures/*.err` | real captured stderr, incl. the `--effort` warning |
| `tests/fixtures/fake_claude.py` | stub binary that replays a fixture — lets `run()` be tested with no API call |
| `tests/test_*.py` | one module per task's deliverable |

**Modified — `.claude/skills/sssf/templates/adws/adw_modules/`:**

| File | Responsibility after this change |
|---|---|
| `utils.py` | shared subprocess + text helpers: `kill_tree`, `claude_code_env`, `clip`, `tool_label`, `stderr_warnings` |
| `data_types.py` | `CodingAgentRequest` / `CodingAgentResult` (+ `PiRequest` / `PiResult` aliases), `ClaudeCodeDefaults` |
| `agent_pi.py` | Pi adapter — stderr→file, uses shared helpers, accepts the renamed request |
| `agent_cc.py` | **the Claude Code adapter** (currently a 15-line stub) |
| `agents.py` | `ADAPTERS` dispatch, per-agent `validate()`, `resume` threading |
| `session.py` | signal handler kills the process tree before finalizing |
| `tracer.py` | `agent_sessions.cost_basis` migration |

**Modified — elsewhere in the skill:** `templates/sssf.config.yaml`, `templates/env.sample`, `references/config.md`, `references/observability.md`, `cookbooks/update_modules.md`, `SKILL.md`, `apps/visualizer/src/components/StatChip.vue`, `apps/visualizer/src/components/PhaseDetail.vue`.

---

## Task 1: Test harness and captured fixtures

Implements: the "zero model calls" constraint. Everything downstream depends on this.

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/run_tests.sh`
- Create: `tests/fixtures/` (copied captures)
- Create: `tests/fixtures/fake_claude.py`
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: nothing.
- Produces: pytest fixtures `templates_dir: Path`, `fixture(name: str) -> list[dict]` (parsed JSONL events), `fixture_path(name: str) -> Path`, `fake_claude(tmp_path, fixture_name, *, exit_code=0, stderr_text="") -> Path` (returns a directory to prepend to `PATH` containing an executable named `claude`).

- [ ] **Step 1: Verify the captured fixtures (already copied)**

The controller copied these out of the session-scoped scratchpad before it
could be deleted, including `empty.jsonl`. **Verify them; do not re-copy** —
the source may no longer exist.

```bash
cd /Users/tylerviles/Documents/projects/factory
ls -la tests/fixtures/
python3 -c "
import json, pathlib
for f in sorted(pathlib.Path('tests/fixtures').glob('*.jsonl')):
    evs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    print(f.name, len(evs), sorted({e.get('type') for e in evs}))
"
```
Expected: `tool_use_roundtrip.jsonl` 10 events including `user`; `resume_run.jsonl` 7;
`restricted_bash.jsonl` 16; `effort_off.jsonl` 9; `empty.jsonl` 0; plus
`effort_off.err` containing a `Warning:` line.

<details><summary>Only if a fixture is missing — the original copy step</summary>

```bash
cd /Users/tylerviles/Documents/projects/factory
SC=/private/tmp/claude-501/-Users-tylerviles-Documents-projects-factory/552e3eee-1dc6-4b30-b4e0-7d0d5835c5af/scratchpad
mkdir -p tests/fixtures
cp "$SC/cc-probe/run1.jsonl"          tests/fixtures/tool_use_roundtrip.jsonl
cp "$SC/cc-probe/r.jsonl"             tests/fixtures/resume_run.jsonl
cp "$SC/out/L6-restricted.jsonl"      tests/fixtures/restricted_bash.jsonl
cp "$SC/out/E-off.jsonl"              tests/fixtures/effort_off.jsonl
cp "$SC/out/E-off.err"                tests/fixtures/effort_off.err
ls -la tests/fixtures/
```

If the scratchpad is gone, regenerate per Task 12's capture recipe — do **not**
hand-write fixtures.

</details>

- [ ] **Step 2: Write `tests/conftest.py`**

```python
"""Test harness for the SSSF skill templates.

The templates are not an installed package — they are files install.py stamps
into a target repo. Tests import them by putting the template dir on sys.path,
which is exactly how a stamped repo imports them (adws/ is the cwd there).
"""
import json
import os
import stat
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / ".claude" / "skills" / "sssf" / "templates" / "adws"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

sys.path.insert(0, str(TEMPLATES))


@pytest.fixture
def templates_dir() -> Path:
    return TEMPLATES


@pytest.fixture
def fixture_path():
    def _path(name: str) -> Path:
        p = FIXTURES / name
        assert p.exists(), f"missing fixture {name}; see Task 1 Step 1"
        return p
    return _path


@pytest.fixture
def fixture(fixture_path):
    """Parse a captured stream-json capture into a list of events."""
    def _load(name: str) -> list[dict]:
        events = []
        for line in fixture_path(name).read_text().splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
        return events
    return _load


@pytest.fixture
def fake_claude(fixture_path):
    """A directory holding an executable `claude` that replays a fixture.

    Lets agent_cc.run() be exercised end to end with no API call. The stub
    writes the fixture to stdout line by line, writes stderr_text to stderr,
    and exits with exit_code. It also records its argv to `argv.json` in the
    same directory so tests can assert on command construction.
    """
    def _make(tmp_path: Path, fixture_name: str, *, exit_code: int = 0,
              stderr_text: str = "") -> Path:
        bindir = tmp_path / "fakebin"
        bindir.mkdir(exist_ok=True)
        script = bindir / "claude"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys, pathlib\n"
            f"pathlib.Path({str(bindir / 'argv.json')!r}).write_text(json.dumps(sys.argv[1:]))\n"
            f"sys.stderr.write({stderr_text!r})\n"
            f"sys.stdout.write(pathlib.Path({str(fixture_path(fixture_name))!r}).read_text())\n"
            f"sys.exit({exit_code})\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return bindir
    return _make


@pytest.fixture
def fake_claude_argv():
    def _read(bindir: Path) -> list[str]:
        return json.loads((bindir / "argv.json").read_text())
    return _read
```

- [ ] **Step 3: Write `tests/run_tests.sh`**

```bash
#!/usr/bin/env bash
# The SSSF templates declare deps inline (PEP 723); tests need the same set
# plus pytest. `uv run --with` assembles it without polluting the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --quiet \
  --with pytest --with pydantic --with pyyaml --with python-dotenv --with rich \
  pytest tests "$@"
```

Then: `chmod +x tests/run_tests.sh`

- [ ] **Step 4: Write the failing test**

`tests/test_harness.py`:

```python
def test_templates_import(templates_dir):
    from adw_modules import agent_cc, agent_pi, agents, data_types, permissions, utils
    assert (templates_dir / "adw_modules" / "agent_cc.py").exists()


def test_roundtrip_fixture_has_the_shapes_we_rely_on(fixture):
    events = fixture("tool_use_roundtrip.jsonl")
    types = [(e.get("type"), e.get("subtype")) for e in events]
    assert ("system", "init") in types
    assert ("result", "success") in types
    tool_uses = [b for e in events if e.get("type") == "assistant"
                 for b in (e.get("message") or {}).get("content", [])
                 if isinstance(b, dict) and b.get("type") == "tool_use"]
    tool_results = [b for e in events if e.get("type") == "user"
                    for b in (e.get("message") or {}).get("content", [])
                    if isinstance(b, dict) and b.get("type") == "tool_result"]
    assert len(tool_uses) == 1 and len(tool_results) == 1
    assert tool_uses[0]["id"] == tool_results[0]["tool_use_id"]


def test_fake_claude_replays_a_fixture(tmp_path, fake_claude, fake_claude_argv):
    import subprocess
    bindir = fake_claude(tmp_path, "tool_use_roundtrip.jsonl", stderr_text="hi\n")
    out = subprocess.run([str(bindir / "claude"), "-p", "x"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert '"type":"result"' in out.stdout.replace(" ", "")
    assert out.stderr == "hi\n"
    assert fake_claude_argv(bindir) == ["-p", "x"]
```

- [ ] **Step 5: Run to verify it fails**

Run: `./tests/run_tests.sh tests/test_harness.py -v`
Expected: FAIL — `conftest.py` / fixtures missing, or `assert p.exists()` on a fixture.

- [ ] **Step 6: Make it pass**

Complete Steps 1–3 above if not already done, then re-run.

- [ ] **Step 7: Run to verify it passes**

Run: `./tests/run_tests.sh tests/test_harness.py -v`
Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add tests/
git commit -m "test: add hermetic harness with captured Claude Code fixtures

Unit tests replay real stream-json captures so the suite makes zero
model calls. fake_claude replays a fixture as an executable stub."
```

---

## Task 2: Shared subprocess safety — stderr to a file, and kill the process tree

Implements: spec §8 "stderr goes to a FILE" and "Kill children first" (decision Q6-4). **Fixes `agent_pi.py` too** — it carries the same latent deadlock.

**Files:**
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/utils.py`
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/agent_pi.py:243-285`
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/session.py:21-35`
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/runner.py` (`Run.live_children`)
- Test: `tests/test_subprocess_io.py`

**Interfaces:**
- Consumes: Task 1 harness.
- Produces: `utils.kill_tree(pid: int, grace: float = 5.0) -> None`; `utils.stderr_warnings(path: Path, limit: int = 20) -> list[str]`; `utils.claude_code_env(inherit_api_key: bool = False) -> dict[str, str]`.

- [ ] **Step 1: Write the failing test**

`tests/test_subprocess_io.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `./tests/run_tests.sh tests/test_subprocess_io.py -v`
Expected: FAIL — `ImportError: cannot import name 'stderr_warnings' from 'adw_modules.utils'`.

- [ ] **Step 3: Add the helpers to `utils.py`**

Append to `.claude/skills/sssf/templates/adws/adw_modules/utils.py`:

```python
import os
import signal
import time

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


def stderr_warnings(path: str | Path, limit: int = 20) -> list[str]:
    """Warning/error lines from a child's stderr log, for the trace.

    The CLI reports real problems here that nothing in this system reads —
    `Warning: Unknown --effort value 'off' …` exits 0 and never reaches the
    trace, so a misconfigured roster looks fine. Returns [] when there is no
    log, which is the common case.
    """
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return []
    hits = [ln.strip() for ln in text.splitlines()
            if ln.strip().startswith(("Warning:", "Error:", "warning:", "error:"))]
    return hits[:limit]
```

- [ ] **Step 4: Run to verify the util tests pass**

Run: `./tests/run_tests.sh tests/test_subprocess_io.py -v`
Expected: 6 passed.

- [ ] **Step 5: Apply the fix to `agent_pi.py`**

In `agent_pi.run()`, replace the Popen block and the trailing stderr read.

Replace:
```python
    process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, bufsize=1, cwd=request.cwd,
                               env=operator_env())
    if on_spawn:
        on_spawn(process.pid)
    with raw_path.open("a") as raw:
```
with:
```python
    # stderr goes to a FILE, not a pipe. With both as pipes and a blocking
    # stdout read, a child that fills the ~64KB stderr buffer blocks writing
    # stderr, stops producing stdout, and both sides wait forever — the same
    # silent 0%-CPU hang the stdin comment below describes, through the other
    # pipe. A file has no fixed-size buffer, so it cannot happen.
    stderr_path = raw_path.with_name("stderr.log")
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stderr_path.open("a") as err:
        process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=err,
                                   text=True, bufsize=1, cwd=request.cwd,
                                   env=operator_env(), start_new_session=True)
    if on_spawn:
        on_spawn(process.pid)
    with raw_path.open("a") as raw:
```

Replace:
```python
    stderr = process.stderr.read() if process.stderr else ""
    result.returncode = process.wait()
```
with:
```python
    result.returncode = process.wait()
    stderr = "\n".join(stderr_warnings(stderr_path)) or \
        Path(stderr_path).read_text(errors="replace")[-800:]
```

Add `stderr_warnings` to the existing `from .utils import …` line.

- [ ] **Step 6: Track this process's own children, and reap them on a signal**

Add to `runner.Run.__init__`:

```python
        # Children THIS process spawned and believes are alive. Deliberately
        # in memory rather than read back from the processes table: the table
        # also holds rows from older runs and from runs that crashed without
        # closing them, and a recycled pid handed to os.killpg can take out an
        # unrelated process GROUP. The question a signal handler must answer is
        # not "what does the trace believe is running" but "what did I start".
        self.live_children: set[int] = set()
```

> The `on_spawn`/`on_exit` wiring that POPULATES this set belongs to **Task 8**,
> which rewrites `send()` wholesale — editing that closure here would guarantee a
> conflict. Until Task 8 lands, the handler reaps an empty set, which is exactly
> today's behaviour, so nothing regresses.

In `session._finalize_when_killed`, replace the handler body:

```python
    def handler(signum, _frame):
        # Reap before closing the trace. Closing the rows first would record
        # the run as finished while its coding agent kept working — a killed
        # ADW that leaves a `claude` or `pi` child running is exactly the pid
        # nobody can find afterwards.
        for pid in list(run.live_children):
            kill_tree(pid)
        run.tracer.session_finish(run.adw_id, ok=False)   # also closes process rows
        raise SystemExit(128 + signum)
```

Update `session.py` imports: `from .utils import engineer_name, kill_tree, new_id`.

> **Deliberately NOT added: `tracer.live_pids()`.** Killing *another* run's
> processes by `adw_id` (the `just kill` recipe `SKILL.md` mentions on the
> example branch) does need the db — and it needs the `command` column too,
> which `tracer.py:76` exists for: *"what the pid was, so a recycled pid is not
> killed by mistake"*. Shipping an unguarded `live_pids()` here would invite
> exactly that mistake. Out of scope; noted so the next person builds it with
> the guard.

- [ ] **Step 7: Run the whole suite**

Run: `./tests/run_tests.sh -v`
Expected: all pass (Task 1's 3 + Task 2's 6).

- [ ] **Step 8: Commit**

```bash
git add .claude/skills/sssf/templates/adws/adw_modules/{utils.py,agent_pi.py,session.py,runner.py} tests/test_subprocess_io.py
git commit -m "fix: stderr to file and kill process tree on signal

agent_pi opened stdout and stderr as pipes but read them sequentially,
deadlocking once a child filled the stderr buffer (reproduced: 200KB to
stderr, parent blocked indefinitely at 0% CPU). Children are now spawned
as their own process group so a killed run can reap grandchildren."
```

---

## Task 3: Shared tool-call helpers, and the renamed request/result types

Implements: spec §3 ("move them to `utils.py` so both adapters share one definition and the tool_call payload cannot drift between harnesses") and §1 Naming.

**Files:**
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/utils.py`
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/data_types.py:373-447`
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/agent_pi.py:26-31,120-138`
- Test: `tests/test_shared_helpers.py`

**Interfaces:**
- Consumes: Task 2's `utils`.
- Produces: `utils.clip(text: str, limit: int) -> str`; `utils.tool_label(tool: str, args: dict) -> str`; constants `utils.RESULT_SNIPPET_CHARS = 20_000`, `utils.ARG_VALUE_CHARS = 20_000`, `utils.LABEL_CHARS = 80`, `utils.PRIMARY_ARGS`. Types `data_types.CodingAgentRequest`, `data_types.CodingAgentResult`, aliases `PiRequest`, `PiResult`, and `data_types.ClaudeCodeDefaults`.

- [ ] **Step 1: Write the failing test**

`tests/test_shared_helpers.py`:

```python
def test_clip_leaves_short_text_alone():
    from adw_modules.utils import clip
    assert clip("hello", 10) == "hello"


def test_clip_truncates_with_ellipsis():
    from adw_modules.utils import clip
    assert clip("abcdefghij", 5) == "abcde…"


def test_tool_label_prefers_command_then_file_path():
    from adw_modules.utils import tool_label
    assert tool_label("Bash", {"command": "ls  -la   src"}) == "Bash: ls -la src"
    assert tool_label("Read", {"file_path": "/a/b.py"}) == "Read: /a/b.py"


def test_tool_label_falls_back_to_any_string_then_bare_name():
    from adw_modules.utils import tool_label
    assert tool_label("X", {"weird": "value"}) == "X: value"
    assert tool_label("X", {"n": 3}) == "X"


def test_pi_helpers_are_the_shared_ones():
    """agent_pi must not keep private copies — the payload would drift."""
    from adw_modules import agent_pi, utils
    assert agent_pi._clip is utils.clip
    assert agent_pi._label is utils.tool_label


def test_renamed_types_with_back_compat_aliases():
    from adw_modules.data_types import (CodingAgentRequest, CodingAgentResult,
                                        PiRequest, PiResult)
    assert PiRequest is CodingAgentRequest
    assert PiResult is CodingAgentResult


def test_request_defaults_for_new_fields():
    from adw_modules.data_types import CodingAgentRequest
    r = CodingAgentRequest(prompt="p", system_prompt="s", model="m",
                           session_id="sid", session_dir="d",
                           raw_output_path="raw.jsonl")
    assert r.resume is False
    assert r.system_prompt_path == ""
    assert r.stderr_path == ""
    assert r.timeout_seconds == 1800


def test_claude_code_defaults():
    from adw_modules.data_types import ClaudeCodeDefaults
    d = ClaudeCodeDefaults()
    assert d.inherit_api_key is False
    assert d.on_overage == "fail"
    assert d.timeout_seconds == 1800
    assert d.max_utilization == 1.0


def test_config_defaults_carry_a_claude_code_block():
    from adw_modules.data_types import SSSFConfig
    cfg = SSSFConfig()
    assert cfg.defaults.claude_code.inherit_api_key is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `./tests/run_tests.sh tests/test_shared_helpers.py -v`
Expected: FAIL — `ImportError: cannot import name 'clip' from 'adw_modules.utils'`.

- [ ] **Step 3: Move the helpers into `utils.py`**

Append to `utils.py`:

```python
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
```

- [ ] **Step 4: Point `agent_pi.py` at them**

Delete `RESULT_SNIPPET_CHARS`, `ARG_VALUE_CHARS`, `LABEL_CHARS`, `PRIMARY_ARGS`, `_clip` and `_label` from `agent_pi.py`, and add to its `from .utils import` line:

```python
from .utils import (ARG_VALUE_CHARS, LABEL_CHARS, PRIMARY_ARGS,
                    RESULT_SNIPPET_CHARS, clip as _clip, now_iso, operator_env,
                    stderr_warnings, tool_label as _label)
```

The `as _clip` / `as _label` aliases keep the existing call sites in
`ToolCallTracker.observe()` unchanged, and make the identity assertion in the
test meaningful.

Also, now that `CodingAgentRequest.stderr_path` exists (Step 5 below), make
`agent_pi` honour it. Task 2 deliberately derived the path unconditionally
because the field did not exist yet; replace that line with:

```python
    stderr_path = Path(request.stderr_path) if request.stderr_path else \
        raw_path.with_name("stderr.log")
```

- [ ] **Step 5: Rename the types in `data_types.py`**

Replace the `class PiRequest(BaseModel):` block with:

```python
class CodingAgentRequest(BaseModel):
    """Everything one non-interactive coding-agent turn needs.

    One type for both adapters. Fields a given harness cannot use are inert
    there rather than duplicated into a parallel type: Pi ignores `resume`
    because `--session-id` already creates-or-continues, and Claude Code
    ignores `session_dir` because it stores transcripts under its own
    projects directory (spec §2).
    """

    prompt: str
    system_prompt: str              # rendered text — Pi passes this in argv
    # The same text, already on disk at {agent_dir}/prompts/system.md. Claude
    # Code takes --append-system-prompt-file, which keeps the largest argument
    # out of argv AND makes the audit copy literally the bytes that were sent.
    system_prompt_path: str = ""
    model: str                      # provider/model-id
    thinking: str = "medium"
    session_id: str                 # sssf id; Claude Code maps it to a uuid
    # False = create the session, True = continue it. Pi ignores this.
    resume: bool = False
    session_dir: str = ""           # Pi only
    raw_output_path: str            # JSONL stream lands here
    stderr_path: str = ""           # child stderr; "" = beside raw_output
    tools: Optional[list[str]] = None
    extensions: list[str] = Field(default_factory=list)
    cwd: str = "."                  # run.repo_root — the codebase agents work in
    timeout_seconds: int = 1800     # wall clock; 0 disables


PiRequest = CodingAgentRequest      # back-compat alias
```

Rename `class PiResult(BaseModel):` to `class CodingAgentResult(BaseModel):`, add these two fields, and add the alias:

```python
    # "billed" (pi: real money) vs "list" (claude_code on a subscription:
    # notional list price). The UI must not present the two the same way.
    cost_basis: str = "billed"
    # The last rate_limit_event's `rate_limit_info`, verbatim. Stored whole
    # rather than as a bare utilisation float because `unifiedWindows` and
    # `resetsAt` are what let a later read tell a live window from one that has
    # since reset — without them the headroom check is guesswork.
    rate_limit: dict = Field(default_factory=dict)


PiResult = CodingAgentResult        # back-compat alias
```

- [ ] **Step 6: Add the Claude Code config block**

In `data_types.py`, above `class ConfigDefaults`:

```python
class ClaudeCodeDefaults(BaseModel):
    """Knobs that only mean something for coding_agent: claude_code."""

    # False strips ANTHROPIC_API_KEY et al from the child env, which is what
    # keeps a run on the Max subscription instead of per-token API billing.
    inherit_api_key: bool = False
    timeout_seconds: int = 1800
    # A subscription past its limits falls through to PAID overage. "fail"
    # aborts the moment that is observed; "warn" logs and continues.
    on_overage: Literal["fail", "warn"] = "fail"
    # Refuse to START a chain when a LIVE rate-limit window was last observed
    # at or above this utilisation. 1.0 (the default) refuses only a window
    # read as fully exhausted — which can never be a false alarm, because a
    # recorded utilisation is a lower bound until its resetsAt passes.
    max_utilization: float = 1.0
```

And inside `ConfigDefaults`, add:

```python
    claude_code: ClaudeCodeDefaults = Field(default_factory=ClaudeCodeDefaults)
```

- [ ] **Step 7: Run to verify it passes**

Run: `./tests/run_tests.sh -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add .claude/skills/sssf/templates/adws/adw_modules/{utils.py,data_types.py,agent_pi.py} tests/test_shared_helpers.py
git commit -m "refactor: share tool-call helpers, rename request/result types

clip/tool_label move to utils so both adapters compute the same trace
label. PiRequest/PiResult become CodingAgentRequest/CodingAgentResult
with aliases kept so nothing upstream breaks."
```

---

## Task 4: `agent_cc` pure mappers — model, effort, tools

Implements: spec §4 (Model mapping), §5 (Thinking-level mapping), §6 (Tool mapping, incl. decision Q6-1).

**Files:**
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/agent_cc.py` (replaces the stub)
- Test: `tests/test_agent_cc_mapping.py`

**Interfaces:**
- Consumes: Task 3's `utils` constants.
- Produces: `agent_cc.resolve_model(pattern: str) -> str` (raises `ValueError`); `agent_cc.map_effort(thinking: str) -> tuple[str, str]` returning `(level, warning_or_empty)`; `agent_cc.map_tools(tools: list[str] | None) -> tuple[list[str], list[str]]` returning `(claude_tool_names, warnings)` and raising `ValueError` on an unknown name; constants `agent_cc.TOOL_MAP`, `agent_cc.PI_EQUIVALENT_TOOLS`, `agent_cc.EFFORT_LEVELS`.

- [ ] **Step 1: Write the failing test**

`tests/test_agent_cc_mapping.py`:

```python
import pytest


# ── model ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pattern,expected", [
    ("anthropic/claude-opus-5", "claude-opus-5"),
    ("anthropic/claude-sonnet-5", "claude-sonnet-5"),
    ("anthropic/claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
    ("anthropic/opus", "opus"),
])
def test_resolve_model_strips_the_anthropic_provider(pattern, expected):
    from adw_modules.agent_cc import resolve_model
    assert resolve_model(pattern) == expected


@pytest.mark.parametrize("bad", [
    "openai/gpt-5.6-terra",
    "google/gemini-3.6-flash",
    "fireworks/accounts/fireworks/models/kimi-k3",
    "claude-opus-5",          # unqualified — provider is not optional
    "",
])
def test_resolve_model_rejects_non_anthropic_and_unqualified(bad):
    from adw_modules.agent_cc import resolve_model
    with pytest.raises(ValueError) as e:
        resolve_model(bad)
    assert "anthropic/" in str(e.value)


# ── effort ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
def test_map_effort_passes_supported_levels_through_silently(level):
    from adw_modules.agent_cc import map_effort
    assert map_effort(level) == (level, "")


@pytest.mark.parametrize("level", ["off", "minimal"])
def test_map_effort_clamps_unsupported_levels_and_warns(level):
    from adw_modules.agent_cc import map_effort
    mapped, warning = map_effort(level)
    assert mapped == "low"
    assert level in warning and "low" in warning


def test_map_effort_rejects_nonsense():
    from adw_modules.agent_cc import map_effort
    with pytest.raises(ValueError):
        map_effort("turbo")


# ── tools ────────────────────────────────────────────────────────────────────

def test_map_tools_translates_the_roster_names():
    from adw_modules.agent_cc import map_tools
    mapped, _ = map_tools(["read", "bash", "edit", "write", "grep", "find"])
    assert mapped == ["Read", "Bash", "Edit", "Write", "Grep", "Glob"]


def test_map_tools_drops_ls_with_a_warning():
    """`ls` has no Claude Code tool. Mapping it to Bash would hand a
    `writes: []` agent shell access it was never granted."""
    from adw_modules.agent_cc import map_tools
    mapped, warnings = map_tools(["read", "ls"])
    assert mapped == ["Read"]
    assert any("ls" in w for w in warnings)


def test_map_tools_none_clamps_to_the_pi_equivalent_set():
    """Decision Q6-1: omitting --tools yields 28 tools including CronCreate,
    RemoteTrigger and EnterWorktree (which moves cwd and breaks both the
    writes boundary and session resumption)."""
    from adw_modules.agent_cc import map_tools, PI_EQUIVALENT_TOOLS
    mapped, _ = map_tools(None)
    assert mapped == PI_EQUIVALENT_TOOLS
    assert mapped == ["Read", "Bash", "Edit", "Write", "Grep", "Glob"]
    for dangerous in ("CronCreate", "RemoteTrigger", "EnterWorktree",
                      "Workflow", "Skill", "PushNotification"):
        assert dangerous not in mapped


def test_map_tools_accepts_exact_claude_code_names():
    from adw_modules.agent_cc import map_tools
    mapped, _ = map_tools(["read", "Task", "WebFetch"])
    assert mapped == ["Read", "Task", "WebFetch"]


def test_map_tools_rejects_an_unknown_name_instead_of_silently_dropping_it():
    """--tools BogusTool is silently dropped by the CLI, so a typo would cost
    a whole run. Catch it in validate() instead."""
    from adw_modules.agent_cc import map_tools
    with pytest.raises(ValueError) as e:
        map_tools(["read", "subagent_create"])
    assert "subagent_create" in str(e.value)


def test_map_tools_rejects_an_empty_list():
    from adw_modules.agent_cc import map_tools
    with pytest.raises(ValueError) as e:
        map_tools([])
    assert "stall" in str(e.value).lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `./tests/run_tests.sh tests/test_agent_cc_mapping.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_model' from 'adw_modules.agent_cc'`.

- [ ] **Step 3: Replace `agent_cc.py`'s stub with the mappers**

```python
"""Claude Code coding-agent interface.

Runs `claude -p --output-format stream-json --verbose` and tails its JSONL
stdout line by line, forwarding each event to a callback WHILE the agent works
— the same streaming contract agent_pi.py provides, so agents.execute() cannot
tell the two apart.

Two things differ from Pi and drive most of this module:

  * `--session-id` CREATES and fails if the id exists; continuing needs
    `--resume`. Pi's one flag does both. `CodingAgentRequest.resume` carries
    the distinction (spec §2).
  * `--tools` is the availability filter, `--allowedTools` is permission
    pre-approval. They are not interchangeable, and an unknown name handed to
    `--tools` is silently dropped (spec §6).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from .data_types import CodingAgentRequest, CodingAgentResult, UsageBreakdown
from .utils import (ARG_VALUE_CHARS, RESULT_SNIPPET_CHARS, claude_code_env,
                    clip, kill_tree, now_iso, stderr_warnings, tool_label)

CLAUDE_PATH = os.environ.get("CLAUDE_PATH", "claude")

# Claude Code's documented reasoning ladder. `off` and `minimal` are absent:
# the CLI accepts them at parse time but warns on stderr at runtime and falls
# back to its OWN default effort, which is neither what the roster asked for
# nor the lowest setting. Clamping to `low` is the closest honouring of intent
# the CLI actually offers (spec §5).
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
EFFORT_CLAMP = {"off": "low", "minimal": "low"}

# Roster (Pi) tool names -> Claude Code tool names.
TOOL_MAP = {
    "read": "Read", "bash": "Bash", "edit": "Edit",
    "write": "Write", "grep": "Grep", "find": "Glob",
}
# `ls` is deliberately absent: Claude Code has no Ls tool, and mapping it to
# Bash would grant shell access to an agent that only asked to list a
# directory — exactly the boundary `writes` exists to keep.
TOOL_DROPPED = {"ls"}

# What `tools: None` resolves to. NOT "omit --tools": that yields 28 tools
# including CronCreate (schedules work outliving the run), RemoteTrigger and
# PushNotification (reach off the machine), Workflow (spawns agent fleets),
# Skill (can reach the sssf skill and recurse into the factory) and
# EnterWorktree (moves cwd, breaking BOTH permissions.enforce and session
# resumption). A roster must mean the same capability on both harnesses.
PI_EQUIVALENT_TOOLS = ["Read", "Bash", "Edit", "Write", "Grep", "Glob"]

# Exact Claude Code tool names a roster may name directly.
CC_TOOL_NAMES = {
    "Task", "Bash", "Edit", "Glob", "Grep", "NotebookEdit", "Read",
    "Skill", "ToolSearch", "WebFetch", "WebSearch", "Write",
}


def resolve_model(pattern: str) -> str:
    """`anthropic/claude-opus-5` -> `claude-opus-5`.

    Rejecting a non-anthropic provider HERE, from validate(), is the point:
    a roster that points a claude_code agent at openai/ must fail before any
    phase opens rather than mid-chain (hard rule 1).
    """
    provider, _, model_id = pattern.partition("/")
    if not model_id or provider != "anthropic":
        raise ValueError(
            f"model {pattern!r} must be written as anthropic/<model-id> when "
            f"coding_agent is claude_code (e.g. anthropic/claude-opus-5)")
    return model_id


def map_effort(thinking: str) -> tuple[str, str]:
    """Returns (effort_level, warning). An empty warning means a clean map."""
    if thinking in EFFORT_LEVELS:
        return thinking, ""
    if thinking in EFFORT_CLAMP:
        mapped = EFFORT_CLAMP[thinking]
        return mapped, (f"thinking {thinking!r} is not a Claude Code effort "
                        f"level; using {mapped!r}")
    raise ValueError(f"thinking {thinking!r} is not a known level "
                     f"(off|minimal|{'|'.join(EFFORT_LEVELS)})")


def map_tools(tools: Optional[list[str]]) -> tuple[list[str], list[str]]:
    """Returns (claude_code_tool_names, warnings).

    Raises on an unknown name rather than passing it through, because the CLI
    drops an unrecognised --tools entry SILENTLY — the run succeeds and the
    tool is simply never offered to the model.
    """
    if tools is None:
        return list(PI_EQUIVALENT_TOOLS), []
    if not tools:
        raise ValueError("tools: [] is a tool-less agent, and it will stall — "
                         "omit the key for the default set, or name the tools")
    mapped, warnings, unknown = [], [], []
    for name in tools:
        if name in TOOL_DROPPED:
            warnings.append(f"tool {name!r} has no Claude Code equivalent and "
                            f"was dropped (Read on a directory and Glob cover it)")
        elif name in TOOL_MAP:
            mapped.append(TOOL_MAP[name])
        elif name in CC_TOOL_NAMES:
            mapped.append(name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"tool(s) {unknown} are not Claude Code tools. --tools drops an "
            f"unknown name silently, so this would cost a run. Known roster "
            f"names: {sorted(TOOL_MAP) + sorted(TOOL_DROPPED)}; or name a "
            f"Claude Code tool exactly: {sorted(CC_TOOL_NAMES)}")
    if not mapped:
        raise ValueError("no usable tools after mapping — the agent will stall")
    return mapped, warnings
```

- [ ] **Step 4: Run to verify it passes**

Run: `./tests/run_tests.sh tests/test_agent_cc_mapping.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/sssf/templates/adws/adw_modules/agent_cc.py tests/test_agent_cc_mapping.py
git commit -m "feat(agent_cc): model, effort and tool mapping

tools: None clamps to the six-tool pi-equivalent set rather than
omitting --tools, which would grant 28 tools including cron, remote
triggers and worktree switching."
```

---

## Task 5: Auth preflight and session identity

Implements: spec §7a (`claude auth status` preflight, decision Q6-6) and §2 (deterministic uuid, create-vs-resume).

**Files:**
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/agent_cc.py`
- Test: `tests/test_agent_cc_auth_session.py`

**Interfaces:**
- Consumes: Task 4's `agent_cc`.
- Produces: `agent_cc.parse_auth_status(raw: str, inherit_api_key: bool) -> dict` (raises `ValueError`); `agent_cc.preflight_auth(inherit_api_key: bool = False) -> dict`; `agent_cc.cc_session_uuid(sssf_session_id: str) -> str`; `agent_cc.CC_NAMESPACE`.

- [ ] **Step 1: Write the failing test**

`tests/test_agent_cc_auth_session.py`:

```python
import json
import uuid

import pytest

SUBSCRIPTION = json.dumps({
    "loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
    "projectsDirectory": "/Users/x/.claude/projects",
    "configDirectory": "/Users/x/.claude", "subscriptionType": "max"})

WITH_API_KEY = json.dumps({
    "loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
    "projectsDirectory": "/Users/x/.claude/projects",
    "apiKeySource": "ANTHROPIC_API_KEY", "subscriptionType": None, "email": None})

LOGGED_OUT = json.dumps({"loggedIn": False, "authMethod": None})

PRO = json.dumps({
    "loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
    "projectsDirectory": "/Users/x/.claude/projects", "subscriptionType": "pro"})


def test_parse_auth_status_accepts_a_subscription():
    from adw_modules.agent_cc import parse_auth_status
    info = parse_auth_status(SUBSCRIPTION, inherit_api_key=False)
    assert info["subscriptionType"] == "max"
    assert info["projectsDirectory"] == "/Users/x/.claude/projects"


def test_parse_auth_status_accepts_pro_because_plan_name_is_never_gated_on():
    """subscriptionType is an Anthropic-controlled string; gating on a value
    would fail a Pro seat, a Team seat, or any renamed plan."""
    from adw_modules.agent_cc import parse_auth_status
    assert parse_auth_status(PRO, inherit_api_key=False)["subscriptionType"] == "pro"


def test_parse_auth_status_rejects_logged_out():
    from adw_modules.agent_cc import parse_auth_status
    with pytest.raises(ValueError) as e:
        parse_auth_status(LOGGED_OUT, inherit_api_key=False)
    assert "not logged in" in str(e.value).lower()


def test_parse_auth_status_rejects_an_api_key_when_not_inheriting():
    """apiKeySource present means the run would bill the API, not the
    subscription — the exact failure this feature exists to prevent."""
    from adw_modules.agent_cc import parse_auth_status
    with pytest.raises(ValueError) as e:
        parse_auth_status(WITH_API_KEY, inherit_api_key=False)
    assert "ANTHROPIC_API_KEY" in str(e.value)


def test_parse_auth_status_allows_an_api_key_when_explicitly_asked():
    from adw_modules.agent_cc import parse_auth_status
    info = parse_auth_status(WITH_API_KEY, inherit_api_key=True)
    assert info["apiKeySource"] == "ANTHROPIC_API_KEY"


def test_parse_auth_status_rejects_garbage():
    from adw_modules.agent_cc import parse_auth_status
    with pytest.raises(ValueError):
        parse_auth_status("not json at all", inherit_api_key=False)


# ── session identity ─────────────────────────────────────────────────────────

def test_cc_session_uuid_is_a_valid_uuid():
    """--session-id errors with 'Must be a valid UUID' otherwise."""
    from adw_modules.agent_cc import cc_session_uuid
    got = cc_session_uuid("sssf-a1b2c3d4-planner-9f8e")
    assert str(uuid.UUID(got)) == got


def test_cc_session_uuid_is_deterministic():
    """An --adw-id rejoin must recompute the SAME uuid from agent_map's sssf
    id, with no extra stored state."""
    from adw_modules.agent_cc import cc_session_uuid
    a = cc_session_uuid("sssf-a1b2c3d4-planner-9f8e")
    b = cc_session_uuid("sssf-a1b2c3d4-planner-9f8e")
    assert a == b


def test_cc_session_uuid_differs_per_agent():
    from adw_modules.agent_cc import cc_session_uuid
    assert cc_session_uuid("sssf-a1-planner-01") != cc_session_uuid("sssf-a1-builder-01")
```

- [ ] **Step 2: Run to verify it fails**

Run: `./tests/run_tests.sh tests/test_agent_cc_auth_session.py -v`
Expected: FAIL — `ImportError: cannot import name 'parse_auth_status'`.

- [ ] **Step 3: Implement**

Append to `agent_cc.py`:

```python
# Fixed namespace so an sssf session id always maps to the same Claude Code
# uuid. Deterministic beats a stored uuid4: agent_map.json keeps its shape,
# the mapping is reproducible from the trace when you need to `claude --resume`
# a dead agent by hand, and an --adw-id rejoin needs no extra state.
CC_NAMESPACE = uuid.UUID("6f1f5b1e-3d0a-5e7c-9a2b-7c4d8e0f1a23")


def cc_session_uuid(sssf_session_id: str) -> str:
    """sssf-<adw_id>-<agent>-<rand4> -> a stable uuid `--session-id` accepts."""
    return str(uuid.uuid5(CC_NAMESPACE, sssf_session_id))


def parse_auth_status(raw: str, inherit_api_key: bool) -> dict:
    """Validate `claude auth status` output. Raises ValueError with the fix.

    Checking `init.apiKeySource == "none"` alone is NOT enough: "none" means
    "no API key in use", and an unauthenticated session reports it too. This
    is the positive check — that a usable credential exists AND that it is the
    subscription rather than an API key.
    """
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"could not parse `claude auth status` output: {error}"
                         f"\n{raw[:400]}") from error
    if not info.get("loggedIn"):
        raise ValueError("Claude Code is not logged in — run `claude auth login`. "
                         "(`claude auth status` reports loggedIn: false)")
    if not inherit_api_key:
        if info.get("apiKeySource"):
            raise ValueError(
                f"Claude Code would bill the API, not your subscription: "
                f"apiKeySource={info['apiKeySource']!r}. Unset it, or set "
                f"defaults.claude_code.inherit_api_key: true to allow it.")
        if info.get("authMethod") != "claude.ai":
            raise ValueError(
                f"expected subscription auth (authMethod 'claude.ai'), got "
                f"{info.get('authMethod')!r}. Set "
                f"defaults.claude_code.inherit_api_key: true to allow it.")
    # subscriptionType is RECORDED, never gated on — it is an Anthropic-
    # controlled plan name, and requiring a value would fail a Pro seat, a
    # Team/Enterprise seat, or any future rename.
    return info


def preflight_auth(inherit_api_key: bool = False) -> dict:
    """Run `claude auth status` under the stripped child env. No model call.

    Belongs in validate(): hard rule 1 says nothing spawns against a
    half-valid config, and an unauthenticated CLI is exactly that.
    """
    try:
        result = subprocess.run(
            [CLAUDE_PATH, "auth", "status"], capture_output=True, text=True,
            timeout=30, stdin=subprocess.DEVNULL,
            env=claude_code_env(inherit_api_key), check=False)
    except FileNotFoundError as error:
        raise ValueError(f"the `claude` CLI was not found on PATH "
                         f"(CLAUDE_PATH={CLAUDE_PATH!r})") from error
    except subprocess.TimeoutExpired as error:
        raise ValueError("`claude auth status` timed out after 30s") from error
    if result.returncode != 0:
        raise ValueError(f"`claude auth status` exited {result.returncode}: "
                         f"{result.stderr.strip()[:400]}")
    return parse_auth_status(result.stdout, inherit_api_key)
```

- [ ] **Step 4: Run to verify it passes**

Run: `./tests/run_tests.sh tests/test_agent_cc_auth_session.py -v`
Expected: 9 passed.

- [ ] **Step 5: Smoke-test the preflight against the real CLI (no model call)**

```bash
cd /Users/tylerviles/Documents/projects/factory
uv run --quiet --with pydantic --with python-dotenv --with pyyaml --with rich python -c "
import sys; sys.path.insert(0, '.claude/skills/sssf/templates/adws')
from adw_modules.agent_cc import preflight_auth
info = preflight_auth()
print('authMethod      :', info.get('authMethod'))
print('subscriptionType:', info.get('subscriptionType'))
print('projectsDirectory:', info.get('projectsDirectory'))
"
```
Expected: `authMethod: claude.ai`, a non-null `subscriptionType`, a real `projectsDirectory`. No model call, no quota spent.

- [ ] **Step 6: Commit**

```bash
git add .claude/skills/sssf/templates/adws/adw_modules/agent_cc.py tests/test_agent_cc_auth_session.py
git commit -m "feat(agent_cc): auth preflight and deterministic session uuid

claude auth status is a free, non-interactive oracle, so the check that
a run will bill the subscription runs in validate() before any phase
opens. apiKeySource alone is insufficient: 'none' also means logged out."
```

---

## Task 6: Event translation — tool calls, usage, context, failure classification

Implements: spec §3 (Event translation) and §8 (Failure classification, decisions Q6-2, Q6-3, Q6-7).

**Files:**
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/agent_cc.py`
- Test: `tests/test_agent_cc_events.py`

**Interfaces:**
- Consumes: Tasks 4–5.
- Produces: exceptions `agent_cc.CodingAgentError`, `RateLimited`, `NotAuthenticated`, `OverageRefused`; `agent_cc.ToolCallTracker` with `observe(event: dict) -> Optional[dict]`; `agent_cc.usage_from_result(ev: dict) -> UsageBreakdown`; `agent_cc.context_window_from_result(ev: dict, model: str) -> int`; `agent_cc.classify(ev: dict, last_rate_limit: dict | None, on_overage: str) -> None`.

- [ ] **Step 1: Write the failing test**

`tests/test_agent_cc_events.py`:

```python
import copy

import pytest


def _result_event(fixture):
    return next(e for e in fixture("tool_use_roundtrip.jsonl")
                if e.get("type") == "result")


# ── tool call folding ────────────────────────────────────────────────────────

def test_tracker_folds_a_tool_use_and_tool_result_into_one_record(fixture):
    from adw_modules.agent_cc import ToolCallTracker
    tracker = ToolCallTracker()
    records = [r for e in fixture("tool_use_roundtrip.jsonl")
               if (r := tracker.observe(e)) is not None]
    assert len(records) == 1
    rec = records[0]
    assert rec["tool"] == "Read"
    assert rec["tool_call_id"].startswith("toolu_")
    assert rec["ok"] is True
    assert rec["args"]["file_path"].endswith("README.md")
    assert "hello" in rec["result_snippet"]
    assert rec["label"].startswith("Read: ")
    assert rec["duration_ms"] >= 0
    assert rec["started_at"] and rec["ended_at"]


def test_tracker_payload_carries_every_contract_key(fixture):
    """spec §1.2: tool_call rows need exactly these."""
    from adw_modules.agent_cc import ToolCallTracker
    tracker = ToolCallTracker()
    rec = next(r for e in fixture("tool_use_roundtrip.jsonl")
               if (r := tracker.observe(e)) is not None)
    for key in ("tool", "tool_call_id", "args", "result_snippet", "ok",
                "duration_ms", "label", "started_at", "ended_at"):
        assert key in rec, f"missing {key}"


def test_tracker_handles_parallel_tool_uses_in_one_message():
    from adw_modules.agent_cc import ToolCallTracker
    tracker = ToolCallTracker()
    assert tracker.observe({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "a", "name": "Read", "input": {"file_path": "/a"}},
        {"type": "tool_use", "id": "b", "name": "Bash", "input": {"command": "ls"}},
    ]}}) is None
    rb = tracker.observe({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "b", "content": "out"}]}})
    ra = tracker.observe({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "x"}]}})
    assert rb["tool"] == "Bash" and ra["tool"] == "Read"


def test_tracker_handles_tool_result_content_as_a_block_list():
    from adw_modules.agent_cc import ToolCallTracker
    tracker = ToolCallTracker()
    tracker.observe({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "a", "name": "Read", "input": {}}]}})
    rec = tracker.observe({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "a",
         "content": [{"type": "text", "text": "block form"}]}]}})
    assert rec["result_snippet"] == "block form"


def test_tracker_marks_an_errored_tool_result_not_ok():
    from adw_modules.agent_cc import ToolCallTracker
    tracker = ToolCallTracker()
    tracker.observe({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "false"}}]}})
    rec = tracker.observe({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "boom",
         "is_error": True}]}})
    assert rec["ok"] is False


def test_tracker_tracks_context_occupancy_deduped_by_message_id():
    """Assistant events repeat the SAME usage object for every content block
    of one message; summing per event double-counts."""
    from adw_modules.agent_cc import ToolCallTracker
    tracker = ToolCallTracker()
    usage = {"input_tokens": 8, "cache_creation_input_tokens": 203,
             "cache_read_input_tokens": 14678, "output_tokens": 4}
    for _ in range(3):
        tracker.observe({"type": "assistant", "message": {
            "id": "msg_1", "content": [{"type": "text", "text": "x"}],
            "usage": usage, "stop_reason": "end_turn"}})
    assert tracker.context_tokens == 8 + 203 + 14678 + 4


def test_tracker_ignores_usage_from_an_errored_turn():
    from adw_modules.agent_cc import ToolCallTracker
    tracker = ToolCallTracker()
    tracker.observe({"type": "assistant", "message": {
        "id": "good", "content": [], "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 1,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}})
    tracker.observe({"type": "assistant", "message": {
        "id": "bad", "content": [], "stop_reason": "error",
        "usage": {"input_tokens": 99999, "output_tokens": 0,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}})
    assert tracker.context_tokens == 11


# ── usage / cost / window ────────────────────────────────────────────────────

def test_usage_from_result_maps_every_component(fixture):
    from adw_modules.agent_cc import usage_from_result
    u = usage_from_result(_result_event(fixture))
    assert u.input_tokens == 18
    assert u.output_tokens == 267
    assert u.cache_read_tokens == 14678
    assert u.cache_write_tokens == 14881
    assert u.reasoning_tokens == 141
    assert u.total_tokens == 18 + 267 + 14678 + 14881
    assert u.total_cost == pytest.approx(0.0325828)
    # Claude Code reports only a lump cost; components are unavailable, not 0.
    assert u.input_cost == 0.0 and u.output_cost == 0.0


def test_context_window_comes_from_model_usage(fixture):
    from adw_modules.agent_cc import context_window_from_result
    ev = _result_event(fixture)
    assert context_window_from_result(ev, "claude-haiku-4-5-20251001") == 200_000


def test_context_window_is_zero_when_unreported(fixture):
    from adw_modules.agent_cc import context_window_from_result
    ev = copy.deepcopy(_result_event(fixture))
    ev["modelUsage"] = {}
    assert context_window_from_result(ev, "whatever") == 0


# ── failure classification ───────────────────────────────────────────────────

def test_classify_passes_a_clean_result(fixture):
    from adw_modules.agent_cc import classify
    classify(_result_event(fixture), None, "fail")   # must not raise


def test_classify_raises_not_authenticated_even_though_there_is_text(fixture):
    """'Not logged in · Please run /login' IS usable text. Letting it reach
    _extract_json burns both correction sends and reports a JSON problem."""
    from adw_modules.agent_cc import classify, NotAuthenticated
    ev = copy.deepcopy(_result_event(fixture))
    ev["is_error"] = True
    ev["result"] = "Not logged in · Please run /login"
    with pytest.raises(NotAuthenticated):
        classify(ev, None, "fail")


def test_classify_raises_rate_limited_on_a_blocked_event(fixture):
    from adw_modules.agent_cc import classify, RateLimited
    ev = copy.deepcopy(_result_event(fixture))
    ev["is_error"] = True
    rl = {"status": "blocked", "rateLimitType": "seven_day",
          "resetsAt": 1790100000, "utilization": 1.0}
    with pytest.raises(RateLimited) as e:
        classify(ev, rl, "fail")
    assert "seven_day" in str(e.value)


def test_rate_limit_wins_over_generic_error(fixture):
    from adw_modules.agent_cc import classify, RateLimited
    ev = copy.deepcopy(_result_event(fixture))
    ev["is_error"] = True
    ev["result"] = "something generic"
    with pytest.raises(RateLimited):
        classify(ev, {"status": "rejected", "rateLimitType": "five_hour",
                      "resetsAt": 1, "utilization": 1.0}, "fail")


def test_classify_refuses_overage_by_default(fixture):
    """A subscription past its limits falls through to PAID overage — the
    per-token billing this whole feature exists to avoid."""
    from adw_modules.agent_cc import classify, OverageRefused
    ev = _result_event(fixture)
    rl = {"status": "allowed", "isUsingOverage": True, "utilization": 1.0}
    with pytest.raises(OverageRefused):
        classify(ev, rl, "fail")


def test_classify_allows_overage_when_configured_to_warn(fixture):
    from adw_modules.agent_cc import classify
    ev = _result_event(fixture)
    rl = {"status": "allowed", "isUsingOverage": True, "utilization": 1.0}
    classify(ev, rl, "warn")     # must not raise


def test_classify_raises_generic_error_with_diagnostics(fixture):
    from adw_modules.agent_cc import classify, CodingAgentError
    ev = copy.deepcopy(_result_event(fixture))
    ev["is_error"] = True
    ev["subtype"] = "error_during_execution"
    ev["terminal_reason"] = "exploded"
    with pytest.raises(CodingAgentError) as e:
        classify(ev, None, "fail")
    assert "error_during_execution" in str(e.value)
    assert "exploded" in str(e.value)
```

- [ ] **Step 2: Run to verify it fails**

Run: `./tests/run_tests.sh tests/test_agent_cc_events.py -v`
Expected: FAIL — `ImportError: cannot import name 'ToolCallTracker'`.

- [ ] **Step 3: Implement**

Append to `agent_cc.py`:

```python
class CodingAgentError(RuntimeError):
    """The coding agent failed in a way re-prompting cannot fix."""


class RateLimited(CodingAgentError):
    """Subscription window exhausted. NOT a JSON problem, NOT retryable."""


class NotAuthenticated(CodingAgentError):
    """The CLI is not logged in. NOT a JSON problem."""


class OverageRefused(CodingAgentError):
    """The subscription fell through to paid overage and we refuse to spend."""


NOT_AUTH_SIGNATURES = ("not logged in", "please run /login", "invalid api key",
                       "authentication_error")


def _result_text(block_content) -> str:
    """tool_result content is `str | list[block]`; normalise both."""
    if isinstance(block_content, str):
        return block_content
    if isinstance(block_content, list):
        return "".join(part.get("text", "") for part in block_content
                       if isinstance(part, dict) and part.get("type") == "text")
    return ""


def usage_from_result(ev: dict) -> UsageBreakdown:
    """Fold the `result` event's authoritative totals into UsageBreakdown.

    `result.usage` is the total for the WHOLE send across every API call it
    made. Per-message usage must not be summed — assistant events repeat one
    message's usage object per content block.

    Per-component COSTS stay 0: Claude Code reports only `total_cost_usd`.
    They are unavailable, not zero, which is why the UI is changed to say so
    rather than render four $0.0000 rows (spec §3).
    """
    u = ev.get("usage") or {}
    details = u.get("output_tokens_details") or {}
    usage = UsageBreakdown()
    usage.input_tokens = u.get("input_tokens") or 0
    usage.output_tokens = u.get("output_tokens") or 0
    usage.cache_read_tokens = u.get("cache_read_input_tokens") or 0
    usage.cache_write_tokens = u.get("cache_creation_input_tokens") or 0
    usage.reasoning_tokens = details.get("thinking_tokens") or 0
    # Pi's convention: cache reads count — cached prompt is still prompt.
    usage.total_tokens = (usage.input_tokens + usage.output_tokens
                          + usage.cache_read_tokens + usage.cache_write_tokens)
    usage.total_cost = ev.get("total_cost_usd") or 0.0
    return usage


def context_window_from_result(ev: dict, model: str) -> int:
    """The model's ceiling, straight from modelUsage. 0 = unknown."""
    entries = ev.get("modelUsage") or {}
    entry = entries.get(model) or next(iter(entries.values()), {})
    return int(entry.get("contextWindow") or 0)


def classify(ev: dict, last_rate_limit: Optional[dict], on_overage: str) -> None:
    """Raise if this `result` event is a failure re-prompting cannot fix.

    Ordered by specificity, and checked BEFORE the envelope text ever reaches
    agents._extract_json. That ordering is the whole point: a rate-limit or
    not-logged-in message is perfectly good TEXT, so left to fall through it
    parses as "bad JSON", burns both correction sends against the same wall,
    and kills the run reporting a prompt-engineering problem.
    """
    rl = last_rate_limit or {}
    if rl.get("isUsingOverage") and on_overage == "fail":
        raise OverageRefused(
            f"the subscription is using PAID overage "
            f"(utilization={rl.get('utilization')}). Refusing to spend. Set "
            f"defaults.claude_code.on_overage: warn to allow it, or disable "
            f"extra usage in your Anthropic account settings.")
    if rl.get("status") in ("rejected", "blocked"):
        raise RateLimited(
            f"Claude Code rate limit reached: {rl.get('rateLimitType')} window "
            f"at utilization={rl.get('utilization')}, resets at "
            f"{rl.get('resetsAt')}. Not retried — a seven_day reset can be days "
            f"away, and falling back to a per-token provider is not automatic.")
    if not ev.get("is_error"):
        return
    text = str(ev.get("result") or "").lower()
    if any(sig in text for sig in NOT_AUTH_SIGNATURES):
        raise NotAuthenticated(
            f"Claude Code is not authenticated: {ev.get('result')!r}. "
            f"Run `claude auth status` to check, `claude auth login` to fix.")
    raise CodingAgentError(
        f"claude exited with is_error=true "
        f"(subtype={ev.get('subtype')!r}, terminal_reason={ev.get('terminal_reason')!r}, "
        f"api_error_status={ev.get('api_error_status')!r}): "
        f"{str(ev.get('result'))[:500]}")


class ToolCallTracker:
    """Folds Claude Code's stream into ONE normalized record per tool call.

    A call appears as a `tool_use` block on an assistant message and closes as
    a `tool_result` block on a user message, joined by id. Only the result
    carries the outcome, so that is where a record is emitted — one trace event
    per real tool call, the moment it returns.

    Also tracks context occupancy as a side effect, because the only place the
    per-turn usage appears is on the assistant messages this already walks.
    """

    def __init__(self) -> None:
        self._open: dict[str, dict] = {}
        self._seen_messages: set[str] = set()
        self.context_tokens = 0

    def observe(self, event: dict) -> Optional[dict]:
        etype = event.get("type")
        if etype == "assistant":
            return self._on_assistant(event)
        if etype == "user":
            return self._on_user(event)
        return None

    def _on_assistant(self, event: dict) -> None:
        message = event.get("message") or {}
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                self._announce(block.get("id"), block.get("name"),
                               block.get("input") or {})
        # Occupancy, deduped: every content block of one message re-reports the
        # SAME usage object, so keying on message id is what stops a
        # three-block message counting its tokens three times.
        message_id = message.get("id")
        usage = message.get("usage") or {}
        if not message_id or message_id in self._seen_messages or not usage:
            return None
        self._seen_messages.add(message_id)
        if message.get("stop_reason") in ("error", "aborted"):
            return None              # an errored turn reports usage you can't trust
        self.context_tokens = ((usage.get("input_tokens") or 0)
                               + (usage.get("cache_creation_input_tokens") or 0)
                               + (usage.get("cache_read_input_tokens") or 0)
                               + (usage.get("output_tokens") or 0))
        return None

    def _on_user(self, event: dict) -> Optional[dict]:
        for block in (event.get("message") or {}).get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            call_id = str(block.get("tool_use_id") or "")
            opened = self._open.pop(call_id, {})
            tool = str(opened.get("tool") or "tool")
            args = opened.get("args") or {}
            record = {
                "tool": tool,
                "tool_call_id": call_id,
                "args": {k: clip(v, ARG_VALUE_CHARS) if isinstance(v, str) else v
                         for k, v in args.items()},
                "ok": not block.get("is_error", False),
                "label": tool_label(tool, args),
                "result_snippet": clip(_result_text(block.get("content")),
                                       RESULT_SNIPPET_CHARS),
                "ended_at": now_iso(),
                "started_at": opened.get("started_at") or now_iso(),
                "duration_ms": int((time.monotonic() - opened["clock"]) * 1000)
                               if opened.get("clock") else 0,
            }
            if event.get("parent_tool_use_id"):
                record["parent_tool_use_id"] = event["parent_tool_use_id"]
            return record
        return None

    def _announce(self, call_id, tool, args) -> None:
        if not call_id:
            return
        self._open[str(call_id)] = {
            "tool": tool or "tool", "args": args or {},
            "started_at": now_iso(),          # wall clock, for the row
            "clock": time.monotonic(),        # monotonic, for duration
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `./tests/run_tests.sh tests/test_agent_cc_events.py -v`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/sssf/templates/adws/adw_modules/agent_cc.py tests/test_agent_cc_events.py
git commit -m "feat(agent_cc): event translation and failure classification

tool_use/tool_result pairs fold into one tool_call row. is_error
short-circuits before the envelope parser so a rate limit or a logged-out
CLI is never misdiagnosed as malformed JSON."
```

---

## Task 7: `agent_cc.run()` — command construction and the subprocess loop

Implements: spec §1 (Command construction), §1b (argv limits), §2 (create vs resume), §7b/§7c (isolation, permissions), §8 (process handling, timeout).

**Files:**
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/agent_cc.py`
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/data_types.py` (two more request fields)
- Test: `tests/test_agent_cc_run.py`

**Interfaces:**
- Consumes: Tasks 4–6.
- Produces: `agent_cc.build_command(request: CodingAgentRequest) -> tuple[list[str], list[str]]` returning `(argv, warnings)`; `agent_cc.run(request, on_event=None, on_spawn=None, on_exit=None) -> CodingAgentResult`; `agent_cc.ARGV_SPILL_THRESHOLD = 96_000`.

- [ ] **Step 1: Add the two missing request fields**

In `data_types.CodingAgentRequest`, after `tools`:

```python
    # Claude Code --restricted, set for agents with `writes: []`. Removes
    # settings-file loading and confines file tools to the working dirs. The
    # roster's tools survive it: --restricted only strips code-running tools
    # that --tools does NOT name.
    restricted: bool = False
    # "fail" aborts the send when the subscription is on PAID overage.
    on_overage: str = "fail"
```

- [ ] **Step 2: Write the failing test**

`tests/test_agent_cc_run.py`:

```python
import json

import pytest

from adw_modules.data_types import CodingAgentRequest


def _req(tmp_path, **over):
    sysmd = tmp_path / "system.md"
    sysmd.write_text("You are a probe agent.")
    base = dict(prompt="do the thing", system_prompt="You are a probe agent.",
                system_prompt_path=str(sysmd),
                model="anthropic/claude-haiku-4-5-20251001", thinking="medium",
                session_id="sssf-a1b2c3d4-scout-9f8e",
                raw_output_path=str(tmp_path / "raw_output.jsonl"),
                stderr_path=str(tmp_path / "stderr.log"),
                tools=["read", "bash"], cwd=str(tmp_path))
    base.update(over)
    return CodingAgentRequest(**base)


# ── command construction ─────────────────────────────────────────────────────

def test_command_has_the_mandatory_print_mode_flags(tmp_path):
    from adw_modules.agent_cc import build_command
    argv, _ = build_command(_req(tmp_path))
    assert "-p" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv, "stream-json without --verbose is a hard CLI error"


def test_command_creates_a_session_by_uuid_when_not_resuming(tmp_path):
    from adw_modules.agent_cc import build_command, cc_session_uuid
    argv, _ = build_command(_req(tmp_path, resume=False))
    assert argv[argv.index("--session-id") + 1] == cc_session_uuid("sssf-a1b2c3d4-scout-9f8e")
    assert "--resume" not in argv


def test_command_resumes_instead_of_creating(tmp_path):
    from adw_modules.agent_cc import build_command, cc_session_uuid
    argv, _ = build_command(_req(tmp_path, resume=True))
    assert argv[argv.index("--resume") + 1] == cc_session_uuid("sssf-a1b2c3d4-scout-9f8e")
    assert "--session-id" not in argv, "--session-id with --resume is a CLI error"


def test_command_uses_the_system_prompt_FILE_not_argv(tmp_path):
    """Keeps the largest fixed argument out of argv (Linux caps one arg at
    128KB) and makes the audit copy the bytes that were actually sent."""
    from adw_modules.agent_cc import build_command
    req = _req(tmp_path)
    argv, _ = build_command(req)
    assert argv[argv.index("--append-system-prompt-file") + 1] == req.system_prompt_path
    assert "--system-prompt" not in argv
    assert req.system_prompt not in argv


def test_command_passes_tools_and_allowed_tools_as_single_comma_joined_args(tmp_path):
    """Both flags are variadic. Space-separating them before the positional
    prompt lets the parser swallow the prompt as another tool name."""
    from adw_modules.agent_cc import build_command
    argv, _ = build_command(_req(tmp_path))
    assert argv[argv.index("--tools") + 1] == "Read,Bash"
    assert argv[argv.index("--allowedTools") + 1] == "Read,Bash"


def test_command_isolates_repo_configuration(tmp_path):
    from adw_modules.agent_cc import build_command
    argv, _ = build_command(_req(tmp_path))
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in argv
    assert "--disable-slash-commands" in argv


def test_command_sets_a_non_blocking_permission_posture(tmp_path):
    from adw_modules.agent_cc import build_command
    argv, _ = build_command(_req(tmp_path))
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
    assert argv[argv.index("--permission-prompts") + 1] == "none"
    assert "--dangerously-skip-permissions" not in argv
    assert "bypassPermissions" not in argv


def test_command_adds_restricted_only_for_read_only_agents(tmp_path):
    from adw_modules.agent_cc import build_command
    assert "--restricted" not in build_command(_req(tmp_path))[0]
    assert "--restricted" in build_command(_req(tmp_path, restricted=True))[0]


def test_command_never_passes_bare_mode(tmp_path):
    """--bare reads auth strictly from ANTHROPIC_API_KEY, defeating the point."""
    from adw_modules.agent_cc import build_command
    assert "--bare" not in build_command(_req(tmp_path))[0]


def test_prompt_is_the_final_positional_argument(tmp_path):
    from adw_modules.agent_cc import build_command
    argv, _ = build_command(_req(tmp_path))
    assert argv[-1] == "do the thing"


def test_a_huge_prompt_spills_to_stdin(tmp_path):
    """Linux caps a single argv entry at 128KB regardless of ARG_MAX."""
    from adw_modules.agent_cc import build_command, ARGV_SPILL_THRESHOLD
    argv, _ = build_command(_req(tmp_path, prompt="x" * (ARGV_SPILL_THRESHOLD + 1)))
    assert argv[argv.index("--input-format") + 1] == "stream-json"
    assert not any(len(a) > ARGV_SPILL_THRESHOLD for a in argv)


def test_effort_clamp_surfaces_as_a_warning(tmp_path):
    from adw_modules.agent_cc import build_command
    argv, warnings = build_command(_req(tmp_path, thinking="off"))
    assert argv[argv.index("--effort") + 1] == "low"
    assert any("off" in w for w in warnings)


# ── the subprocess loop ──────────────────────────────────────────────────────

def test_run_replays_a_capture_end_to_end(tmp_path, fake_claude, monkeypatch):
    from adw_modules import agent_cc
    bindir = fake_claude(tmp_path, "tool_use_roundtrip.jsonl")
    monkeypatch.setattr(agent_cc, "CLAUDE_PATH", str(bindir / "claude"))
    seen, spawned, exited = [], [], []
    result = agent_cc.run(_req(tmp_path), on_event=seen.append,
                          on_spawn=spawned.append, on_exit=exited.append)
    assert result.text == "Just says hello."
    assert result.returncode == 0
    assert result.tokens == 18 + 267 + 14678 + 14881
    assert result.context_window == 200_000
    assert result.context_tokens == 8 + 203 + 14678 + 4
    assert result.cost_basis == "list"
    assert result.session_id == "sssf-a1b2c3d4-scout-9f8e"
    assert spawned and exited and spawned == exited
    assert any(e.get("type") == "result" for e in seen)


def test_run_writes_the_raw_stream_to_disk(tmp_path, fake_claude, monkeypatch):
    from adw_modules import agent_cc
    bindir = fake_claude(tmp_path, "tool_use_roundtrip.jsonl")
    monkeypatch.setattr(agent_cc, "CLAUDE_PATH", str(bindir / "claude"))
    req = _req(tmp_path)
    agent_cc.run(req)
    raw = (tmp_path / "raw_output.jsonl").read_text()
    assert '"type":"result"' in raw.replace(" ", "")


def test_run_emits_one_tool_call_record_through_on_event(tmp_path, fake_claude, monkeypatch):
    from adw_modules import agent_cc
    bindir = fake_claude(tmp_path, "tool_use_roundtrip.jsonl")
    monkeypatch.setattr(agent_cc, "CLAUDE_PATH", str(bindir / "claude"))
    tracker = agent_cc.ToolCallTracker()
    records = []
    agent_cc.run(_req(tmp_path),
                 on_event=lambda e: records.append(r) if (r := tracker.observe(e)) else None)
    assert len(records) == 1 and records[0]["tool"] == "Read"


def test_run_surfaces_stderr_warnings(tmp_path, fake_claude, monkeypatch):
    from adw_modules import agent_cc
    warning = "Warning: Unknown --effort value 'off' — ignoring it.\n"
    bindir = fake_claude(tmp_path, "tool_use_roundtrip.jsonl", stderr_text=warning)
    monkeypatch.setattr(agent_cc, "CLAUDE_PATH", str(bindir / "claude"))
    result = agent_cc.run(_req(tmp_path))
    assert any("Unknown --effort value" in w for w in result.warnings)


def test_run_raises_on_a_nonzero_exit_with_no_text(tmp_path, fake_claude, monkeypatch):
    from adw_modules import agent_cc
    bindir = fake_claude(tmp_path, "empty.jsonl", exit_code=1,
                         stderr_text="Error: Session ID abc is already in use.\n")
    monkeypatch.setattr(agent_cc, "CLAUDE_PATH", str(bindir / "claude"))
    with pytest.raises(agent_cc.CodingAgentError) as e:
        agent_cc.run(_req(tmp_path))
    assert "already in use" in str(e.value)
```

Also create the empty fixture this needs:

```bash
: > /Users/tylerviles/Documents/projects/factory/tests/fixtures/empty.jsonl
```

- [ ] **Step 3: Run to verify it fails**

Run: `./tests/run_tests.sh tests/test_agent_cc_run.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_command'`.

- [ ] **Step 4: Add `warnings` to the result type**

In `data_types.CodingAgentResult`, after `utilization`:

```python
    # Child stderr lines worth surfacing. The CLI reports real problems here
    # (`Warning: Unknown --effort value …`) while exiting 0, so without this
    # the trace is blind to them.
    warnings: list[str] = Field(default_factory=list)
```

- [ ] **Step 5: Implement `build_command` and `run`**

Append to `agent_cc.py`:

```python
# Linux caps a SINGLE argv entry at MAX_ARG_STRLEN (128KB) regardless of how
# much ARG_MAX headroom there is. Stay under it with room to spare; above this
# the prompt goes to stdin instead (spec §1b).
ARGV_SPILL_THRESHOLD = 96_000


def build_command(request: CodingAgentRequest) -> tuple[list[str], list[str]]:
    """Returns (argv, warnings). Pure — no subprocess, no filesystem."""
    model = resolve_model(request.model)
    effort, effort_warning = map_effort(request.thinking)
    tools, warnings = map_tools(request.tools)
    if effort_warning:
        warnings.append(effort_warning)

    cmd = [CLAUDE_PATH, "-p",
           "--output-format", "stream-json",
           "--verbose",                       # mandatory with stream-json
           "--model", model,
           "--effort", effort]

    # --session-id CREATES and fails if the id exists; --resume continues.
    # Passing both is a CLI error without --fork-session.
    session_uuid = cc_session_uuid(request.session_id)
    cmd += ["--resume", session_uuid] if request.resume else ["--session-id", session_uuid]

    if request.system_prompt_path:
        cmd += ["--append-system-prompt-file", request.system_prompt_path]
    else:
        cmd += ["--append-system-prompt", request.system_prompt]

    # --tools decides what EXISTS; --allowedTools decides what needs no
    # approval. Comma-joined into ONE argument each: both flags are variadic,
    # so space-separated values before the positional prompt let the parser
    # swallow the prompt as another tool name.
    joined = ",".join(tools)
    cmd += ["--tools", joined, "--allowedTools", joined]

    # Non-blocking without being reckless: edits proceed, granted tools are
    # pre-approved, and anything else that WOULD prompt is denied rather than
    # hanging a headless run forever.
    cmd += ["--permission-mode", "acceptEdits", "--permission-prompts", "none"]

    # The agent runs with cwd at the repo root (permissions.enforce and session
    # resumption both require it), so the repo's own CLAUDE.md, hooks, skills
    # and MCP servers are in reach. Isolate the CONFIGURATION, not the cwd.
    # NOT --bare: it reads auth strictly from ANTHROPIC_API_KEY.
    cmd += ["--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands"]

    if request.restricted:
        cmd.append("--restricted")

    if len(request.prompt) > ARGV_SPILL_THRESHOLD:
        cmd += ["--input-format", "stream-json"]
    else:
        cmd.append(request.prompt)
    return cmd, warnings


def run(request: CodingAgentRequest,
        on_event: Optional[Callable[[dict], None]] = None,
        on_spawn: Optional[Callable[[int], None]] = None,
        on_exit: Optional[Callable[[int], None]] = None) -> CodingAgentResult:
    """Run one non-interactive Claude Code turn.

    Same contract as agent_pi.run(): stream events to on_event as they happen,
    bracket the child with on_spawn/on_exit so a hung agent is a pid the trace
    can name, and return the shape agents.execute() already consumes.
    """
    cmd, warnings = build_command(request)
    model = resolve_model(request.model)

    raw_path = Path(request.raw_output_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = Path(request.stderr_path or raw_path.with_name("stderr.log"))
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    result = CodingAgentResult(session_id=request.session_id, cost_basis="list")
    tracker = ToolCallTracker()
    last_rate_limit: Optional[dict] = None
    result_event: Optional[dict] = None
    spill = len(request.prompt) > ARGV_SPILL_THRESHOLD

    # stderr to a FILE, never a second pipe: with both piped and a blocking
    # stdout read, a chatty child deadlocks both sides (see agent_pi.py).
    with stderr_path.open("a") as err:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if spill else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
            cwd=request.cwd, env=claude_code_env(request.inherit_api_key),
            start_new_session=True)      # own process group, so kill_tree works
    if on_spawn:
        on_spawn(process.pid)

    if spill:
        # The prompt was too large for argv, so it travels as one stream-json
        # user message. Closing stdin is what ends the turn.
        assert process.stdin is not None
        process.stdin.write(json.dumps({
            "type": "user",
            "message": {"role": "user",
                        "content": [{"type": "text", "text": request.prompt}]},
        }) + "\n")
        process.stdin.close()

    deadline = (time.monotonic() + request.timeout_seconds
                if request.timeout_seconds else None)
    try:
        with raw_path.open("a") as raw:
            assert process.stdout is not None
            for line in process.stdout:
                raw.write(line)
                raw.flush()                  # events land on disk as they happen
                if deadline and time.monotonic() > deadline:
                    raise CodingAgentError(
                        f"claude exceeded timeout_seconds="
                        f"{request.timeout_seconds} and was killed")
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "rate_limit_event":
                    last_rate_limit = event.get("rate_limit_info") or {}
                elif event.get("type") == "result":
                    result_event = event
                tracker.observe(event)
                if on_event:
                    on_event(event)
    except BaseException:
        kill_tree(process.pid)
        raise
    finally:
        if process.poll() is None:
            kill_tree(process.pid)
        result.returncode = process.wait()
        if on_exit:
            on_exit(process.pid)

    result.warnings = warnings + stderr_warnings(stderr_path)

    if result_event is None:
        tail = stderr_path.read_text(errors="replace").strip()[-800:]
        raise CodingAgentError(
            f"claude exited {result.returncode} without a result event: {tail}")

    # Classify BEFORE handing any text upstream: a rate limit or a logged-out
    # CLI is perfectly good text, and left to fall through it would parse as
    # "bad JSON" and burn the correction budget against the same wall.
    classify(result_event, last_rate_limit, request.on_overage)

    result.text = str(result_event.get("result") or "")
    result.usage = usage_from_result(result_event)
    result.tokens = result.usage.total_tokens
    result.cost = result.usage.total_cost
    result.context_tokens = tracker.context_tokens
    result.context_window = context_window_from_result(result_event, model)
    if last_rate_limit:
        result.rate_limit = last_rate_limit
    if result.returncode != 0 and not result.text:
        raise CodingAgentError(f"claude exited {result.returncode}")
    return result
```

Add `inherit_api_key: bool = False` to `CodingAgentRequest` (used above).

- [ ] **Step 6: Add the create/resume fallback (spec §2)**

State can disagree with reality in both directions: a stale `agent_map.json`
points at a transcript that was deleted, or a phase created a session and then
failed before the map was written. Each is a one-shot retry, never a loop.

Add to `tests/test_agent_cc_run.py`:

```python
def test_run_falls_back_to_resume_when_the_session_already_exists(
        tmp_path, fake_claude, fake_claude_argv, monkeypatch):
    from adw_modules import agent_cc
    calls = []
    real_popen = agent_cc.subprocess.Popen
    first = fake_claude(tmp_path / "a", "empty.jsonl", exit_code=1,
                        stderr_text="Error: Session ID abc is already in use.\n")
    second = fake_claude(tmp_path / "b", "tool_use_roundtrip.jsonl")

    def _popen(cmd, **kw):
        calls.append(list(cmd))
        cmd = [str((first if len(calls) == 1 else second) / "claude")] + list(cmd[1:])
        return real_popen(cmd, **kw)

    monkeypatch.setattr(agent_cc.subprocess, "Popen", _popen)
    result = agent_cc.run(_req(tmp_path, resume=False))
    assert result.text == "Just says hello."
    assert "--session-id" in calls[0] and "--resume" not in calls[0]
    assert "--resume" in calls[1] and "--session-id" not in calls[1]


def test_run_falls_back_to_create_when_the_transcript_is_gone(
        tmp_path, fake_claude, monkeypatch):
    from adw_modules import agent_cc
    calls = []
    real_popen = agent_cc.subprocess.Popen
    first = fake_claude(tmp_path / "a", "empty.jsonl", exit_code=1,
                        stderr_text="Error: No conversation found with session ID abc\n")
    second = fake_claude(tmp_path / "b", "tool_use_roundtrip.jsonl")

    def _popen(cmd, **kw):
        calls.append(list(cmd))
        cmd = [str((first if len(calls) == 1 else second) / "claude")] + list(cmd[1:])
        return real_popen(cmd, **kw)

    monkeypatch.setattr(agent_cc.subprocess, "Popen", _popen)
    agent_cc.run(_req(tmp_path, resume=True))
    assert "--resume" in calls[0]
    assert "--session-id" in calls[1]


def test_run_does_not_retry_twice(tmp_path, fake_claude, monkeypatch):
    from adw_modules import agent_cc
    calls = []
    real_popen = agent_cc.subprocess.Popen
    bad = fake_claude(tmp_path / "a", "empty.jsonl", exit_code=1,
                      stderr_text="Error: Session ID abc is already in use.\n")

    def _popen(cmd, **kw):
        calls.append(list(cmd))
        return real_popen([str(bad / "claude")] + list(cmd[1:]), **kw)

    monkeypatch.setattr(agent_cc.subprocess, "Popen", _popen)
    with pytest.raises(agent_cc.CodingAgentError):
        agent_cc.run(_req(tmp_path, resume=False))
    assert len(calls) == 2, "one retry, never a loop"
```

Rename the body of `run()` to `_run_once(request, on_event, on_spawn, on_exit)`
and make `run()` the retry wrapper:

```python
SESSION_EXISTS = "already in use"
SESSION_MISSING = ("no conversation found", "session not found")


def run(request: CodingAgentRequest,
        on_event: Optional[Callable[[dict], None]] = None,
        on_spawn: Optional[Callable[[int], None]] = None,
        on_exit: Optional[Callable[[int], None]] = None) -> CodingAgentResult:
    """One turn, with a single create/resume correction.

    Our belief about whether the session exists comes from agent_map.json,
    which is only written after a phase SUCCEEDS — so a phase that created a
    session and then failed leaves the map saying "new" about an id Claude
    Code already knows, and a cleared ~/.claude leaves it saying "existing"
    about one that is gone. Both are one retry, never a loop.
    """
    try:
        return _run_once(request, on_event, on_spawn, on_exit)
    except CodingAgentError as error:
        text = str(error).lower()
        if request.resume and any(sig in text for sig in SESSION_MISSING):
            flipped = request.model_copy(update={"resume": False})
        elif not request.resume and SESSION_EXISTS in text:
            flipped = request.model_copy(update={"resume": True})
        else:
            raise
        return _run_once(flipped, on_event, on_spawn, on_exit)
```

`_run_once` must include the stderr tail in the `CodingAgentError` it raises on
a non-zero exit (Step 5 already does), or these signatures never match.

- [ ] **Step 7: Run to verify it passes**

Run: `./tests/run_tests.sh tests/test_agent_cc_run.py -v`
Expected: all pass.

- [ ] **Step 8: Run the whole suite**

Run: `./tests/run_tests.sh -v`
Expected: all pass, zero model calls.

- [ ] **Step 9: Commit**

```bash
git add .claude/skills/sssf/templates/adws/adw_modules/{agent_cc.py,data_types.py} tests/test_agent_cc_run.py tests/fixtures/empty.jsonl
git commit -m "feat(agent_cc): command construction and the streaming run loop

System prompt travels by file (Linux caps one argv entry at 128KB) and a
huge user prompt spills to stdin. Config is isolated with
--setting-sources '' rather than by moving cwd, which permissions.enforce
and session resumption both require to stay at the repo root."
```

---

## Task 8: Wire it into `agents.py` — dispatch, validation, resume threading

Implements: spec §1.3 (there is no dispatch point today), §4 (validate dispatches per agent), §2 (resume threading), §6 (harness_engineering fails validation).

**Files:**
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/agents.py:18-73,112-137`
- Test: `tests/test_agents_dispatch.py`

**Interfaces:**
- Consumes: Tasks 4–7.
- Produces: `agents.ADAPTERS: dict[str, ModuleType]`; `agents.validate(cfg, required)` unchanged signature, new behaviour.

- [ ] **Step 1: Write the failing test**

`tests/test_agents_dispatch.py`:

```python
import pytest


def _cfg(tmp_path, **agent_over):
    from adw_modules.data_types import SSSFConfig
    sysmd, usermd = tmp_path / "system.md", tmp_path / "user.md"
    sysmd.write_text("s"); usermd.write_text("u")
    agent = dict(name="scout", coding_agent="claude_code",
                 model="anthropic/claude-sonnet-5", thinking="medium",
                 prompt_engineering={"system": str(sysmd), "user": str(usermd)},
                 tools=["read", "bash"])
    agent.update(agent_over)
    return SSSFConfig(agents=[agent])


@pytest.fixture(autouse=True)
def _no_real_preflight(monkeypatch):
    """validate() must not shell out during unit tests."""
    from adw_modules import agent_cc
    monkeypatch.setattr(agent_cc, "preflight_auth",
                        lambda inherit_api_key=False: {
                            "loggedIn": True, "authMethod": "claude.ai",
                            "subscriptionType": "max",
                            "projectsDirectory": "/tmp/projects"})


def test_adapters_table_has_both_harnesses():
    from adw_modules import agent_cc, agent_pi, agents
    assert agents.ADAPTERS == {"pi": agent_pi, "claude_code": agent_cc}


def test_validate_accepts_a_well_formed_claude_code_agent(tmp_path):
    from adw_modules import agents
    agents.validate(_cfg(tmp_path), ["scout"])       # must not raise


def test_validate_rejects_a_non_anthropic_model_at_startup(tmp_path):
    from adw_modules import agents
    with pytest.raises(SystemExit) as e:
        agents.validate(_cfg(tmp_path, model="openai/gpt-5.6-terra"), ["scout"])
    assert "anthropic/" in str(e.value)


def test_validate_does_not_consult_pi_catalog_for_a_claude_code_agent(tmp_path, monkeypatch):
    """agents.validate() called agent_pi.resolve_model for EVERY agent, which
    shells out to `pi --list-models`. A claude_code agent would fail for the
    wrong reason on a machine without pi."""
    from adw_modules import agent_pi, agents

    def _boom(pattern):
        raise AssertionError("pi catalog must not be consulted")

    monkeypatch.setattr(agent_pi, "resolve_model", _boom)
    agents.validate(_cfg(tmp_path), ["scout"])


def test_validate_rejects_pi_extensions_on_a_claude_code_agent(tmp_path):
    from adw_modules import agents
    with pytest.raises(SystemExit) as e:
        agents.validate(_cfg(tmp_path, harness_engineering=["x/subagents.ts"]), ["scout"])
    msg = str(e.value)
    assert "subagents.ts" in msg and "Task" in msg


def test_validate_rejects_an_unknown_tool_name(tmp_path):
    from adw_modules import agents
    with pytest.raises(SystemExit) as e:
        agents.validate(_cfg(tmp_path, tools=["read", "subagent_create"]), ["scout"])
    assert "subagent_create" in str(e.value)


def test_validate_rejects_an_empty_tool_list(tmp_path):
    from adw_modules import agents
    with pytest.raises(SystemExit):
        agents.validate(_cfg(tmp_path, tools=[]), ["scout"])


def test_validate_rejects_unknown_thinking(tmp_path):
    from adw_modules import agents
    with pytest.raises(SystemExit):
        agents.validate(_cfg(tmp_path, thinking="turbo"), ["scout"])


def test_validate_surfaces_a_failed_auth_preflight(tmp_path, monkeypatch):
    from adw_modules import agent_cc, agents

    def _fail(inherit_api_key=False):
        raise ValueError("Claude Code is not logged in — run `claude auth login`")

    monkeypatch.setattr(agent_cc, "preflight_auth", _fail)
    with pytest.raises(SystemExit) as e:
        agents.validate(_cfg(tmp_path), ["scout"])
    assert "not logged in" in str(e.value)


def test_preflight_runs_once_even_with_several_claude_code_agents(tmp_path, monkeypatch):
    from adw_modules import agent_cc, agents
    from adw_modules.data_types import SSSFConfig
    calls = []
    monkeypatch.setattr(agent_cc, "preflight_auth",
                        lambda inherit_api_key=False: calls.append(1) or {
                            "loggedIn": True, "authMethod": "claude.ai"})
    sysmd, usermd = tmp_path / "s.md", tmp_path / "u.md"
    sysmd.write_text("s"); usermd.write_text("u")
    pe = {"system": str(sysmd), "user": str(usermd)}
    cfg = SSSFConfig(agents=[
        dict(name="a", coding_agent="claude_code", model="anthropic/claude-sonnet-5",
             prompt_engineering=pe, tools=["read"]),
        dict(name="b", coding_agent="claude_code", model="anthropic/claude-opus-5",
             prompt_engineering=pe, tools=["read"]),
    ])
    agents.validate(cfg, ["a", "b"])
    assert len(calls) == 1, "one CLI call, not one per agent"


def test_pi_agents_still_validate_through_the_pi_resolver(tmp_path, monkeypatch):
    from adw_modules import agent_pi, agents
    seen = []
    monkeypatch.setattr(agent_pi, "resolve_model",
                        lambda p: seen.append(p) or ("google", "gemini-3.6-flash"))
    agents.validate(_cfg(tmp_path, coding_agent="pi",
                         model="google/gemini-3.6-flash"), ["scout"])
    assert seen == ["google/gemini-3.6-flash"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `./tests/run_tests.sh tests/test_agents_dispatch.py -v`
Expected: FAIL — `AttributeError: module 'adw_modules.agents' has no attribute 'ADAPTERS'`.

- [ ] **Step 3: Replace `validate()` and add the dispatch table**

In `agents.py`, change the import line to:

```python
from . import agent_cc, agent_pi, permissions, prompts
```

and add after `JSON_FIX_ATTEMPTS`:

```python
# The dispatch point v1 never had: agents.py imported agent_pi directly and
# called it unconditionally, so `coding_agent` was recorded in the trace and
# then ignored at the call site.
ADAPTERS = {"pi": agent_pi, "claude_code": agent_cc}
```

Replace `validate()` entirely:

```python
def validate(cfg: SSSFConfig, required: list[str]) -> None:
    """Fail fast: every required name must resolve to a usable agent.

    Dispatches per agent. The v1 version called `agent_pi.resolve_model` for
    EVERY agent, which shells out to `pi --list-models` — a claude_code agent
    would have failed for the wrong reason on a machine without pi, and no
    Anthropic model is in pi's catalog anyway.
    """
    problems: list[str] = []
    needs_cc_preflight = False
    for name in required:
        try:
            agent = resolve(cfg, name)
        except SystemExit as e:
            problems.append(str(e))
            continue
        adapter = ADAPTERS.get(agent.coding_agent)
        if adapter is None:
            problems.append(f"agent {name!r}: unknown coding_agent "
                            f"{agent.coding_agent!r} (known: {sorted(ADAPTERS)})")
            continue
        for label, ref in (("system", agent.prompt_engineering.system),
                           ("user", agent.prompt_engineering.user)):
            if not Path(ref).is_file():
                problems.append(f"agent {name!r}: {label} prompt not found: {ref}")
        try:
            adapter.resolve_model(agent.model)
        except ValueError as e:
            problems.append(f"agent {name!r}: {e}")
        if agent.coding_agent != "claude_code":
            continue

        needs_cc_preflight = True
        # Everything below is checkable without spawning anything, so it
        # belongs here rather than surfacing mid-chain (hard rule 1).
        try:
            agent_cc.map_effort(agent.thinking)
        except ValueError as e:
            problems.append(f"agent {name!r}: {e}")
        try:
            agent_cc.map_tools(agent.tools)
        except ValueError as e:
            problems.append(f"agent {name!r}: {e}")
        if agent.harness_engineering:
            problems.append(
                f"agent {name!r}: harness_engineering entries "
                f"{agent.harness_engineering} are Pi extensions and cannot load "
                f"under coding_agent: claude_code. Claude Code has a built-in "
                f"Task tool — drop the entries and the four subagent_* tools, "
                f"and add 'Task' to this agent's tools.")

    # One CLI call for the whole roster, not one per agent.
    if needs_cc_preflight and not problems:
        try:
            agent_cc.preflight_auth(cfg.defaults.claude_code.inherit_api_key)
        except ValueError as e:
            problems.append(str(e))

    if problems:
        raise SystemExit("config validation failed:\n- " + "\n- ".join(problems))
```

- [ ] **Step 4: Thread `resume` and the adapter through `execute()`**

In `agents.execute()`, replace the session id line and the `send` closure:

```python
    entry = run.agent_map.get(agent.name)
    reused = bool(entry and entry.get("model") == agent.model)
    session_id = _agent_session_id(run, agent)
    adapter = ADAPTERS[agent.coding_agent]
```

```python
    # Claude Code's --session-id CREATES and errors if the id exists, so the
    # adapter must be told which send this is. Only execute() knows: send #1
    # continues iff we rejoined a prior session, and every send after it —
    # JSON corrections, gate corrections — is by definition a continuation.
    # Pi ignores the flag; its one flag already does both.
    resumed = reused

    def send(prompt_text: str) -> agent_pi.PiResult:
        nonlocal latest, resumed
        request = CodingAgentRequest(
            prompt=prompt_text,
            system_prompt=system_text,
            system_prompt_path=str((agent_dir / "prompts" / "system.md").resolve()),
            model=agent.model,
            thinking=agent.thinking,
            session_id=session_id,
            resume=resumed,
            session_dir=str((agent_dir / "pi_sessions").resolve()),
            raw_output_path=str((agent_dir / "raw_output.jsonl").resolve()),
            stderr_path=str((agent_dir / "stderr.log").resolve()),
            tools=agent.tools,
            extensions=agent.harness_engineering,
            cwd=str(run.repo_root),
            restricted=agent.writes == [],
            inherit_api_key=run.cfg.defaults.claude_code.inherit_api_key,
            on_overage=run.cfg.defaults.claude_code.on_overage,
            timeout_seconds=run.cfg.defaults.claude_code.timeout_seconds,
        )
        result = adapter.run(
            request,
            on_event=_event_forwarder(run, phase, agent.name, adapter),
            # live_children is what the signal handler reaps (Task 2). It is
            # tracked in memory rather than read back from the processes table
            # because that table also holds rows from crashed runs, and a
            # recycled pid handed to os.killpg can signal an unrelated GROUP.
            on_spawn=lambda pid: (run.live_children.add(pid),
                                  run.tracer.process_start(
                                      run.adw_id, "agent", agent.name, pid,
                                      f"{agent.coding_agent} {agent.name} {agent.model}")),
            on_exit=lambda pid: (run.live_children.discard(pid),
                                 run.tracer.process_end(run.adw_id, pid)))
        resumed = True                      # every later send continues
        for warning in getattr(result, "warnings", []):
            run.tracer.event(EventRecord(adw_id=run.adw_id, phase_id=phase.phase_id,
                                         type="log", name="coding_agent_warning",
                                         payload={"agent": agent.name,
                                                  "message": warning}))
        run.add_usage(result.tokens, result.cost)
        spent.merge(result.usage)
        latest = result
        return result
```

Change `_event_forwarder` to take the adapter so each harness folds its own stream:

```python
def _event_forwarder(run, phase: Phase, agent_name: str, adapter):
    """One tool_call event per real tool call, with its exact args and result."""
    tracker = adapter.ToolCallTracker()
    ...
```

Import `CodingAgentRequest` in `agents.py`'s `data_types` import list.

- [ ] **Step 5: Run to verify it passes**

Run: `./tests/run_tests.sh -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add .claude/skills/sssf/templates/adws/adw_modules/agents.py tests/test_agents_dispatch.py
git commit -m "feat(agents): dispatch on coding_agent, validate per harness

coding_agent was recorded in the trace and then ignored at the call site.
validate() now routes model resolution to the right adapter instead of
always consulting pi's catalog, and resume is threaded through the
send sequence so Claude Code's create-only --session-id works."
```

---

## Task 9: Trace the cost basis, and stop the UI claiming false dollars

Implements: spec §3 (decision Q6-3).

**Files:**
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/tracer.py:80-99,251-271`
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/agents.py` (`agent_end` payload)
- Modify: `.claude/skills/sssf/apps/visualizer/src/components/StatChip.vue:25`
- Modify: `.claude/skills/sssf/apps/visualizer/src/components/PhaseDetail.vue:96-128`
- Modify: `.claude/skills/sssf/references/observability.md:30`
- Test: `tests/test_tracer_cost_basis.py`

**Interfaces:**
- Consumes: Task 8.
- Produces: `tracer.agent_session_row(..., cost_basis: str = "billed")`; `agent_sessions.cost_basis` column.

- [ ] **Step 1: Write the failing test**

`tests/test_tracer_cost_basis.py`:

```python
import sqlite3


def _tracer(tmp_path):
    from adw_modules.tracer import Tracer
    return Tracer(tmp_path / "sssf.db", tmp_path / "events.jsonl")


def _agent(name="scout", coding_agent="claude_code"):
    from adw_modules.data_types import AgentConfig
    return AgentConfig(name=name, coding_agent=coding_agent,
                       model="anthropic/claude-sonnet-5",
                       prompt_engineering={"system": "s", "user": "u"})


def test_cost_basis_column_exists(tmp_path):
    t = _tracer(tmp_path)
    cols = {row[1] for row in t.conn.execute("PRAGMA table_info(agent_sessions)")}
    assert "cost_basis" in cols


def test_cost_basis_is_persisted(tmp_path):
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    t.agent_session_row("adw1", _agent(), "sess-1", context_tokens=10,
                        context_window=200_000, cost_basis="list")
    row = t.conn.execute(
        "SELECT cost_basis FROM agent_sessions WHERE adw_id='adw1'").fetchone()
    assert row[0] == "list"


def test_cost_basis_defaults_to_billed_for_pi(tmp_path):
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    t.agent_session_row("adw1", _agent(coding_agent="pi"), "sess-1")
    row = t.conn.execute(
        "SELECT cost_basis FROM agent_sessions WHERE adw_id='adw1'").fetchone()
    assert row[0] == "billed"


def test_migration_adds_the_column_to_an_older_db(tmp_path):
    """A db from an older SSSF must still open. CREATE TABLE IF NOT EXISTS
    never revisits an existing table, hence the explicit ALTER list."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE agent_sessions (adw_id TEXT, agent TEXT, "
                 "coding_agent TEXT, model TEXT, session_id TEXT, "
                 "created_at TEXT, last_used_at TEXT, "
                 "PRIMARY KEY (adw_id, agent))")
    conn.commit(); conn.close()
    from adw_modules.tracer import Tracer
    t = Tracer(db, tmp_path / "events.jsonl")
    cols = {row[1] for row in t.conn.execute("PRAGMA table_info(agent_sessions)")}
    assert "cost_basis" in cols
```

- [ ] **Step 2: Run to verify it fails**

Run: `./tests/run_tests.sh tests/test_tracer_cost_basis.py -v`
Expected: FAIL — `assert 'cost_basis' in cols`.

- [ ] **Step 3: Add the column and the migration**

In `tracer.py`'s `SCHEMA`, inside `agent_sessions`, after `context_window`:

```sql
  cost_basis    TEXT DEFAULT 'billed',   -- 'billed' (real money) | 'list' (subscription notional)
```

Append to `MIGRATIONS`:

```python
              ("agent_sessions", "cost_basis", "TEXT DEFAULT 'billed'")]
```

Change `agent_session_row`'s signature and SQL:

```python
    def agent_session_row(self, adw_id: str, agent: AgentConfig, session_id: str,
                          context_tokens: int = 0, context_window: int = 0,
                          cost_basis: str = "billed") -> None:
```

Add `cost_basis` to the INSERT column list, the `VALUES` placeholders, the
`ON CONFLICT … DO UPDATE SET` list (`cost_basis=excluded.cost_basis`), and the
parameter tuple.

- [ ] **Step 4: Pass it from `agents.execute()`**

Replace the `agent_session_row` call:

```python
    run.tracer.agent_session_row(run.adw_id, agent, session_id,
                                 context_tokens=context.context_tokens,
                                 context_window=context.context_window,
                                 cost_basis=getattr(context, "cost_basis", "billed"))
```

And add the basis and utilisation to the `agent_end` payload:

```python
                                 payload={"cost": spent.total_cost,
                                          "usage": spent.model_dump(),
                                          "cost_basis": getattr(context, "cost_basis", "billed"),
                                          "rate_limit": getattr(context, "rate_limit", {}),
                                          "context_tokens": context.context_tokens,
                                          "context_window": context.context_window}))
```

- [ ] **Step 5: Stop the UI asserting these are billed dollars**

`StatChip.vue:25` — replace the tooltip:

```js
  cost: 'Cost — dollars for this run, all agents combined. Subscription-billed agents report list price, not money charged.',
```

`PhaseDetail.vue` — **extend the note mechanism that is already there** rather
than adding a second one. The file already handles "this run's cost data is
incomplete" with a `partial: boolean` (line 98) and a conditional `<p>` (line
582); widening that flag into the message itself covers both cases with one
mechanism. `partial` is read in exactly one place, so nothing else breaks.

Three edits.

1. `UsageRow.cost` becomes optional (line 71), so an unavailable cost is
   visibly unavailable rather than a confident `$0`:

```ts
interface UsageRow {
  label: string
  tokens: number
  /** Undefined when the harness reports no per-component cost — rendered as an em dash. */
  cost?: number
  kind?: 'total' | 'nested'
  title?: string
}
```

2. `phaseUsage` returns a `note` instead of a `partial` flag (line 84). The
   pre-breakdown branch keeps its existing sentence; the list-price branch adds
   its own:

```ts
const NO_COMPONENT_COSTS =
  'list price — this harness reports only a total, not per-component costs'
const PRE_BREAKDOWN =
  'this run predates the per-component breakdown — only the total was recorded'

const phaseUsage = computed<{ rows: UsageRow[]; note: string } | null>(() => {
  ...
  if (!u) {
    return { note: PRE_BREAKDOWN,
             rows: [{ label: 'total', tokens: end.tokens ?? 0, cost: payload.cost ?? 0, kind: 'total' }] }
  }
  // Claude Code on a subscription reports one lump total_cost_usd. The
  // component costs are UNAVAILABLE, not zero — and money(0) renders '$0',
  // which reads as "this was free". Tokens are real either way, so only the
  // dollars drop out.
  const listOnly = (payload.cost_basis ?? 'billed') === 'list'
  const money_ = (n: number) => (listOnly ? undefined : n)
  const rows: UsageRow[] = [
    { label: 'input', tokens: u.input_tokens, cost: money_(u.input_cost) },
    { label: 'output', tokens: u.output_tokens, cost: money_(u.output_cost) },
  ]
  ...   // thinking / cache rows unchanged except for the same money_() wrap
  rows.push({ label: 'total', tokens: u.total_tokens, cost: u.total_cost, kind: 'total' })
  return { rows, note: listOnly ? NO_COMPONENT_COSTS : '' }
})
```

   The `thinking` row's share calculation (line 109) already divides by
   `u.output_cost`, which is `0` here — wrap it the same way so it reports
   `undefined` rather than a computed `$0`.

3. The table body and the note (lines 578, 582):

```html
                <td class="u-c">{{ r.cost === undefined ? '—' : money(r.cost) }}</td>
```
```html
          <p v-if="phaseUsage.note" class="faint u-note">{{ phaseUsage.note }}</p>
```

- [ ] **Step 6: Correct the documented invariant**

`references/observability.md:30` — replace *"The four components sum to `total_tokens`, and their costs sum to `total_cost`"* with:

```markdown
The four components sum to `total_tokens` on every harness. **Their costs sum
to `total_cost` only when `cost_basis` is `billed` (Pi).** A `claude_code`
agent on a subscription reports `cost_basis: list`: the CLI gives one lump
`total_cost_usd` at notional list price and no per-component split, so the
component costs are unavailable rather than zero, and the UI says so instead of
rendering `$0.0000` rows.
```

- [ ] **Step 7: Run to verify it passes**

Run: `./tests/run_tests.sh -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add .claude/skills/sssf/templates/adws/adw_modules/{tracer.py,agents.py} .claude/skills/sssf/apps/visualizer/src/components/{StatChip.vue,PhaseDetail.vue} .claude/skills/sssf/references/observability.md tests/test_tracer_cost_basis.py
git commit -m "feat(tracer): record cost_basis, stop the UI claiming false dollars

A Max-subscription run reports notional list price. The chip tooltip said
'dollars billed' and the component rows rendered \$0 — which reads as free
rather than unreported. Costs are now optional and render as an em dash,
reusing the note mechanism the component already had for partial data."
```

---

## Task 10: Headroom pre-check — refuse a chain a live window cannot finish

Implements: spec §8 `max_utilization`. Without this the key is declared, documented, and never read.

**Files:**
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/tracer.py` (module-level reader)
- Modify: `.claude/skills/sssf/templates/adws/adw_modules/agents.py` (`validate`)
- Test: `tests/test_headroom.py`

**Interfaces:**
- Consumes: Task 9's `agent_end.rate_limit` payload.
- Produces: `tracer.last_rate_limit(db_path) -> dict | None`; `agents.exhausted_windows(rl: dict, max_utilization: float, now: float | None = None) -> list[tuple[str, float, int]]`.

**Why this is sound rather than guesswork:** a recorded utilisation is a **lower bound** on the current one — within a window it only ever rises, and `resetsAt` says exactly when it returns to zero. An expired window is therefore dropped on fact, not on a guess, and a live one can only understate. **A false refusal is impossible**; the only error mode is the safe one (allowing a run that will fail anyway).

- [ ] **Step 1: Write the failing test**

`tests/test_headroom.py`:

```python
import json
import sqlite3

import pytest

CAPTURED = {
    "status": "allowed_warning", "resetsAt": 1790100000,
    "rateLimitType": "seven_day", "utilization": 0.88, "isUsingOverage": False,
    "unifiedWindows": {"five_hour": {"utilization": 0.0, "resetsAt": 1790040000},
                       "seven_day": {"utilization": 0.88, "resetsAt": 1790100000}},
}
BEFORE_RESET = 1790000000
AFTER_RESET = 1790200000


def _db(tmp_path, *payloads):
    db = tmp_path / "sssf.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE events (event_id TEXT, type TEXT, payload_json TEXT)")
    for i, payload in enumerate(payloads):
        conn.execute("INSERT INTO events VALUES (?,?,?)",
                     (f"e{i}", "agent_end", json.dumps(payload)))
    conn.commit(); conn.close()
    return db


def test_last_rate_limit_returns_none_without_a_db(tmp_path):
    from adw_modules.tracer import last_rate_limit
    assert last_rate_limit(tmp_path / "nope.db") is None


def test_last_rate_limit_returns_none_when_no_agent_recorded_one(tmp_path):
    from adw_modules.tracer import last_rate_limit
    assert last_rate_limit(_db(tmp_path, {"cost": 0.01})) is None


def test_last_rate_limit_skips_pi_rows_and_finds_the_newest(tmp_path):
    from adw_modules.tracer import last_rate_limit
    db = _db(tmp_path, {"cost": 0.5, "rate_limit": {"utilization": 0.1}},
             {"cost": 0.01},                      # a pi agent: no rate_limit key
             {"cost": 0.2, "rate_limit": CAPTURED})
    assert last_rate_limit(db) == CAPTURED


def test_last_rate_limit_does_not_create_the_db(tmp_path):
    """validate() must not bring a trace db into existence as a side effect."""
    from adw_modules.tracer import last_rate_limit
    missing = tmp_path / "absent.db"
    last_rate_limit(missing)
    assert not missing.exists()


def test_exhausted_windows_flags_a_live_window_over_the_bar():
    from adw_modules.agents import exhausted_windows
    hits = exhausted_windows(CAPTURED, 0.80, now=BEFORE_RESET)
    assert [h[0] for h in hits] == ["seven_day"]


def test_exhausted_windows_ignores_a_window_that_has_reset():
    """The reading is void once resetsAt passes — this is fact, not a guess."""
    from adw_modules.agents import exhausted_windows
    assert exhausted_windows(CAPTURED, 0.80, now=AFTER_RESET) == []


def test_exhausted_windows_default_bar_only_catches_full_exhaustion():
    from adw_modules.agents import exhausted_windows
    assert exhausted_windows(CAPTURED, 1.0, now=BEFORE_RESET) == []
    full = {"unifiedWindows": {"seven_day": {"utilization": 1.0, "resetsAt": 1790100000}}}
    assert [h[0] for h in exhausted_windows(full, 1.0, now=BEFORE_RESET)] == ["seven_day"]


def test_exhausted_windows_falls_back_to_the_flat_shape():
    from adw_modules.agents import exhausted_windows
    flat = {"rateLimitType": "five_hour", "utilization": 0.99, "resetsAt": 1790100000}
    assert [h[0] for h in exhausted_windows(flat, 0.9, now=BEFORE_RESET)] == ["five_hour"]


def test_validate_refuses_when_a_live_window_is_exhausted(tmp_path, monkeypatch):
    from adw_modules import agent_cc, agents
    from adw_modules.data_types import SSSFConfig
    monkeypatch.setattr(agent_cc, "preflight_auth",
                        lambda inherit_api_key=False: {"loggedIn": True,
                                                       "authMethod": "claude.ai"})
    monkeypatch.setattr(agents, "_now", lambda: BEFORE_RESET)
    sysmd, usermd = tmp_path / "s.md", tmp_path / "u.md"
    sysmd.write_text("s"); usermd.write_text("u")
    db = _db(tmp_path, {"rate_limit": CAPTURED})
    cfg = SSSFConfig(
        defaults={"claude_code": {"max_utilization": 0.80}},
        observability={"db": str(db)},
        agents=[dict(name="scout", coding_agent="claude_code",
                     model="anthropic/claude-sonnet-5", tools=["read"],
                     prompt_engineering={"system": str(sysmd), "user": str(usermd)})])
    with pytest.raises(SystemExit) as e:
        agents.validate(cfg, ["scout"])
    assert "seven_day" in str(e.value) and "0.88" in str(e.value)


def test_validate_does_not_check_headroom_for_a_pi_only_chain(tmp_path, monkeypatch):
    """A claude_code agent elsewhere in the roster must not block a pi chain."""
    from adw_modules import agent_pi, agents
    from adw_modules.data_types import SSSFConfig
    monkeypatch.setattr(agent_pi, "resolve_model", lambda p: ("google", "g"))
    monkeypatch.setattr(agents, "_now", lambda: BEFORE_RESET)
    sysmd, usermd = tmp_path / "s.md", tmp_path / "u.md"
    sysmd.write_text("s"); usermd.write_text("u")
    pe = {"system": str(sysmd), "user": str(usermd)}
    db = _db(tmp_path, {"rate_limit": CAPTURED})
    cfg = SSSFConfig(
        defaults={"claude_code": {"max_utilization": 0.80}},
        observability={"db": str(db)},
        agents=[dict(name="builder", coding_agent="pi", model="google/g",
                     prompt_engineering=pe, tools=["read"]),
                dict(name="scout", coding_agent="claude_code",
                     model="anthropic/claude-sonnet-5", prompt_engineering=pe,
                     tools=["read"])])
    agents.validate(cfg, ["builder"])          # must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `./tests/run_tests.sh tests/test_headroom.py -v`
Expected: FAIL — `ImportError: cannot import name 'last_rate_limit' from 'adw_modules.tracer'`.

- [ ] **Step 3: Add the reader to `tracer.py`**

Module level, beside the `Tracer` class — deliberately **not** a method:

```python
def last_rate_limit(db_path: str | Path) -> Optional[dict]:
    """The most recent rate_limit_info any claude_code agent recorded.

    A plain function opening a READ-ONLY connection, so validate() can consult
    the trace without constructing a Tracer — which would create the db file
    and run migrations as a side effect of a check that is supposed to be
    inert. Returns None when there is no db yet, or no claude_code history:
    the first run of a fresh repo simply has nothing to go on.
    """
    if not Path(db_path).exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        for (payload,) in conn.execute(
                "SELECT payload_json FROM events WHERE type='agent_end' "
                "ORDER BY rowid DESC LIMIT 25"):
            try:
                rate_limit = (json.loads(payload or "{}") or {}).get("rate_limit")
            except json.JSONDecodeError:
                continue
            if rate_limit:
                return rate_limit
        return None
    except sqlite3.Error:
        return None            # an older db without the column is not an error
    finally:
        conn.close()
```

Add `from typing import Optional` to `tracer.py` if absent.

- [ ] **Step 4: Add the check to `agents.py`**

```python
def _now() -> float:
    """Seeded in tests; a module-level seam beats monkeypatching time."""
    return time.time()


def exhausted_windows(rate_limit: dict, max_utilization: float,
                      now: Optional[float] = None) -> list[tuple[str, float, int]]:
    """Live windows at or above the bar, as (name, utilization, resets_at).

    A recorded utilisation only rises until its resetsAt, so an unexpired
    reading is a LOWER bound on the current one and an expired one is simply
    void. That is why this cannot raise a false alarm.
    """
    now = _now() if now is None else now
    windows = rate_limit.get("unifiedWindows") or {}
    if not windows and rate_limit.get("resetsAt"):
        windows = {rate_limit.get("rateLimitType", "window"): {
            "utilization": rate_limit.get("utilization", 0.0),
            "resetsAt": rate_limit["resetsAt"]}}
    hits = []
    for name, window in windows.items():
        resets_at = window.get("resetsAt") or 0
        if now >= resets_at:
            continue                                # window reset; reading void
        utilization = window.get("utilization") or 0.0
        if utilization >= max_utilization:
            hits.append((name, utilization, resets_at))
    return hits
```

In `validate()`, inside the `if needs_cc_preflight and not problems:` block, after the auth preflight:

```python
        rate_limit = tracer_mod.last_rate_limit(cfg.observability.db)
        bar = cfg.defaults.claude_code.max_utilization
        for name, utilization, resets_at in exhausted_windows(rate_limit or {}, bar):
            when = datetime.fromtimestamp(resets_at, timezone.utc).isoformat()
            problems.append(
                f"Claude Code subscription: the {name} window was last observed "
                f"at utilization={utilization:.2f} (limit {bar:.2f}) and does not "
                f"reset until {when}. Refusing to start a chain that cannot "
                f"finish. Raise defaults.claude_code.max_utilization to override.")
```

Imports: `import time`, `from datetime import datetime, timezone`, and
`from . import tracer as tracer_mod` (aliased so it does not shadow the
`tracer` attribute ADWs already use on `run`).

- [ ] **Step 5: Run to verify it passes**

Run: `./tests/run_tests.sh tests/test_headroom.py -v`
Expected: 10 passed.

- [ ] **Step 6: Run the whole suite**

Run: `./tests/run_tests.sh -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add .claude/skills/sssf/templates/adws/adw_modules/{tracer.py,agents.py} tests/test_headroom.py
git commit -m "feat(agents): refuse a chain a live rate-limit window cannot finish

A recorded utilisation is a lower bound until its resetsAt passes, so an
expired window is dropped on fact and a live one can only understate.
False refusals are impossible; the default bar catches only full
exhaustion."
```

---

## Task 11: Config template and documentation

Implements: spec "Files touched" — the operator-facing half. No test file; verified by Task 11's live run loading the stamped config.

**Files:**
- Modify: `.claude/skills/sssf/templates/sssf.config.yaml`
- Modify: `.claude/skills/sssf/templates/env.sample`
- Modify: `.claude/skills/sssf/references/config.md`
- Modify: `.claude/skills/sssf/cookbooks/update_modules.md`
- Modify: `.claude/skills/sssf/SKILL.md:74-76`
- Test: `tests/test_config_template.py`

**Interfaces:**
- Consumes: Task 9.
- Produces: nothing new in code; the stamped config gains a `defaults.claude_code` block and a commented `claude_code` example agent.

- [ ] **Step 1: Write the failing test**

`tests/test_config_template.py`:

```python
from pathlib import Path

import yaml


def _template(templates_dir: Path) -> Path:
    return templates_dir.parent / "sssf.config.yaml"


def test_template_config_still_parses_into_the_schema(templates_dir):
    from adw_modules.data_types import SSSFConfig
    raw = yaml.safe_load(_template(templates_dir).read_text())
    cfg = SSSFConfig(**raw)
    assert [a.name for a in cfg.agents]


def test_template_declares_the_claude_code_defaults_block(templates_dir):
    raw = yaml.safe_load(_template(templates_dir).read_text())
    cc = raw["defaults"]["claude_code"]
    assert cc["inherit_api_key"] is False
    assert cc["on_overage"] == "fail"
    assert cc["max_utilization"] == 1.0


def test_default_roster_is_still_pi_so_existing_installs_are_unchanged(templates_dir):
    raw = yaml.safe_load(_template(templates_dir).read_text())
    assert raw["defaults"]["coding_agent"] == "pi"


def test_skill_md_no_longer_calls_claude_code_stubbed(templates_dir):
    skill = (templates_dir.parent.parent / "SKILL.md").read_text()
    assert "stubbed until v2" not in skill
    assert "claude_code" in skill
```

- [ ] **Step 2: Run to verify it fails**

Run: `./tests/run_tests.sh tests/test_config_template.py -v`
Expected: FAIL — `KeyError: 'claude_code'`.

- [ ] **Step 3: Add the defaults block to `sssf.config.yaml`**

After the `data_dir:` line in `defaults:`:

```yaml
  # Only read when an agent sets coding_agent: claude_code.
  claude_code:
    # false strips ANTHROPIC_API_KEY (and ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL,
    # the Bedrock/Vertex switches, CLAUDE_EFFORT) from the child's environment.
    # THIS is what keeps a run on your Max subscription: with the key present,
    # Claude Code reports apiKeySource: ANTHROPIC_API_KEY and bills the API.
    # agents.validate() refuses to start unless `claude auth status` agrees.
    inherit_api_key: false
    timeout_seconds: 1800        # wall clock per send; 0 disables
    # A subscription past its limits falls through to PAID overage. `fail`
    # aborts the moment that is seen. The authoritative control is account-side:
    # disable extra usage in your Anthropic settings.
    on_overage: fail             # fail | warn
    # Refuse to START a chain when a LIVE rate-limit window was last seen at or
    # above this. 1.0 refuses only a window read as fully exhausted. This cannot
    # false-alarm: a recorded utilisation only rises until its resetsAt, so an
    # expired window is ignored on fact and a live one can only understate.
    # It is a pre-filter, not a guarantee — it knows nothing about a fresh repo,
    # or about `claude` you ran interactively outside the factory.
    max_utilization: 1.0
```

- [ ] **Step 4: Add the commented example agent**

At the end of `agents:`:

```yaml
  # ── Running an agent on Claude Code instead of Pi ──────────────────────────
  # Uncomment to move the reviewer onto your Claude subscription. Note what
  # CHANGES versus a pi agent, all of it enforced by agents.validate():
  #   * model MUST be anthropic/<id> — any other provider fails at startup
  #   * harness_engineering MUST be empty — Pi .ts extensions cannot load;
  #     Claude Code's built-in Task tool replaces subagents.ts
  #   * tool names map read->Read, bash->Bash, edit->Edit, write->Write,
  #     grep->Grep, find->Glob. `ls` has no equivalent and is dropped.
  #     Omitting `tools` gives the same six, NOT all 28 Claude Code tools.
  #   * thinking off/minimal clamp to low (Claude Code's ladder starts at low)
  #
  # - name: reviewer_cc
  #   coding_agent: claude_code
  #   model: anthropic/claude-sonnet-5
  #   thinking: high
  #   color: "#fb7185"
  #   purpose: Confirm that what was built is what was asked for; change nothing.
  #   prompt_engineering:
  #     system: adws/adw_data/prompt_engineering/reviewer/system.md
  #     user: adws/adw_data/prompt_engineering/reviewer/user.md
  #   writes: []                 # also adds --restricted to the child
  #   tools:
  #     - read
  #     - grep
  #     - find
  #     - bash
  #     - write
```

- [ ] **Step 5: Update `env.sample`**

Append:

```
# ── Claude Code agents (coding_agent: claude_code) ───────────────────────────
# These need NO key. They run on your Claude subscription through the `claude`
# CLI, and agents.validate() refuses to start unless `claude auth status`
# reports loggedIn with authMethod "claude.ai" and no API key in play.
#
# ANTHROPIC_API_KEY is deliberately STRIPPED from the child environment. With
# it set, Claude Code bills the API instead of your subscription. Set
# defaults.claude_code.inherit_api_key: true only if that is what you want.
#
# A subscription past its limits can fall through to PAID overage;
# defaults.claude_code.on_overage: fail (the default) aborts rather than spend.
# The authoritative control is disabling extra usage in your Anthropic account.
#
# DATA RESIDENCY: Claude Code writes full conversation transcripts OUTSIDE this
# repo, under the `projectsDirectory` that `claude auth status` reports
# (normally ~/.claude/projects/<slug>/). They contain whatever the agent read.
# Unlike Pi, whose sessions live under data_dir, these are not covered by
# .gitignore and are not removed with the repo.
#
# CLAUDE_PATH=claude               # claude binary if not on PATH
```

- [ ] **Step 6: Update `references/config.md`**

Four edits:

1. `coding_agent` row: replace *"v1 implements `pi` only…"* with `Which interface runs the agent. `pi` -> `agent_pi.py`; `claude_code` -> `agent_cc.py` (headless `claude -p`, billed to your Claude subscription).`
2. Model resolution: add `**For `coding_agent: claude_code` the provider must be `anthropic`** — `anthropic/claude-opus-5`, `anthropic/claude-sonnet-5`, or an alias like `anthropic/opus`. There is no catalog lookup; the id after the slash is handed to `claude --model`. Any other provider fails in `agents.validate()` before a phase opens.`
3. Thinking levels: add `On Claude Code this maps to `--effort`, whose ladder is `low|medium|high|xhigh|max`. `off` and `minimal` clamp to `low` with a logged warning — passing them through makes the CLI warn on stderr and silently use its OWN default, which is neither what you asked for nor the lowest setting.`
4. Tools: add the mapping table and — critically — correct the escape hatch at `:196`:

```markdown
> **This escape hatch does not exist for `coding_agent: claude_code`.** There,
> `tools: None` resolves to the six-tool Pi-equivalent set
> (`Read, Bash, Edit, Write, Grep, Glob`), *not* "every tool". Omitting
> `--tools` would hand the agent all 28 Claude Code tools, including
> `CronCreate` (schedules work that outlives the run), `RemoteTrigger` and
> `PushNotification` (reach off the machine), `Workflow`, `Skill` (can reach
> the sssf skill itself) and `EnterWorktree` — which moves cwd and breaks both
> the `writes` boundary and session resumption. Name what you need.
```

Also replace the `harness_engineering` Claude Code sentence with: `**Not supported for `coding_agent: claude_code`** — entries are Pi extension paths and cannot load. A non-empty list fails `agents.validate()`. Claude Code's built-in `Task` tool replaces `subagents.ts`; name `Task` in `tools` instead.`

- [ ] **Step 7: Update `update_modules.md` and `SKILL.md`**

`cookbooks/update_modules.md` module table:
- `data_types.py` row: replace `PiRequest`/`PiResult` with `CodingAgentRequest`/`CodingAgentResult` (aliased as `PiRequest`/`PiResult`).
- `agent_cc.py` row: replace *"stubbed in v1, lands in v2"* with `the Claude Code interface — headless `claude -p --output-format stream-json --verbose`, JSONL stream tailed live, `--session-id` creates / `--resume` continues`.
- `agents.py` row: add `dispatches `coding_agent` through `ADAPTERS``.

`SKILL.md` "v1 scope" section — replace the paragraph with:

```markdown
## Coding agents

Two interfaces. `coding_agent: pi` (default) runs the Pi coding agent against
whatever provider its model names. `coding_agent: claude_code` runs headless
Claude Code on your Claude subscription — the model must be `anthropic/<id>`,
`harness_engineering` must be empty, and `agents.validate()` verifies with
`claude auth status` that the run will bill the subscription rather than the
API before any phase opens. See [references/config.md](references/config.md).
```

- [ ] **Step 8: Run to verify it passes**

Run: `./tests/run_tests.sh -v`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add .claude/skills/sssf/templates/{sssf.config.yaml,env.sample} .claude/skills/sssf/references/config.md .claude/skills/sssf/cookbooks/update_modules.md .claude/skills/sssf/SKILL.md tests/test_config_template.py
git commit -m "docs: document coding_agent: claude_code

Corrects the tools: None escape hatch, which on Claude Code would grant
28 tools rather than the documented seven."
```

---

## Task 12: Live integration against the real CLI

Implements: spec §9 tests 4–19. **This is the only task that spends subscription quota.**

**Files:**
- Create: `tests/live/README.md`
- Create: `tests/live/run_live.sh`
- Test: manual, scripted — results recorded in `tests/live/README.md`

**Interfaces:**
- Consumes: Tasks 1–10 complete and green.
- Produces: a stamped scratch repo proving the adapter end to end.

> **Budget discipline.** The account was at `seven_day: 0.88` when this was
> designed. Check headroom first, use `anthropic/claude-haiku-4-5-20251001`
> throughout, and stop at the first red. `just` is NOT installed — use the
> `uv run` forms below.

- [ ] **Step 1: Check headroom before spending anything**

```bash
claude auth status
```
Expected: `loggedIn: true`, `authMethod: "claude.ai"`, a non-null `subscriptionType`. If utilisation is known to be near 1.0, **stop and wait for the window to reset** — a red run here is indistinguishable from a bug.

- [ ] **Step 2: Stamp a scratch repo**

```bash
SCRATCH=$(mktemp -d)/sssf-live && mkdir -p "$SCRATCH" && cd "$SCRATCH"
git init -q && printf 'def add(a, b):\n    return a + b\n' > calc.py
git add -A && git commit -qm "initial"
uv run /Users/tylerviles/Documents/projects/factory/.claude/skills/sssf/scripts/install.py
```
Expected: `sssf installed into …`, `adws/` present.

- [ ] **Step 3: Switch scout to Claude Code**

Edit `adws/adw_sssf_config/sssf.config.yaml` — on the `scout` agent set
`coding_agent: claude_code`, `model: anthropic/claude-haiku-4-5-20251001`,
**delete its `harness_engineering:` block and the four `subagent_*` entries
from its `tools:` list** (validation fails otherwise, by design).

- [ ] **Step 4: Prove validation fails fast (zero model calls) — spec tests 1–3**

```bash
cd "$SCRATCH"
# non-anthropic model on a claude_code agent
sed -i '' 's|model: anthropic/claude-haiku-4-5-20251001|model: openai/gpt-5.6-terra|' adws/adw_sssf_config/sssf.config.yaml
uv run adws/adw_scout.py "x" 2>&1 | head -5
sed -i '' 's|model: openai/gpt-5.6-terra|model: anthropic/claude-haiku-4-5-20251001|' adws/adw_sssf_config/sssf.config.yaml
```
Expected: `config validation failed:` naming `anthropic/`. **No `claude` process spawns.**

- [ ] **Step 5: First real run — spec tests 4–7, 18**

```bash
cd "$SCRATCH"
uv run adws/adw_prompt.py --agent scout "reply with a one-line summary of this repo"
sqlite3 adws/adw_data/sssf.db "select adw_id,status,total_tokens,round(total_cost,4) from sessions order by started_at desc limit 1;"
sqlite3 adws/adw_data/sssf.db "select agent,coding_agent,cost_basis,context_tokens,context_window from agent_sessions;"
sqlite3 adws/adw_data/sssf.db "select type,count(*) from events group by type;"
sqlite3 adws/adw_data/sssf.db "select json_extract(payload_json,'\$.tool'),json_extract(payload_json,'\$.ok'),json_extract(payload_json,'\$.duration_ms') from events where type='tool_call';"
```
Expected: session `success`; `cost_basis` = `list`; `context_window` = 200000; `tool_call` rows present with non-null tool/ok/duration_ms; `processes` rows all closed.

Confirm billing provenance in the captured stream:
```bash
grep -o '"apiKeySource":"[^"]*"' adws/adw_data/sessions/*/scout/raw_output.jsonl | head -1
grep -c 'rate_limit_event' adws/adw_data/sessions/*/scout/raw_output.jsonl
```
Expected: `"apiKeySource":"none"` and **at least one `rate_limit_event`** — API-key calls do not consume subscription windows, so its presence is the proof that this billed the subscription.

- [ ] **Step 6: Prove the key is stripped — spec test 11**

```bash
cd "$SCRATCH" && ANTHROPIC_API_KEY=sk-ant-not-a-real-key uv run adws/adw_prompt.py --agent scout "say ok"
grep -o '"apiKeySource":"[^"]*"' adws/adw_data/sessions/*/scout/raw_output.jsonl | tail -1
```
Expected: still `"apiKeySource":"none"`. **If this shows `ANTHROPIC_API_KEY`, stop — the core guarantee is broken.**

- [ ] **Step 7: Isolation — spec test 12**

```bash
cd "$SCRATCH"
printf '# Rules\nThe project codeword is XYZZY-PLUGH-7741.\n' > CLAUDE.md
mkdir -p .claude && cat > .claude/settings.json <<'EOF'
{"hooks":{"SessionStart":[{"matcher":"startup","hooks":[{"type":"command","command":"touch /tmp/sssf-hook-fired"}]}]}}
EOF
rm -f /tmp/sssf-hook-fired
uv run adws/adw_prompt.py --agent scout "say ok"
grep -c 'XYZZY-PLUGH-7741' adws/adw_data/sessions/*/scout/raw_output.jsonl
test -f /tmp/sssf-hook-fired && echo "HOOK FIRED (FAIL)" || echo "hook suppressed (pass)"
git checkout -- . && rm -rf .claude CLAUDE.md
```
Expected: sentinel count `0`, hook suppressed. **Do not ask the model whether it knows the codeword** — with file tools it will simply read `CLAUDE.md`, which tests reachability, not injection.

- [ ] **Step 8: Mixed chain and the correction loop — spec tests 8–10**

```bash
cd "$SCRATCH"
uv run adws/adw_plan_build.py "add a subtract(a, b) function to calc.py with a docstring"
cat adws/adw_data/sessions/*/agent_map.json
sqlite3 adws/adw_data/sssf.db "select agent,coding_agent,session_id from agent_sessions;"
```
Expected: chain completes; two `agent_map.json` entries with different `coding_agent`; planner's `writes: [specs/]` held.

Then force the correction loop:
```bash
cd "$SCRATCH"
cp adws/adw_data/prompt_engineering/scout/user.md /tmp/scout-user.bak
python3 - <<'EOF'
import re, pathlib
p = pathlib.Path("adws/adw_data/prompt_engineering/scout/user.md")
t = p.read_text()
p.write_text(re.sub(r"## Report.*", "## Report\n\nAnswer in prose. Do NOT emit JSON.", t, flags=re.S))
EOF
uv run adws/adw_prompt.py --agent scout "summarise this repo" ; echo "exit=$?"
sqlite3 adws/adw_data/sssf.db "select agent,valid,attempt from envelopes order by rowid;"
SESSDIR=$(claude auth status | python3 -c "import json,sys;print(json.load(sys.stdin)['projectsDirectory'])")
ls "$SESSDIR"/*sssf-live*/ | wc -l
cp /tmp/scout-user.bak adws/adw_data/prompt_engineering/scout/user.md
```
Expected: `envelopes` shows `valid=0` rows then either a `valid=1` row or a clean failure after 3 attempts — **and exactly ONE transcript file** in the project's session directory, proving the corrections resumed the same session rather than cold-starting.

- [ ] **Step 9: Kill and timeout — spec tests 13–14**

```bash
cd "$SCRATCH"
uv run adws/adw_plan_build.py "read every file in this repo and write an exhaustive design document" &
ADW=$!; sleep 25; kill $ADW; sleep 8
sqlite3 adws/adw_data/sssf.db "select status from sessions order by started_at desc limit 1;"
sqlite3 adws/adw_data/sssf.db "select count(*) from processes where ended_at is null;"
pgrep -fl "claude -p" || echo "no orphans (pass)"
```
Expected: session `fail`, zero open `processes` rows, **no orphaned `claude` process**.

- [ ] **Step 10: Record the results**

Write `tests/live/README.md` with, for each step: the command, the observed output, and PASS/FAIL. An inconclusive result is recorded as inconclusive — never as a pass.

- [ ] **Step 11: Commit**

```bash
cd /Users/tylerviles/Documents/projects/factory
git add tests/live/
git commit -m "test: record live integration results for the Claude Code adapter"
```

---

## Self-Review

**Spec coverage.** §1 Command construction → Task 7. §1b argv limits → Task 7. §1.1–1.2 the `agent_pi` contract → Tasks 3, 6. §1.3 dispatch → Task 8. §1.4 sessions → Tasks 5, 8. §1.5 tracer → Tasks 6, 9. §1.6 validate → Task 8. §1.7 `writes` → unchanged by design; exercised in Task 11 Step 8. §2 session semantics → Tasks 5, 7, 8. §3 event translation + cost → Tasks 6, 9. §4 model → Task 4. §5 thinking → Task 4. §6 tools + harness_engineering → Tasks 4, 8. §7a env/auth → Tasks 2, 5. §7b isolation → Task 7, verified Task 11 Step 7. §7c permissions → Task 7. §7d append mode → Task 7. §8 process handling, stderr, failure classification, overage → Tasks 2, 6, 7. §9 test plan → Task 11.

**Gaps found and closed inline:**
- §2's create/resume *fallback* was specified but had no task. Now **Task 7 Step 6**, with three tests (exists→resume, missing→create, and a "one retry, never a loop" guard).
- `max_utilization` (formerly `min_headroom`) was a declared, documented field that **nothing read**. Now **Task 10**. Closing it also forced Task 9 to record the whole `rate_limit_info` rather than a bare utilisation float — `unifiedWindows` and `resetsAt` are what make the check sound rather than a guess.
- The `PhaseDetail.vue` edit did not typecheck against the real file (`UsageRow.cost` is required, and `phaseUsage` returns `{rows, partial}`), and it invented a second note mechanism beside the one already there. Rewritten in **Task 9 Step 5** to widen `partial` into `note` and make `cost` optional.
- The kill path read pids from the `processes` table and handed them to `os.killpg`, which takes a process GROUP id — a recycled pid could have signalled an unrelated group, and it ignored the `command` guard `tracer.py:76` exists for. Replaced in **Task 2 Step 6** with an in-memory `run.live_children`, plus a group-leadership check inside `kill_tree`.

**Placeholder scan:** clean — every code step carries real code, every command is one that ran on this machine during design.

**Type consistency:** `CodingAgentRequest` fields are introduced across Tasks 3 (`system_prompt_path`, `resume`, `stderr_path`, `timeout_seconds`), 7 (`restricted`, `on_overage`, `inherit_api_key`) — Task 7 Step 1 and Step 5 make those edits explicit, so no task consumes a field before it exists. `CodingAgentResult` gains `cost_basis`/`rate_limit` in Task 3 and `warnings` in Task 7 Step 4, both before first use. `utils.clip` / `utils.tool_label` are defined in Task 3 and used in Task 6. `agent_cc.ToolCallTracker` is required by `_event_forwarder` in Task 8 and defined in Task 6. `run.live_children` is defined in Task 2 and populated by Task 8's callbacks. `tracer.last_rate_limit` and `agents.exhausted_windows` are defined in Task 10 and consume the `agent_end.rate_limit` payload Task 9 writes — Task 9 precedes Task 10 deliberately.

---

## Execution Handoff

Plan complete and saved to `specs/claude-code-adapter-plan.md` — 12 tasks, of which only Task 12 spends subscription quota. Two execution options:

**1. Subagent-Driven (recommended)** — a fresh subagent per task, reviewed between tasks, fast iteration.

**2. Inline Execution** — tasks executed in this session via `superpowers:executing-plans`, batched with checkpoints.

Which approach?
