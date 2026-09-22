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
    # observe() returns a LIST (a Claude Code event can close several parallel
    # tool calls at once), so the callback must extend, not append — appending
    # would nest one list inside records and break records[0]["tool"].
    agent_cc.run(_req(tmp_path), on_event=lambda e: records.extend(tracker.observe(e)))
    assert len(records) == 1 and records[0]["tool"] == "Read"


def test_run_surfaces_stderr_warnings(tmp_path, fake_claude, monkeypatch):
    from adw_modules import agent_cc
    warning = "Warning: Unknown --effort value 'off' — ignoring it.\n"
    bindir = fake_claude(tmp_path, "tool_use_roundtrip.jsonl", stderr_text=warning)
    monkeypatch.setattr(agent_cc, "CLAUDE_PATH", str(bindir / "claude"))
    result = agent_cc.run(_req(tmp_path))
    assert any("Unknown --effort value" in w for w in result.warnings)


def test_run_accepts_the_subscription_init_event(tmp_path, fake_claude, monkeypatch):
    """The captured fixture is a real subscription run: init.apiKeySource=='none'."""
    from adw_modules import agent_cc
    bindir = fake_claude(tmp_path, "tool_use_roundtrip.jsonl")
    monkeypatch.setattr(agent_cc, "CLAUDE_PATH", str(bindir / "claude"))
    agent_cc.run(_req(tmp_path))          # must not raise


def test_run_refuses_when_the_child_reports_api_key_billing(tmp_path, fixture_path,
                                                            fake_claude, monkeypatch):
    """The per-send cross-check on the feature's core claim. `preflight_auth`
    ran at validate() time; this catches a key that appeared since."""
    import json
    from adw_modules import agent_cc
    events = [json.loads(l) for l in fixture_path("tool_use_roundtrip.jsonl")
              .read_text().splitlines() if l.strip()]
    for e in events:
        if e.get("subtype") == "init":
            e["apiKeySource"] = "ANTHROPIC_API_KEY"
    doctored = tmp_path / "billed.jsonl"
    doctored.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    bindir = tmp_path / "fakebin"; bindir.mkdir(exist_ok=True)
    script = bindir / "claude"
    script.write_text("#!/usr/bin/env python3\nimport sys,pathlib\n"
                      f"sys.stdout.write(pathlib.Path({str(doctored)!r}).read_text())\n")
    script.chmod(0o755)
    monkeypatch.setattr(agent_cc, "CLAUDE_PATH", str(script))
    with pytest.raises(agent_cc.NotAuthenticated) as e:
        agent_cc.run(_req(tmp_path))
    assert "ANTHROPIC_API_KEY" in str(e.value)


def test_run_scans_only_this_attempts_stderr_on_retry(tmp_path, fake_claude, monkeypatch):
    """send() is called repeatedly against ONE append-mode log, so attempt 2
    must not report attempt 1's warnings. Same defect Task 2 fixed in
    agent_pi.py — pinned here so the new adapter cannot reintroduce it."""
    from adw_modules import agent_cc
    first = fake_claude(tmp_path / "a", "tool_use_roundtrip.jsonl",
                        stderr_text="Warning: FIRST attempt only\n")
    second = fake_claude(tmp_path / "b", "tool_use_roundtrip.jsonl",
                         stderr_text="Warning: SECOND attempt\n")
    req = _req(tmp_path)
    monkeypatch.setattr(agent_cc, "CLAUDE_PATH", str(first / "claude"))
    agent_cc.run(req)
    monkeypatch.setattr(agent_cc, "CLAUDE_PATH", str(second / "claude"))
    result = agent_cc.run(req)
    joined = " ".join(result.warnings)
    assert "SECOND attempt" in joined
    assert "FIRST attempt" not in joined, "leaked a previous attempt's stderr"


def test_run_raises_on_a_nonzero_exit_with_no_text(tmp_path, fake_claude, monkeypatch):
    from adw_modules import agent_cc
    bindir = fake_claude(tmp_path, "empty.jsonl", exit_code=1,
                         stderr_text="Error: Session ID abc is already in use.\n")
    monkeypatch.setattr(agent_cc, "CLAUDE_PATH", str(bindir / "claude"))
    with pytest.raises(agent_cc.CodingAgentError) as e:
        agent_cc.run(_req(tmp_path))
    assert "already in use" in str(e.value)


# ── create/resume fallback ───────────────────────────────────────────────────

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
