import json
import types
from unittest.mock import MagicMock

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


def test_validate_caches_projects_directory_from_the_preflight(tmp_path, monkeypatch):
    """review Important #6: validate() used to parse projectsDirectory out of
    `claude auth status` and throw it away. It must now cache it so execute()
    can record it without a second CLI call."""
    from adw_modules import agent_cc, agents
    monkeypatch.setattr(agent_cc, "_preflight_projects_directory", "")
    agents.validate(_cfg(tmp_path), ["scout"])
    assert agent_cc.projects_directory() == "/tmp/projects"   # from _no_real_preflight


def test_pi_agents_still_validate_through_the_pi_resolver(tmp_path, monkeypatch):
    from adw_modules import agent_pi, agents
    seen = []
    monkeypatch.setattr(agent_pi, "resolve_model",
                        lambda p: seen.append(p) or ("google", "gemini-3.6-flash"))
    agents.validate(_cfg(tmp_path, coding_agent="pi",
                         model="google/gemini-3.6-flash"), ["scout"])
    assert seen == ["google/gemini-3.6-flash"]


# ── execute() dispatch + resume threading ───────────────────────────────────
#
# `validate()` never spawns anything, so none of the tests above touch the
# resume predicate at all. This is the riskiest code in the task — swap the
# order of `_agent_session_id`'s `(session_id, reused)` tuple and every test
# above still passes, while every claude_code run would either create a
# session that already exists or resume one that does not. A fake adapter
# module, injected through `agents.ADAPTERS`, exercises `execute()` for real
# — no subprocess, no API call — and pins the four behaviours this task added.

def _agent_config(tmp_path, **overrides):
    from adw_modules.data_types import AgentConfig
    sysmd, usermd = tmp_path / "system.md", tmp_path / "user.md"
    sysmd.write_text("s"); usermd.write_text("u")
    fields = dict(name="scout", coding_agent="claude_code",
                  model="anthropic/claude-sonnet-5", thinking="medium",
                  prompt_engineering={"system": str(sysmd), "user": str(usermd)},
                  harness_engineering=[], tools=None, writes=None)
    fields.update(overrides)
    return AgentConfig(**fields)


def _exec_cfg(tmp_path, **agent_overrides):
    from adw_modules.data_types import SSSFConfig
    return SSSFConfig(agents=[_agent_config(tmp_path, **agent_overrides)])


def _fake_run(tmp_path, cfg):
    """Minimal stand-in for runner.Run — only what agents.execute() touches.

    `repo_root` points OUTSIDE any git checkout (tmp_path), so
    `permissions.snapshot()`'s `git diff`/`git ls-files` calls fail closed and
    return {} on both sides of the phase — nothing for `enforce()` to flag,
    without needing a real repo.
    """
    run = types.SimpleNamespace()
    run.cfg = cfg
    run.adw_id = "test-adw"
    run.tracer = MagicMock()
    run.console = MagicMock()
    run.repo_root = tmp_path
    run.session_dir = tmp_path / "session"
    run.context_handoff_dir = tmp_path / "session" / "context_handoff"
    run.context_handoff_dir.mkdir(parents=True, exist_ok=True)
    run.agent_map = {}
    run.live_children = set()
    run.tokens = 0
    run.cost = 0.0

    def add_usage(tokens, cost):
        run.tokens += tokens
        run.cost += cost
    run.add_usage = add_usage

    def save_agent_map(name, entry):
        run.agent_map[name] = entry
    run.save_agent_map = save_agent_map
    return run


def _phase(owner, retries=0):
    from adw_modules.data_types import Phase, PhaseParams
    params = PhaseParams(name="build", kind="agent", owner=owner,
                         description="test phase", retries=retries)
    return Phase(phase_id="test-adw_01_build", adw_id="test-adw", seq=1,
                params=params, status="running")


def _result(text="", warnings=None):
    from adw_modules.data_types import CodingAgentResult
    return CodingAgentResult(text=text, warnings=warnings or [])


def _fake_adapter(results):
    """Records every CodingAgentRequest sent, in order; returns queued results."""
    class _Tracker:
        def observe(self, event):
            return []

    requests = []

    def run(request, on_event=None, on_spawn=None, on_exit=None):
        requests.append(request)
        return results[len(requests) - 1]

    return types.SimpleNamespace(run=run, ToolCallTracker=_Tracker), requests


def test_execute_resume_is_false_then_true_across_a_json_retry(tmp_path, monkeypatch):
    from adw_modules import agents
    from adw_modules.data_types import AgentCall, GenericOutput

    cfg = _exec_cfg(tmp_path)
    run = _fake_run(tmp_path, cfg)
    phase = _phase("scout")
    bad = _result(text="not json at all")
    good = _result(text=json.dumps({"status": "success", "summary": "ok"}))
    fake_adapter, requests = _fake_adapter([bad, good])
    monkeypatch.setitem(agents.ADAPTERS, "claude_code", fake_adapter)

    envelope = agents.execute(run, phase, AgentCall(output_type=GenericOutput, prompt="go"))

    assert [r.resume for r in requests] == [False, True]
    assert envelope.status == "success"


def test_execute_resumes_immediately_when_rejoining_a_prior_session(tmp_path, monkeypatch):
    from adw_modules import agents
    from adw_modules.data_types import AgentCall, GenericOutput

    cfg = _exec_cfg(tmp_path)
    run = _fake_run(tmp_path, cfg)
    run.agent_map["scout"] = {"session_id": "sssf-test-adw-scout-abcd",
                              "model": "anthropic/claude-sonnet-5",
                              "coding_agent": "claude_code"}
    phase = _phase("scout")
    good = _result(text=json.dumps({"status": "success", "summary": "ok"}))
    fake_adapter, requests = _fake_adapter([good])
    monkeypatch.setitem(agents.ADAPTERS, "claude_code", fake_adapter)

    agents.execute(run, phase, AgentCall(output_type=GenericOutput, prompt="go"))

    assert requests[0].resume is True
    assert requests[0].session_id == "sssf-test-adw-scout-abcd"


def test_execute_sets_restricted_only_for_a_read_only_agent(tmp_path, monkeypatch):
    from adw_modules import agents
    from adw_modules.data_types import AgentCall, GenericOutput

    for writes, expected in ((list(), True), (None, False)):
        cfg = _exec_cfg(tmp_path, writes=writes)
        run = _fake_run(tmp_path, cfg)
        phase = _phase("scout")
        good = _result(text=json.dumps({"status": "success", "summary": "ok"}))
        fake_adapter, requests = _fake_adapter([good])
        monkeypatch.setitem(agents.ADAPTERS, "claude_code", fake_adapter)

        agents.execute(run, phase, AgentCall(output_type=GenericOutput, prompt="go"))

        assert requests[0].restricted is expected, f"writes={writes!r}"


def test_execute_records_cc_session_uuid_and_projects_directory(tmp_path, monkeypatch):
    """review Important #6: spec §1b's mitigation for the containment
    regression (Claude Code transcripts live outside data_dir) was recording
    projectsDirectory + the session uuid in agent_start and agent_map.json —
    neither shipped. Pinned here for both destinations."""
    from adw_modules import agent_cc, agents
    from adw_modules.data_types import AgentCall, GenericOutput

    monkeypatch.setattr(agent_cc, "_preflight_projects_directory",
                        "/Users/x/.claude/projects")
    cfg = _exec_cfg(tmp_path)
    run = _fake_run(tmp_path, cfg)
    phase = _phase("scout")
    good = _result(text=json.dumps({"status": "success", "summary": "ok"}))
    fake_adapter, requests = _fake_adapter([good])
    monkeypatch.setitem(agents.ADAPTERS, "claude_code", fake_adapter)

    agents.execute(run, phase, AgentCall(output_type=GenericOutput, prompt="go"))

    started = next(c.args[0] for c in run.tracer.event.call_args_list
                  if c.args[0].type == "agent_start")
    expected_uuid = agent_cc.cc_session_uuid(requests[0].session_id)
    assert started.payload["cc_session_uuid"] == expected_uuid
    assert started.payload["cc_projects_directory"] == "/Users/x/.claude/projects"

    entry = run.agent_map["scout"]
    assert entry["cc_session_uuid"] == expected_uuid
    assert entry["cc_projects_directory"] == "/Users/x/.claude/projects"


def test_execute_leaves_cc_fields_empty_for_a_pi_agent(tmp_path, monkeypatch):
    """The uuid/projectsDirectory mitigation is a claude_code-only concept —
    a pi agent (whose sessions already live inside data_dir) must not get a
    fabricated Claude Code uuid or a stale cached projects directory."""
    from adw_modules import agent_cc, agents
    from adw_modules.data_types import AgentCall, GenericOutput

    monkeypatch.setattr(agent_cc, "_preflight_projects_directory",
                        "/Users/x/.claude/projects")
    cfg = _exec_cfg(tmp_path, coding_agent="pi", model="google/gemini-3.6-flash")
    run = _fake_run(tmp_path, cfg)
    phase = _phase("scout")
    good = _result(text=json.dumps({"status": "success", "summary": "ok"}))
    fake_adapter, requests = _fake_adapter([good])
    monkeypatch.setitem(agents.ADAPTERS, "pi", fake_adapter)

    agents.execute(run, phase, AgentCall(output_type=GenericOutput, prompt="go"))

    started = next(c.args[0] for c in run.tracer.event.call_args_list
                  if c.args[0].type == "agent_start")
    assert started.payload["cc_session_uuid"] == ""
    assert started.payload["cc_projects_directory"] == ""
    assert run.agent_map["scout"]["cc_session_uuid"] == ""
    assert run.agent_map["scout"]["cc_projects_directory"] == ""


def test_execute_forwards_warnings_to_both_the_trace_and_the_console(tmp_path, monkeypatch):
    """Tasks 5-6 added these warnings (the --effort clamp, the missing-init-
    event notice) specifically for operator visibility — a trace-only event
    would leave a live run silent about them."""
    from adw_modules import agents
    from adw_modules.data_types import AgentCall, GenericOutput

    cfg = _exec_cfg(tmp_path)
    run = _fake_run(tmp_path, cfg)
    phase = _phase("scout")
    warning = "thinking 'off' is not a Claude Code effort level; using 'low'"
    result = _result(text=json.dumps({"status": "success", "summary": "ok"}),
                     warnings=[warning])
    fake_adapter, requests = _fake_adapter([result])
    monkeypatch.setitem(agents.ADAPTERS, "claude_code", fake_adapter)

    agents.execute(run, phase, AgentCall(output_type=GenericOutput, prompt="go"))

    traced = [c.args[0] for c in run.tracer.event.call_args_list
             if c.args[0].name == "coding_agent_warning"]
    assert len(traced) == 1
    assert traced[0].payload == {"agent": "scout", "message": warning}
    run.console.note.assert_called_once_with(f"scout: {warning}")


def test_execute_persists_rate_limit_observed_when_the_send_raises(tmp_path, monkeypatch):
    """review Important #5: RateLimited/OverageRefused propagate straight out
    of execute() — agent_end never fires on this path — so send() must
    persist the observation itself, or the reading that actually trips
    agents.py's headroom guard can never reach the trace."""
    from adw_modules import agent_cc, agents
    from adw_modules.data_types import AgentCall, GenericOutput

    cfg = _exec_cfg(tmp_path)
    run = _fake_run(tmp_path, cfg)
    phase = _phase("scout")
    rate_limit = {"status": "rejected", "rateLimitType": "seven_day",
                  "utilization": 1.0, "resetsAt": 1790100000}

    def _boom_run(request, on_event=None, on_spawn=None, on_exit=None):
        raise agent_cc.RateLimited("rate limited", rate_limit=rate_limit)

    fake_adapter = types.SimpleNamespace(run=_boom_run, ToolCallTracker=lambda: types.SimpleNamespace(observe=lambda e: []))
    monkeypatch.setitem(agents.ADAPTERS, "claude_code", fake_adapter)

    with pytest.raises(agent_cc.RateLimited):
        agents.execute(run, phase, AgentCall(output_type=GenericOutput, prompt="go"))

    observed = [c.args[0] for c in run.tracer.event.call_args_list
               if c.args[0].name == "rate_limit_observed"]
    assert len(observed) == 1
    assert observed[0].type == "log"
    assert observed[0].payload == {"agent": "scout", "rate_limit": rate_limit}


def test_execute_does_not_persist_rate_limit_observed_on_an_ordinary_error(tmp_path, monkeypatch):
    """An adapter error with no .rate_limit attribute (a plain pi RuntimeError,
    or a CodingAgentError with none set) must not fabricate a reading."""
    from adw_modules import agents
    from adw_modules.data_types import AgentCall, GenericOutput

    cfg = _exec_cfg(tmp_path)
    run = _fake_run(tmp_path, cfg)
    phase = _phase("scout")

    def _boom_run(request, on_event=None, on_spawn=None, on_exit=None):
        raise RuntimeError("claude exited 1")

    fake_adapter = types.SimpleNamespace(run=_boom_run, ToolCallTracker=lambda: types.SimpleNamespace(observe=lambda e: []))
    monkeypatch.setitem(agents.ADAPTERS, "claude_code", fake_adapter)

    with pytest.raises(RuntimeError):
        agents.execute(run, phase, AgentCall(output_type=GenericOutput, prompt="go"))

    assert not [c.args[0] for c in run.tracer.event.call_args_list
               if c.args[0].name == "rate_limit_observed"]


def test_execute_records_permission_denials_in_agent_end_payload(tmp_path, monkeypatch):
    """spec §7c: --permission-prompts none silently denies anything that
    would have prompted; result.permission_denials must reach the agent_end
    payload so a mysteriously-stalled agent is diagnosable from the trace."""
    from adw_modules import agents
    from adw_modules.data_types import AgentCall, CodingAgentResult, GenericOutput

    cfg = _exec_cfg(tmp_path)
    run = _fake_run(tmp_path, cfg)
    phase = _phase("scout")
    denials = [{"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}]
    result = CodingAgentResult(text=json.dumps({"status": "success", "summary": "ok"}),
                               permission_denials=denials)
    fake_adapter, requests = _fake_adapter([result])
    monkeypatch.setitem(agents.ADAPTERS, "claude_code", fake_adapter)

    agents.execute(run, phase, AgentCall(output_type=GenericOutput, prompt="go"))

    agent_end = next(c.args[0] for c in run.tracer.event.call_args_list
                     if c.args[0].type == "agent_end")
    assert agent_end.payload["permission_denials"] == denials


def test_execute_records_the_spawned_process_via_on_spawn(tmp_path, monkeypatch):
    """agents.py's _on_spawn (agents.py:264) constructs a real ProcessRecord
    on every coding-agent spawn — post-review M4: neither the fake adapters
    above nor the tracer suite's hand-written ProcessRecord exercise this
    construction site, so a mistyped kwarg there would pass the whole suite.
    None of the other dispatch tests' fake adapters call on_spawn/on_exit;
    this one does, so the real call site in agents.py runs for real."""
    from adw_modules import agents
    from adw_modules.data_types import AgentCall, GenericOutput, ProcessRecord

    cfg = _exec_cfg(tmp_path)
    run = _fake_run(tmp_path, cfg)
    phase = _phase("scout")
    good = _result(text=json.dumps({"status": "success", "summary": "ok"}))

    def _run_and_spawn(request, on_event=None, on_spawn=None, on_exit=None):
        if on_spawn:
            on_spawn(4242)
        if on_exit:
            on_exit(4242)
        return good

    fake_adapter = types.SimpleNamespace(
        run=_run_and_spawn,
        ToolCallTracker=lambda: types.SimpleNamespace(observe=lambda e: []))
    monkeypatch.setitem(agents.ADAPTERS, "claude_code", fake_adapter)

    agents.execute(run, phase, AgentCall(output_type=GenericOutput, prompt="go"))

    (record,), _ = run.tracer.process_start.call_args
    assert isinstance(record, ProcessRecord)
    assert record.adw_id == "test-adw"
    assert record.kind == "agent"
    assert record.name == "scout"
    assert record.pid == 4242
    assert record.command == "claude_code scout anthropic/claude-sonnet-5"

    run.tracer.process_end.assert_called_once_with("test-adw", 4242)


# ── hard rule 4: _persist_envelope's param count ─────────────────────────────
#
# dafa60b/ed85936 moved Tracer.agent_session_row, process_start, and
# envelope_row onto one record each. _persist_envelope (agents.py) was the
# same violation, deferred because it is private — eight loose params (run,
# phase, agent_name, call, envelope, attempt, valid, raw). Fixed the same way:
# one EnvelopePersistence object besides `run`, matching the pattern's own
# ProcessRecord/AgentSessionRecord/EnvelopeRecord shape.

def test_persist_envelope_takes_one_param_besides_run():
    import inspect
    from adw_modules import agents
    params = list(inspect.signature(agents._persist_envelope).parameters)
    assert params == ["run", "params"]


def test_persist_envelope_writes_the_envelope_record_on_success(tmp_path, monkeypatch):
    """The move to EnvelopePersistence must not silently drop or default a
    field an existing caller relies on — mirrors
    test_envelope_row_persists_all_fields's real-write shape, but through the
    actual agents.py call site rather than constructing EnvelopeRecord by
    hand."""
    from adw_modules import agents
    from adw_modules.data_types import AgentCall, EnvelopeRecord, GenericOutput

    cfg = _exec_cfg(tmp_path)
    run = _fake_run(tmp_path, cfg)
    phase = _phase("scout")
    good = _result(text=json.dumps({"status": "success", "summary": "ok"}))
    fake_adapter, requests = _fake_adapter([good])
    monkeypatch.setitem(agents.ADAPTERS, "claude_code", fake_adapter)

    agents.execute(run, phase, AgentCall(output_type=GenericOutput, prompt="go"))

    (record,), _ = run.tracer.envelope_row.call_args
    assert isinstance(record, EnvelopeRecord)
    assert record.agent == "scout"
    assert record.output_type == "GenericOutput"
    assert record.valid is True
    assert json.loads(record.payload_json)["summary"] == "ok"

    on_disk = json.loads((run.session_dir / "scout" / "envelope.json").read_text())
    assert on_disk["agent_name"] == "scout"
    assert on_disk["summary"] == "ok"


def test_persist_envelope_writes_invalid_rows_from_the_retry_loop(tmp_path, monkeypatch):
    """_parse_with_retries calls _persist_envelope on every failed parse
    attempt too (EnvelopeRecord's own docstring) — the raw text must survive
    the move into EnvelopePersistence's `raw` field."""
    from adw_modules import agents
    from adw_modules.data_types import AgentCall, GenericOutput

    cfg = _exec_cfg(tmp_path)
    run = _fake_run(tmp_path, cfg)
    phase = _phase("scout")
    bad = _result(text="not json at all")
    good = _result(text=json.dumps({"status": "success", "summary": "ok"}))
    fake_adapter, requests = _fake_adapter([bad, good])
    monkeypatch.setitem(agents.ADAPTERS, "claude_code", fake_adapter)

    agents.execute(run, phase, AgentCall(output_type=GenericOutput, prompt="go"))

    invalid_calls = [c.args[0] for c in run.tracer.envelope_row.call_args_list
                     if not c.args[0].valid]
    assert len(invalid_calls) == 1
    assert invalid_calls[0].attempt == 1
    assert "not json at all" in invalid_calls[0].payload_json
    assert 4242 not in run.live_children   # on_exit discards what on_spawn added
