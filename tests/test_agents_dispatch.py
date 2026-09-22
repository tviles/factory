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
