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
