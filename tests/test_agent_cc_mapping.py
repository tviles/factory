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
