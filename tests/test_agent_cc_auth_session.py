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

# `claude auth status` never emits this string itself (it omits the key
# entirely) — but the stream's `init` event does, for the same "no API key"
# state. Asserting it here pins the asymmetry the guard has to tolerate.
SUBSCRIPTION_WITH_NONE_KEY_SOURCE = json.dumps({
    "loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
    "projectsDirectory": "/Users/x/.claude/projects",
    "apiKeySource": "none", "subscriptionType": "max"})


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


def test_parse_auth_status_accepts_api_key_source_none_as_absent():
    """apiKeySource: "none" means "no API key in use" on this surface, not
    "an API key named none" — it must pass exactly like an omitted key."""
    from adw_modules.agent_cc import parse_auth_status
    info = parse_auth_status(SUBSCRIPTION_WITH_NONE_KEY_SOURCE, inherit_api_key=False)
    assert info["subscriptionType"] == "max"


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


# ── projectsDirectory caching (review Important #6) ──────────────────────────

def test_projects_directory_defaults_to_empty(monkeypatch):
    from adw_modules import agent_cc
    monkeypatch.setattr(agent_cc, "_preflight_projects_directory", "")
    assert agent_cc.projects_directory() == ""


def test_remember_projects_directory_caches_the_field(monkeypatch):
    from adw_modules import agent_cc
    monkeypatch.setattr(agent_cc, "_preflight_projects_directory", "")
    agent_cc.remember_projects_directory(
        {"loggedIn": True, "projectsDirectory": "/Users/x/.claude/projects"})
    assert agent_cc.projects_directory() == "/Users/x/.claude/projects"


def test_remember_projects_directory_tolerates_a_missing_field(monkeypatch):
    """An inherit_api_key auth_status (WITH_API_KEY above) has no
    projectsDirectory at all — must degrade to '', not raise."""
    from adw_modules import agent_cc
    monkeypatch.setattr(agent_cc, "_preflight_projects_directory", "stale")
    agent_cc.remember_projects_directory({"loggedIn": True})
    assert agent_cc.projects_directory() == ""
