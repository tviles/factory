"""context_window() must survive a default `pi` install.

~/.pi/agent/models.json is for CUSTOM models; a fresh `pi` install does not
have one, and `pi` works fine without it. Every run crashed with
FileNotFoundError before ever reaching the _pi_catalog() fallback the
function already contains — confirmed by a real end-to-end run dying here
before pi even launched.
"""


def test_context_window_falls_back_when_models_json_is_missing(tmp_path, monkeypatch):
    """RED against the pre-fix code: an absent MODELS_JSON must fall through
    to the _pi_catalog() scan, not raise FileNotFoundError."""
    from adw_modules import agent_pi

    monkeypatch.setattr(agent_pi, "MODELS_JSON", str(tmp_path / "does-not-exist.json"))
    monkeypatch.setattr(agent_pi, "_pi_catalog",
                        lambda: [("anthropic", "claude-sonnet-5", 200_000)])

    window = agent_pi.context_window("anthropic", "claude-sonnet-5")

    assert window == 200_000


def test_context_window_falls_back_when_models_json_is_malformed(tmp_path, monkeypatch):
    """A models.json that exists but isn't valid JSON must also fall through
    to the catalog scan instead of raising."""
    from adw_modules import agent_pi

    bad = tmp_path / "models.json"
    bad.write_text("{not valid json")
    monkeypatch.setattr(agent_pi, "MODELS_JSON", str(bad))
    monkeypatch.setattr(agent_pi, "_pi_catalog",
                        lambda: [("anthropic", "claude-sonnet-5", 200_000)])

    window = agent_pi.context_window("anthropic", "claude-sonnet-5")

    assert window == 200_000


def test_context_window_returns_zero_when_nothing_knows_the_model(tmp_path, monkeypatch):
    """The final "unknown" answer stays 0, even once the file read is guarded."""
    from adw_modules import agent_pi

    monkeypatch.setattr(agent_pi, "MODELS_JSON", str(tmp_path / "does-not-exist.json"))
    monkeypatch.setattr(agent_pi, "_pi_catalog", lambda: [])

    assert agent_pi.context_window("anthropic", "claude-sonnet-5") == 0


def test_context_window_still_prefers_the_custom_registry_when_present(tmp_path, monkeypatch):
    """Fixing the missing-file case must not change behavior when the file IS
    present and valid — the custom registry stays the first place checked."""
    from adw_modules import agent_pi
    import json

    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({
        "providers": {"anthropic": {"models": [
            {"id": "claude-sonnet-5", "contextWindow": 1_000_000},
        ]}}
    }))
    monkeypatch.setattr(agent_pi, "MODELS_JSON", str(registry))
    # If the fallback ran instead of the registry hit, this would answer.
    monkeypatch.setattr(agent_pi, "_pi_catalog",
                        lambda: [("anthropic", "claude-sonnet-5", 200_000)])

    assert agent_pi.context_window("anthropic", "claude-sonnet-5") == 1_000_000
