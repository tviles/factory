"""context_window() must survive a default `pi` install.

~/.pi/agent/models.json is for CUSTOM models; a fresh `pi` install does not
have one, and `pi` works fine without it. Every run crashed with
FileNotFoundError before ever reaching the _pi_catalog() fallback the
function already contains — confirmed by a real end-to-end run dying here
before pi even launched.

The tests below also pin the follow-up fix: the guard must survive every
"unreadable as JSON" shape (missing file, non-UTF-8 bytes, malformed JSON,
valid-JSON-non-object) silently on the DEFAULT path, but must warn instead of
staying silent when PI_MODELS_PATH was explicitly configured to something
broken.
"""

import pytest


def test_context_window_falls_back_when_models_json_is_missing(tmp_path, monkeypatch):
    """RED against the pre-fix code: an absent MODELS_JSON must fall through
    to the _pi_catalog() scan, not raise FileNotFoundError."""
    from adw_modules import agent_pi

    monkeypatch.delenv("PI_MODELS_PATH", raising=False)
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
    monkeypatch.delenv("PI_MODELS_PATH", raising=False)
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


def test_context_window_falls_back_silently_on_non_utf8_bytes_by_default(tmp_path, monkeypatch, recwarn):
    """UnicodeDecodeError is a ValueError subclass the old (OSError,
    json.JSONDecodeError) tuple missed entirely — a binary or latin-1
    models.json used to crash the run even after the missing-file fix."""
    from adw_modules import agent_pi

    bad = tmp_path / "models.json"
    bad.write_bytes(b"\xff\xfe\x00\x01not utf-8")
    monkeypatch.delenv("PI_MODELS_PATH", raising=False)
    monkeypatch.setattr(agent_pi, "MODELS_JSON", str(bad))
    monkeypatch.setattr(agent_pi, "_pi_catalog",
                        lambda: [("anthropic", "claude-sonnet-5", 200_000)])

    window = agent_pi.context_window("anthropic", "claude-sonnet-5")

    assert window == 200_000
    assert len(recwarn) == 0   # default path: absence is normal, stay quiet


def test_context_window_falls_back_silently_when_json_is_not_an_object(tmp_path, monkeypatch, recwarn):
    """Valid JSON that isn't a dict (`null`, `[]`, a bare string, ...) must
    not reach `.get()` on a list/None/str — that raised AttributeError even
    once the file-read guard was widened."""
    from adw_modules import agent_pi

    not_an_object = tmp_path / "models.json"
    not_an_object.write_text("[]")
    monkeypatch.delenv("PI_MODELS_PATH", raising=False)
    monkeypatch.setattr(agent_pi, "MODELS_JSON", str(not_an_object))
    monkeypatch.setattr(agent_pi, "_pi_catalog",
                        lambda: [("anthropic", "claude-sonnet-5", 200_000)])

    window = agent_pi.context_window("anthropic", "claude-sonnet-5")

    assert window == 200_000
    assert len(recwarn) == 0   # default path: absence is normal, stay quiet


def test_context_window_warns_when_an_explicit_models_path_is_missing(tmp_path, monkeypatch):
    """A default MODELS_JSON silently falling back is normal — but an
    operator who explicitly set PI_MODELS_PATH to a wrong path made a
    mistake, not an absence, and used to crash loudly before the fix and now
    must at least warn rather than fail completely silently."""
    from adw_modules import agent_pi

    missing = tmp_path / "does-not-exist.json"
    monkeypatch.setenv("PI_MODELS_PATH", str(missing))
    monkeypatch.setattr(agent_pi, "MODELS_JSON", str(missing))
    monkeypatch.setattr(agent_pi, "_pi_catalog",
                        lambda: [("anthropic", "claude-sonnet-5", 200_000)])

    with pytest.warns(RuntimeWarning, match="PI_MODELS_PATH"):
        window = agent_pi.context_window("anthropic", "claude-sonnet-5")

    assert window == 200_000   # still falls back — the warning is additive


def test_context_window_warns_when_an_explicit_models_path_is_malformed(tmp_path, monkeypatch):
    """Same as the missing-path case, but the explicitly-set file exists and
    is garbage JSON rather than absent."""
    from adw_modules import agent_pi

    bad = tmp_path / "models.json"
    bad.write_text("{not valid json")
    monkeypatch.setenv("PI_MODELS_PATH", str(bad))
    monkeypatch.setattr(agent_pi, "MODELS_JSON", str(bad))
    monkeypatch.setattr(agent_pi, "_pi_catalog",
                        lambda: [("anthropic", "claude-sonnet-5", 200_000)])

    with pytest.warns(RuntimeWarning, match="PI_MODELS_PATH"):
        window = agent_pi.context_window("anthropic", "claude-sonnet-5")

    assert window == 200_000
