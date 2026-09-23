"""agent_pi must not misreport a provider failure as malformed JSON.

Real incident: a run against google/gemini-3.6-flash died with "scout never
produced valid GenericOutput JSON: no JSON object found in the response".
The actual cause was an HTTP 429 (quota exhausted): the provider refused the
final turn three times (initial send + both JSON-correction retries), each
returning empty text. pi's own stream already carried `stopReason: "error"`
plus a populated `errorMessage` on the failing turn — agent_pi.run() ignored
both and let the empty text fall through to agents._extract_json, which
diagnosed it as bad JSON and burned both correction attempts against a wall
retrying could not fix.

Fake `pi` binaries in the style of test_subprocess_io.py's `_fake_pi_script`
— no network, no real provider.
"""

import json
from pathlib import Path

import pytest


def _fake_pi_script(tmp_path: Path, body: str, name: str = "fake_pi.py") -> Path:
    """A stand-in `pi` binary: ignores every argv it is given and just runs
    `body`. See test_subprocess_io.py's helper of the same name/shape."""
    script = tmp_path / name
    script.write_text("#!/usr/bin/env python3\n" + body)
    script.chmod(0o755)
    return script


def _patch_pi_resolution(monkeypatch, agent_pi, pi_path: Path) -> None:
    """Point agent_pi.run() at the fake binary, and bypass `pi --list-models`
    / `~/.pi/agent/models.json` — orthogonal to what this suite tests."""
    monkeypatch.setattr(agent_pi, "PI_PATH", str(pi_path))
    monkeypatch.setattr(agent_pi, "resolve_model", lambda pattern: ("test", "test-model"))
    monkeypatch.setattr(agent_pi, "context_window", lambda provider, model_id: 0)


def _pi_request(tmp_path: Path, **overrides):
    from adw_modules.data_types import PiRequest
    defaults = dict(
        prompt="hi", system_prompt="sys", model="unused",
        session_id="test-session", session_dir=str(tmp_path / "pi_sessions"),
        raw_output_path=str(tmp_path / "agent" / "raw_output.jsonl"),
        cwd=str(tmp_path),
    )
    defaults.update(overrides)
    return PiRequest(**defaults)


# The exact event shape the task's raw_output.jsonl carried on the failing
# turns: role/api/provider/model, stopReason: "error", and errorMessage as a
# JSON-encoded provider error body.
_ERRORED_TURN = json.dumps({
    "type": "message_end",
    "message": {
        "role": "assistant", "api": "google-generative-ai", "provider": "google",
        "model": "gemini-3.6-flash", "stopReason": "error", "content": [],
        "errorMessage": json.dumps({
            "error": {"message": "429 Too Many Requests. You exceeded your "
                                  "current quota, please check your plan and "
                                  "billing details."}
        }),
    },
}) + "\n"


def test_errored_turn_with_no_text_raises_provider_error_not_bad_json(tmp_path, monkeypatch):
    """Pins the fix at its source: run() itself must raise, naming the
    provider's message, before any text ever reaches the JSON parser."""
    from adw_modules import agent_pi

    fake_pi = _fake_pi_script(tmp_path, (
        "import sys\n"
        f"sys.stdout.write({_ERRORED_TURN!r})\n"
        "sys.stdout.flush()\n"
    ))
    _patch_pi_resolution(monkeypatch, agent_pi, fake_pi)
    request = _pi_request(tmp_path)

    with pytest.raises(agent_pi.ProviderError) as exc_info:
        agent_pi.run(request)

    message = str(exc_info.value)
    assert "429" in message
    assert "quota" in message.lower()
    # The old misdiagnosis: agents._extract_json's "no JSON object found in
    # the response", wrapped by agents._parse_with_retries into "never
    # produced valid ... JSON". Neither phrase belongs in a provider-error
    # message — that conflation is exactly the bug.
    assert "no JSON object found" not in message
    assert "never produced valid" not in message


def test_errored_turn_does_not_mask_a_real_result_from_an_earlier_turn(tmp_path, monkeypatch):
    """If an EARLIER turn already produced usable text, a later errored turn
    must not discard it — only "no usable text at all" triggers
    ProviderError; this must not become a new way to lose a good answer."""
    from adw_modules import agent_pi

    good_turn = json.dumps({
        "type": "message_end",
        "message": {"role": "assistant", "stopReason": "stop",
                    "content": [{"type": "text", "text": '{"status": "success"}'}],
                    "usage": {}},
    })
    fake_pi = _fake_pi_script(tmp_path, (
        "import sys\n"
        f"sys.stdout.write({good_turn!r} + '\\n')\n"
        f"sys.stdout.write({_ERRORED_TURN!r})\n"
        "sys.stdout.flush()\n"
    ))
    _patch_pi_resolution(monkeypatch, agent_pi, fake_pi)
    request = _pi_request(tmp_path)

    result = agent_pi.run(request)   # must NOT raise
    assert result.text == '{"status": "success"}'


def test_malformed_but_present_text_is_unaffected_by_the_fix(tmp_path, monkeypatch):
    """A legitimate malformed-JSON response (no provider error, just bad
    text) must still return normally so agents._parse_with_retries can retry
    it — the fix must not change behaviour for a genuine formatting mistake."""
    from adw_modules import agent_pi

    fake_pi = _fake_pi_script(tmp_path, (
        "import sys, json\n"
        "sys.stdout.write(json.dumps({'type': 'message_end', 'message': "
        "{'role': 'assistant', 'stopReason': 'stop', "
        "'content': [{'type': 'text', 'text': 'not json at all'}], "
        "'usage': {}}}) + '\\n')\n"
        "sys.stdout.flush()\n"
    ))
    _patch_pi_resolution(monkeypatch, agent_pi, fake_pi)
    request = _pi_request(tmp_path)

    result = agent_pi.run(request)   # must NOT raise
    assert result.text == "not json at all"
