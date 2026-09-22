import copy

import pytest


def _result_event(fixture):
    return next(e for e in fixture("tool_use_roundtrip.jsonl")
                if e.get("type") == "result")


# ── tool call folding ────────────────────────────────────────────────────────

def test_tracker_folds_a_tool_use_and_tool_result_into_one_record(fixture):
    from adw_modules.agent_cc import ToolCallTracker
    tracker = ToolCallTracker()
    records = []
    for e in fixture("tool_use_roundtrip.jsonl"):
        records.extend(tracker.observe(e))
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
    records = []
    for e in fixture("tool_use_roundtrip.jsonl"):
        records.extend(tracker.observe(e))
    rec = records[0]
    for key in ("tool", "tool_call_id", "args", "result_snippet", "ok",
                "duration_ms", "label", "started_at", "ended_at"):
        assert key in rec, f"missing {key}"


def test_tracker_handles_parallel_tool_uses_in_one_message():
    from adw_modules.agent_cc import ToolCallTracker
    tracker = ToolCallTracker()
    assert tracker.observe({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "a", "name": "Read", "input": {"file_path": "/a"}},
        {"type": "tool_use", "id": "b", "name": "Bash", "input": {"command": "ls"}},
    ]}}) == []
    rb = tracker.observe({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "b", "content": "out"}]}})
    ra = tracker.observe({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "x"}]}})
    assert rb[0]["tool"] == "Bash" and ra[0]["tool"] == "Read"


def test_tracker_returns_every_completed_call_when_one_event_carries_several():
    """Synthetic event: no fixture shows this shape (all four captures put one
    tool_result per user event), but Claude Code CAN close a parallel tool
    batch in a single event — review Task 6 I-2. observe() must emit every
    completed call from that one event, not just the first, and must not
    leave the later ones stuck open."""
    from adw_modules.agent_cc import ToolCallTracker
    tracker = ToolCallTracker()
    tracker.observe({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "a", "name": "Read", "input": {"file_path": "/a"}},
        {"type": "tool_use", "id": "b", "name": "Bash", "input": {"command": "ls"}},
    ]}})
    records = tracker.observe({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "x"},
        {"type": "tool_result", "tool_use_id": "b", "content": "out"},
    ]}})
    assert len(records) == 2
    assert {r["tool"] for r in records} == {"Read", "Bash"}
    assert {r["tool_call_id"] for r in records} == {"a", "b"}
    assert tracker._open == {}   # neither call is left leaked open


def test_tracker_handles_tool_result_content_as_a_block_list():
    from adw_modules.agent_cc import ToolCallTracker
    tracker = ToolCallTracker()
    tracker.observe({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "a", "name": "Read", "input": {}}]}})
    records = tracker.observe({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "a",
         "content": [{"type": "text", "text": "block form"}]}]}})
    assert records[0]["result_snippet"] == "block form"


def test_tracker_marks_an_errored_tool_result_not_ok():
    from adw_modules.agent_cc import ToolCallTracker
    tracker = ToolCallTracker()
    tracker.observe({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "false"}}]}})
    records = tracker.observe({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "boom",
         "is_error": True}]}})
    assert records[0]["ok"] is False


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
