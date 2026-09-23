import json
import sqlite3

import pytest

CAPTURED = {
    "status": "allowed_warning", "resetsAt": 1790100000,
    "rateLimitType": "seven_day", "utilization": 0.88, "isUsingOverage": False,
    "unifiedWindows": {"five_hour": {"utilization": 0.0, "resetsAt": 1790040000},
                       "seven_day": {"utilization": 0.88, "resetsAt": 1790100000}},
}
BEFORE_RESET = 1790000000
AFTER_RESET = 1790200000


def _db(tmp_path, *payloads):
    return _db_rows(tmp_path, *[("agent_end", "", payload) for payload in payloads])


def _db_rows(tmp_path, *rows):
    """Like `_db`, but each row is (type, name, payload) — for exercising the
    rate_limit_observed log event `last_rate_limit()` also scans (review
    Important #5: a RateLimited/OverageRefused raise propagates out of
    execute() before agent_end ever fires, so that failure path needs its own
    event shape to reach the trace at all)."""
    db = tmp_path / "sssf.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE events (event_id TEXT, type TEXT, name TEXT, payload_json TEXT)")
    for i, (etype, name, payload) in enumerate(rows):
        conn.execute("INSERT INTO events VALUES (?,?,?,?)",
                     (f"e{i}", etype, name, json.dumps(payload)))
    conn.commit(); conn.close()
    return db


def test_last_rate_limit_returns_none_without_a_db(tmp_path):
    from adw_modules.tracer import last_rate_limit
    assert last_rate_limit(tmp_path / "nope.db") is None


def test_last_rate_limit_returns_none_when_no_agent_recorded_one(tmp_path):
    from adw_modules.tracer import last_rate_limit
    assert last_rate_limit(_db(tmp_path, {"cost": 0.01})) is None


def test_last_rate_limit_skips_pi_rows_and_finds_the_newest(tmp_path):
    from adw_modules.tracer import last_rate_limit
    db = _db(tmp_path, {"cost": 0.5, "rate_limit": {"utilization": 0.1}},
             {"cost": 0.01},                      # a pi agent: no rate_limit key
             {"cost": 0.2, "rate_limit": CAPTURED})
    assert last_rate_limit(db) == CAPTURED


def test_last_rate_limit_does_not_create_the_db(tmp_path):
    """validate() must not bring a trace db into existence as a side effect."""
    from adw_modules.tracer import last_rate_limit
    missing = tmp_path / "absent.db"
    last_rate_limit(missing)
    assert not missing.exists()


def test_exhausted_windows_flags_a_live_window_over_the_bar():
    from adw_modules.agents import exhausted_windows
    hits = exhausted_windows(CAPTURED, 0.80, now=BEFORE_RESET)
    assert [h[0] for h in hits] == ["seven_day"]


def test_exhausted_windows_ignores_a_window_that_has_reset():
    """The reading is void once resetsAt passes — this is fact, not a guess."""
    from adw_modules.agents import exhausted_windows
    assert exhausted_windows(CAPTURED, 0.80, now=AFTER_RESET) == []


def test_exhausted_windows_default_bar_only_catches_full_exhaustion():
    from adw_modules.agents import exhausted_windows
    assert exhausted_windows(CAPTURED, 1.0, now=BEFORE_RESET) == []
    full = {"unifiedWindows": {"seven_day": {"utilization": 1.0, "resetsAt": 1790100000}}}
    assert [h[0] for h in exhausted_windows(full, 1.0, now=BEFORE_RESET)] == ["seven_day"]


def test_exhausted_windows_falls_back_to_the_flat_shape():
    from adw_modules.agents import exhausted_windows
    flat = {"rateLimitType": "five_hour", "utilization": 0.99, "resetsAt": 1790100000}
    assert [h[0] for h in exhausted_windows(flat, 0.9, now=BEFORE_RESET)] == ["five_hour"]


def test_validate_refuses_when_a_live_window_is_exhausted(tmp_path, monkeypatch):
    from adw_modules import agent_cc, agents
    from adw_modules.data_types import SSSFConfig
    monkeypatch.setattr(agent_cc, "preflight_auth",
                        lambda inherit_api_key=False: {"loggedIn": True,
                                                       "authMethod": "claude.ai"})
    monkeypatch.setattr(agents, "_now", lambda: BEFORE_RESET)
    sysmd, usermd = tmp_path / "s.md", tmp_path / "u.md"
    sysmd.write_text("s"); usermd.write_text("u")
    db = _db(tmp_path, {"rate_limit": CAPTURED})
    cfg = SSSFConfig(
        defaults={"claude_code": {"max_utilization": 0.80}},
        observability={"db": str(db)},
        agents=[dict(name="scout", coding_agent="claude_code",
                     model="anthropic/claude-sonnet-5", tools=["read"],
                     prompt_engineering={"system": str(sysmd), "user": str(usermd)})])
    with pytest.raises(SystemExit) as e:
        agents.validate(cfg, ["scout"])
    assert "seven_day" in str(e.value) and "0.88" in str(e.value)


def test_last_rate_limit_skips_a_row_whose_payload_is_a_json_list(tmp_path):
    """Valid JSON that is not an object is as unusable as no payload at all."""
    from adw_modules.tracer import last_rate_limit
    db = _db(tmp_path, [1, 2, 3], {"rate_limit": CAPTURED})
    assert last_rate_limit(db) == CAPTURED


def test_last_rate_limit_skips_a_row_whose_payload_is_a_json_scalar(tmp_path):
    from adw_modules.tracer import last_rate_limit
    db = _db(tmp_path, "3", {"rate_limit": CAPTURED})
    assert last_rate_limit(db) == CAPTURED


def test_last_rate_limit_returns_none_when_only_bad_shapes_exist(tmp_path):
    from adw_modules.tracer import last_rate_limit
    assert last_rate_limit(_db(tmp_path, [1, 2, 3], "3")) is None


def test_exhausted_windows_returns_empty_for_a_non_dict_reading():
    from adw_modules.agents import exhausted_windows
    assert exhausted_windows([1, 2, 3], 0.80, now=BEFORE_RESET) == []
    assert exhausted_windows("not a dict", 0.80, now=BEFORE_RESET) == []
    assert exhausted_windows(None, 0.80, now=BEFORE_RESET) == []


def test_exhausted_windows_skips_a_non_dict_window_entry():
    """A window value that isn't a dict must be treated as unknown, not raise."""
    from adw_modules.agents import exhausted_windows
    bad = {"unifiedWindows": {"seven_day": "not-a-dict",
                              "five_hour": {"utilization": 0.99,
                                            "resetsAt": 1790100000}}}
    assert [h[0] for h in exhausted_windows(bad, 0.80, now=BEFORE_RESET)] == ["five_hour"]


def test_last_rate_limit_finds_a_rate_limit_observed_log_row(tmp_path):
    """review Important #5: the only reading likely to be >= max_utilization
    (a rejected/blocked window) comes from a FAILED send, which never reaches
    agent_end — agents.py's send() persists it as a rate_limit_observed log
    event instead, and last_rate_limit() must find that too."""
    from adw_modules.tracer import last_rate_limit
    db = _db_rows(tmp_path,
                  ("agent_end", "scout", {"cost": 0.2}),
                  ("log", "rate_limit_observed", {"agent": "scout", "rate_limit": CAPTURED}))
    assert last_rate_limit(db) == CAPTURED


def test_last_rate_limit_ignores_an_unrelated_log_row(tmp_path):
    """A rate_limit_observed row is matched by NAME, not just type='log' — an
    ordinary console/warning log row must not be mistaken for one."""
    from adw_modules.tracer import last_rate_limit
    db = _db_rows(tmp_path, ("log", "coding_agent_warning", {"message": "hi"}))
    assert last_rate_limit(db) is None


def test_validate_refuses_using_a_rate_limit_observed_from_a_failed_send(tmp_path, monkeypatch):
    """The headroom guard must be reachable from the failure path, not only
    from a successful phase's agent_end — this is what makes the default of
    1.0 mean something (review Important #5)."""
    from adw_modules import agent_cc, agents
    from adw_modules.data_types import SSSFConfig
    monkeypatch.setattr(agent_cc, "preflight_auth",
                        lambda inherit_api_key=False: {"loggedIn": True,
                                                       "authMethod": "claude.ai"})
    monkeypatch.setattr(agents, "_now", lambda: BEFORE_RESET)
    sysmd, usermd = tmp_path / "s.md", tmp_path / "u.md"
    sysmd.write_text("s"); usermd.write_text("u")
    db = _db_rows(tmp_path,
                  ("log", "rate_limit_observed", {"agent": "scout", "rate_limit": CAPTURED}))
    cfg = SSSFConfig(
        defaults={"claude_code": {"max_utilization": 0.80}},
        observability={"db": str(db)},
        agents=[dict(name="scout", coding_agent="claude_code",
                     model="anthropic/claude-sonnet-5", tools=["read"],
                     prompt_engineering={"system": str(sysmd), "user": str(usermd)})])
    with pytest.raises(SystemExit) as e:
        agents.validate(cfg, ["scout"])
    assert "seven_day" in str(e.value) and "0.88" in str(e.value)


def test_validate_does_not_check_headroom_for_a_pi_only_chain(tmp_path, monkeypatch):
    """A claude_code agent elsewhere in the roster must not block a pi chain."""
    from adw_modules import agent_pi, agents
    from adw_modules.data_types import SSSFConfig
    monkeypatch.setattr(agent_pi, "resolve_model", lambda p: ("google", "g"))
    monkeypatch.setattr(agents, "_now", lambda: BEFORE_RESET)
    sysmd, usermd = tmp_path / "s.md", tmp_path / "u.md"
    sysmd.write_text("s"); usermd.write_text("u")
    pe = {"system": str(sysmd), "user": str(usermd)}
    db = _db(tmp_path, {"rate_limit": CAPTURED})
    cfg = SSSFConfig(
        defaults={"claude_code": {"max_utilization": 0.80}},
        observability={"db": str(db)},
        agents=[dict(name="builder", coding_agent="pi", model="google/g",
                     prompt_engineering=pe, tools=["read"]),
                dict(name="scout", coding_agent="claude_code",
                     model="anthropic/claude-sonnet-5", prompt_engineering=pe,
                     tools=["read"])])
    agents.validate(cfg, ["builder"])          # must not raise
