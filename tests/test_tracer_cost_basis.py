import sqlite3


def _tracer(tmp_path):
    from adw_modules.tracer import Tracer
    return Tracer(tmp_path / "sssf.db", tmp_path / "events.jsonl")


def _agent(name="scout", coding_agent="claude_code"):
    from adw_modules.data_types import AgentConfig
    return AgentConfig(name=name, coding_agent=coding_agent,
                       model="anthropic/claude-sonnet-5",
                       prompt_engineering={"system": "s", "user": "u"})


def test_cost_basis_column_exists(tmp_path):
    t = _tracer(tmp_path)
    cols = {row[1] for row in t.conn.execute("PRAGMA table_info(agent_sessions)")}
    assert "cost_basis" in cols


def test_cost_basis_is_persisted(tmp_path):
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    t.agent_session_row("adw1", _agent(), "sess-1", context_tokens=10,
                        context_window=200_000, cost_basis="list")
    row = t.conn.execute(
        "SELECT cost_basis FROM agent_sessions WHERE adw_id='adw1'").fetchone()
    assert row[0] == "list"


def test_cost_basis_defaults_to_billed_for_pi(tmp_path):
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    t.agent_session_row("adw1", _agent(coding_agent="pi"), "sess-1")
    row = t.conn.execute(
        "SELECT cost_basis FROM agent_sessions WHERE adw_id='adw1'").fetchone()
    assert row[0] == "billed"


def test_migration_adds_the_column_to_an_older_db(tmp_path):
    """A db from an older SSSF must still open. CREATE TABLE IF NOT EXISTS
    never revisits an existing table, hence the explicit ALTER list."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE agent_sessions (adw_id TEXT, agent TEXT, "
                 "coding_agent TEXT, model TEXT, session_id TEXT, "
                 "created_at TEXT, last_used_at TEXT, "
                 "PRIMARY KEY (adw_id, agent))")
    conn.commit(); conn.close()
    from adw_modules.tracer import Tracer
    t = Tracer(db, tmp_path / "events.jsonl")
    cols = {row[1] for row in t.conn.execute("PRAGMA table_info(agent_sessions)")}
    assert "cost_basis" in cols
