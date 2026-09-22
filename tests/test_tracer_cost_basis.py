import inspect
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
    from adw_modules.data_types import AgentSessionRecord
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    t.agent_session_row(AgentSessionRecord(
        adw_id="adw1", agent=_agent(), session_id="sess-1", context_tokens=10,
        context_window=200_000, cost_basis="list"))
    row = t.conn.execute(
        "SELECT cost_basis FROM agent_sessions WHERE adw_id='adw1'").fetchone()
    assert row[0] == "list"


def test_cost_basis_defaults_to_billed_for_pi(tmp_path):
    from adw_modules.data_types import AgentSessionRecord
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    t.agent_session_row(AgentSessionRecord(
        adw_id="adw1", agent=_agent(coding_agent="pi"), session_id="sess-1"))
    row = t.conn.execute(
        "SELECT cost_basis FROM agent_sessions WHERE adw_id='adw1'").fetchone()
    assert row[0] == "billed"


def test_agent_session_row_takes_one_object(tmp_path):
    """Hard rule 4: >4 params takes one concrete type instead — agent_session_row
    grew to six loose params (adw_id, agent, session_id, context_tokens,
    context_window, cost_basis) before this. It must accept exactly one
    positional param (record) besides self."""
    from adw_modules.tracer import Tracer
    params = list(inspect.signature(Tracer.agent_session_row).parameters)
    assert params == ["self", "record"]


def test_process_start_takes_one_object(tmp_path):
    """Same rule, same shape — process_start (adw_id, kind, name, pid, command)
    was five loose params, pre-existing and untouched by the cost_basis change,
    fixed alongside agent_session_row rather than left as the next violation."""
    from adw_modules.tracer import Tracer
    params = list(inspect.signature(Tracer.process_start).parameters)
    assert params == ["self", "record"]


def test_process_start_persists_command_for_pid_reuse_safety(tmp_path):
    """process_start's whole point is recording `command` so a recycled pid is
    not killed by mistake (see the docstring) — the move to ProcessRecord must
    not silently drop or default that field for an existing caller."""
    from adw_modules.data_types import ProcessRecord
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    t.process_start(ProcessRecord(adw_id="adw1", kind="agent", name="scout",
                                  pid=4242, command="claude_code scout gemini"))
    row = t.conn.execute(
        "SELECT kind, name, pid, command FROM processes WHERE adw_id='adw1'"
    ).fetchone()
    assert row == ("agent", "scout", 4242, "claude_code scout gemini")


def _phase(adw_id="adw1"):
    from adw_modules.data_types import Phase, PhaseParams
    params = PhaseParams(name="build", kind="agent", owner="scout",
                         description="test phase")
    return Phase(phase_id=f"{adw_id}_01_build", adw_id=adw_id, seq=1,
                params=params, status="running")


def test_envelope_row_takes_one_object(tmp_path):
    """Same rule, same shape — envelope_row (phase, agent, output_type,
    payload_json, valid, attempt) was six loose params, surveyed alongside
    agent_session_row and process_start in dafa60b and left for a follow-up
    rather than bundled in. It must accept exactly one positional param
    (record) besides self."""
    from adw_modules.tracer import Tracer
    params = list(inspect.signature(Tracer.envelope_row).parameters)
    assert params == ["self", "record"]


def test_envelope_row_persists_all_fields(tmp_path):
    """The move to EnvelopeRecord must not silently drop or default a field
    an existing caller relies on."""
    from adw_modules.data_types import EnvelopeRecord
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    t.envelope_row(EnvelopeRecord(
        phase=_phase("adw1"), agent="scout", output_type="GenericOutput",
        payload_json='{"status": "success"}', valid=True, attempt=1))
    row = t.conn.execute(
        "SELECT adw_id, phase_id, agent, output_type, payload_json, valid,"
        " attempt FROM envelopes WHERE adw_id='adw1'").fetchone()
    assert row == ("adw1", "adw1_01_build", "scout", "GenericOutput",
                   '{"status": "success"}', 1, 1)


def test_agent_session_row_persists_every_column(tmp_path):
    """The move to AgentSessionRecord must not silently drop or default a
    field an existing caller relies on — supplements the signature-only
    check above with a real read-back of every non-timestamp column."""
    from adw_modules.data_types import AgentSessionRecord
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    agent = _agent(name="scout", coding_agent="claude_code")
    agent.color = "#ff0000"
    t.agent_session_row(AgentSessionRecord(
        adw_id="adw1", agent=agent, session_id="sess-1", context_tokens=123,
        context_window=200_000, cost_basis="list"))

    row = t.conn.execute(
        "SELECT adw_id, agent, coding_agent, model, color, session_id,"
        " context_tokens, context_window, cost_basis"
        " FROM agent_sessions WHERE adw_id='adw1' AND agent='scout'").fetchone()

    assert row == ("adw1", "scout", "claude_code", "anthropic/claude-sonnet-5",
                   "#ff0000", "sess-1", 123, 200_000, "list")


def test_agent_session_row_upserts_on_conflict_instead_of_duplicating(tmp_path):
    """agent_session_row's INSERT carries an ON CONFLICT(adw_id, agent) DO
    UPDATE — a second call for the same (adw_id, agent) must update the
    existing row's mutable columns in place, not raise or duplicate it.
    created_at is untouched by the update clause (identity of the row), so it
    must survive across the second call while last_used_at moves forward."""
    import time

    from adw_modules.data_types import AgentSessionRecord
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    agent = _agent(name="scout", coding_agent="claude_code")

    t.agent_session_row(AgentSessionRecord(
        adw_id="adw1", agent=agent, session_id="sess-1", context_tokens=10,
        context_window=100_000, cost_basis="billed"))
    first_created_at = t.conn.execute(
        "SELECT created_at FROM agent_sessions WHERE adw_id='adw1' AND agent='scout'"
    ).fetchone()[0]

    time.sleep(0.01)   # created_at/last_used_at are second-resolution ISO stamps
    agent.model = "anthropic/claude-opus-5"
    agent.color = "#00ff00"
    t.agent_session_row(AgentSessionRecord(
        adw_id="adw1", agent=agent, session_id="sess-2", context_tokens=999,
        context_window=500_000, cost_basis="list"))

    rows = t.conn.execute(
        "SELECT model, color, session_id, context_tokens, context_window,"
        " cost_basis, created_at, last_used_at"
        " FROM agent_sessions WHERE adw_id='adw1' AND agent='scout'").fetchall()

    assert len(rows) == 1   # upsert, not a second row
    (model, color, session_id, context_tokens, context_window, cost_basis,
     created_at, last_used_at) = rows[0]
    assert (model, color, session_id, context_tokens, context_window, cost_basis) == (
        "anthropic/claude-opus-5", "#00ff00", "sess-2", 999, 500_000, "list")
    assert created_at == first_created_at   # row identity: untouched by the update
    assert last_used_at != first_created_at   # but the update did land


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
