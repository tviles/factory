"""Tracer owns a sqlite connection and must release it.

`-W error::ResourceWarning` reported one "unclosed database" warning per
Tracer built in the suite before Tracer.close() existed — a connection per
Tracer object, leaked for the life of the process. session_finish() is the
one chokepoint every run-ending path already funnels through (Run.finish's
normal path, Run.phase's except branch, and session._finalize_when_killed's
SIGTERM/SIGINT handler — see runner.py and session.py), so that is where the
close is wired in.
"""

import sqlite3


def _tracer(tmp_path):
    from adw_modules.tracer import Tracer
    return Tracer(tmp_path / "sssf.db", tmp_path / "events.jsonl")


def test_close_releases_the_connection(tmp_path):
    t = _tracer(tmp_path)
    t.close()
    assert t.conn is None


def test_close_is_idempotent(tmp_path):
    """The signal handler can race an already-finished run; a defensive
    second close() must not raise."""
    t = _tracer(tmp_path)
    t.close()
    t.close()   # must not raise


def test_session_finish_closes_the_connection(tmp_path):
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    t.session_finish("adw1", ok=True)
    assert t.conn is None


def test_session_finish_called_twice_does_not_raise(tmp_path):
    """Mirrors the real double-call risk: Run.phase's except branch and
    session._finalize_when_killed's handler both call session_finish, and a
    killed run that had already finished normally must not crash instead of
    just being a no-op."""
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    t.session_finish("adw1", ok=True)
    t.session_finish("adw1", ok=False)   # must not raise, must not resurrect conn
    assert t.conn is None


def test_writes_before_close_are_durable(tmp_path):
    """The close must come AFTER every write session_finish makes — a
    dropped or half-committed UPDATE would be a worse bug than the warning
    this fixes. Read back through a FRESH connection, the way the visualizer
    would, rather than trusting the closed Tracer's own state."""
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    t.session_finish("adw1", ok=True)

    reader = sqlite3.connect(str(t.db_path))
    row = reader.execute(
        "SELECT status FROM sessions WHERE adw_id='adw1'").fetchone()
    assert row == ("success",)
    reader.close()


def test_writer_close_does_not_disturb_a_concurrent_wal_reader(tmp_path):
    """The visualizer polls sssf.db from its OWN connection, concurrently,
    in WAL mode (Tracer.__init__ sets journal_mode=WAL). Closing the
    WRITER's handle in session_finish must not affect a reader that already
    had the file open — simulated here by opening the second connection
    BEFORE the writer closes, same as a live visualizer poll would."""
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    reader = sqlite3.connect(str(t.db_path))

    t.session_finish("adw1", ok=True)   # closes the writer's own connection only

    row = reader.execute(
        "SELECT status FROM sessions WHERE adw_id='adw1'").fetchone()
    assert row == ("success",)
    # the reader can keep going after the writer is gone
    assert reader.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
    reader.close()
