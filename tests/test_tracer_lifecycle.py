"""Tracer owns a sqlite connection and must release it — but NOT from inside
session_finish().

`-W error::ResourceWarning` reported one "unclosed database" warning per
Tracer built in the suite before Tracer.close() existed — a connection per
Tracer object, leaked for the life of the process. The first fix (0db53dd)
wired close() into session_finish() on the theory that it was "the one
chokepoint every run-ending path already funnels through" — but it is not:
Run.finish() and Run.phase's except branch (runner.py) both call
session_finish() and then keep writing (Console.phase_ended /
session_finished, which trace through Console._emit -> tracer.event). A
close() inside session_finish handed those later writes a dead connection —
`AttributeError: 'NoneType' object has no attribute 'execute'` — burying
whatever real error a failed phase was reporting (see test_runner_lifecycle.py
for the end-to-end regression test through the real Run.phase path).

The corrected contract: session_finish() only ever WRITES (status +
processes_end_all). Closing is the caller's job, done in runner.py's
Run.finish and Run.phase's except branch, each once their own last
console/tracer write for that path is done. session._finalize_when_killed's
signal handler (session.py) never closes either, for the same reason — it
cannot know whether Run.phase's except branch still needs the connection.
session_finish keeps its "no-op once already closed" guard regardless, since
that handler can still race an already-finished run that DID close via one
of the runner.py tails.
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
    """A defensive second close() (or two racing paths that both reach one)
    must not raise."""
    t = _tracer(tmp_path)
    t.close()
    t.close()   # must not raise


def test_session_finish_does_not_close_the_connection(tmp_path):
    """The corrected contract: session_finish only writes. Closing is done
    by the caller (runner.py), once its own later writes are also done —
    session_finish itself cannot know whether more writes are coming."""
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    t.session_finish("adw1", ok=True)
    assert t.conn is not None
    t.close()


def test_session_finish_after_an_explicit_close_does_not_raise(tmp_path):
    """The real double-call race this guards: runner.py's tail closes the
    connection once its writes are done, and session._finalize_when_killed's
    SIGTERM/SIGINT handler can still fire afterward and call session_finish
    unconditionally, not knowing the run already finished. That must stay a
    safe no-op, not an AttributeError inside a signal handler."""
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    t.session_finish("adw1", ok=True)
    t.close()                              # what runner.py's tail does
    t.session_finish("adw1", ok=False)     # the signal handler racing in — must not raise
    assert t.conn is None


def test_writes_from_session_finish_are_durable_without_a_close(tmp_path):
    """session_finish's UPDATE must be visible immediately — Tracer opens
    its connection with isolation_level=None (autocommit), so durability
    never depended on close() flushing anything. Read back through a FRESH
    connection, the way the visualizer would, while the writer's own
    connection is still open."""
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    t.session_finish("adw1", ok=True)

    reader = sqlite3.connect(str(t.db_path))
    row = reader.execute(
        "SELECT status FROM sessions WHERE adw_id='adw1'").fetchone()
    assert row == ("success",)
    reader.close()
    t.close()


def test_writer_close_does_not_disturb_a_concurrent_wal_reader(tmp_path):
    """The visualizer polls sssf.db from its OWN connection, concurrently,
    in WAL mode (Tracer.__init__ sets journal_mode=WAL). Closing the
    WRITER's handle (now done by the caller, after session_finish) must not
    affect a reader that already had the file open — simulated here by
    opening the second connection BEFORE the writer closes, same as a live
    visualizer poll would."""
    t = _tracer(tmp_path)
    t.session_start("adw1", "eng")
    reader = sqlite3.connect(str(t.db_path))

    t.session_finish("adw1", ok=True)
    t.close()   # what runner.py's tail does, after session_finish

    row = reader.execute(
        "SELECT status FROM sessions WHERE adw_id='adw1'").fetchone()
    assert row == ("success",)
    # the reader can keep going after the writer is gone
    assert reader.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
    reader.close()
