"""Run.phase's failure path must report the phase's REAL error, not a crash
in the tracer/console plumbing that is supposed to just be recording it.

0db53dd wired Tracer.close() into session_finish() as "the one chokepoint
every run-ending path funnels through" — but session_finish() is not the end
of a run on EITHER path that matters: runner.py's `except BaseException`
branch (phase failure) and Run.finish (the ordinary SUCCESS path) both call
it and then keep writing (console.phase_ended, console.session_finished,
both of which trace through Console._emit -> tracer.event). Once
session_finish has closed the connection, every one of those later writes
hits a `None` conn and raises AttributeError.

This is worse than "a failed phase crashes": a real run of a trivial,
fully-successful ADW script crashed with exit code 1 even though the db row
said "success" and the terminal banner printed "✓ success" — because
Console._emit PRINTS first, then traces, so the banner renders fine and the
process dies immediately after. Run.finish's own docstring exists precisely
to keep the db status, the terminal banner, and the exit code from
disagreeing (after an earlier bug where a run was recorded green while
exiting 1) — this regression recreated exactly that failure shape from the
other direction. So every test below that exercises a successful run asserts
the actual exit code Run.finish returns, not just the absence of a
traceback.

These tests drive a phase to failure AND to success through the real
`Run.phase`/`Run.finish` path with a LIVE Tracer (no mocks) — exactly the
path nothing in the existing 187-test suite exercised end-to-end.
"""

from __future__ import annotations

import sqlite3

import pytest


def _run(tmp_path):
    from adw_modules.data_types import SSSFConfig
    from adw_modules.runner import Run
    from adw_modules.tracer import Tracer

    cfg = SSSFConfig()
    cfg.defaults.data_dir = str(tmp_path / "data")
    cfg.observability.db = str(tmp_path / "data" / "sssf.db")
    tracer = Tracer(cfg.observability.db, tmp_path / "data" / "events.jsonl")
    tracer.session_start("test-adw", "tester")   # session.ensure() always does this first
    run = Run(cfg=cfg, adw_id="test-adw", tracer=tracer, engineer="tester")
    return run, tracer


def test_a_failing_phase_reports_its_own_error_not_a_tracer_crash(tmp_path):
    """RED against current code: the phase's ValueError must propagate
    unchanged. Before the fix, Run.phase's except branch calls
    tracer.session_finish (which closes the connection), then
    console.phase_ended/console.session_finished try to trace through the
    now-closed connection and raise AttributeError instead."""
    from adw_modules.data_types import PhaseParams

    run, tracer = _run(tmp_path)
    try:
        with pytest.raises(ValueError, match="provider 429"):
            with run.phase(PhaseParams(name="build", kind="code", owner="tester",
                                       description="test phase")) as ph:
                raise ValueError("provider 429")
    finally:
        tracer.close()


def test_a_failing_phase_still_closes_the_tracer_connection(tmp_path):
    """The fix must not just avoid the crash — it must still release the
    connection once the failure path's writes are done, or the
    ResourceWarning 0db53dd fixed comes back for every failed run."""
    from adw_modules.data_types import PhaseParams

    run, tracer = _run(tmp_path)
    with pytest.raises(ValueError):
        with run.phase(PhaseParams(name="build", kind="code", owner="tester",
                                   description="test phase")) as ph:
            raise ValueError("boom")

    assert tracer.conn is None


def test_a_failing_phase_persists_the_error_to_sqlite_before_closing(tmp_path):
    """The close must come AFTER the failure's own writes, not instead of
    them — read back through a FRESH connection, the way the visualizer
    would, so a half-written trace would fail this even if close() timing
    alone looked right."""
    from adw_modules.data_types import PhaseParams

    run, tracer = _run(tmp_path)
    with pytest.raises(ValueError):
        with run.phase(PhaseParams(name="build", kind="code", owner="tester",
                                   description="test phase")) as ph:
            raise ValueError("boom")

    reader = sqlite3.connect(str(run.cfg.observability.db))
    try:
        phase_row = reader.execute(
            "SELECT status, error FROM phases WHERE adw_id='test-adw'").fetchone()
        assert phase_row == ("fail", "boom")
        session_row = reader.execute(
            "SELECT status FROM sessions WHERE adw_id='test-adw'").fetchone()
        assert session_row == ("fail",)
    finally:
        reader.close()


def test_run_finish_returns_exit_code_0_and_closes_on_a_real_successful_run(tmp_path):
    """Path 1 (Run.finish): the escalated finding — a fully-successful run
    crashed with exit code 1 pre-fix, because console.session_finished
    traces AFTER session_finish had already closed the connection. Assert
    the actual return value `sys.exit(main())` would receive, not just the
    absence of a traceback: a green run must exit 0, matching its own db
    status and terminal banner (the exact agreement Run.finish's docstring
    promises). Also confirms the fix didn't just delete the close — it still
    releases the connection once this path's writes are done."""
    from adw_modules.data_types import PhaseParams

    run, tracer = _run(tmp_path)
    with run.phase(PhaseParams(name="build", kind="code", owner="tester",
                               description="test phase")):
        pass
    exit_code = run.finish()

    assert exit_code == 0
    assert tracer.conn is None


def test_run_finish_returns_exit_code_1_and_still_closes_when_not_accepted(tmp_path):
    """The phases-passed-but-not-accepted case (e.g. a builder whose tests
    never went green) must keep disagreeing correctly — exit 1 — and must
    still close cleanly, not just the all-phases-succeeded case above."""
    from adw_modules.data_types import PhaseParams

    run, tracer = _run(tmp_path)
    with run.phase(PhaseParams(name="build", kind="code", owner="tester",
                               description="test phase")):
        pass
    exit_code = run.finish(accepted=False, reason="acceptance test never passed")

    assert exit_code == 1
    assert tracer.conn is None


def test_signal_handler_style_session_finish_does_not_crash_after_a_normal_finish(tmp_path):
    """Path 3 (session._finalize_when_killed): its handler calls
    session_finish(ok=False) unconditionally and does not know whether the
    run already finished normally. That race must stay a safe no-op, not an
    AttributeError inside a signal handler."""
    from adw_modules.data_types import PhaseParams

    run, tracer = _run(tmp_path)
    with run.phase(PhaseParams(name="build", kind="code", owner="tester",
                               description="test phase")):
        pass
    run.finish()

    tracer.session_finish(run.adw_id, ok=False)   # must not raise
