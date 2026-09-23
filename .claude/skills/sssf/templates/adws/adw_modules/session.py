"""Session lifecycle: pin-or-create an adw_id, build the Run object.

`ensure(cfg, adw_id)` joins the session if it exists or creates it under
exactly that id (pinned ids for repeatable runs); omitted, a fresh id is
minted and printed so the next ADW can pick it up.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

from .data_types import ProcessRecord, SSSFConfig
from .runner import Run
from .tracer import Tracer
from .utils import engineer_name, kill_tree, new_id


def _finalize_when_killed(run: Run) -> None:
    """A killed run still closes its own trace.

    Python's default SIGTERM handling exits without unwinding, so `just kill`
    (or any `kill <pid>`) would leave the session reading `running` forever and
    its process rows open — the trace would claim work is in flight that is
    already dead. Turning the signal into SystemExit both finalizes here and
    lets the phase context manager record the phase as failed on the way out.
    """
    def handler(signum, _frame):
        # Reap before closing the trace. Closing the rows first would record
        # the run as finished while its coding agent kept working — a killed
        # ADW that leaves a `claude` or `pi` child running is exactly the pid
        # nobody can find afterwards.
        for pid in list(run.live_children):
            kill_tree(pid)
        # Writes the session/process rows only — deliberately does NOT close
        # the tracer's connection. This handler can preempt the main thread
        # mid-phase, in which case the SystemExit raised below is caught by
        # Run.phase's `except BaseException` branch (runner.py), which still
        # needs a live connection to record the failure and trace the
        # console output that follows. Closing here would hand that branch a
        # dead connection — the exact bug this split exists to avoid. Each
        # runner.py tail (Run.finish, Run.phase's except branch) closes the
        # connection itself once ITS writes are done; if neither runs after
        # this handler (e.g. the signal lands between phases), the
        # connection is simply left for process teardown to reclaim.
        run.tracer.session_finish(run.adw_id, ok=False)   # also ends process rows
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)


def ensure(cfg: SSSFConfig, adw_id: str | None = None) -> Run:
    adw_id = adw_id or new_id(8)
    tracer = Tracer(cfg.observability.db,
                    f"{cfg.defaults.data_dir}/sessions/{adw_id}/events.jsonl")
    run = Run(cfg=cfg, adw_id=adw_id, tracer=tracer, engineer=engineer_name())
    tracer.session_start(adw_id, run.engineer, adw_name=Path(sys.argv[0]).stem)
    # This process is the run. Record it before any phase opens, so a run that
    # hangs in its first agent call is still killable by adw_id.
    tracer.process_start(ProcessRecord(
        adw_id=adw_id, kind="adw", name="", pid=os.getpid(),
        command=" ".join([Path(sys.argv[0]).name, *sys.argv[1:]])))
    _finalize_when_killed(run)
    run.console.session_started(adw_id, run.engineer)
    return run
