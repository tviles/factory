"""Test harness for the SSSF skill templates.

The templates are not an installed package — they are files install.py stamps
into a target repo. Tests import them by putting the template dir on sys.path,
which is exactly how a stamped repo imports them (adws/ is the cwd there).
"""
import json
import os
import stat
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / ".claude" / "skills" / "sssf" / "templates" / "adws"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

sys.path.insert(0, str(TEMPLATES))


@pytest.fixture
def templates_dir() -> Path:
    return TEMPLATES


@pytest.fixture
def fixture_path():
    def _path(name: str) -> Path:
        p = FIXTURES / name
        assert p.exists(), f"missing fixture {name}; see Task 1 Step 1"
        return p
    return _path


@pytest.fixture
def fixture(fixture_path):
    """Parse a captured stream-json capture into a list of events."""
    def _load(name: str) -> list[dict]:
        events = []
        for line in fixture_path(name).read_text().splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
        return events
    return _load


@pytest.fixture
def fake_claude(fixture_path):
    """A directory holding an executable `claude` that replays a fixture.

    Lets agent_cc.run() be exercised end to end with no API call. The stub
    writes the fixture to stdout line by line, writes stderr_text to stderr,
    and exits with exit_code. It also records its argv to `argv.json` in the
    same directory so tests can assert on command construction.
    """
    def _make(tmp_path: Path, fixture_name: str, *, exit_code: int = 0,
              stderr_text: str = "") -> Path:
        bindir = tmp_path / "fakebin"
        bindir.mkdir(exist_ok=True)
        script = bindir / "claude"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys, pathlib\n"
            f"pathlib.Path({str(bindir / 'argv.json')!r}).write_text(json.dumps(sys.argv[1:]))\n"
            f"sys.stderr.write({stderr_text!r})\n"
            f"sys.stdout.write(pathlib.Path({str(fixture_path(fixture_name))!r}).read_text())\n"
            f"sys.exit({exit_code})\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return bindir
    return _make


@pytest.fixture
def fake_claude_argv():
    def _read(bindir: Path) -> list[str]:
        return json.loads((bindir / "argv.json").read_text())
    return _read
