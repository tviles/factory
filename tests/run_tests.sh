#!/usr/bin/env bash
# The SSSF templates declare deps inline (PEP 723); tests need the same set
# plus pytest. `uv run --with` assembles it without polluting the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
# A path argument REPLACES the default target; bare flags (-v, -q, -k foo) are
# appended to it. `pytest tests tests/test_x.py` ran the whole suite, which
# made every "run just this test" instruction in this project a no-op.
targets=(); flags=()
# Flags pytest expects a separate value after (-k EXPR, -m MARKEXPR): without
# tracking these, `-k foo` misfiles `foo` as a target ("foo" doesn't start
# with `-`), which silently reintroduces the same "runs everything" bug this
# fix exists to kill.
value_flags=("-k" "-m")
want_value=""
for arg in "$@"; do
  if [ -n "$want_value" ]; then
    flags+=("$arg")
    want_value=""
    continue
  fi
  case "$arg" in
    -*)
      flags+=("$arg")
      for vf in "${value_flags[@]}"; do
        [ "$arg" = "$vf" ] && want_value=1
      done
      ;;
    *) targets+=("$arg") ;;
  esac
done
[ ${#targets[@]} -eq 0 ] && targets=(tests)
exec uv run --quiet \
  --with pytest --with pydantic --with pyyaml --with python-dotenv --with rich \
  pytest "${targets[@]}" "${flags[@]}"
