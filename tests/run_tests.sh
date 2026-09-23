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
# `set -u` + an EMPTY array expansion ("${flags[@]}" with nothing appended) is
# an unbound-variable error on bash <= 4.2 — stock macOS ships 3.2, and the
# shebang's `env bash` resolves to it on any Mac with no newer bash earlier on
# PATH. `"$@"` is specifically exempt from `set -u`; a hand-rolled array is
# not, so the flagless and path-only forms (the two Extra A exists to fix)
# aborted with `flags[@]: unbound variable`. `${arr[@]+"${arr[@]}"}` is the
# standard set-u-safe idiom: expands to nothing when the array is empty
# instead of tripping the unset check. `targets` can't currently go empty
# (defaulted just above), but the same guard costs nothing and removes the
# assumption.
exec uv run --quiet \
  --with pytest --with pydantic --with pyyaml --with python-dotenv --with rich \
  pytest ${targets[@]+"${targets[@]}"} ${flags[@]+"${flags[@]}"}
