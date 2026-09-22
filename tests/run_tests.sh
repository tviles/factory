#!/usr/bin/env bash
# The SSSF templates declare deps inline (PEP 723); tests need the same set
# plus pytest. `uv run --with` assembles it without polluting the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --quiet \
  --with pytest --with pydantic --with pyyaml --with python-dotenv --with rich \
  pytest tests "$@"
