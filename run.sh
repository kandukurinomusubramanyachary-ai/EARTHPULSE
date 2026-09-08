#!/usr/bin/env bash
# EarthPulse — one-command start for macOS / Linux
#
#   chmod +x run.sh
#   ./run.sh
#
# Creates a virtual environment, installs dependencies the first time,
# then starts the server on http://localhost:8000

set -e
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.10+ from https://python.org"
  exit 1
fi

# Refuse to run on unsupported Python.
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    sys.exit("ERROR: Python 3.10+ required, found %d.%d" % sys.version_info[:2])
PY

if [ ! -d ".venv" ]; then
  echo "==> Creating virtual environment (.venv)"
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f ".venv/.deps-installed" ]; then
  echo "==> Installing dependencies (first run only, ~1-2 minutes)"
  python -m pip install --upgrade pip --quiet
  python -m pip install -r requirements.txt
  touch .venv/.deps-installed
fi

echo ""
echo "  EarthPulse is starting..."
echo "  Open http://localhost:8000 in your browser"
echo "  Press Ctrl+C to stop"
echo ""

cd backend
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
