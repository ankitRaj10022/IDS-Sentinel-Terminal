#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WHEEL="$ROOT/dist/ids_sentinel_terminal-0.2.1-py3-none-any.whl"

if [ "${1:-}" = "--rebuild" ] || [ ! -f "$WHEEL" ]; then
  cd "$ROOT"
  python3 scripts/build_python_package.py
  python3 scripts/build_distributions.py
fi

if [ ! -f "$WHEEL" ]; then
  echo "Wheel not found: $WHEEL" >&2
  exit 1
fi

python3 -m pip install --user --force-reinstall "$WHEEL"

cat <<'EOF'

IDS Sentinel Terminal installed.
Run these commands:
  ids-sentinel --version
  ids-sentinel status
  ids-sentinel gui

If ids-sentinel is not found, run:
  export PATH="$HOME/.local/bin:$PATH"
EOF
