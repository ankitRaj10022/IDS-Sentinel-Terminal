#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TTY_ARGS=()
if [[ ! -t 0 ]]; then
  TTY_ARGS=(-T)
fi

if docker compose ps --status running ids-automation >/dev/null 2>&1; then
  docker compose exec "${TTY_ARGS[@]}" ids-automation python -m ids_app.terminal "$@"
else
  docker compose run --rm "${TTY_ARGS[@]}" ids-automation python -m ids_app.terminal "$@"
fi
