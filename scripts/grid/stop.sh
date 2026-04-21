#!/usr/bin/env bash
PID_DIR="$(cd "$(dirname "$0")" && pwd)/.pids"

if [[ ! -f "$PID_DIR/all.pids" ]]; then
  echo "No grid running (no $PID_DIR/all.pids)"
  exit 0
fi

echo "Stopping grid..."
while read -r pid; do
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" && echo "  killed $pid"
  fi
done < "$PID_DIR/all.pids"

rm -f "$PID_DIR/all.pids"
echo "Done."
