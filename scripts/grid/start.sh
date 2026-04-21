#!/usr/bin/env bash
# Start a 10-server grid with 3x layer redundancy.
#
# Topology (34-layer Gemma 3 4B, 3 ranges × 3 replicas + 1 extra):
#   layers  0-11 → ports 8080 8081 8082          (3 replicas)
#   layers 12-23 → ports 8083 8084 8085          (3 replicas)
#   layers 24-33 → ports 8086 8087 8088 8089     (4 replicas, intentionally uneven)
#
# Router HTTP: 9090    gRPC grid: 50052
#
# Usage:
#   ./scripts/grid/start.sh [VINDEX_PATH]

set -euo pipefail

VINDEX="${1:-/Users/christopherhay/chris-source/larql/output/gemma3-4b-q4k-v2.vindex}"
ROUTER_HTTP=9090
ROUTER_GRPC=50052
ROUTER_HOST=127.0.0.1
BIN_DIR="$(cd "$(dirname "$0")/../.." && pwd)/target/release"
PID_DIR="$(cd "$(dirname "$0")" && pwd)/.pids"

if [[ ! -d "$VINDEX" ]]; then
  echo "error: vindex not found at $VINDEX" >&2
  echo "Usage: $0 [VINDEX_PATH]" >&2
  exit 1
fi

if [[ ! -x "$BIN_DIR/larql-router" ]]; then
  echo "error: $BIN_DIR/larql-router not found — run: cargo build --release -p larql-router -p larql-server" >&2
  exit 1
fi

mkdir -p "$PID_DIR"

# Kill any stale processes from a previous run
if [[ -f "$PID_DIR/all.pids" ]]; then
  echo "Stopping previous grid..."
  while read -r pid; do
    kill "$pid" 2>/dev/null || true
  done < "$PID_DIR/all.pids"
  rm -f "$PID_DIR/all.pids"
  sleep 1
fi

start_server() {
  local port="$1"
  local layers="$2"
  local label="shard-${port}"
  local logfile="$PID_DIR/${label}.log"

  "$BIN_DIR/larql-server" "$VINDEX" \
    --ffn-only \
    --layers "$layers" \
    --port "$port" \
    --join "http://${ROUTER_HOST}:${ROUTER_GRPC}" \
    --public-url "http://${ROUTER_HOST}:${port}" \
    > "$logfile" 2>&1 &

  local pid=$!
  echo "$pid" >> "$PID_DIR/all.pids"
  echo "  [shard layers=$layers port=$port pid=$pid]"
}

echo "Starting larql-router (HTTP=$ROUTER_HTTP gRPC=$ROUTER_GRPC)..."
"$BIN_DIR/larql-router" \
  --grid-port "$ROUTER_GRPC" \
  --port "$ROUTER_HTTP" \
  > "$PID_DIR/router.log" 2>&1 &
ROUTER_PID=$!
echo "$ROUTER_PID" >> "$PID_DIR/all.pids"
echo "  [router pid=$ROUTER_PID]"
sleep 1

echo ""
echo "Starting 10 shard servers..."

# Range A: layers 0-11 (3 replicas)
start_server 8080 "0-11"
start_server 8081 "0-11"
start_server 8082 "0-11"

# Range B: layers 12-23 (3 replicas)
start_server 8083 "12-23"
start_server 8084 "12-23"
start_server 8085 "12-23"

# Range C: layers 24-33 (4 replicas — intentionally uneven to test load balancing)
start_server 8086 "24-33"
start_server 8087 "24-33"
start_server 8088 "24-33"
start_server 8089 "24-33"

echo ""
echo "Waiting for shards to load and announce (~15s)..."
sleep 15

echo ""
echo "Grid status:"
echo "  Router HTTP : http://${ROUTER_HOST}:${ROUTER_HTTP}"
echo "  Router gRPC : ${ROUTER_HOST}:${ROUTER_GRPC}"
echo "  Health      : $(curl -sf http://${ROUTER_HOST}:${ROUTER_HTTP}/v1/health 2>/dev/null || echo 'UNREACHABLE')"
echo ""
echo "Registered servers (from router logs):"
grep "Grid: server joined\|Registered with router" "$PID_DIR"/router.log 2>/dev/null | tail -20 || true
echo ""
echo "PIDs saved to $PID_DIR/all.pids"
echo "Logs in $PID_DIR/*.log"
echo ""
echo "To stop: ./scripts/grid/stop.sh"
