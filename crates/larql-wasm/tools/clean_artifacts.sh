#!/usr/bin/env bash
# clean_artifacts.sh — reclaim build artifacts from the wasm gating campaign.
#
# Tiers:
#   (default)   safe cleanup that does NOT disturb incremental builds:
#               - /tmp/*-pipe.log scratch logs from pipeline runs
#               - target/**/examples, target/**/doc, target/**/*.d depfiles
#               - target subdirs for targets the pipeline never uses
#                 (wasm32-unknown-unknown, wasm32-wasip1, wasm32-unknown-emscripten)
#   --deep      additionally `cargo clean` BOTH workspaces (native debug +
#               wasm32v1-none). Reclaims the most space but forces full
#               recompiles. REFUSED while a cargo/gate build is running.
#
# Usage:
#   tools/clean_artifacts.sh          # safe tier
#   tools/clean_artifacts.sh --deep   # + cargo clean (only when idle)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WASM_DIR="$ROOT/crates/larql-wasm"
DEEP=0
[ "${1:-}" = "--deep" ] && DEEP=1

build_running() { pgrep -f 'cargo |gate_cascade.py|pipeline.sh' >/dev/null 2>&1; }

freed_before=$(du -sk "$ROOT" 2>/dev/null | cut -f1)

echo "── safe cleanup ──"
# 1. Scratch pipeline logs.
rm -f /tmp/*-pipe.log 2>/dev/null && echo "  removed /tmp/*-pipe.log"

# 2. Pipeline never builds these targets locally — drop their target subdirs.
for unused in wasm32-unknown-unknown wasm32-wasip1 wasm32-unknown-emscripten; do
    for td in "$ROOT/target/$unused" "$WASM_DIR/target/$unused" \
              "$ROOT/crates/larql-experts/target/$unused"; do
        if [ -d "$td" ]; then
            sz=$(du -sh "$td" 2>/dev/null | cut -f1)
            rm -rf "$td" && echo "  removed $td ($sz)"
        fi
    done
done

# 3. examples/doc/depfiles are not needed for `--lib` incremental checks.
for base in "$ROOT/target" "$WASM_DIR/target"; do
    [ -d "$base" ] || continue
    find "$base" -type d -name examples -prune -exec rm -rf {} + 2>/dev/null
    find "$base" -type d -name doc      -prune -exec rm -rf {} + 2>/dev/null
done
echo "  pruned examples/ and doc/ under target dirs"

if [ "$DEEP" = "1" ]; then
    echo "── deep cleanup (cargo clean) ──"
    if build_running; then
        echo "  REFUSED: a cargo/gate build is running. Re-run --deep when idle."
    else
        cargo clean --manifest-path "$ROOT/Cargo.toml" 2>/dev/null && echo "  cargo clean: root workspace"
        cargo clean --manifest-path "$WASM_DIR/Cargo.toml" 2>/dev/null && echo "  cargo clean: larql-wasm workspace"
    fi
fi

freed_after=$(du -sk "$ROOT" 2>/dev/null | cut -f1)
freed_mb=$(( (freed_before - freed_after) / 1024 ))
echo "── done: reclaimed ~${freed_mb} MB (repo now $(du -sh "$ROOT" 2>/dev/null | cut -f1)) ──"
