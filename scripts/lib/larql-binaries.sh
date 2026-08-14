#!/usr/bin/env bash
# Shared binary acquisition for DEC stage drivers (docs/adr/0026-tagged-release-binaries.md).
#
# Hard policy, not an optimisation: GPU-provisioned hosts NEVER build larql
# from source. A cold `cargo build --release` is 20-40 minutes of pure CPU
# work (dozens of dependency crates, wasmtime transitively) during which the
# GPU allocation sits idle. If no release artifact exists for the platform,
# that is a blocker on the dispatch, not something to work around by building
# on the box anyway.
#
# Usage — set any of the inputs below, then source and call:
#
#     LARQL_LOG_PREFIX=dec0-arm-l
#     . "${SCRIPT_DIR}/lib/larql-binaries.sh"
#     larql_acquire_binaries
#
# Afterwards LARQL_BIN and LARQL_SERVER_BIN hold usable paths.
#
# Inputs (all optional):
#   LARQL_RELEASE_VERSION       release tag to fetch        (default: v0.1.0)
#   LARQL_BIN_DIR               where fetched binaries land (default: ./larql-bin)
#   LARQL_BIN                   use this `larql`, skip acquisition
#   LARQL_SERVER_BIN            use this `larql-server`, skip acquisition
#   LARQL_ALLOW_SOURCE_BUILD=1  permit `cargo build` even on a GPU host
#   LARQL_LOG_PREFIX            log tag                     (default: larql)
#
# Resolution order: operator-supplied → reuse populated BIN_DIR → fetch the
# release archive → build from source (GPU-gated).

LARQL_RELEASE_VERSION="${LARQL_RELEASE_VERSION:-v0.1.0}"
LARQL_BIN_DIR="${LARQL_BIN_DIR:-./larql-bin}"
LARQL_BIN="${LARQL_BIN:-}"
LARQL_SERVER_BIN="${LARQL_SERVER_BIN:-}"
LARQL_ALLOW_SOURCE_BUILD="${LARQL_ALLOW_SOURCE_BUILD:-0}"
LARQL_LOG_PREFIX="${LARQL_LOG_PREFIX:-larql}"

_larql_log() { echo "[${LARQL_LOG_PREFIX}] $*"; }

larql_detect_target() {
    case "$(uname -s)/$(uname -m)" in
        Linux/x86_64)  echo "x86_64-unknown-linux-gnu" ;;
        Darwin/arm64)  echo "aarch64-apple-darwin" ;;
        *)             echo "" ;;
    esac
}

larql_fetch_binaries() {
    local target="$1"
    local url="https://github.com/chrishayuk/larql/releases/download/${LARQL_RELEASE_VERSION}/larql-${target}.tar.gz"
    local tmp
    _larql_log "fetching prebuilt binaries: ${url}"
    tmp="$(mktemp)"
    if ! curl -fsSL "${url}" -o "${tmp}"; then
        rm -f "${tmp}"
        return 1
    fi
    mkdir -p "${LARQL_BIN_DIR}"
    tar -xzf "${tmp}" -C "${LARQL_BIN_DIR}" --strip-components=1
    rm -f "${tmp}"
    chmod +x "${LARQL_BIN_DIR}/larql" "${LARQL_BIN_DIR}/larql-server"
}

larql_acquire_binaries() {
    local target

    if [ -n "${LARQL_BIN}" ] && [ -n "${LARQL_SERVER_BIN}" ]; then
        _larql_log "using operator-supplied binaries."
    elif [ -x "${LARQL_BIN_DIR}/larql" ] && [ -x "${LARQL_BIN_DIR}/larql-server" ]; then
        LARQL_BIN="${LARQL_BIN_DIR}/larql"
        LARQL_SERVER_BIN="${LARQL_BIN_DIR}/larql-server"
        _larql_log "reusing binaries already in ${LARQL_BIN_DIR}."
    else
        target="$(larql_detect_target)"
        if [ -n "${target}" ] && larql_fetch_binaries "${target}"; then
            LARQL_BIN="${LARQL_BIN_DIR}/larql"
            LARQL_SERVER_BIN="${LARQL_BIN_DIR}/larql-server"
        elif command -v nvidia-smi >/dev/null 2>&1 && [ "${LARQL_ALLOW_SOURCE_BUILD}" != "1" ]; then
            echo "[${LARQL_LOG_PREFIX}] no release artifact for '${target:-unsupported platform}'" \
                 "at ${LARQL_RELEASE_VERSION}, and this host has a GPU (nvidia-smi present)." >&2
            echo "[${LARQL_LOG_PREFIX}] REFUSING to build larql from source on a GPU box —" \
                 "ADR-0026 policy. Publish a release for this platform, pass LARQL_BIN/" \
                 "LARQL_SERVER_BIN, or re-run on a CPU-only host. Override with" \
                 "LARQL_ALLOW_SOURCE_BUILD=1 only if you accept burning the GPU" \
                 "allocation on compilation." >&2
            exit 1
        else
            # Two ways to land here, and they are NOT the same event — say which.
            if command -v nvidia-smi >/dev/null 2>&1; then
                _larql_log "no release artifact for '${target:-unsupported platform}' at ${LARQL_RELEASE_VERSION};" \
                           "building from source on a host WITH a GPU because" \
                           "LARQL_ALLOW_SOURCE_BUILD=1 was set — the ADR-0026 policy was" \
                           "deliberately overridden, and the GPU is idle for this build."
            else
                _larql_log "no release artifact for '${target:-unsupported platform}' at ${LARQL_RELEASE_VERSION};" \
                           "building from source (no GPU detected, so ADR-0026 permits it)…"
            fi
            # --no-default-features mirrors larql-cli.yml's Linux CI job: default
            # features enable Metal (an empty crate off-macOS, but it still drags
            # larql-{inference,kv,vindex}/gpu and a wider dependency graph —
            # aws-lc-sys among others) that the CPU-only expert-server path never uses.
            cargo build --release -p larql-cli -p larql-server --no-default-features
            LARQL_BIN="./target/release/larql"
            LARQL_SERVER_BIN="./target/release/larql-server"
        fi
    fi

    # Gate on the binary actually EXECUTING, not just existing. A fetched
    # archive can be intact and still unloadable on this host — a
    # `GLIBC_2.38 not found` on Colab was logged as if it were a version
    # string, and the driver went on to launch a server that could never
    # start, costing a full session to diagnose. No measurement may be taken
    # on a path whose binary was never proven to run.
    # NB: capture the binary's own status, NOT a pipeline's — `… | head -1`
    # would report head's success and silently defeat this gate.
    local version_out version_rc=0
    version_out="$("${LARQL_BIN}" --version 2>&1)" || version_rc=$?
    version_out="$(printf '%s\n' "${version_out}" | head -1)"
    if [ "${version_rc}" -ne 0 ]; then
        _larql_log "larql: FATAL — fetched binary will not execute on this host:" >&2
        _larql_log "  ${version_out}" >&2
        _larql_log "  binary: ${LARQL_BIN}" >&2
        _larql_log "  If this is a glibc mismatch, the release was built against a" >&2
        _larql_log "  newer glibc than this host provides — rebuild the release on an" >&2
        _larql_log "  older runner rather than working around it here (ADR-0026)." >&2
        exit 1
    fi
    _larql_log "larql: ${version_out}"
}
