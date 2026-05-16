#!/usr/bin/env bash
# SPDX-FileCopyrightText: Contributors to the larql-to-sparql project
# SPDX-License-Identifier: Apache-2.0
#
# Native ChromeOS/crosh build script using the Chromium OS SDK (cros_sdk).
#
# Builds one crate inside the cros_sdk chroot against a CrOS target triple.
# No 'repo sync' is needed; cros_sdk --download bootstraps a self-contained
# chroot from a ~5 GB SDK tarball fetched from storage.googleapis.com.
# cargo is pre-installed in the chroot via the dev-lang/rust Portage ebuild;
# do NOT invoke rustup inside the chroot.
#
# Assumptions:
#   - chromite/bin is on PATH (provides cros_sdk; clone chromite to ~/chromiumos/chromite)
#   - Running on Linux (ubuntu-latest or equivalent)
#
# Usage:
#   # Workspace smoke build (local / comprehensive.sh / make platform-test-crosh):
#   ./scripts/ci/build-crosh.sh
#
#   # Single-crate CI matrix build (chromeos.yml):
#   CRATE=larql-core CROS_TARGET=x86_64-cros-linux-gnu ./scripts/ci/build-crosh.sh
#
# Environment variables:
#   CRATE: Cargo package name (optional; if unset, builds the whole workspace)
#   CROS_TARGET: Rust target triple (optional; required when CRATE is set)
#   VERBOSE: Enable verbose output (set to 1)

set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}"

if [[ "${VERBOSE:-0}" == "1" ]]; then
  set -x
fi

# Validate: CRATE and CROS_TARGET must be set together or not at all.
if [[ -n "${CRATE:-}" && -z "${CROS_TARGET:-}" ]] || \
   [[ -z "${CRATE:-}" && -n "${CROS_TARGET:-}" ]]; then
  echo "Error: set both CRATE and CROS_TARGET, or neither" >&2
  exit 1
fi

# When running outside a full ChromiumOS source tree (e.g. on CI runners where
# only chromite is cloned), cros_sdk cannot auto-detect the SDK version.
# The workflow extracts SDK_VERSION from chromite/lib/constants.py and exports
# it as CHROMEOS_SDK_VERSION so we can pass it explicitly here.
SDK_VER_FLAG=()
if [[ -n "${CHROMEOS_SDK_VERSION:-}" ]]; then
  SDK_VER_FLAG=(--sdk-version "${CHROMEOS_SDK_VERSION}")
fi

# Ensure the cros_sdk chroot is bootstrapped (downloads SDK tarball if needed).
cros_sdk "${SDK_VER_FLAG[@]}" --download

# Build inside the chroot.
# --working-dir bind-mounts REPO_ROOT and cds into it inside the chroot.
if [[ -n "${CRATE:-}" ]]; then
  # Single-crate mode (CI matrix).
  cros_sdk "${SDK_VER_FLAG[@]}" --working-dir="$REPO_ROOT" -- \
    cargo build -p "$CRATE" --target "$CROS_TARGET"
else
  # Workspace smoke build (local use / comprehensive.sh).
  cros_sdk "${SDK_VER_FLAG[@]}" --working-dir="$REPO_ROOT" -- \
    cargo build --workspace
fi
