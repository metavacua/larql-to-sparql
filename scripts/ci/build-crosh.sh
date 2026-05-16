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
#   - depot_tools is on PATH (cros_sdk command available)
#   - Running on Linux (ubuntu-latest or equivalent)
#
# Usage (via chromeos.yml workflow):
#   CRATE=larql-core CROS_TARGET=x86_64-cros-linux-gnu ./scripts/ci/build-crosh.sh
#
# Environment variables:
#   CRATE: Cargo package name to build (required)
#   CROS_TARGET: Rust target triple to compile for (required)
#   VERBOSE: Enable verbose output (set to 1)

set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}"

if [[ -z "${CRATE:-}" ]]; then
  echo "Error: CRATE environment variable is required" >&2
  exit 1
fi

if [[ -z "${CROS_TARGET:-}" ]]; then
  echo "Error: CROS_TARGET environment variable is required" >&2
  exit 1
fi

if [[ "${VERBOSE:-0}" == "1" ]]; then
  set -x
fi

# Ensure the cros_sdk chroot is bootstrapped (downloads SDK tarball if needed).
cros_sdk --download

# Build the crate inside the chroot.
# --working-dir bind-mounts REPO_ROOT and cds into it inside the chroot.
cros_sdk --working-dir="$REPO_ROOT" -- \
  cargo build -p "$CRATE" --target "$CROS_TARGET"
