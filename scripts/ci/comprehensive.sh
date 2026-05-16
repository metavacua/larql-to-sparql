#!/usr/bin/env bash
# SPDX-FileCopyrightText: Contributors to the larql-to-sparql project
# SPDX-License-Identifier: Apache-2.0
#
# Extra-platforms CI orchestrator (Android).
#
# Linux, macOS, and Windows are covered by upstream's per-crate workflows
# (.github/workflows/larql-*.yml). This script drives the Android platform.
#
# Usage:
#   ./scripts/ci/comprehensive.sh          # Auto-detect and run
#   PLATFORM=android ./scripts/ci/comprehensive.sh  # Force platform
#
# Supported platforms: android
# Environment variables:
#   PLATFORM: Override auto-detected platform (optional)
#   VERBOSE: Enable verbose output (set to 1)

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

# Detect platform if not explicitly set. Only Android is in scope here;
# everything else is delegated to upstream's per-crate CI.
detect_platform() {
  local os_name
  os_name="$(uname -s)"

  case "${os_name}" in
    Linux)
      if [[ -f /system/build.prop ]] 2>/dev/null; then
        echo "android"
      else
        echo "unsupported-linux"
      fi
      ;;
    *)
      echo "unsupported" >&2
      return 1
      ;;
  esac
}

run_platform_script() {
  local platform="$1"
  local script="${SCRIPT_DIR}/build-${platform}.sh"

  if [[ ! -f "${script}" ]]; then
    echo -e "${RED}x Platform script not found: ${script}${NC}" >&2
    exit 1
  fi

  echo -e "${YELLOW}-> Running: ${platform}${NC}"

  if bash "${script}"; then
    echo -e "${GREEN}ok ${platform} build and test passed${NC}"
    return 0
  else
    echo -e "${RED}x ${platform} build and test failed${NC}"
    return 1
  fi
}

main() {
  local platform="${PLATFORM:-}"

  if [[ -z "${platform}" ]]; then
    platform="$(detect_platform)"
  fi

  if [[ "${VERBOSE:-0}" == "1" ]]; then
    echo "Platform detected: ${platform}"
    echo "Repository root: ${REPO_ROOT}"
  fi

  case "${platform}" in
    android)
      ;;
    unsupported-linux)
      echo -e "${YELLOW}Skipping: generic Linux is covered by upstream's per-crate workflows.${NC}" >&2
      exit 0
      ;;
    *)
      echo -e "${RED}x Unsupported platform: ${platform}${NC}" >&2
      echo "Supported: android. Linux/macOS/Windows are covered by"
      echo "upstream's per-crate workflows (.github/workflows/larql-*.yml)."
      exit 1
      ;;
  esac

  cd "${REPO_ROOT}"
  run_platform_script "${platform}"
}

main "$@"
