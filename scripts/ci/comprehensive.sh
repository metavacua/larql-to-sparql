#!/usr/bin/env bash
# SPDX-FileCopyrightText: Contributors to the larql-to-sparql project
# SPDX-License-Identifier: Apache-2.0
#
# Extra-platforms CI orchestrator (Android, Crostini, crosh).
#
# Linux, macOS, and Windows are covered by upstream's per-crate workflows
# (.github/workflows/larql-*.yml). This script drives only the extra
# platforms unique to the fork.
#
# Usage:
#   ./scripts/ci/comprehensive.sh              # Auto-detect and run
#   PLATFORM=crostini ./scripts/ci/comprehensive.sh  # Force platform
#
# Supported platforms: android, crostini, crosh
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

# Detect platform if not explicitly set. Only ChromeOS and Android are
# in scope here; everything else is delegated to upstream's per-crate CI.
detect_platform() {
  local os_name
  os_name="$(uname -s)"

  case "${os_name}" in
    Linux)
      if [[ -f /etc/lsb-release ]] && grep -qi "CHROMEOS" /etc/lsb-release 2>/dev/null; then
        echo "crostini"
      elif [[ -f /system/build.prop ]] 2>/dev/null; then
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
    android|crostini|crosh)
      ;;
    unsupported-linux)
      echo -e "${YELLOW}Skipping: generic Linux is covered by upstream's per-crate workflows.${NC}" >&2
      exit 0
      ;;
    *)
      echo -e "${RED}x Unsupported platform: ${platform}${NC}" >&2
      echo "Supported: android, crostini, crosh. Linux/macOS/Windows are covered by"
      echo "upstream's per-crate workflows (.github/workflows/larql-*.yml)."
      exit 1
      ;;
  esac

  cd "${REPO_ROOT}"
  run_platform_script "${platform}"
}

main "$@"
