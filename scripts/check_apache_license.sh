#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: Apache-2.0
#
# Apache-2.0 §4 mechanical gate.
#
# Enforces the file-state predicates that are deterministically checkable
# from the Apache License, Version 2.0:
#
#   §4(a)  A copy of the License is distributed with the Work.
#   §4(b)  Modified files carry a prominent modification notice.
#   §4(c)  Source forms retain copyright/license notices (delegated to
#          `reuse lint`; this script only spot-checks the identifier).
#   §4(d)  If the Work has a NOTICE file, redistributions include it.
#
# The license-identifier allow-list is currently {Apache-2.0}. Any other
# value is rejected.
#
# Usage:
#   scripts/check_apache_license.sh                  # strict, no §4(b)
#   scripts/check_apache_license.sh BASE..HEAD       # also runs §4(b)
#
# Exit codes:
#   0   all checked clauses satisfied
#   1   at least one clause violated (diagnostics printed)
#   2   toolchain/repository misconfiguration

set -euo pipefail

ALLOWED_LICENSE_IDS=("Apache-2.0")
range="${1:-}"
fail=0

# ---------------------------------------------------------------------------
# §4(a): canonical license text present and non-empty.
# ---------------------------------------------------------------------------
if [[ ! -s LICENSE ]]; then
  echo "::error file=LICENSE::Apache-2.0 §4(a): LICENSE file missing or empty" >&2
  fail=1
fi
if [[ ! -s LICENSES/Apache-2.0.txt ]]; then
  echo "::error file=LICENSES/Apache-2.0.txt::Apache-2.0 §4(a): canonical license text missing" >&2
  fail=1
fi

# ---------------------------------------------------------------------------
# §4(d): NOTICE file present and non-empty. Adopting a NOTICE file is a
# project policy decision; once present it must propagate.
# ---------------------------------------------------------------------------
if [[ ! -s NOTICE ]]; then
  echo "::error file=NOTICE::Apache-2.0 §4(d): NOTICE file missing or empty" >&2
  fail=1
fi

# ---------------------------------------------------------------------------
# §4 (license-identifier allow-list): every SPDX-License-Identifier
# declaration in the repo must be in ALLOWED_LICENSE_IDS.
# ---------------------------------------------------------------------------
allowed_re="$(IFS='|'; echo "${ALLOWED_LICENSE_IDS[*]}")"
# A line is an actual SPDX declaration (rather than a prose mention in
# documentation) iff, after the `path:lineno:` prefix, the only characters
# before `SPDX-License-Identifier:` are whitespace and standard
# single-line/block comment markers (#, //, <!--, *, ;, --). Markdown
# table cells and inline code fences contain `|` or backticks before the
# token and are correctly skipped.
declaration_re='^[^:]+:[0-9]+:[[:space:]]*([#;*<!/-]|//|<!--)?[[:space:]]*SPDX-License-Identifier:'
violations="$(
  grep -RIn 'SPDX-License-Identifier:' \
    --exclude-dir=.git \
    --exclude-dir=target \
    --exclude-dir=node_modules \
    --exclude-dir=LICENSES \
    --exclude=LICENSE \
    --exclude="check_apache_license.sh" \
    . \
  | grep -E "$declaration_re" \
  | grep -vE "SPDX-License-Identifier:[[:space:]]*(${allowed_re})([[:space:]]|$)" \
  || true
)"
if [[ -n "$violations" ]]; then
  echo "::error::Apache-2.0 license-identifier allow-list violation (allowed: ${ALLOWED_LICENSE_IDS[*]}):" >&2
  printf '%s\n' "$violations" >&2
  fail=1
fi

# ---------------------------------------------------------------------------
# §4(b): modification notices on PR-modified pre-existing files.
#
# Only runs when an explicit base..head range is supplied; on push to main
# we do not have a meaningful base to diff against.
# ---------------------------------------------------------------------------
if [[ -n "$range" ]]; then
  modified="$(git diff --diff-filter=M --name-only "$range" || true)"
  if [[ -z "$modified" ]]; then
    echo "check_apache_license: §4(b) vacuous (no pre-existing files modified)"
  else
    missing=()
    while IFS= read -r f; do
      [[ -f "$f" ]] || continue
      if grep -qE '(SPDX-FileContributor|Modifications:|Modified by:)' "$f"; then
        continue
      fi
      # REUSE.toml-driven attribution is acceptable: an explicit path entry
      # in REUSE.toml signals a deliberate provenance declaration.
      if grep -qF "\"$f\"" REUSE.toml; then
        continue
      fi
      missing+=("$f")
    done <<<"$modified"
    if (( ${#missing[@]} > 0 )); then
      echo "::error::Apache-2.0 §4(b): modified files lack a modification notice:" >&2
      printf '  %s\n' "${missing[@]}" >&2
      echo "Remediation: add an SPDX-FileContributor line, a 'Modified by:' comment," >&2
      echo "or an explicit REUSE.toml [[annotations]] block for each listed file." >&2
      fail=1
    fi
  fi
fi

if (( fail )); then
  exit 1
fi
echo "check_apache_license: OK"
