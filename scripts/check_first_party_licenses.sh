#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: Apache-2.0
#
# First-party license gate.
#
# Replaces the original Apache-2.0-only `scripts/check_apache_license.sh`.
# The audit (audit/first-party-report.md, audit/dependency-licenses.md)
# established that the project uses more than one license:
#
#   * Apache-2.0                — pre-existing upstream code and the
#                                 compliance toolchain.
#   * AGPL-3.0-or-later         — forward code contributions per the fork's
#                                 documented licensing posture.
#   * CC-BY-SA-4.0              — forward documentation contributions
#                                 (including everything under audit/).
#
# Rather than duplicate this allow-list inside the script, the script reads
# it directly from REUSE.toml. REUSE.toml is the authoritative manifest;
# any SPDX-License-Identifier value that appears in any [[annotations]]
# block is admissible. This eliminates the drift risk that bit the original
# script (it enforced Apache-2.0-only despite the audit revealing otherwise).
#
# Apache-2.0 §4 mechanical predicates the script still enforces verbatim:
#
#   §4(a)  A copy of the License is distributed with the Work.
#   §4(b)  Modified files carry a prominent modification notice.
#   §4(c)  Source forms retain copyright/license notices (delegated to
#          `reuse lint`; this script only spot-checks the identifier).
#   §4(d)  If the Work has a NOTICE file, redistributions include it.
#
# Apache-2.0 obligations apply only to portions of the work that are
# actually licensed under Apache-2.0; AGPL-3.0-or-later and CC-BY-SA-4.0
# files are not subject to §4(a)/(b)/(d). LICENSE/NOTICE existence is
# enforced unconditionally because the upstream-derived portion of the
# tree is Apache-2.0 and §4 still applies to it.
#
# Usage:
#   scripts/check_first_party_licenses.sh                  # strict, no §4(b)
#   scripts/check_first_party_licenses.sh BASE..HEAD       # also runs §4(b)
#
# Exit codes:
#   0   all checked clauses satisfied
#   1   at least one clause violated (diagnostics printed)
#   2   toolchain/repository misconfiguration

set -euo pipefail

range="${1:-}"
fail=0

# ---------------------------------------------------------------------------
# Read the allow-list from REUSE.toml. Every value of `SPDX-License-Identifier`
# appearing in an [[annotations]] block is admissible. This makes REUSE.toml
# the single source of truth.
# ---------------------------------------------------------------------------
if [[ ! -f REUSE.toml ]]; then
  echo "::error file=REUSE.toml::REUSE.toml missing; cannot derive allow-list" >&2
  exit 2
fi
# REUSE-IgnoreStart
mapfile -t ALLOWED_LICENSE_IDS < <(
  grep -E '^SPDX-License-Identifier[[:space:]]*=' REUSE.toml \
    | sed -E 's/^SPDX-License-Identifier[[:space:]]*=[[:space:]]*"([^"]+)"[[:space:]]*$/\1/' \
    | sort -u
)
# REUSE-IgnoreEnd
if (( ${#ALLOWED_LICENSE_IDS[@]} == 0 )); then
  echo "::error file=REUSE.toml::no SPDX-License-Identifier values found" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# §4(a): canonical Apache-2.0 license text present and non-empty.
# Enforced unconditionally: the upstream-derived portion of the tree is
# Apache-2.0, and §4(a) requires its license text to be distributed.
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
# §4(d): NOTICE file present and non-empty.
# ---------------------------------------------------------------------------
if [[ ! -s NOTICE ]]; then
  echo "::error file=NOTICE::Apache-2.0 §4(d): NOTICE file missing or empty" >&2
  fail=1
fi

# ---------------------------------------------------------------------------
# REUSE 3.x layout: every SPDX-License-Identifier value referenced in
# REUSE.toml must have a corresponding text under LICENSES/.
# ---------------------------------------------------------------------------
for id in "${ALLOWED_LICENSE_IDS[@]}"; do
  [[ "$id" == "NONE" ]] && continue
  if [[ ! -s "LICENSES/${id}.txt" ]]; then
    echo "::error file=LICENSES/${id}.txt::REUSE 3.x: canonical license text missing for ${id}" >&2
    fail=1
  fi
done

# ---------------------------------------------------------------------------
# License-identifier allow-list: every SPDX-License-Identifier declaration
# in the repo must be in ALLOWED_LICENSE_IDS (i.e., must also appear in
# REUSE.toml so the manifest covers it deterministically).
# ---------------------------------------------------------------------------
allowed_re="$(IFS='|'; echo "${ALLOWED_LICENSE_IDS[*]}")"
# A line is an actual SPDX declaration (rather than a prose mention in
# documentation) iff, after the `path:lineno:` prefix, the only characters
# # REUSE-IgnoreStart
# before SPDX-License-Identifier are whitespace and standard
# # REUSE-IgnoreEnd
# single-line/block comment markers (#, //, <!--, *, ;, --). Markdown
# table cells and inline code fences contain `|` or backticks before the
# token and are correctly skipped.
# REUSE-IgnoreStart
declaration_re='^[^:]+:[0-9]+:[[:space:]]*([#;*<!/-]|//|<!--)?[[:space:]]*SPDX-License-Identifier:'
# Pre-process every candidate file with awk to blank out lines between
# REUSE-IgnoreStart and REUSE-IgnoreEnd markers (REUSE 3.x mechanism for
# excluding illustrative SPDX text from compliance checks). We then run
# grep over the in-memory virtual contents so file:line numbers remain
# accurate to the original tree.
violations="$(
  while IFS= read -r f; do
    awk -v fname="$f" '
      BEGIN { skip = 0 }
      /REUSE-IgnoreStart/ { skip = 1 }
      { if (skip) print fname ":" NR ":"; else print fname ":" NR ":" $0 }
      /REUSE-IgnoreEnd/   { skip = 0 }
    ' "$f"
  done < <(
    grep -RIl 'SPDX-License-Identifier:' \
      --exclude-dir=.git \
      --exclude-dir=target \
      --exclude-dir=node_modules \
      --exclude-dir=LICENSES \
      --exclude-dir=audit \
      --exclude=LICENSE \
      --exclude="check_first_party_licenses.sh" \
      --exclude="check_apache_license.sh" \
      .
  ) \
  | grep -E "$declaration_re" \
  | grep -vE "SPDX-License-Identifier:[[:space:]]*(${allowed_re})([[:space:]]|$)" \
  || true
)"
# REUSE-IgnoreEnd
if [[ -n "$violations" ]]; then
  echo "::error::license-identifier allow-list violation (allowed: ${ALLOWED_LICENSE_IDS[*]}):" >&2
  printf '%s\n' "$violations" >&2
  echo "Remediation: add the license to REUSE.toml as an [[annotations]] block, or change the file's SPDX-License-Identifier." >&2
  fail=1
fi

# ---------------------------------------------------------------------------
# §4(b): modification notices on PR-modified pre-existing files.
# Applies only to Apache-2.0-licensed files; CC-BY-SA-4.0 and AGPL-3.0
# files are not subject to Apache §4(b).
#
# Only runs when an explicit base..head range is supplied; on push to main
# we do not have a meaningful base to diff against.
# ---------------------------------------------------------------------------
if [[ -n "$range" ]]; then
  modified="$(git diff --diff-filter=M --name-only "$range" || true)"
  if [[ -z "$modified" ]]; then
    echo "check_first_party_licenses: §4(b) vacuous (no pre-existing files modified)"
  else
    missing=()
    while IFS= read -r f; do
      [[ -f "$f" ]] || continue
      # Skip non-Apache-2.0 files: §4(b) does not apply to them.
      # Best effort: read the file's declared SPDX-License-Identifier; if
      # it's not Apache-2.0, the §4(b) modification-notice obligation is
      # vacuous for that file.
      file_id="$(grep -m1 -oE 'SPDX-License-Identifier:[[:space:]]*[A-Za-z0-9.+_-]+' "$f" 2>/dev/null \
        | sed -E 's/^SPDX-License-Identifier:[[:space:]]*//')" || true
      if [[ -n "$file_id" && "$file_id" != "Apache-2.0" ]]; then
        continue
      fi
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
echo "check_first_party_licenses: OK (allow-list: ${ALLOWED_LICENSE_IDS[*]})"
