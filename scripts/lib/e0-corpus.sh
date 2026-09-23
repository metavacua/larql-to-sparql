# E0 corpus definition — the rows, the prompts, and the normalisation.
#
# Sourced by both `e0-capture-goldens.sh` (writes the record) and
# `e0-verify-goldens.sh` (replays it against a current binary).
#
# WHY THIS IS SHARED RATHER THAN DUPLICATED
#
# The capture script's own preamble warns that comparing two builds to each
# other is close to circular. The same hazard applies one level up: if capture
# and verify each carried their own copy of the prompts, the row list or the
# normalisation, they would drift, and a passing E0-FULL would mean "the two
# scripts agree" rather than "behaviour is unchanged". A prompt reworded on one
# side alone turns every downstream row into a false diff; a normalisation rule
# added on one side alone silently hides a real one.
#
# So the corpus is defined once, here, and neither script may redefine it.

# Fixed prompt set. Short, deterministic, and spanning factual recall, code and
# multi-token continuation so a regression in any one does not hide in the
# others.
readonly E0_PROMPTS=(
  "The capital of France is"
  "def fibonacci(n):"
  "Water boils at"
)

readonly E0_SLICE_PRESETS=(client attn embed server browse router expert-server all)

# Strip anything that varies between runs or machines but is not behaviour:
# absolute paths, wall-clock durations, throughput rates, ISO timestamps.
#
# Takes the vindex path and scratch dir as arguments rather than reading
# globals, so a caller cannot accidentally normalise against the wrong paths.
#
# # No `\b`
#
# The previous timing rule ended in `\b`, which is a GNU extension. BSD
# `sed -E` does not support it and does not complain — it just never matches.
# So on macOS every duration passed through unstripped, and each replay
# differed from its golden on the timing line alone. A normaliser that
# silently does nothing is worse than no normaliser: it produces confident
# diffs that read exactly like the regression under test.
#
# The portable form matches the unit followed by a non-alphanumeric character
# or end-of-line, and puts that character back via the capture group. `ms`
# precedes bare `s` so the longer unit wins, and `MB`/`GB` on their own are
# deliberately absent — those are file sizes, which are behaviour and must
# survive normalisation.
e0_normalise() {
  local vindex="$1" tmp="$2"
  sed -E \
    -e "s#${vindex}#<VINDEX>#g" \
    -e "s#${tmp}#<TMP>#g" \
    -e 's#/[^ ]*/(larql|target)[^ ]*#<BIN>#g' \
    -e 's/[0-9]+(\.[0-9]+)? *(ms|us|ns|tok\/s|MB\/s|GB\/s|s)([^A-Za-z0-9]|$)/<TIME>\3/g' \
    -e 's/[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z?/<TS>/g'
}

# Emit `<row-name>\t<command...>` for every row in the corpus, NUL-free and
# one per line. Both scripts iterate this, so the row set cannot diverge.
#
# Arguments: binary, vindex, scratch dir, token count, walk top-K.
e0_rows() {
  local bin="$1" vindex="$2" tmp="$3" tokens="$4" walk_k="$5"
  printf '%s\n' "verify|${bin}|verify|${vindex}"
  printf '%s\n' "show|${bin}|show|${vindex}"

  local i
  for i in "${!E0_PROMPTS[@]}"; do
    printf '%s\n' "walk_p${i}|${bin}|walk|--index|${vindex}|--prompt|${E0_PROMPTS[$i]}|--top-k|${walk_k}"
  done
  for i in "${!E0_PROMPTS[@]}"; do
    printf '%s\n' "decode_p${i}|${bin}|run|${vindex}|${E0_PROMPTS[$i]}|-n|${tokens}"
  done
  local preset
  for preset in "${E0_SLICE_PRESETS[@]}"; do
    printf '%s\n' "slice_${preset}|${bin}|slice|${vindex}|-o|${tmp}/${preset}|--preset|${preset}|--dry-run"
  done
}

# Run one row and print its normalised output plus exit status.
#
# The exit status is part of the record on purpose: E0 must notice if a path
# that used to fail starts succeeding, or vice versa.
# Rows whose output is a *set* of findings rather than a sequence.
#
# `verify` is one: it reports a status per file, and which line comes first
# carries no meaning. The incumbent binary rendered them in `HashMap` order, so
# any golden captured from it pins an arbitrary permutation that no other run
# reproduces. Sorting such rows before comparison makes the row test what it is
# actually about — which files, what status, what size — and leaves ordering to
# be policed where it belongs, in `checksums.rs`'s unit tests.
#
# WALK rows are emphatically NOT in this set: their whole content is a ranking,
# and sorting one would discard the result being checked.
e0_row_is_unordered() {
  case "$1" in
    verify) return 0 ;;
    *) return 1 ;;
  esac
}

e0_run_row() {
  local name="$1" vindex="$2" tmp="$3"; shift 3
  # Not `status`: that name is read-only in zsh, so sourcing this file into an
  # interactive zsh to debug a row fails with an error that looks like a script
  # bug rather than a shell collision.
  local out exit_code
  out="$("$@" 2>&1)"; exit_code=$?
  if e0_row_is_unordered "$name"; then
    printf '%s\n' "$out" | e0_normalise "$vindex" "$tmp" | sort
  else
    printf '%s\n' "$out" | e0_normalise "$vindex" "$tmp"
  fi
  # Always last, so a sorted row still ends with its status.
  echo "exit_status=${exit_code}"
}

# The index.json fields that decide generation dispatch (spec §12.1). Pinned
# explicitly so a stray version bump is a diff rather than a load-time surprise.
e0_index_version() {
  python3 - "$1/index.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("version", "vindex_spec_version", "extract_level", "quant_format"):
    print(f"{k}={d.get(k)}")
PY
}
