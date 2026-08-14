#!/usr/bin/env bash
# Recursive filesystem index, so "what did this command touch?" is DATA rather
# than an inference about which paths were worth listing.
#
# Why this exists: on a hosted runner there is no interactive shell. Every
# earlier attempt to attribute a file operation guessed a path, listed it, and
# reasoned from the absence of anything there — which cannot distinguish "wrote
# nothing" from "wrote somewhere I did not look". COMPILE INTO MODEL builds
# into a staging directory named `<basename>.tmp.<pid>` beside its target and
# renames it into place; listing only the target sees an empty directory and
# concludes, wrongly, that nothing was written.
#
# Snapshots are whole subtrees, taken before and after, and diffed. A path that
# appears, disappears, or changes size shows up whether or not anyone predicted
# it.
#
#   fs-snapshot.sh snap <out-file> <root>...     record one subtree index
#   fs-snapshot.sh diff <before> <after>         report +added -removed ~changed
#
# Format is `type<TAB>size<TAB>path`, sorted by path, so `diff`, `comm` and
# `join` all work on it directly and no parser is needed to read it.
set -uo pipefail

case "${1:-}" in
  snap)
    out="$2"; shift 2
    : > "$out"
    for root in "$@"; do
      if [ -e "$root" ]; then
        # -xdev: do not cross into other filesystems (a runner mounts several).
        # Symlinks are recorded as links, never followed — following them turns
        # one HF blob into dozens of duplicate paths and hides the indirection.
        find "$root" -xdev \
          \( -type f -printf 'f\t%s\t%p\n' \
          -o -type d -printf 'd\t0\t%p\n' \
          -o -type l -printf 'l\t0\t%p -> %l\n' \) 2>>"$out.err"
      else
        echo -e "?\t0\t$root (does not exist)"
      fi
    done | LC_ALL=C sort -t"$(printf '\t')" -k3 >> "$out"
    wc -l < "$out"
    ;;
  diff)
    before="$2"; after="$3"
    # comm needs both sides sorted the same way; they are, by construction.
    echo "=== ADDED (present after, absent before) ==="
    LC_ALL=C comm -13 <(cut -f3 "$before" | LC_ALL=C sort) <(cut -f3 "$after" | LC_ALL=C sort)
    echo "=== REMOVED (present before, absent after) ==="
    LC_ALL=C comm -23 <(cut -f3 "$before" | LC_ALL=C sort) <(cut -f3 "$after" | LC_ALL=C sort)
    echo "=== SIZE CHANGED ==="
    LC_ALL=C join -t"$(printf '\t')" -j 3 -o 0,1.2,2.2 \
      <(LC_ALL=C sort -t"$(printf '\t')" -k3 "$before") \
      <(LC_ALL=C sort -t"$(printf '\t')" -k3 "$after") \
      | awk -F'\t' '$2 != $3 { print $1 "\t" $2 " -> " $3 }'
    ;;
  *)
    echo "usage: fs-snapshot.sh snap <out> <root>... | diff <before> <after>" >&2
    exit 2
    ;;
esac
