"""Peak RSS across one or more `/usr/bin/time -v` output files.

The strategy-matrix produce step runs several commands (e.g. extract → quantize).
The true peak RSS is the MAX 'Maximum resident set size' across ALL of them — not
just the last/wrapped one. Wrapping only the final command under-reports the peak
(a failed fast quantize hides a multi-GB extract), corrupting the resource audit.
"""
import re
import sys
from pathlib import Path

_RSS = re.compile(r"Maximum resident set size.*?:\s*(\d+)", re.I)


def parse_rss_kb(text):
    """The 'Maximum resident set size (kbytes)' from one /usr/bin/time -v output, or None."""
    m = _RSS.search(text or "")
    return int(m.group(1)) if m else None


def peak_rss_kb(paths):
    """Max RSS (kB) across the given time-output files; None if none parse.
    Missing/unparseable files are ignored (not an error)."""
    vals = []
    for p in paths:
        try:
            v = parse_rss_kb(Path(p).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            v = None
        if v is not None:
            vals.append(v)
    return max(vals) if vals else None


def main():
    peak = peak_rss_kb(sys.argv[1:])
    print(peak if peak is not None else "")


if __name__ == "__main__":
    main()
