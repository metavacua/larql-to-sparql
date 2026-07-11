import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import peak_rss as P

# A representative `/usr/bin/time -v` output block.
TIME_EXTRACT = """\
\tCommand being timed: "larql extract qwen15 -o out/x.vindex.src --level inference"
\tUser time (seconds): 700.12
\tMaximum resident set size (kbytes): 4400000
\tExit status: 0
"""
TIME_QUANTIZE_FAIL = """\
\tCommand being timed: "larql convert quantize fp4 --input out/x.vindex.src"
\tUser time (seconds): 0.10
\tMaximum resident set size (kbytes): 17808
\tExit status: 1
"""


def test_parse_rss_from_time_output():
    assert P.parse_rss_kb(TIME_EXTRACT) == 4400000
    assert P.parse_rss_kb(TIME_QUANTIZE_FAIL) == 17808


def test_parse_rss_none_when_absent():
    assert P.parse_rss_kb("no rss line here") is None
    assert P.parse_rss_kb("") is None
    assert P.parse_rss_kb(None) is None


def test_peak_is_max_across_files(tmp_path):
    # the fp4 case: extract command peaks high, the failed quantize peaks tiny —
    # the true peak is the extract's, not the last (quantize) command's.
    a = tmp_path / "t.src"
    b = tmp_path / "t"
    a.write_text(TIME_EXTRACT)
    b.write_text(TIME_QUANTIZE_FAIL)
    assert P.peak_rss_kb([str(a), str(b)]) == 4400000
    assert P.peak_rss_kb([str(b), str(a)]) == 4400000  # order-independent


def test_missing_and_unparseable_files_ignored(tmp_path):
    good = tmp_path / "g"
    good.write_text(TIME_EXTRACT)
    missing = tmp_path / "nope"
    empty = tmp_path / "e"
    empty.write_text("garbage")
    assert P.peak_rss_kb([str(missing), str(good), str(empty)]) == 4400000
    assert P.peak_rss_kb([str(missing)]) is None
    assert P.peak_rss_kb([]) is None
