import json
from pathlib import Path

from scripts.target_analysis_common import error_sites, load_json, load_jsonl, unit_graph_units_named

FIXTURES = Path(__file__).parent / "fixtures" / "target_analysis"


def test_load_json_reads_a_real_file():
    data = load_json(FIXTURES / "unit_graph_serde_default.json")
    assert data["version"] == 1


def test_unit_graph_units_named_finds_serde_only():
    unit_graph = load_json(FIXTURES / "unit_graph_serde_default.json")
    units = unit_graph_units_named(unit_graph, "serde")
    assert len(units) == 1
    assert units[0]["features"] == ["default", "std", "derive"]


def test_unit_graph_units_named_returns_empty_for_absent_crate():
    unit_graph = load_json(FIXTURES / "unit_graph_serde_default.json")
    assert unit_graph_units_named(unit_graph, "not-a-real-crate") == []


def test_error_sites_extracts_only_error_level_primary_spans():
    messages = load_json(FIXTURES / "compiler_messages_baseline.json")
    sites = error_sites(messages)
    assert sites == {
        ("crates/larql-boundary/src/lib.rs", 12, "E0433"),
        ("crates/larql-boundary/src/lib.rs", 47, "E0433"),
    }


def test_load_jsonl_reads_cargos_real_line_delimited_wire_format():
    # cargo's --message-format=json writes one JSON object per line, not a
    # single JSON array (confirmed elsewhere in this pipeline's own workflow
    # comments) -- load_json() cannot parse this; load_jsonl() must.
    messages = load_jsonl(FIXTURES / "compiler_messages_baseline.jsonl")
    assert len(messages) == 3
    sites = error_sites(messages)
    assert sites == {
        ("crates/larql-boundary/src/lib.rs", 12, "E0433"),
        ("crates/larql-boundary/src/lib.rs", 47, "E0433"),
    }


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "with-blanks.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n\n', encoding="utf-8")
    assert load_jsonl(path) == [{"a": 1}, {"a": 2}]
