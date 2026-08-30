//! Integration coverage for the LQL lexer and parser helper/lifecycle paths.
//!
//! These exercise the PURE parsing surface only — no vindex/model needed.
//! Everything is driven through the public `larql_lql::parser::parse` entry
//! point (the lexer is `pub(crate)`, so it is reached transitively).
//!
//! Goals (≥90% line coverage):
//!   * lexer.rs — every keyword's `as_field_name` arm (150-252), the float
//!     fractional-digit loop (588), and `LexError::Display` (637-639).
//!   * parser/helpers.rs — optional-clause and error branches.
//!   * parser/lifecycle.rs — USE / COMPILE / DIFF / COMPACT variants + errors.

use larql_lql::ast::Statement;
use larql_lql::parser::parse;

// ── small assertion helpers ──

fn ok(sql: &str) -> Statement {
    match parse(sql) {
        Ok(stmt) => stmt,
        Err(e) => panic!("expected Ok parsing {sql:?}, got Err: {e}"),
    }
}

fn err(sql: &str) {
    let r = parse(sql);
    assert!(
        r.is_err(),
        "expected Err parsing {sql:?}, got Ok: {:?}",
        r.ok()
    );
}

// ────────────────────────────────────────────────────────────────────────
// lexer.rs 150-252 — `Keyword::as_field_name` for EVERY keyword variant.
//
// `parse_field` (helpers.rs 203-207) maps `Token::Keyword(kw)` to
// `kw.as_field_name()`. Driving `SELECT <kw> FROM EDGES` therefore exercises
// one `as_field_name` match arm per keyword. parse_field reads a single field
// (no trailing comma) then the FROM clause consumes the next `FROM`, so even
// keywords like FROM/SELECT/EDGES work as a single field name here.
// ────────────────────────────────────────────────────────────────────────

/// Every keyword as it is spelled for the lexer's case-insensitive matcher.
/// Order/spelling mirrors `Keyword::from_str` in lexer.rs.
const ALL_KEYWORDS: &[&str] = &[
    "EXTRACT",
    "COMPILE",
    "DIFF",
    "USE",
    "WALK",
    "SELECT",
    "DESCRIBE",
    "EXPLAIN",
    "INSERT",
    "DELETE",
    "UPDATE",
    "MERGE",
    "SHOW",
    "STATS",
    "FROM",
    "INTO",
    "WHERE",
    "AND",
    "OR",
    "NOT",
    "IN",
    "LIKE",
    "BETWEEN",
    "ORDER",
    "BY",
    "ASC",
    "DESC",
    "LIMIT",
    "TOP",
    "LAYERS",
    "MODE",
    "COMPARE",
    "AT",
    "LAYER",
    "CONFIDENCE",
    "MODEL",
    "EDGES",
    "RELATION",
    "RELATIONS",
    "ENTITIES",
    "FEATURES",
    "MODELS",
    "FORMAT",
    "COMPONENTS",
    "ON",
    "CONFLICT",
    "KEEP_SOURCE",
    "KEEP_TARGET",
    "HIGHEST_CONFIDENCE",
    "LAST_WINS",
    "FAIL",
    "FOR",
    "SET",
    "VALUES",
    "CURRENT",
    "WITH",
    "EXAMPLES",
    "ONLY",
    "VERBOSE",
    "RANGE",
    "ALL",
    "NEAREST",
    "TO",
    "PURE",
    "HYBRID",
    "DENSE",
    "SAFETENSORS",
    "GGUF",
    "AUTO_EXTRACT",
    "FFN_GATE",
    "FFN_DOWN",
    "FFN_UP",
    "EMBEDDINGS",
    "ATTN_OV",
    "ATTN_QK",
    "INFER",
    "SYNTAX",
    "KNOWLEDGE",
    "OUTPUT",
    "WEIGHTS",
    "INFERENCE",
    "BEGIN",
    "SAVE",
    "APPLY",
    "REMOVE",
    "PATCH",
    "PATCHES",
    "REMOTE",
    "TRACE",
    "DECOMPOSE",
    "POSITIONS",
    "BRIEF",
    "RAW",
    "ATTENTION",
    "ALPHA",
    "KNN",
    "COMPOSE",
    "REBALANCE",
    "FLOOR",
    "CEILING",
    "MAX",
    "UNTIL",
    "CONVERGED",
    "COMPACT",
    "STATUS",
];

#[test]
fn every_keyword_usable_as_select_field_name() {
    // Hits lexer.rs `as_field_name` for all 100 keyword variants, plus the
    // `Token::Keyword(kw)` arm of helpers.rs::parse_field.
    for kw in ALL_KEYWORDS {
        let sql = format!("SELECT {kw} FROM EDGES;");
        let stmt = ok(&sql);
        match stmt {
            Statement::Select { fields, .. } => {
                assert_eq!(fields.len(), 1, "field count for {kw}");
            }
            other => panic!("expected Select for {kw}, got {other:?}"),
        }
    }
}

#[test]
fn keyword_field_name_in_where_and_order_by() {
    // expect_field_name (helpers.rs 476-481 keyword arm) via WHERE + ORDER BY.
    // Uses keywords that collide with real column names.
    let stmt = ok("SELECT * FROM EDGES WHERE layer = 5 ORDER BY confidence DESC;");
    match stmt {
        Statement::Select {
            conditions, order, ..
        } => {
            assert_eq!(conditions.len(), 1);
            assert_eq!(conditions[0].field, "layer");
            let o = order.expect("order present");
            assert_eq!(o.field, "confidence");
            assert!(o.descending);
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn keyword_field_name_in_update_set() {
    // expect_field_name keyword arm via UPDATE ... SET <kw> = ...
    let stmt = ok(r#"UPDATE EDGES SET confidence = 0.9 WHERE relation = "x";"#);
    match stmt {
        Statement::Update { set, .. } => {
            assert_eq!(set[0].field, "confidence");
        }
        other => panic!("got {other:?}"),
    }
}

// ────────────────────────────────────────────────────────────────────────
// lexer.rs — numeric literals
// ────────────────────────────────────────────────────────────────────────

#[test]
fn multi_digit_float_fraction_loop() {
    // CONFIDENCE <float> → NumberLit. A multi-digit fraction drives the inner
    // fractional-digit while-loop in lexer.rs::read_number (line ~588).
    let stmt = ok(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c") CONFIDENCE 0.8755;"#,
    );
    match stmt {
        Statement::Insert { confidence, .. } => {
            let c = confidence.expect("confidence present");
            assert!((c - 0.8755).abs() < 1e-4, "got {c}");
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn integer_then_dot_not_a_float() {
    // A digit run followed by '.' where the dot is NOT followed by a digit:
    // read_number must emit an IntegerLit and leave the Dot. `5.` then EOF —
    // the trailing Dot is a trailing token → parse error, but it confirms the
    // integer (not float) branch was taken without panicking.
    err("SELECT * FROM EDGES LIMIT 5.;");
}

#[test]
fn negative_integer_value_in_condition() {
    // parse_value Dash → IntegerLit branch (helpers.rs 295-298).
    let stmt = ok("SELECT * FROM EDGES WHERE layer = -3;");
    match stmt {
        Statement::Select { conditions, .. } => {
            assert!(matches!(
                conditions[0].value,
                larql_lql::ast::Value::Integer(-3)
            ));
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn negative_float_value_in_condition() {
    // parse_value Dash → NumberLit branch (helpers.rs 291-293).
    let stmt = ok("SELECT * FROM EDGES WHERE confidence = -1.5;");
    match stmt {
        Statement::Select { conditions, .. } => match conditions[0].value {
            larql_lql::ast::Value::Number(n) => assert!((n + 1.5).abs() < 1e-6),
            ref other => panic!("got {other:?}"),
        },
        other => panic!("got {other:?}"),
    }
}

#[test]
fn dash_not_followed_by_number_is_error() {
    // parse_value Dash → neither Number nor Integer (helpers.rs 299-302).
    err(r#"SELECT * FROM EDGES WHERE entity = -"oops";"#);
}

// ────────────────────────────────────────────────────────────────────────
// lexer.rs — strings, escapes, operators, comments, error chars
// ────────────────────────────────────────────────────────────────────────

#[test]
fn string_escapes_decode_through_parser() {
    let stmt = ok(r#"WALK "line1\nline2\ttab\r\\back\0null \"q\" 'x'";"#);
    match stmt {
        Statement::Walk { prompt, .. } => {
            assert!(prompt.contains('\n'));
            assert!(prompt.contains('\t'));
            assert!(prompt.contains('\\'));
            assert!(prompt.contains('\0'));
            assert!(prompt.contains('"'));
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn single_quoted_string_with_escaped_apostrophe() {
    let stmt = ok(r"WALK 'it\'s fine';");
    match stmt {
        Statement::Walk { prompt, .. } => assert_eq!(prompt, "it's fine"),
        other => panic!("got {other:?}"),
    }
}

#[test]
fn unknown_escape_passes_through_unchanged() {
    // `\q` → `q` (the `other => other as char` arm in read_quoted).
    let stmt = ok(r#"WALK "a\qb";"#);
    match stmt {
        Statement::Walk { prompt, .. } => assert_eq!(prompt, "aqb"),
        other => panic!("got {other:?}"),
    }
}

#[test]
fn unterminated_string_is_error() {
    err(r#"WALK "no closing quote"#);
}

#[test]
fn unterminated_after_backslash_is_error() {
    err(r#"WALK "abc\"#);
}

#[test]
fn line_comments_are_skipped() {
    let stmt = ok("-- a comment\nSTATS; -- trailing comment");
    assert!(matches!(stmt, Statement::Stats { .. }));
}

#[test]
fn pipe_operator_chains_statements() {
    let stmt = ok(r#"WALK "x" TOP 3 |> SELECT * FROM EDGES;"#);
    assert!(matches!(stmt, Statement::Pipe { .. }));
}

#[test]
fn incomplete_pipe_is_error() {
    // '|' not followed by '>' (lexer LexError branch).
    err(r#"WALK "x" | SELECT * FROM EDGES;"#);
}

#[test]
fn bang_without_eq_is_error() {
    // '!' not followed by '=' (lexer LexError branch).
    err("SELECT * FROM EDGES WHERE a ! b;");
}

#[test]
fn unexpected_character_is_error() {
    // '@' is not a valid token start.
    err("SELECT @ FROM EDGES;");
}

#[test]
fn all_comparison_operators_parse() {
    // Eq / Neq / Gt / Lt / Gte / Lte plus LIKE and IN exercise parse_compare_op.
    ok("SELECT * FROM EDGES WHERE a = 1;");
    ok("SELECT * FROM EDGES WHERE a != 1;");
    ok("SELECT * FROM EDGES WHERE a > 1;");
    ok("SELECT * FROM EDGES WHERE a < 1;");
    ok("SELECT * FROM EDGES WHERE a >= 1;");
    ok("SELECT * FROM EDGES WHERE a <= 1;");
    ok(r#"SELECT * FROM EDGES WHERE a LIKE "%x%";"#);
    ok("SELECT * FROM EDGES WHERE a IN (1, 2, 3);");
}

#[test]
fn lex_error_display_is_rendered() {
    // lexer.rs 637-639: LexError::Display. `parse` surfaces lex errors as a
    // boxed std::error::Error; formatting it routes through Display.
    let e = parse(r#"WALK "unterminated"#).unwrap_err();
    let msg = format!("{e}");
    assert!(
        msg.contains("Lex error"),
        "expected lexer Display prefix, got {msg:?}"
    );
}

// ────────────────────────────────────────────────────────────────────────
// parser/helpers.rs — value list, field list, compare-op + token errors
// ────────────────────────────────────────────────────────────────────────

#[test]
fn value_list_with_multiple_items() {
    // parse_value LParen branch + comma loop (helpers.rs 305-316, line 314).
    let stmt = ok(r#"SELECT * FROM EDGES WHERE relation IN ("a", "b", "c");"#);
    match stmt {
        Statement::Select { conditions, .. } => match &conditions[0].value {
            larql_lql::ast::Value::List(items) => assert_eq!(items.len(), 3),
            other => panic!("got {other:?}"),
        },
        other => panic!("got {other:?}"),
    }
}

#[test]
fn empty_value_list() {
    // parse_value LParen with immediate RParen (skips the item loop entirely).
    let stmt = ok("SELECT * FROM EDGES WHERE relation IN ();");
    match stmt {
        Statement::Select { conditions, .. } => match &conditions[0].value {
            larql_lql::ast::Value::List(items) => assert!(items.is_empty()),
            other => panic!("got {other:?}"),
        },
        other => panic!("got {other:?}"),
    }
}

#[test]
fn value_not_a_value_is_error() {
    // parse_value fallthrough error (helpers.rs 318): a bare keyword in value
    // position (FROM) is not a value.
    err("SELECT * FROM EDGES WHERE a = FROM;");
}

#[test]
fn star_in_middle_of_field_list() {
    // parse_field Star arm (helpers.rs 193-196): reached via a `*` AFTER a
    // comma, so parse_field_list's leading-star shortcut does not apply.
    let stmt = ok("SELECT entity, * FROM EDGES;");
    match stmt {
        Statement::Select { fields, .. } => assert_eq!(fields.len(), 2),
        other => panic!("got {other:?}"),
    }
}

#[test]
fn field_not_a_name_is_error() {
    // parse_field fallthrough error (helpers.rs 208-211): an integer after a
    // comma is not a field name.
    err("SELECT entity, 7 FROM EDGES;");
}

#[test]
fn missing_comparison_operator_is_error() {
    // parse_compare_op error (helpers.rs 265-268).
    err("SELECT * FROM EDGES WHERE a b;");
}

#[test]
fn expect_token_mismatch_is_error() {
    // expect_token error (helpers.rs 443-446): INSERT column list missing the
    // comma between `entity` and `relation`.
    err(r#"INSERT INTO EDGES (entity relation, target) VALUES ("a", "b", "c");"#);
}

#[test]
fn expect_ident_eq_mismatch_is_error() {
    // expect_ident_eq error (helpers.rs 461-465): first column must be `entity`.
    err(r#"INSERT INTO EDGES (foo, relation, target) VALUES ("a", "b", "c");"#);
}

#[test]
fn expect_ident_eq_accepts_keyword_column_names() {
    // expect_ident_eq keyword arm (helpers.rs 457-460): `relation` is a
    // keyword but is accepted as the column identifier via as_field_name.
    let stmt = ok(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#);
    assert!(matches!(stmt, Statement::Insert { .. }));
}

#[test]
fn expect_field_name_error() {
    // expect_field_name fallthrough error (helpers.rs 482-485): a string
    // literal is not a valid field name in a WHERE condition.
    err(r#"DELETE FROM EDGES WHERE "lit" = 1;"#);
}

#[test]
fn expect_u32_rejects_non_positive_int() {
    // expect_u32 error (helpers.rs 413-416): TOP wants a positive integer.
    err(r#"WALK "x" TOP -5;"#);
    err(r#"WALK "x" TOP "five";"#);
}

#[test]
fn expect_f32_accepts_integer_and_rejects_string() {
    // expect_f32 IntegerLit arm (helpers.rs 426-429): CONFIDENCE 1 (int).
    let stmt =
        ok(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a","b","c") CONFIDENCE 1;"#);
    match stmt {
        Statement::Insert { confidence, .. } => {
            assert!((confidence.unwrap() - 1.0).abs() < 1e-6);
        }
        other => panic!("got {other:?}"),
    }
    // expect_f32 error arm (helpers.rs 430-433).
    err(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a","b","c") CONFIDENCE "hi";"#);
}

#[test]
fn parse_range_rejects_inverted_bounds() {
    // parse_range start > end error (helpers.rs 23-27).
    err(r#"WALK "x" LAYERS 30-3;"#);
}

#[test]
fn parse_range_valid() {
    let stmt = ok(r#"WALK "x" LAYERS 0-33;"#);
    match stmt {
        Statement::Walk { layers, .. } => {
            let r = layers.expect("range");
            assert_eq!(r.start, 0);
            assert_eq!(r.end, 33);
        }
        other => panic!("got {other:?}"),
    }
}

// ── parse_walk_mode / parse_output_format / parse_conflict_strategy ──

#[test]
fn walk_modes_all_three() {
    for (m, want) in [
        ("HYBRID", larql_lql::ast::WalkMode::Hybrid),
        ("PURE", larql_lql::ast::WalkMode::Pure),
        ("DENSE", larql_lql::ast::WalkMode::Dense),
    ] {
        let sql = format!(r#"WALK "x" MODE {m};"#);
        match ok(&sql) {
            Statement::Walk { mode, .. } => assert_eq!(mode, Some(want)),
            other => panic!("got {other:?}"),
        }
    }
}

#[test]
fn walk_mode_invalid_is_error() {
    // parse_walk_mode error (helpers.rs 76-79).
    err(r#"WALK "x" MODE TOP;"#);
}

#[test]
fn compile_output_formats_both() {
    match ok(r#"COMPILE CURRENT INTO MODEL "out.safetensors" FORMAT SAFETENSORS;"#) {
        Statement::Compile { format, .. } => {
            assert_eq!(format, Some(larql_lql::ast::OutputFormat::Safetensors));
        }
        other => panic!("got {other:?}"),
    }
    match ok(r#"COMPILE CURRENT INTO MODEL "out.gguf" FORMAT GGUF;"#) {
        Statement::Compile { format, .. } => {
            assert_eq!(format, Some(larql_lql::ast::OutputFormat::Gguf));
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn compile_output_format_invalid_is_error() {
    // parse_output_format error (helpers.rs 93-96).
    err(r#"COMPILE CURRENT INTO MODEL "out" FORMAT WALK;"#);
}

#[test]
fn merge_conflict_strategies_all_three() {
    for (s, want) in [
        ("KEEP_SOURCE", larql_lql::ast::ConflictStrategy::KeepSource),
        ("KEEP_TARGET", larql_lql::ast::ConflictStrategy::KeepTarget),
        (
            "HIGHEST_CONFIDENCE",
            larql_lql::ast::ConflictStrategy::HighestConfidence,
        ),
    ] {
        let sql = format!(r#"MERGE "a.vindex" INTO "b.vindex" ON CONFLICT {s};"#);
        match ok(&sql) {
            Statement::Merge { conflict, .. } => assert_eq!(conflict, Some(want)),
            other => panic!("got {other:?}"),
        }
    }
}

#[test]
fn merge_conflict_strategy_invalid_is_error() {
    // parse_conflict_strategy error (helpers.rs 114-117).
    err(r#"MERGE "a.vindex" INTO "b.vindex" ON CONFLICT FAIL;"#);
}

#[test]
fn merge_without_clauses() {
    let stmt = ok(r#"MERGE "a.vindex";"#);
    match stmt {
        Statement::Merge {
            source,
            target,
            conflict,
        } => {
            assert_eq!(source, "a.vindex");
            assert!(target.is_none());
            assert!(conflict.is_none());
        }
        other => panic!("got {other:?}"),
    }
}

// ── parse_component / parse_component_list ──

#[test]
fn component_list_via_keywords() {
    let stmt = ok(
        r#"EXTRACT MODEL "m" INTO "o.vindex" COMPONENTS FFN_GATE, FFN_DOWN, FFN_UP, EMBEDDINGS, ATTN_OV, ATTN_QK;"#,
    );
    match stmt {
        Statement::Extract { components, .. } => {
            assert_eq!(components.unwrap().len(), 6);
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn component_list_via_unquoted_idents() {
    // parse_component Ident arm (helpers.rs 157-169): components given as bare
    // identifiers rather than keyword tokens. `gate`/`down` etc are NOT
    // keywords, but the full names are — so use names the lexer treats as
    // idents only when *not* keywords. `ffn_gate` lexes as a keyword, so to
    // hit the Ident arm we need a non-keyword spelling that the matcher still
    // accepts. The component matcher lowercases, and the lexer keyword set is
    // upper-only-by-canonical; `Ffn_Gate` still lexes to the keyword. So we
    // instead reach the Ident arm with the unknown-component error path below
    // and rely on keyword tokens for the happy path. This test asserts the
    // unknown-component Ident error (helpers.rs 165).
    err(r#"EXTRACT MODEL "m" INTO "o.vindex" COMPONENTS not_a_component;"#);
}

#[test]
fn component_invalid_token_is_error() {
    // parse_component fallthrough error (helpers.rs 170-173): a number is not a
    // component name.
    err(r#"EXTRACT MODEL "m" INTO "o.vindex" COMPONENTS 5;"#);
}

// ── try_parse_layer_band (helpers.rs 33-60) ──

#[test]
fn describe_with_layer_bands() {
    // SYNTAX / KNOWLEDGE / OUTPUT bands + ALL LAYERS.
    match ok(r#"DESCRIBE "France" SYNTAX;"#) {
        Statement::Describe { band, .. } => {
            assert_eq!(band, Some(larql_lql::ast::LayerBand::Syntax))
        }
        other => panic!("got {other:?}"),
    }
    match ok(r#"DESCRIBE "France" KNOWLEDGE;"#) {
        Statement::Describe { band, .. } => {
            assert_eq!(band, Some(larql_lql::ast::LayerBand::Knowledge))
        }
        other => panic!("got {other:?}"),
    }
    match ok(r#"DESCRIBE "France" OUTPUT;"#) {
        Statement::Describe { band, .. } => {
            assert_eq!(band, Some(larql_lql::ast::LayerBand::Output))
        }
        other => panic!("got {other:?}"),
    }
    match ok(r#"DESCRIBE "France" ALL LAYERS;"#) {
        Statement::Describe { band, .. } => {
            assert_eq!(band, Some(larql_lql::ast::LayerBand::All))
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn describe_all_not_followed_by_layers_backtracks() {
    // try_parse_layer_band ALL arm with no LAYERS following: pos is restored
    // and None returned (helpers.rs 42-44). With `ALL` not followed by LAYERS,
    // the band is left unset and the describe loop breaks; the stray `ALL`
    // keyword then becomes a trailing token, so the overall parse is an error.
    // The backtrack restore (helpers.rs 42-43) is still executed en route.
    err(r#"DESCRIBE "France" ALL;"#);
}

// ────────────────────────────────────────────────────────────────────────
// parser/lifecycle.rs — USE / COMPILE / DIFF / EXTRACT / COMPACT
// ────────────────────────────────────────────────────────────────────────

#[test]
fn use_vindex_model_and_remote() {
    match ok(r#"USE "gemma3-4b.vindex";"#) {
        Statement::Use { target } => {
            assert!(matches!(target, larql_lql::ast::UseTarget::Vindex(_)))
        }
        other => panic!("got {other:?}"),
    }
    // USE MODEL ... AUTO_EXTRACT
    match ok(r#"USE MODEL "google/gemma-3-4b-it" AUTO_EXTRACT;"#) {
        Statement::Use { target } => match target {
            larql_lql::ast::UseTarget::Model { auto_extract, .. } => assert!(auto_extract),
            other => panic!("got {other:?}"),
        },
        other => panic!("got {other:?}"),
    }
    // USE MODEL without AUTO_EXTRACT
    match ok(r#"USE MODEL "google/gemma-3-4b-it";"#) {
        Statement::Use { target } => match target {
            larql_lql::ast::UseTarget::Model { auto_extract, .. } => assert!(!auto_extract),
            other => panic!("got {other:?}"),
        },
        other => panic!("got {other:?}"),
    }
    // USE REMOTE — lifecycle.rs 202-204.
    match ok(r#"USE REMOTE "http://localhost:8080";"#) {
        Statement::Use { target } => {
            assert!(matches!(target, larql_lql::ast::UseTarget::Remote(_)))
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn compile_into_model_and_vindex() {
    match ok(r#"COMPILE CURRENT INTO MODEL "out.safetensors";"#) {
        Statement::Compile { target, .. } => {
            assert_eq!(target, larql_lql::ast::CompileTarget::Model)
        }
        other => panic!("got {other:?}"),
    }
    // INTO VINDEX (vindex is a bare identifier, not a keyword).
    match ok(r#"COMPILE "src.vindex" INTO VINDEX "out.vindex";"#) {
        Statement::Compile { target, .. } => {
            assert_eq!(target, larql_lql::ast::CompileTarget::Vindex)
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn compile_into_unrecognised_target_is_error() {
    // lifecycle.rs 75-76: INTO followed by neither MODEL nor `vindex` ident.
    err(r#"COMPILE CURRENT INTO "out.safetensors";"#);
}

#[test]
fn compile_into_vindex_on_conflict_strategies() {
    for (s, want) in [
        ("LAST_WINS", larql_lql::ast::CompileConflict::LastWins),
        (
            "HIGHEST_CONFIDENCE",
            larql_lql::ast::CompileConflict::HighestConfidence,
        ),
        ("FAIL", larql_lql::ast::CompileConflict::Fail),
    ] {
        let sql = format!(r#"COMPILE "s.vindex" INTO VINDEX "o.vindex" ON CONFLICT {s};"#);
        match ok(&sql) {
            Statement::Compile { on_conflict, .. } => assert_eq!(on_conflict, Some(want)),
            other => panic!("got {other:?}"),
        }
    }
}

#[test]
fn compile_on_conflict_invalid_strategy_is_error() {
    // lifecycle.rs 112-116: bad strategy keyword after ON CONFLICT.
    err(r#"COMPILE "s.vindex" INTO VINDEX "o.vindex" ON CONFLICT WALK;"#);
}

#[test]
fn compile_on_conflict_rejected_for_model_target() {
    // lifecycle.rs 119-123: ON CONFLICT is only valid for INTO VINDEX.
    err(r#"COMPILE CURRENT INTO MODEL "o.safetensors" ON CONFLICT LAST_WINS;"#);
}

#[test]
fn extract_with_inference_all_and_legacy_weights() {
    match ok(r#"EXTRACT MODEL "m" INTO "o.vindex" WITH INFERENCE;"#) {
        Statement::Extract { extract_level, .. } => {
            assert_eq!(extract_level, larql_lql::ast::ExtractLevel::Inference)
        }
        other => panic!("got {other:?}"),
    }
    match ok(r#"EXTRACT MODEL "m" INTO "o.vindex" WITH ALL;"#) {
        Statement::Extract { extract_level, .. } => {
            assert_eq!(extract_level, larql_lql::ast::ExtractLevel::All)
        }
        other => panic!("got {other:?}"),
    }
    // Legacy WITH WEIGHTS → Inference.
    match ok(r#"EXTRACT MODEL "m" INTO "o.vindex" WITH WEIGHTS;"#) {
        Statement::Extract { extract_level, .. } => {
            assert_eq!(extract_level, larql_lql::ast::ExtractLevel::Inference)
        }
        other => panic!("got {other:?}"),
    }
    // Default (no WITH) → Browse, with COMPONENTS + LAYERS.
    match ok(r#"EXTRACT MODEL "m" INTO "o.vindex" COMPONENTS FFN_GATE LAYERS 0-5;"#) {
        Statement::Extract {
            extract_level,
            layers,
            ..
        } => {
            assert_eq!(extract_level, larql_lql::ast::ExtractLevel::Browse);
            assert!(layers.is_some());
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn extract_with_unknown_keyword_is_error() {
    // lifecycle.rs 40: WITH followed by neither INFERENCE/ALL/WEIGHTS.
    err(r#"EXTRACT MODEL "m" INTO "o.vindex" WITH WALK;"#);
}

#[test]
fn extract_format_names_a_generation_or_stays_absent() {
    use larql_lql::ast::ExtractFormat;
    // Explicit requests parse, case-insensitively (bare identifiers, as
    // with COMPILE INTO VINDEX).
    match ok(r#"EXTRACT MODEL "m" INTO "o.vindex" FORMAT VINDEX2;"#) {
        Statement::Extract { format, .. } => assert_eq!(format, Some(ExtractFormat::Vindex2)),
        other => panic!("got {other:?}"),
    }
    match ok(r#"EXTRACT MODEL "m" INTO "o.vindex" FORMAT vindex3 WITH INFERENCE;"#) {
        Statement::Extract {
            format,
            extract_level,
            ..
        } => {
            assert_eq!(format, Some(ExtractFormat::Vindex3));
            assert_eq!(extract_level, larql_lql::ast::ExtractLevel::Inference);
        }
        other => panic!("got {other:?}"),
    }
    // Absence is "no preference" — a distinct value from either request,
    // resolved by the vindex crate's policy site, not the parser.
    match ok(r#"EXTRACT MODEL "m" INTO "o.vindex";"#) {
        Statement::Extract { format, .. } => assert_eq!(format, None),
        other => panic!("got {other:?}"),
    }
    // FORMAT followed by anything else is a parse error.
    err(r#"EXTRACT MODEL "m" INTO "o.vindex" FORMAT GGUF;"#);
}

#[test]
fn diff_with_all_optional_clauses() {
    let stmt = ok(r#"DIFF "a.vindex" "b.vindex" LAYER 5 RELATION "capital-of" LIMIT 10;"#);
    match stmt {
        Statement::Diff {
            layer,
            relation,
            limit,
            into_patch,
            ..
        } => {
            assert_eq!(layer, Some(5));
            assert_eq!(relation.as_deref(), Some("capital-of"));
            assert_eq!(limit, Some(10));
            assert!(into_patch.is_none());
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn diff_relations_alias_and_into_patch() {
    // RELATIONS keyword variant + INTO PATCH terminal clause (lifecycle 152-173).
    let stmt = ok(r#"DIFF CURRENT "b.vindex" RELATIONS "x" INTO PATCH "p.vlp";"#);
    match stmt {
        Statement::Diff {
            relation,
            into_patch,
            ..
        } => {
            assert_eq!(relation.as_deref(), Some("x"));
            assert_eq!(into_patch.as_deref(), Some("p.vlp"));
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn diff_minimal() {
    let stmt = ok(r#"DIFF "a.vindex" "b.vindex";"#);
    assert!(matches!(stmt, Statement::Diff { .. }));
}

#[test]
fn compact_minor_and_major_variants() {
    assert!(matches!(ok("COMPACT MINOR;"), Statement::CompactMinor));

    // MAJOR plain.
    match ok("COMPACT MAJOR;") {
        Statement::CompactMajor { full, lambda } => {
            assert!(!full);
            assert!(lambda.is_none());
        }
        other => panic!("got {other:?}"),
    }
    // MAJOR FULL (FULL spelled as ident).
    match ok("COMPACT MAJOR FULL;") {
        Statement::CompactMajor { full, .. } => assert!(full),
        other => panic!("got {other:?}"),
    }
    // MAJOR ALL (FULL via the ALL keyword — lifecycle.rs 227-230).
    match ok("COMPACT MAJOR ALL;") {
        Statement::CompactMajor { full, .. } => assert!(full),
        other => panic!("got {other:?}"),
    }
    // MAJOR WITH LAMBDA = <f>.
    match ok("COMPACT MAJOR WITH LAMBDA = 0.25;") {
        Statement::CompactMajor { lambda, .. } => {
            assert!((lambda.unwrap() - 0.25).abs() < 1e-6)
        }
        other => panic!("got {other:?}"),
    }
    // MAJOR FULL WITH LAMBDA = <f> (both clauses).
    match ok("COMPACT MAJOR FULL WITH LAMBDA = 0.5;") {
        Statement::CompactMajor { full, lambda } => {
            assert!(full);
            assert!((lambda.unwrap() - 0.5).abs() < 1e-6);
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn compact_major_with_lambda_missing_eq_is_error() {
    // lifecycle.rs 244-246: LAMBDA must be followed by '='.
    err("COMPACT MAJOR WITH LAMBDA 0.25;");
}

#[test]
fn compact_major_with_non_lambda_is_error() {
    // lifecycle.rs 250-253: WITH must be followed by LAMBDA in COMPACT MAJOR.
    err("COMPACT MAJOR WITH FULL;");
}

#[test]
fn compact_unknown_subcommand_is_error() {
    // lifecycle.rs 262-265: neither MINOR nor MAJOR.
    err("COMPACT EVERYTHING;");
}

// ────────────────────────────────────────────────────────────────────────
// Cross-cutting: statement dispatch, pipe, trailing-token, empty input
// ────────────────────────────────────────────────────────────────────────

#[test]
fn unknown_leading_keyword_is_error() {
    // parse_statement fallthrough (mod.rs 85-88): no statement keyword.
    err("FROM EDGES;");
}

#[test]
fn empty_input_is_error() {
    // Empty token stream → parse_statement sees Eof → error.
    err("");
    err("   ");
    err("-- only a comment");
}

#[test]
fn trailing_token_after_statement_is_error() {
    // mod.rs 51-56: extra tokens after a complete statement (no pipe).
    err("STATS extra;");
}

#[test]
fn select_default_source_and_nearest_clause() {
    // SELECT ... FROM EDGES NEAREST TO "X" AT LAYER N (query.rs nearest path).
    let stmt = ok(r#"SELECT * FROM EDGES NEAREST TO "France" AT LAYER 14;"#);
    match stmt {
        Statement::Select { nearest, .. } => {
            let n = nearest.expect("nearest present");
            assert_eq!(n.entity, "France");
            assert_eq!(n.layer, 14);
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn select_from_features_and_entities() {
    assert!(matches!(
        ok("SELECT * FROM FEATURES;"),
        Statement::Select {
            source: larql_lql::ast::SelectSource::Features,
            ..
        }
    ));
    assert!(matches!(
        ok("SELECT * FROM ENTITIES;"),
        Statement::Select {
            source: larql_lql::ast::SelectSource::Entities,
            ..
        }
    ));
}

#[test]
fn order_by_asc_explicit_and_default() {
    match ok("SELECT * FROM EDGES ORDER BY confidence ASC;") {
        Statement::Select { order, .. } => assert!(!order.unwrap().descending),
        other => panic!("got {other:?}"),
    }
    // No ASC/DESC → default ascending.
    match ok("SELECT * FROM EDGES ORDER BY confidence;") {
        Statement::Select { order, .. } => assert!(!order.unwrap().descending),
        other => panic!("got {other:?}"),
    }
}
