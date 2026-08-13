//! Extraction for the arithmetic expert (spec §4).
//!
//! **Explicit path:** symbolic parse of operand digit spans and operators off
//! the prompt surface — exact by construction, zero tokens. The same scanner
//! is the tier-0 gate (`gate.rs`), so a tier-0 fire implies the symbolic
//! extract succeeds: fire ⇒ extraction, the A10 invariant, holds by
//! construction.
//!
//! **Disguised path:** 2-shot rewrite prompt → parse the *emitted expression*.
//! The parser reads the model's expression, never its sum (rigging-proofed by
//! design — anything after `=` is discarded).

use super::alu::{BigInt, Expr, Op};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// One lexed token of the prompt surface. `Other` breaks operand/operator
/// adjacency so unrelated numbers never join into an expression. Spans are
/// char indices into the input, used for the weak-chain cue check.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Num {
        digits: String,
        start: usize,
        end: usize,
    },
    Op {
        op: Op,
        weak: bool,
    },
    Other,
}

/// Scan `text` for the longest explicit integer chain `N op N (op N)*`
/// written in *math notation* — tier-0 stays symbolic and dumb by design.
///
/// Operator rules (gate specificity is the contract; intent inference is
/// NOT tier-0's job — ambiguous surface forms are the model's territory,
/// and no-fire ⇒ native is the designed fallthrough, spec §3):
/// - **Strong** operators — `+`, `*`, `×`, `−` (U+2212) — are math glyphs
///   wherever they appear; a chain containing any of them fires bare.
/// - **Weak** operators — ASCII `-` with whitespace on both sides, and
///   standalone `x`/`X` — are ordinary prose syntax (ranges `5 - 10`,
///   scores `3 - 1`, shifts `9 - 5`, spaced dates `2026 - 06 - 11`,
///   dimensions `4 x 4`). A chain whose operators are ALL weak fires only
///   when followed by an explicit `=` — the one cue that is itself math
///   notation. (`?` is sentence punctuation, not notation: "Are you
///   available 9 - 5?" must never fire.) Everything else falls through to
///   native untouched. Adversarial prose corpus: 0 false fires
///   (`scanner_adversarial` example).
/// - Unspaced `-` never counts (dates `2026-06-11`, ranges `5-10`, phones);
///   `/` never counts — division is OPEN in v0.1 and `06/11` is a date.
///
/// Numbers absorb `1,234,567`-style thousands separators and `_` separators.
pub fn find_expression(text: &str) -> Option<Expr> {
    find_expression_with_policy(text, true)
}

/// `require_notation_cue = false` relaxes the weak-chain `=` requirement —
/// used by [`parse_rewrite`] only, where the line being parsed is a
/// model-emitted expression by instruction (the rewrite prompt IS the
/// cue), not user prose.
fn find_expression_with_policy(text: &str, require_notation_cue: bool) -> Option<Expr> {
    let chars: Vec<char> = text.chars().collect();
    let toks = lex(&chars);

    // Collect every maximal chain, then pick the longest QUALIFYING one —
    // a weak unqualified range earlier in the text must not shadow a real
    // expression later in it.
    let mut best: Option<(usize, usize)> = None; // (start tok idx, op_count)
    let mut i = 0;
    while i < toks.len() {
        if matches!(toks[i], Tok::Num { .. }) {
            let mut j = i;
            let mut ops = 0usize;
            while matches!(toks.get(j + 1), Some(Tok::Op { .. }))
                && matches!(toks.get(j + 2), Some(Tok::Num { .. }))
            {
                ops += 1;
                j += 2;
            }
            if ops > 0
                && (!require_notation_cue || chain_qualifies(&chars, &toks, i, ops))
                && best.map(|(_, b)| ops > b).unwrap_or(true)
            {
                best = Some((i, ops));
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }

    let (start, op_count) = best?;
    build_expr(&toks, start, op_count)
}

fn build_expr(toks: &[Tok], start: usize, op_count: usize) -> Option<Expr> {
    let mut operands = Vec::with_capacity(op_count + 1);
    let mut ops = Vec::with_capacity(op_count);
    for k in 0..=op_count {
        let Tok::Num { digits, .. } = &toks[start + 2 * k] else {
            return None;
        };
        operands.push(BigInt::parse(digits)?);
        if k < op_count {
            let Tok::Op { op, .. } = &toks[start + 2 * k + 1] else {
                return None;
            };
            ops.push(*op);
        }
    }
    Some(Expr { operands, ops })
}

/// Stream-trigger scan: every maximal chain immediately followed by `=`
/// (whitespace allowed) — the moment a generating model has restated a
/// problem in notation and positioned its cursor at the answer slot.
/// Returns `(expr, char index just past the '=')` per trigger, in text
/// order. The trailing `=` IS the cue, so weak-operator chains qualify
/// here (the model writing `9 - 5 =` is notation by its own hand).
///
/// This is the read primitive for mid-stream dispatch; the
/// `ave_stream_trigger_probe` harness measures whether the model's
/// spontaneous restatement reflex is frequent and faithful enough to
/// gate on (fire rate × emitted-expression fidelity × position).
pub fn find_triggers(text: &str) -> Vec<(Expr, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let toks = lex(&chars);
    let mut out = Vec::new();

    let mut i = 0;
    while i < toks.len() {
        if matches!(toks[i], Tok::Num { .. }) {
            let mut j = i;
            let mut ops = 0usize;
            while matches!(toks.get(j + 1), Some(Tok::Op { .. }))
                && matches!(toks.get(j + 2), Some(Tok::Num { .. }))
            {
                ops += 1;
                j += 2;
            }
            if ops > 0 {
                let span_end = match &toks[j] {
                    Tok::Num { end, .. } => *end,
                    _ => unreachable!("chain ends on Num"),
                };
                let eq_pos = chars[span_end..]
                    .iter()
                    .position(|c| !c.is_whitespace())
                    .map(|off| span_end + off)
                    .filter(|p| chars[*p] == '=');
                if let (Some(p), Some(expr)) = (eq_pos, build_expr(&toks, i, ops)) {
                    out.push((expr, p + 1));
                }
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// Strong-op chains qualify bare; all-weak chains qualify only when the
/// next non-whitespace char after the chain is `=` (see
/// [`find_expression`] docs).
fn chain_qualifies(chars: &[char], toks: &[Tok], start: usize, op_count: usize) -> bool {
    let has_strong =
        (0..op_count).any(|k| matches!(toks[start + 2 * k + 1], Tok::Op { weak: false, .. }));
    if has_strong {
        return true;
    }
    let span_end = match &toks[start + 2 * op_count] {
        Tok::Num { end, .. } => *end,
        _ => return false,
    };
    chars[span_end..]
        .iter()
        .find(|c| !c.is_whitespace())
        .is_some_and(|c| *c == '=')
}

fn lex(chars: &[char]) -> Vec<Tok> {
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() {
            let start = i;
            let mut digits = String::new();
            while i < chars.len() {
                let c = chars[i];
                if c.is_ascii_digit() {
                    digits.push(c);
                    i += 1;
                } else if (c == ',' || c == '_')
                    && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit())
                {
                    // Separator inside a number; keep digits only.
                    i += 1;
                } else {
                    break;
                }
            }
            toks.push(Tok::Num {
                digits,
                start,
                end: i,
            });
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        let ws_before = i == 0 || chars[i - 1].is_whitespace();
        let ws_after = i + 1 >= chars.len() || chars[i + 1].is_whitespace();
        let op = match c {
            '+' => Some((Op::Add, false)),
            '*' | '×' => Some((Op::Mul, false)),
            '−' => Some((Op::Sub, false)),
            '-' if ws_before && ws_after => Some((Op::Sub, true)),
            'x' | 'X' if ws_before && ws_after => Some((Op::Mul, true)),
            _ => None,
        };
        match op {
            Some((op, weak)) => toks.push(Tok::Op { op, weak }),
            None => toks.push(Tok::Other),
        }
        i += 1;
    }
    toks
}

/// The 2-shot rewrite prompt (the measured A8 floor — deliberately untuned;
/// structured-output extraction is the OPEN improvement, not this prompt).
pub fn rewrite_prompt(question: &str) -> String {
    format!(
        "Rewrite each question as a bare arithmetic expression. Do not solve it.\n\
         Q: If you have 7 apples and pick 5 more, how many apples do you have?\n\
         E: 7 + 5\n\
         Q: A crate holds 240 bottles. How many bottles are in 3 crates?\n\
         E: 240 * 3\n\
         Q: {question}\n\
         E:"
    )
}

/// Parse the model-emitted rewrite. First emitted line only, truncated at
/// `=` so the model's own sum — if it volunteers one — is never consumed.
/// Weak-operator chains parse bare here: the rewrite instruction is the
/// notation cue, so `10 - 4` on the rewrite line is an expression, not a
/// range.
pub fn parse_rewrite(generated: &str) -> Option<Expr> {
    let line = generated.trim_start().lines().next()?;
    let line = line.split(['=', '\u{ff1d}']).next().unwrap_or(line);
    find_expression_with_policy(line, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Option<String> {
        find_expression(text).map(|e| format!("{e} -> {}", e.eval()))
    }

    #[test]
    fn explicit_forms_parse_exactly() {
        assert_eq!(parse("12 + 7 ="), Some("12 + 7 -> 19".into()));
        assert_eq!(
            parse("What is 123456 + 654321?"),
            Some("123456 + 654321 -> 777777".into())
        );
        assert_eq!(
            parse("12345 * 6789"),
            Some("12345 * 6789 -> 83810205".into())
        );
        assert_eq!(parse("12×34"), Some("12 * 34 -> 408".into()));
        assert_eq!(parse("3 x 4 ="), Some("3 * 4 -> 12".into()));
        assert_eq!(parse("100000 - 1 ="), Some("100000 - 1 -> 99999".into()));
        assert_eq!(parse("47−5"), Some("47 - 5 -> 42".into()));
    }

    #[test]
    fn two_op_chains_parse() {
        assert_eq!(
            parse("999 + 111 - 222 ="),
            Some("999 + 111 - 222 -> 888".into())
        );
        assert_eq!(parse("2 + 3 * 4"), Some("2 + 3 * 4 -> 14".into()));
    }

    #[test]
    fn thousands_separators_absorb_into_one_operand() {
        assert_eq!(
            parse("1,234,567 + 1"),
            Some("1234567 + 1 -> 1234568".into())
        );
        assert_eq!(parse("1_000 + 24"), Some("1000 + 24 -> 1024".into()));
    }

    #[test]
    fn expression_stops_at_equals_never_reads_the_sum() {
        // The chain is "12 + 7"; the 19 after '=' must not join it.
        let e = find_expression("12 + 7 = 19").expect("expr");
        assert_eq!(e.to_string(), "12 + 7");
    }

    #[test]
    fn distractors_do_not_parse() {
        for text in [
            "My phone number is 4415550172.",
            "The meeting is on 2026-06-11.",
            "Trains depart at 18:45 from platform 3.",
            "Order 66 was executed in 19 BBY.",
            "What is the capital of France?",
            "Account 123456789012345678901234567890 is active.",
            "It takes 5-10 days to ship.",
            "The score was 3/4.",
            "version 1.2.3 released",
        ] {
            assert!(
                find_expression(text).is_none(),
                "false parse on distractor: {text:?}"
            );
        }
    }

    #[test]
    fn hyphen_needs_whitespace_both_sides_and_equals_notation() {
        assert!(
            find_expression("100-1 =").is_none(),
            "unspaced never counts"
        );
        assert!(find_expression("100- 1 =").is_none());
        assert!(find_expression("100 -1 =").is_none());
        assert!(find_expression("100 - 1 =").is_some());
    }

    #[test]
    fn x_must_be_standalone_and_notation_cued() {
        assert!(find_expression("3x4").is_none(), "3x4 could be a label");
        // Standalone x is a WEAK operator: only explicit `=` notation
        // fires it — bare forms and prose dimensions are the model's
        // territory (designed fallthrough).
        assert!(find_expression("3 x 4 =").is_some(), "= notation");
        assert!(find_expression("3 x 4").is_none(), "bare → native");
        assert!(find_expression("matrix 3 x 4").is_none(), "prose dimension");
        assert!(
            find_expression("what is 3 x 4?").is_none(),
            "? is not notation"
        );
    }

    #[test]
    fn weak_chains_fire_only_on_explicit_equals_notation() {
        // The adversarial-prose corpus: ranges, scores, idioms, spaced
        // dates, dimensions, question forms — all carried digits around a
        // spaced hyphen or standalone x and all used to fire (e.g.
        // "9 - 5 job" → 4, "dated 2026 - 06 - 11" → 2009). Tier-0 does
        // not infer intent: no `=`, no fire.
        for text in [
            "It takes 5 - 10 business days.",
            "They won 3 - 1 at home.",
            "I work a 9 - 5 job.",
            "Open Monday - Friday, 9 - 17.",
            "pages 12 - 48 cover the appendix",
            "the score was 2 - 2 after extra time",
            "ages 18 - 25 only",
            "dated 2026 - 06 - 11 in the ledger",
            "a 4 x 4 truck",
            "2 x 4 lumber at the yard",
            "a 3 x 5 index card",
            "room is 12 x 14 feet",
            "Are you available 9 - 5?",
            "9 - 5",
            "100 - 1",
            "what is 100 - 7?",
        ] {
            assert!(
                find_expression(text).is_none(),
                "weak chain fired without `=` notation: {text:?}"
            );
        }
        // Explicit `=` notation fires.
        assert!(find_expression("100000 - 1 =").is_some(), "= notation");
        assert!(find_expression("9 - 5 =").is_some(), "= notation");
        // Any strong glyph in the chain is notation wherever it appears.
        assert!(
            find_expression("she computed 999 + 111 - 222 quickly").is_some(),
            "strong + in chain"
        );
        assert!(
            find_expression("what is 123456 + 654321?").is_some(),
            "question form rides the strong op, not the ?"
        );
    }

    #[test]
    fn qualifying_chain_wins_over_earlier_unqualified_range() {
        // The prose range must not shadow the real expression behind it.
        let e = find_expression("ages 18 - 25 only, so 12 + 7 =").expect("expr");
        assert_eq!(e.to_string(), "12 + 7");
    }

    #[test]
    fn longest_chain_wins() {
        // Two candidate chains; the 3-operand one is the expression.
        let e = find_expression("page 7 + 1, then 10 + 20 + 30").expect("expr");
        assert_eq!(e.to_string(), "10 + 20 + 30");
    }

    #[test]
    fn rewrite_prompt_embeds_question_and_two_shots() {
        let p = rewrite_prompt("If a box holds 12 eggs, how many in 4 boxes?");
        assert!(p.contains("7 + 5"));
        assert!(p.contains("240 * 3"));
        assert!(p.ends_with("E:"));
        assert!(p.contains("how many in 4 boxes?"));
    }

    #[test]
    fn parse_rewrite_reads_first_line_only() {
        let e = parse_rewrite(" 240 * 3\nQ: another question\nE: 1 + 1").expect("expr");
        assert_eq!(e.to_string(), "240 * 3");
    }

    #[test]
    fn parse_rewrite_discards_a_volunteered_sum() {
        let e = parse_rewrite(" 7 + 5 = 12").expect("expr");
        assert_eq!(e.to_string(), "7 + 5");
    }

    #[test]
    fn parse_rewrite_misses_on_garbage() {
        assert!(parse_rewrite("I cannot rewrite that.").is_none());
        assert!(parse_rewrite("").is_none());
    }

    #[test]
    fn find_triggers_reads_chains_at_their_equals_marker() {
        // The canonical restatement shape: prose, then `expr = `.
        let t = find_triggers("Sure!\n\n123456 + 654321 = ");
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].0.to_string(), "123456 + 654321");
        // Weak ops qualify here — the model's own `=` is the cue.
        let t = find_triggers("so 9 - 5 = ");
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].0.to_string(), "9 - 5");
        // No `=` yet → no trigger (mid-restatement).
        assert!(find_triggers("123456 + 654321").is_empty());
        // Multiple triggers arrive in text order (the chained case).
        let t = find_triggers("12 + 7 = 19\nthen 19 * 2 = ");
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].0.to_string(), "12 + 7");
        assert_eq!(t[1].0.to_string(), "19 * 2");
        // Plain prose with `=` but no chain stays silent.
        assert!(find_triggers("x = y in the limit").is_empty());
    }

    #[test]
    fn find_triggers_position_points_past_the_equals() {
        let text = "ok 12 + 7 = ";
        let t = find_triggers(text);
        assert_eq!(t.len(), 1);
        let chars: Vec<char> = text.chars().collect();
        assert_eq!(chars[t[0].1 - 1], '=', "index is just past the '='");
    }
}
