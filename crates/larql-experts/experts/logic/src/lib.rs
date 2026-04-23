//! # Propositional-logic expert
//!
//! Boolean expressions in a formal syntax: single-letter variables `A..Z`,
//! constants `TRUE` / `FALSE`, operators `NOT` (also `~`, `¬`, `!`),
//! `AND` (`&`, `∧`), `OR` (`|`, `∨`), `IMPLIES` (`→`), `IFF` (`↔`), with
//! parentheses. Precedence: NOT > AND > OR > IMPLIES > IFF.
//!
//! ## Ops
//!
//! - `eval {expr: string, assignments: {var: bool, ...}} → bool`
//! - `simplify {expr: string} → string`
//! - `truth_table {expr: string} → {vars: [string], rows: [{inputs: [bool], output: bool}]}`
//! - `classify {expr: string} → "tautology" | "contradiction" | "contingent"`

use expert_interface::{arg_str, expert_exports, json, Value};

expert_exports!(
    id = "logic",
    tier = 1,
    description = "Propositional logic: eval, simplify, truth table, classify (tautology/contradiction/contingent)",
    version = "0.2.0",
    ops = [
        ("eval",        ["expr", "assignments"]),
        ("simplify",    ["expr"]),
        ("truth_table", ["expr"]),
        ("classify",    ["expr"]),
    ],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    let expr = parse(arg_str(args, "expr")?)?;
    match op {
        "eval" => {
            let assignments = args.get("assignments")?.as_object()?;
            let vars: Vec<(char, bool)> = assignments
                .iter()
                .filter_map(|(k, v)| Some((k.chars().next()?, v.as_bool()?)))
                .collect();
            Some(json!(expr.eval(&vars)))
        }
        "simplify" => Some(json!(simplify(&expr).to_str())),
        "truth_table" => {
            let vars = expr.vars();
            let rows: Vec<Value> = (0..(1usize << vars.len()))
                .map(|row| {
                    let assignment: Vec<(char, bool)> = vars
                        .iter()
                        .enumerate()
                        .map(|(i, &c)| (c, (row >> (vars.len() - 1 - i)) & 1 == 1))
                        .collect();
                    let inputs: Vec<bool> = assignment.iter().map(|(_, b)| *b).collect();
                    json!({"inputs": inputs, "output": expr.eval(&assignment)})
                })
                .collect();
            Some(json!({
                "vars": vars.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
                "rows": rows,
            }))
        }
        "classify" => {
            let vars = expr.vars();
            let mut all_true = true;
            let mut all_false = true;
            for row in 0..(1usize << vars.len()) {
                let assignment: Vec<(char, bool)> = vars
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| (c, (row >> (vars.len() - 1 - i)) & 1 == 1))
                    .collect();
                let r = expr.eval(&assignment);
                if !r { all_true = false; }
                if r { all_false = false; }
            }
            Some(json!(if all_true { "tautology" } else if all_false { "contradiction" } else { "contingent" }))
        }
        _ => None,
    }
}

// ── Expression model ─────────────────────────────────────────────────────────

#[derive(Clone, PartialEq)]
enum Tok { Var(char), And, Or, Not, Imp, Iff, LP, RP, T, F }

fn tokenize(s: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let upper: String = s.to_uppercase();
    let chars: Vec<char> = upper.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\n' => { i += 1; }
            '(' => { out.push(Tok::LP); i += 1; }
            ')' => { out.push(Tok::RP); i += 1; }
            '¬' | '~' | '!' => { out.push(Tok::Not); i += 1; }
            '∧' | '&' => { out.push(Tok::And); i += 1; }
            '∨' | '|' => { out.push(Tok::Or); i += 1; }
            '→' => { out.push(Tok::Imp); i += 1; }
            '↔' => { out.push(Tok::Iff); i += 1; }
            'A'..='Z' => {
                let rest: String = chars[i..].iter().collect();
                let boundary = |len: usize| rest.len() == len
                    || !chars.get(i + len).is_some_and(|c| c.is_alphabetic());
                if rest.starts_with("AND") && boundary(3) { out.push(Tok::And); i += 3; }
                else if rest.starts_with("OR") && boundary(2) { out.push(Tok::Or); i += 2; }
                else if rest.starts_with("NOT") && boundary(3) { out.push(Tok::Not); i += 3; }
                else if rest.starts_with("IMPLIES") && boundary(7) { out.push(Tok::Imp); i += 7; }
                else if rest.starts_with("IFF") && boundary(3) { out.push(Tok::Iff); i += 3; }
                else if rest.starts_with("TRUE") && boundary(4) { out.push(Tok::T); i += 4; }
                else if rest.starts_with("FALSE") && boundary(5) { out.push(Tok::F); i += 5; }
                else { out.push(Tok::Var(c)); i += 1; }
            }
            _ => { i += 1; }
        }
    }
    out
}

#[derive(Clone)]
enum Expr {
    Var(char),
    Const(bool),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Imp(Box<Expr>, Box<Expr>),
    Iff(Box<Expr>, Box<Expr>),
}

impl Expr {
    fn vars(&self) -> Vec<char> {
        let mut v = Vec::new();
        self.collect(&mut v);
        v.sort_unstable();
        v.dedup();
        v
    }
    fn collect(&self, out: &mut Vec<char>) {
        match self {
            Expr::Var(c) => out.push(*c),
            Expr::Const(_) => {}
            Expr::Not(e) => e.collect(out),
            Expr::And(a, b) | Expr::Or(a, b) | Expr::Imp(a, b) | Expr::Iff(a, b) => {
                a.collect(out); b.collect(out);
            }
        }
    }
    fn eval(&self, vars: &[(char, bool)]) -> bool {
        match self {
            Expr::Var(c) => vars.iter().find(|(v, _)| v == c).map(|(_, b)| *b).unwrap_or(false),
            Expr::Const(b) => *b,
            Expr::Not(e) => !e.eval(vars),
            Expr::And(a, b) => a.eval(vars) && b.eval(vars),
            Expr::Or(a, b) => a.eval(vars) || b.eval(vars),
            Expr::Imp(a, b) => !a.eval(vars) || b.eval(vars),
            Expr::Iff(a, b) => a.eval(vars) == b.eval(vars),
        }
    }
    fn to_str(&self) -> String {
        match self {
            Expr::Var(c) => format!("{}", c),
            Expr::Const(true) => String::from("TRUE"),
            Expr::Const(false) => String::from("FALSE"),
            Expr::Not(e) => format!("NOT {}", e.to_str()),
            Expr::And(a, b) => format!("({} AND {})", a.to_str(), b.to_str()),
            Expr::Or(a, b) => format!("({} OR {})", a.to_str(), b.to_str()),
            Expr::Imp(a, b) => format!("({} → {})", a.to_str(), b.to_str()),
            Expr::Iff(a, b) => format!("({} ↔ {})", a.to_str(), b.to_str()),
        }
    }
}

struct Parser { toks: Vec<Tok>, pos: usize }

impl Parser {
    fn peek(&self) -> Option<&Tok> { self.toks.get(self.pos) }
    fn eat(&mut self) { self.pos += 1; }

    fn iff(&mut self) -> Option<Expr> {
        let mut lhs = self.implies()?;
        while self.peek() == Some(&Tok::Iff) {
            self.eat();
            lhs = Expr::Iff(Box::new(lhs), Box::new(self.implies()?));
        }
        Some(lhs)
    }
    fn implies(&mut self) -> Option<Expr> {
        let mut lhs = self.or()?;
        while self.peek() == Some(&Tok::Imp) {
            self.eat();
            lhs = Expr::Imp(Box::new(lhs), Box::new(self.or()?));
        }
        Some(lhs)
    }
    fn or(&mut self) -> Option<Expr> {
        let mut lhs = self.and()?;
        while self.peek() == Some(&Tok::Or) {
            self.eat();
            lhs = Expr::Or(Box::new(lhs), Box::new(self.and()?));
        }
        Some(lhs)
    }
    fn and(&mut self) -> Option<Expr> {
        let mut lhs = self.not()?;
        while self.peek() == Some(&Tok::And) {
            self.eat();
            lhs = Expr::And(Box::new(lhs), Box::new(self.not()?));
        }
        Some(lhs)
    }
    fn not(&mut self) -> Option<Expr> {
        if self.peek() == Some(&Tok::Not) { self.eat(); Some(Expr::Not(Box::new(self.not()?))) }
        else { self.atom() }
    }
    fn atom(&mut self) -> Option<Expr> {
        match self.peek()?.clone() {
            Tok::LP => { self.eat(); let e = self.iff()?; if self.peek() == Some(&Tok::RP) { self.eat(); } Some(e) }
            Tok::Var(c) => { self.eat(); Some(Expr::Var(c)) }
            Tok::T => { self.eat(); Some(Expr::Const(true)) }
            Tok::F => { self.eat(); Some(Expr::Const(false)) }
            _ => None,
        }
    }
}

fn parse(s: &str) -> Option<Expr> {
    let mut p = Parser { toks: tokenize(s), pos: 0 };
    p.iff()
}

fn simplify(e: &Expr) -> Expr {
    match e {
        Expr::Not(inner) => {
            let s = simplify(inner);
            match s {
                Expr::Const(b) => Expr::Const(!b),
                Expr::Not(i2) => *i2,
                other => Expr::Not(Box::new(other)),
            }
        }
        Expr::And(a, b) => {
            let sa = simplify(a); let sb = simplify(b);
            match (&sa, &sb) {
                (Expr::Const(false), _) | (_, Expr::Const(false)) => Expr::Const(false),
                (Expr::Const(true), _) => sb,
                (_, Expr::Const(true)) => sa,
                _ => Expr::And(Box::new(sa), Box::new(sb)),
            }
        }
        Expr::Or(a, b) => {
            let sa = simplify(a); let sb = simplify(b);
            match (&sa, &sb) {
                (Expr::Const(true), _) | (_, Expr::Const(true)) => Expr::Const(true),
                (Expr::Const(false), _) => sb,
                (_, Expr::Const(false)) => sa,
                _ => Expr::Or(Box::new(sa), Box::new(sb)),
            }
        }
        Expr::Imp(a, b) => Expr::Imp(Box::new(simplify(a)), Box::new(simplify(b))),
        Expr::Iff(a, b) => Expr::Iff(Box::new(simplify(a)), Box::new(simplify(b))),
        other => other.clone(),
    }
}
