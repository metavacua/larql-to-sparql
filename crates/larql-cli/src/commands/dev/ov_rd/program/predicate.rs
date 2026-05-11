use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::context::{fields, PositionContext};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Predicate {
    Eq(Vec<Value>),
    And(Vec<Predicate>),
    Or(Vec<Predicate>),
    Not(Box<Predicate>),
}

impl Predicate {
    pub fn eval(&self, ctx: &PositionContext) -> bool {
        match self {
            Predicate::Eq(args) if args.len() == 2 => eval_eq(&args[0], &args[1], ctx),
            Predicate::Eq(_) => false,
            Predicate::And(preds) => preds.iter().all(|p| p.eval(ctx)),
            Predicate::Or(preds) => preds.iter().any(|p| p.eval(ctx)),
            Predicate::Not(pred) => !pred.eval(ctx),
        }
    }

    pub fn complexity(&self) -> usize {
        match self {
            Predicate::Eq(args) => complexity_eq(args),
            Predicate::And(preds) => preds.iter().map(|p| p.complexity()).sum::<usize>() + 1,
            Predicate::Or(preds) => preds.iter().map(|p| p.complexity()).sum::<usize>() + 2,
            Predicate::Not(pred) => pred.complexity() + 1,
        }
    }
}

fn eval_eq(field: &Value, expected: &Value, ctx: &PositionContext) -> bool {
    match field.as_str().unwrap_or("") {
        fields::STRATUM => ctx.stratum == expected.as_str().unwrap_or(""),
        fields::ATTENDS_BOS => ctx.attends_bos == expected.as_bool().unwrap_or(false),
        fields::ATTENDS_PREV => ctx.attends_prev == expected.as_bool().unwrap_or(false),
        fields::POSITION => ctx.position == expected.as_u64().unwrap_or(u64::MAX) as usize,
        fields::CURRENT_CODE => ctx.current_code == expected.as_u64().unwrap_or(u64::MAX) as usize,
        fields::ORIGINAL_CODE => {
            ctx.original_code == expected.as_u64().unwrap_or(u64::MAX) as usize
        }
        fields::TOKEN_ID => ctx.token_id == expected.as_u64().unwrap_or(u64::MAX) as u32,
        fields::PREV_TOKEN_ID => ctx.prev_token_id == expected.as_u64().map(|v| v as u32),
        _ => false,
    }
}

fn complexity_eq(args: &[Value]) -> usize {
    let field = args.first().and_then(|v| v.as_str()).unwrap_or("");
    if field.starts_with(fields::PREV_LAYER_FFN_TOP1_ID) {
        3
    } else if args.get(1).map(|v| v.is_number()).unwrap_or(false) {
        2
    } else {
        1
    }
}
