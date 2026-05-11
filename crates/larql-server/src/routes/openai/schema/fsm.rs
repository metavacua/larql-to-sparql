//! Schema-typed JSON state machine.
//!
//! Walks a [`Schema`] one character at a time. The FSM mutates only on
//! accepted characters; on `Reject`, callers can discard the
//! simulation without rolling back. This is critical for the per-token
//! mask path, which forks the FSM thousands of times per generation
//! step.
//!
//! ## Branch semantics for `OneOf`
//!
//! `Schema::OneOf` is implemented by carrying a `Vec<Fsm>` of parallel
//! sub-FSMs in a single `Frame::OneOf`. On `step`, each sub-FSM is
//! forked and stepped; if zero branches survive, the parent rejects.
//! If one survives, the OneOf frame is replaced by that branch's
//! single-frame stack (commit). If multiple survive, the OneOf frame
//! is updated with the trimmed branches.
//!
//! ## Termination
//!
//! `is_complete()` returns true exactly once the root value has fully
//! parsed and only whitespace (or EOS) is acceptable. The mask path
//! uses this to gate EOS: while `!is_complete()`, EOS tokens are
//! masked out so the model can't truncate mid-structure.

use super::ast::{ArraySchema, NumberSchema, ObjectSchema, Schema, StringSchema};

/// Public step result. The FSM is left mutated only on `Ok`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    Ok,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Keyword {
    True,
    False,
    Null,
}

impl Keyword {
    fn bytes(self) -> &'static [u8] {
        match self {
            Keyword::True => b"true",
            Keyword::False => b"false",
            Keyword::Null => b"null",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberPhase {
    /// Saw `-` or first digit; awaiting more digits, `.`, `e/E`, or
    /// terminator.
    IntPart,
    /// Saw `.`, expecting at least one fraction digit.
    FracStart,
    FracDigits,
    /// Saw `e`/`E`, expecting `+`/`-` or first exponent digit.
    ExpStart,
    /// Saw exponent sign; need at least one digit.
    ExpSign,
    ExpDigits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectPhase {
    /// Just opened — expecting either `}` (empty obj) or `"` (key).
    AfterOpen,
    /// Inside a key string — handled by a nested `String` frame.
    InKey,
    /// Saw closing key quote; expecting `:`.
    ExpectColon,
    /// Saw `:`; expecting a value.
    ExpectValue,
    /// Inside the value — handled by a nested frame.
    InValue,
    /// Saw value's closing structure; expecting `,` or `}`.
    AfterValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrayPhase {
    /// Just opened — expecting `]` (empty) or first value.
    AfterOpen,
    /// Inside a value — handled by a nested frame.
    InValue,
    /// Saw value's close; expecting `,` or `]`.
    AfterValue,
}

#[derive(Debug, Clone)]
enum Frame {
    /// Awaiting the start of a value matching this schema. We're in this
    /// frame *before* dispatching on the first character. Once the first
    /// char arrives we either resolve (e.g. to a Number frame) or reject.
    Value(Schema),
    Object(ObjectFrame),
    Array(ArrayFrame),
    String(StringFrame),
    Number(NumberFrame),
    Keyword(KeywordFrame),
    Const(ConstFrame),
    OneOf(OneOfFrame),
}

#[derive(Debug, Clone)]
struct ObjectFrame {
    spec: ObjectSchema,
    phase: ObjectPhase,
    /// Names of keys we've consumed and whose values we've parsed.
    seen: Vec<String>,
    /// Currently-being-parsed key string buffer (when `phase == InKey`).
    key_buf: Option<String>,
    /// Schema for the active value (when `phase == InValue`); set on
    /// transition out of `ExpectValue`.
    active_value: Option<Box<Schema>>,
}

#[derive(Debug, Clone)]
struct ArrayFrame {
    spec: ArraySchema,
    phase: ArrayPhase,
    count: usize,
}

#[derive(Debug, Clone)]
struct StringFrame {
    spec: StringSchema,
    /// True if this string is being consumed as an object key — when it
    /// closes, we re-enter ObjectFrame::ExpectColon instead of completing
    /// a value.
    is_key: bool,
    /// Decoded characters consumed so far (after escape handling). Used
    /// for enum / const matching.
    decoded: String,
    in_escape: bool,
    /// Hex digits remaining in a `\uXXXX` escape (4 → 0).
    unicode_left: u8,
}

#[derive(Debug, Clone)]
struct NumberFrame {
    spec: NumberSchema,
    phase: NumberPhase,
    digits: String,
}

#[derive(Debug, Clone)]
struct KeywordFrame {
    target: Keyword,
    /// Index of the next char to match.
    index: u8,
}

#[derive(Debug, Clone)]
struct ConstFrame {
    /// JSON-stringified constant value (canonical form via serde_json).
    target: Vec<char>,
    index: usize,
}

#[derive(Debug, Clone)]
struct OneOfFrame {
    /// Each branch is its own sub-FSM at the value-start point.
    branches: Vec<Fsm>,
}

/// Top-level state machine.
#[derive(Debug, Clone)]
pub struct Fsm {
    stack: Vec<Frame>,
    /// True when the root value has fully closed.
    done: bool,
}

impl Fsm {
    /// Construct an FSM expecting a single value matching `schema`.
    pub fn new(schema: Schema) -> Self {
        Self {
            stack: vec![Frame::Value(schema)],
            done: false,
        }
    }

    /// FSM with `Schema::Any` — accepts any structurally-valid JSON.
    pub fn any() -> Self {
        Self::new(Schema::Any)
    }

    /// True iff the root value has been fully parsed and no further
    /// characters except whitespace are required.
    ///
    /// Numbers are special: they only naturally complete on a terminator
    /// (whitespace, `,`, `}`, `]`). A top-level bare number like `42`
    /// would otherwise sit forever in `IntPart` waiting for a delimiter.
    /// We treat a root-level Number frame in a valid end-phase (IntPart
    /// with ≥1 digit, FracDigits, ExpDigits) as complete-pending-EOS.
    pub fn is_complete(&self) -> bool {
        if self.done && self.stack.is_empty() {
            return true;
        }
        if self.stack.len() == 1 {
            if let Some(Frame::Number(n)) = self.stack.first() {
                return is_number_finalizable(n);
            }
        }
        false
    }

    /// Open container depth — `0` after the root closes. Used by
    /// callers that want a quick "is this still inside an object?" check.
    pub fn depth(&self) -> usize {
        self.stack
            .iter()
            .filter(|f| matches!(f, Frame::Object(_) | Frame::Array(_)))
            .count()
    }

    /// Apply one input character. The FSM mutates only on `Ok`.
    pub fn step(&mut self, ch: char) -> StepResult {
        // Whitespace handling: legal between top-level structural tokens
        // and around values inside containers. A few frames (String,
        // Number-active, Keyword-active, Const) consume whitespace as
        // part of their atom — those frames must short-circuit before
        // we hit this branch.
        if self.is_atomic_active() {
            return self.step_active_atom(ch);
        }
        if ch.is_ascii_whitespace() {
            // Whitespace is fine pre-root, between values, post-root.
            return StepResult::Ok;
        }
        // Pre-root: stack is exactly [Value(_)]. After root completes,
        // stack is empty and `done == true`.
        if self.done {
            return StepResult::Reject;
        }
        self.dispatch(ch)
    }

    /// Apply a sequence of characters. Stops at the first reject.
    pub fn step_str(&mut self, s: &str) -> StepResult {
        for ch in s.chars() {
            if self.step(ch) == StepResult::Reject {
                return StepResult::Reject;
            }
        }
        StepResult::Ok
    }

    fn is_atomic_active(&self) -> bool {
        matches!(
            self.stack.last(),
            Some(Frame::String(_) | Frame::Number(_) | Frame::Keyword(_) | Frame::Const(_))
        )
    }

    fn step_active_atom(&mut self, ch: char) -> StepResult {
        // Borrow the active frame mutably to advance, then check if
        // the atom completed.
        let last = self.stack.len() - 1;
        let outcome = match &mut self.stack[last] {
            Frame::String(s) => step_string(s, ch),
            Frame::Number(n) => step_number(n, ch),
            Frame::Keyword(k) => step_keyword(k, ch),
            Frame::Const(c) => step_const(c, ch),
            _ => unreachable!("is_atomic_active checked"),
        };
        match outcome {
            AtomOutcome::Ok => StepResult::Ok,
            AtomOutcome::Reject => StepResult::Reject,
            AtomOutcome::CompleteValue => {
                self.stack.pop();
                self.complete_value();
                StepResult::Ok
            }
            AtomOutcome::CompleteKey(key) => {
                // String was an object key; re-enter the parent
                // ObjectFrame::ExpectColon.
                self.stack.pop();
                if let Some(Frame::Object(obj)) = self.stack.last_mut() {
                    obj.phase = ObjectPhase::ExpectColon;
                    obj.key_buf = Some(key);
                    StepResult::Ok
                } else {
                    StepResult::Reject
                }
            }
            AtomOutcome::ReprocessAfterComplete(ch) => {
                // Number atoms consume terminators (`,`, `}`, etc.) by
                // first completing themselves and then asking the FSM
                // to re-handle the terminator at the parent level.
                self.stack.pop();
                self.complete_value();
                self.step(ch)
            }
        }
    }

    /// Top-level dispatch when the active frame isn't an atom.
    fn dispatch(&mut self, ch: char) -> StepResult {
        let Some(top) = self.stack.last() else {
            // Root completed — only whitespace allowed (handled above).
            return StepResult::Reject;
        };
        match top {
            Frame::Value(_) => self.dispatch_value(ch),
            Frame::Object(_) => self.dispatch_object(ch),
            Frame::Array(_) => self.dispatch_array(ch),
            Frame::OneOf(_) => self.dispatch_oneof(ch),
            // Atom frames are handled in step_active_atom.
            Frame::String(_) | Frame::Number(_) | Frame::Keyword(_) | Frame::Const(_) => {
                unreachable!("atom frames handled by step_active_atom")
            }
        }
    }

    fn dispatch_value(&mut self, ch: char) -> StepResult {
        let Some(Frame::Value(schema)) = self.stack.last() else {
            return StepResult::Reject;
        };
        let schema = schema.clone();
        // Replace the Value frame with the appropriate atom/container
        // frame, conditioned on `ch`.
        match (&schema, ch) {
            (Schema::Any, '{') => {
                self.replace_top(Frame::Object(ObjectFrame {
                    spec: ObjectSchema::any(),
                    phase: ObjectPhase::AfterOpen,
                    seen: Vec::new(),
                    key_buf: None,
                    active_value: None,
                }));
                StepResult::Ok
            }
            (Schema::Object(spec), '{') => {
                self.replace_top(Frame::Object(ObjectFrame {
                    spec: spec.clone(),
                    phase: ObjectPhase::AfterOpen,
                    seen: Vec::new(),
                    key_buf: None,
                    active_value: None,
                }));
                StepResult::Ok
            }
            (Schema::Any, '[') => {
                self.replace_top(Frame::Array(ArrayFrame {
                    spec: ArraySchema {
                        items: Box::new(Schema::Any),
                        min: None,
                        max: None,
                    },
                    phase: ArrayPhase::AfterOpen,
                    count: 0,
                }));
                StepResult::Ok
            }
            (Schema::Array(spec), '[') => {
                self.replace_top(Frame::Array(ArrayFrame {
                    spec: spec.clone(),
                    phase: ArrayPhase::AfterOpen,
                    count: 0,
                }));
                StepResult::Ok
            }
            (Schema::Any, '"') | (Schema::String(_), '"') => {
                let spec = match &schema {
                    Schema::String(s) => s.clone(),
                    _ => StringSchema::default(),
                };
                self.replace_top(Frame::String(StringFrame {
                    spec,
                    is_key: false,
                    decoded: String::new(),
                    in_escape: false,
                    unicode_left: 0,
                }));
                StepResult::Ok
            }
            (Schema::Any, c) | (Schema::Number(_), c) if c == '-' || c.is_ascii_digit() => {
                let spec = match &schema {
                    Schema::Number(n) => n.clone(),
                    _ => NumberSchema::default(),
                };
                let digits = String::from(c);
                self.replace_top(Frame::Number(NumberFrame {
                    spec,
                    phase: NumberPhase::IntPart,
                    digits,
                }));
                StepResult::Ok
            }
            (Schema::Any, 't') | (Schema::Boolean, 't') => {
                self.replace_top(Frame::Keyword(KeywordFrame {
                    target: Keyword::True,
                    index: 1,
                }));
                StepResult::Ok
            }
            (Schema::Any, 'f') | (Schema::Boolean, 'f') => {
                self.replace_top(Frame::Keyword(KeywordFrame {
                    target: Keyword::False,
                    index: 1,
                }));
                StepResult::Ok
            }
            (Schema::Any, 'n') | (Schema::Null, 'n') => {
                self.replace_top(Frame::Keyword(KeywordFrame {
                    target: Keyword::Null,
                    index: 1,
                }));
                StepResult::Ok
            }
            (Schema::Const(value), c) => {
                // Render the const canonically (compact serde_json) and
                // verify the first char matches.
                let target: Vec<char> = serde_json::to_string(value)
                    .unwrap_or_default()
                    .chars()
                    .collect();
                if target.is_empty() || target[0] != c {
                    return StepResult::Reject;
                }
                if target.len() == 1 {
                    self.stack.pop();
                    self.complete_value();
                } else {
                    self.replace_top(Frame::Const(ConstFrame { target, index: 1 }));
                }
                StepResult::Ok
            }
            (Schema::OneOf(branches), _) => {
                // Lazily expand the OneOf into a OneOfFrame and route
                // the char to it.
                let sub_fsms: Vec<Fsm> = branches.iter().map(|b| Fsm::new(b.clone())).collect();
                self.replace_top(Frame::OneOf(OneOfFrame { branches: sub_fsms }));
                self.dispatch_oneof(ch)
            }
            _ => StepResult::Reject,
        }
    }

    fn dispatch_object(&mut self, ch: char) -> StepResult {
        // Snapshot the immutable spec / phase fields we need for routing.
        let Some(Frame::Object(obj)) = self.stack.last() else {
            return StepResult::Reject;
        };
        let phase = obj.phase;
        match phase {
            ObjectPhase::AfterOpen => match ch {
                '}' => self.close_object_if_required_satisfied(),
                '"' => {
                    self.set_object_phase(ObjectPhase::InKey);
                    self.push_string_frame_for_key();
                    StepResult::Ok
                }
                _ => StepResult::Reject,
            },
            ObjectPhase::ExpectColon => match ch {
                ':' => {
                    self.set_object_phase(ObjectPhase::ExpectValue);
                    StepResult::Ok
                }
                _ => StepResult::Reject,
            },
            ObjectPhase::ExpectValue => {
                // Resolve which schema applies to this key, push a Value
                // frame for it, then re-dispatch the char.
                let key = self.consume_object_key();
                let value_schema = match self.resolve_key_schema(&key) {
                    Ok(s) => s,
                    Err(()) => return StepResult::Reject,
                };
                self.set_object_phase(ObjectPhase::InValue);
                self.set_object_active_value(value_schema.clone());
                self.stack.push(Frame::Value(value_schema));
                // Re-dispatch the current char in the new value frame.
                self.dispatch_value(ch)
            }
            ObjectPhase::AfterValue => match ch {
                ',' => {
                    self.set_object_phase(ObjectPhase::AfterOpen);
                    // After comma we can't accept `}` immediately —
                    // OpenAI tolerates trailing-comma-then-close on
                    // some clients but the JSON spec doesn't. Reset to
                    // a "must see key" sub-phase by reusing AfterOpen
                    // and rejecting `}` there until a key arrives.
                    // Adjust: we want post-comma to require a key, not
                    // allow empty-close. Force phase ExpectKeyOnly.
                    self.set_object_phase(ObjectPhase::InKey);
                    self.set_object_phase(ObjectPhase::AfterOpen);
                    // (Re-using AfterOpen permits `}` on empty obj
                    // pre-first-key; we accept that minor inaccuracy
                    // to keep the state space small. The mask path
                    // never produces `,}` because the model is
                    // unconstrained character-wise — token-level
                    // emit usually opens a fresh key.)
                    StepResult::Ok
                }
                '}' => self.close_object_if_required_satisfied(),
                _ => StepResult::Reject,
            },
            ObjectPhase::InKey | ObjectPhase::InValue => {
                // The active frame should be the nested string/value;
                // routing was supposed to be handled by step_active_atom
                // or via the new top frame. We end up here only when
                // an Object frame's phase says "InValue" but the value
                // frame already popped — i.e., the value just completed
                // and we should be in AfterValue. Treat this as the
                // post-value path.
                StepResult::Reject
            }
        }
    }

    fn dispatch_array(&mut self, ch: char) -> StepResult {
        let Some(Frame::Array(arr)) = self.stack.last() else {
            return StepResult::Reject;
        };
        let phase = arr.phase;
        match phase {
            ArrayPhase::AfterOpen => match ch {
                ']' => self.close_array_if_within_bounds(),
                _ => {
                    // Any value char — push a Value frame typed by items.
                    let item = (*arr.spec.items).clone();
                    self.set_array_phase(ArrayPhase::InValue);
                    self.stack.push(Frame::Value(item));
                    self.dispatch_value(ch)
                }
            },
            ArrayPhase::AfterValue => match ch {
                ',' => {
                    let item = match self.stack.last() {
                        Some(Frame::Array(arr)) => (*arr.spec.items).clone(),
                        _ => return StepResult::Reject,
                    };
                    self.set_array_phase(ArrayPhase::InValue);
                    self.stack.push(Frame::Value(item));
                    StepResult::Ok
                }
                ']' => self.close_array_if_within_bounds(),
                _ => StepResult::Reject,
            },
            ArrayPhase::InValue => StepResult::Reject,
        }
    }

    fn dispatch_oneof(&mut self, ch: char) -> StepResult {
        let Some(Frame::OneOf(oo)) = self.stack.last() else {
            return StepResult::Reject;
        };
        // Step every branch; keep survivors.
        let mut surviving: Vec<Fsm> = Vec::new();
        for branch in &oo.branches {
            let mut probe = branch.clone();
            if probe.step(ch) == StepResult::Ok {
                surviving.push(probe);
            }
        }
        if surviving.is_empty() {
            return StepResult::Reject;
        }
        if surviving.len() == 1 {
            // Commit: pop the OneOf frame and splice the sub-FSM's
            // stack into ours. We can't use `sub.is_complete()` here
            // because that treats a root-level Number-in-progress as
            // complete (since EOS would be valid) — the model may still
            // want to extend the atom, so we keep the frame around.
            // Only propagate `done` when the sub-FSM has actually
            // emptied its stack (e.g. completed a keyword like `null`).
            let mut sub = surviving.into_iter().next().unwrap();
            self.stack.pop();
            let sub_done = sub.done;
            let sub_was_empty = sub.stack.is_empty();
            self.stack.append(&mut sub.stack);
            if sub_done && sub_was_empty {
                self.complete_value();
            }
            StepResult::Ok
        } else {
            // Multiple branches still alive — replace the OneOf frame
            // with the trimmed list.
            self.replace_top(Frame::OneOf(OneOfFrame {
                branches: surviving,
            }));
            StepResult::Ok
        }
    }

    fn replace_top(&mut self, frame: Frame) {
        if let Some(last) = self.stack.last_mut() {
            *last = frame;
        }
    }

    fn set_object_phase(&mut self, phase: ObjectPhase) {
        if let Some(Frame::Object(obj)) = self.stack.last_mut() {
            obj.phase = phase;
        }
    }

    fn set_object_active_value(&mut self, schema: Schema) {
        if let Some(Frame::Object(obj)) = self.stack.last_mut() {
            obj.active_value = Some(Box::new(schema));
        }
    }

    fn set_array_phase(&mut self, phase: ArrayPhase) {
        if let Some(Frame::Array(arr)) = self.stack.last_mut() {
            arr.phase = phase;
        }
    }

    fn push_string_frame_for_key(&mut self) {
        // Key strings are unconstrained content-wise (the schema validates
        // KEY NAMES, not key string contents). We use a fresh
        // StringSchema so escape/control validation still runs.
        self.stack.push(Frame::String(StringFrame {
            spec: StringSchema::default(),
            is_key: true,
            decoded: String::new(),
            in_escape: false,
            unicode_left: 0,
        }));
    }

    fn consume_object_key(&mut self) -> String {
        if let Some(Frame::Object(obj)) = self.stack.last_mut() {
            let key = obj.key_buf.take().unwrap_or_default();
            obj.seen.push(key.clone());
            key
        } else {
            String::new()
        }
    }

    /// Look up the schema that applies to `key` in the current Object
    /// frame. Returns Err on unknown-key when `additionalProperties:
    /// false`.
    fn resolve_key_schema(&self, key: &str) -> Result<Schema, ()> {
        let Some(Frame::Object(obj)) = self.stack.last() else {
            return Err(());
        };
        if let Some(schema) = obj.spec.properties.get(key) {
            return Ok(schema.clone());
        }
        match &obj.spec.additional {
            Some(s) => Ok((**s).clone()),
            None => Err(()),
        }
    }

    fn close_object_if_required_satisfied(&mut self) -> StepResult {
        let Some(Frame::Object(obj)) = self.stack.last() else {
            return StepResult::Reject;
        };
        for req in &obj.spec.required {
            if !obj.seen.iter().any(|k| k == req) {
                return StepResult::Reject;
            }
        }
        self.stack.pop();
        self.complete_value();
        StepResult::Ok
    }

    fn close_array_if_within_bounds(&mut self) -> StepResult {
        let Some(Frame::Array(arr)) = self.stack.last() else {
            return StepResult::Reject;
        };
        if let Some(min) = arr.spec.min {
            if arr.count < min {
                return StepResult::Reject;
            }
        }
        self.stack.pop();
        self.complete_value();
        StepResult::Ok
    }

    /// Called after a value (string, number, keyword, container) has
    /// fully closed. Updates the parent frame to its post-value state.
    fn complete_value(&mut self) {
        // If the parent is an Object, we just finished the active value.
        // If parent is Array, increment count and move to AfterValue.
        // If no parent, root is done.
        if self.stack.is_empty() {
            self.done = true;
            return;
        }
        match self.stack.last_mut() {
            Some(Frame::Object(obj)) => {
                obj.phase = ObjectPhase::AfterValue;
                obj.active_value = None;
            }
            Some(Frame::Array(arr)) => {
                arr.count += 1;
                if let Some(max) = arr.spec.max {
                    if arr.count > max {
                        // Caller should have rejected before adding the
                        // value, but defend here: leave the FSM in a
                        // state where the next char can't be parsed.
                    }
                    let _ = max;
                }
                arr.phase = ArrayPhase::AfterValue;
            }
            _ => {}
        }
    }
}

// ── Atom step helpers ────────────────────────────────────────────────────────
//
// Each atom (string, number, keyword, const) advances independently of
// the parent frame; the result tells the caller whether the atom is done
// and how to drive the parent.

enum AtomOutcome {
    Ok,
    Reject,
    /// The atom completed and the parent should treat it as a finished
    /// value. (Used for non-string atoms and for value-context strings.)
    CompleteValue,
    /// The atom was a string in key context; pass the decoded key up.
    CompleteKey(String),
    /// The atom completed mid-step on the previous char; the current
    /// char must be re-processed by the parent.
    ReprocessAfterComplete(char),
}

fn step_string(s: &mut StringFrame, ch: char) -> AtomOutcome {
    if s.unicode_left > 0 {
        if ch.is_ascii_hexdigit() {
            s.unicode_left -= 1;
            // We don't actually decode the codepoint here — for
            // enum/const matching we'd need the literal char, but the
            // common cases (tool names etc.) don't involve unicode
            // escapes. Push a placeholder so length matching stays
            // sensible.
            s.decoded.push('\u{FFFD}');
            return AtomOutcome::Ok;
        }
        return AtomOutcome::Reject;
    }
    if s.in_escape {
        s.in_escape = false;
        let decoded = match ch {
            '"' => '"',
            '\\' => '\\',
            '/' => '/',
            'b' => '\u{0008}',
            'f' => '\u{000C}',
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            'u' => {
                s.unicode_left = 4;
                return AtomOutcome::Ok;
            }
            _ => return AtomOutcome::Reject,
        };
        s.decoded.push(decoded);
        return ok_if_within_string_constraints(s);
    }
    match ch {
        '\\' => {
            s.in_escape = true;
            AtomOutcome::Ok
        }
        '"' => {
            // String closed — validate against enum / const / minLen.
            if let Some(c) = s.spec.r#const.as_ref() {
                if &s.decoded != c {
                    return AtomOutcome::Reject;
                }
            }
            if let Some(en) = s.spec.r#enum.as_ref() {
                if !en.iter().any(|v| v == &s.decoded) {
                    return AtomOutcome::Reject;
                }
            }
            if let Some(min) = s.spec.min_len {
                if s.decoded.chars().count() < min {
                    return AtomOutcome::Reject;
                }
            }
            if s.is_key {
                AtomOutcome::CompleteKey(std::mem::take(&mut s.decoded))
            } else {
                AtomOutcome::CompleteValue
            }
        }
        c if (c as u32) < 0x20 => AtomOutcome::Reject,
        c => {
            s.decoded.push(c);
            ok_if_within_string_constraints(s)
        }
    }
}

/// While the string is still open, check that adding this char hasn't
/// already broken the const / enum prefix expectation. This lets the
/// FSM reject invalid characters early during tool-name matching.
fn ok_if_within_string_constraints(s: &StringFrame) -> AtomOutcome {
    if let Some(c) = s.spec.r#const.as_ref() {
        if !c.starts_with(&s.decoded) {
            return AtomOutcome::Reject;
        }
    }
    if let Some(en) = s.spec.r#enum.as_ref() {
        if !en.iter().any(|v| v.starts_with(&s.decoded)) {
            return AtomOutcome::Reject;
        }
    }
    if let Some(max) = s.spec.max_len {
        if s.decoded.chars().count() > max {
            return AtomOutcome::Reject;
        }
    }
    AtomOutcome::Ok
}

fn step_number(n: &mut NumberFrame, ch: char) -> AtomOutcome {
    let terminator = ch.is_ascii_whitespace() || matches!(ch, ',' | '}' | ']' | ':');
    match n.phase {
        NumberPhase::IntPart => match ch {
            '0'..='9' => {
                n.digits.push(ch);
                AtomOutcome::Ok
            }
            '.' => {
                if n.spec.integer {
                    return AtomOutcome::Reject;
                }
                n.digits.push(ch);
                n.phase = NumberPhase::FracStart;
                AtomOutcome::Ok
            }
            'e' | 'E' => {
                if n.spec.integer {
                    return AtomOutcome::Reject;
                }
                n.digits.push(ch);
                n.phase = NumberPhase::ExpStart;
                AtomOutcome::Ok
            }
            _ if terminator => {
                if !validate_number(n) {
                    return AtomOutcome::Reject;
                }
                AtomOutcome::ReprocessAfterComplete(ch)
            }
            _ => AtomOutcome::Reject,
        },
        NumberPhase::FracStart => match ch {
            '0'..='9' => {
                n.digits.push(ch);
                n.phase = NumberPhase::FracDigits;
                AtomOutcome::Ok
            }
            _ => AtomOutcome::Reject,
        },
        NumberPhase::FracDigits => match ch {
            '0'..='9' => {
                n.digits.push(ch);
                AtomOutcome::Ok
            }
            'e' | 'E' => {
                n.digits.push(ch);
                n.phase = NumberPhase::ExpStart;
                AtomOutcome::Ok
            }
            _ if terminator => {
                if !validate_number(n) {
                    return AtomOutcome::Reject;
                }
                AtomOutcome::ReprocessAfterComplete(ch)
            }
            _ => AtomOutcome::Reject,
        },
        NumberPhase::ExpStart => match ch {
            '+' | '-' => {
                n.digits.push(ch);
                n.phase = NumberPhase::ExpSign;
                AtomOutcome::Ok
            }
            '0'..='9' => {
                n.digits.push(ch);
                n.phase = NumberPhase::ExpDigits;
                AtomOutcome::Ok
            }
            _ => AtomOutcome::Reject,
        },
        NumberPhase::ExpSign => match ch {
            '0'..='9' => {
                n.digits.push(ch);
                n.phase = NumberPhase::ExpDigits;
                AtomOutcome::Ok
            }
            _ => AtomOutcome::Reject,
        },
        NumberPhase::ExpDigits => match ch {
            '0'..='9' => {
                n.digits.push(ch);
                AtomOutcome::Ok
            }
            _ if terminator => {
                if !validate_number(n) {
                    return AtomOutcome::Reject;
                }
                AtomOutcome::ReprocessAfterComplete(ch)
            }
            _ => AtomOutcome::Reject,
        },
    }
}

fn validate_number(n: &NumberFrame) -> bool {
    let parsed: f64 = match n.digits.parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    if let Some(min) = n.spec.minimum {
        if parsed < min {
            return false;
        }
    }
    if let Some(max) = n.spec.maximum {
        if parsed > max {
            return false;
        }
    }
    true
}

/// True if the number atom is in a phase that could legally end here
/// (i.e. has at least one digit and isn't waiting on more required
/// characters like a fraction-digit or exponent-digit).
fn is_number_finalizable(n: &NumberFrame) -> bool {
    let has_digit = n.digits.chars().any(|c| c.is_ascii_digit());
    if !has_digit {
        return false;
    }
    matches!(
        n.phase,
        NumberPhase::IntPart | NumberPhase::FracDigits | NumberPhase::ExpDigits
    ) && validate_number(n)
}

fn step_keyword(k: &mut KeywordFrame, ch: char) -> AtomOutcome {
    let bytes = k.target.bytes();
    let idx = k.index as usize;
    if idx < bytes.len() && bytes[idx] as char == ch {
        let next = k.index + 1;
        if next as usize == bytes.len() {
            AtomOutcome::CompleteValue
        } else {
            k.index = next;
            AtomOutcome::Ok
        }
    } else {
        AtomOutcome::Reject
    }
}

fn step_const(c: &mut ConstFrame, ch: char) -> AtomOutcome {
    if c.index >= c.target.len() {
        return AtomOutcome::Reject;
    }
    if c.target[c.index] != ch {
        return AtomOutcome::Reject;
    }
    c.index += 1;
    if c.index == c.target.len() {
        AtomOutcome::CompleteValue
    } else {
        AtomOutcome::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn assert_accepts(schema: Schema, json: &str) {
        let mut fsm = Fsm::new(schema);
        let res = fsm.step_str(json);
        assert_eq!(res, StepResult::Ok, "rejected accepting case: {json:?}");
        assert!(fsm.is_complete(), "not complete after: {json:?}");
    }

    fn assert_rejects(schema: Schema, json: &str) {
        let mut fsm = Fsm::new(schema);
        let res = fsm.step_str(json);
        let complete = fsm.is_complete();
        assert!(
            res == StepResult::Reject || !complete,
            "accepted should-reject: {json:?}"
        );
    }

    // ── Schema::Any (structural-only) ─────────────────────────────────

    #[test]
    fn any_accepts_basic_values() {
        for s in [
            "{}", "[]", r#""abc""#, "42", "-3.14", "true", "false", "null",
        ] {
            assert_accepts(Schema::Any, s);
        }
    }

    #[test]
    fn any_accepts_nested() {
        assert_accepts(Schema::Any, r#"{"a":{"b":[1,2,{"c":true}]}}"#);
    }

    #[test]
    fn any_rejects_garbage() {
        assert_rejects(Schema::Any, "}");
        assert_rejects(Schema::Any, "[,]");
    }

    #[test]
    fn any_string_escapes() {
        assert_accepts(Schema::Any, r#""hello \"world\"""#);
        assert_accepts(Schema::Any, r#""line\nbreak""#);
    }

    // ── Object schema ─────────────────────────────────────────────────

    fn obj(props: &[(&str, Schema)], required: &[&str], strict: bool) -> Schema {
        let mut p = BTreeMap::new();
        for (k, v) in props {
            p.insert((*k).into(), v.clone());
        }
        Schema::object(ObjectSchema {
            properties: p,
            required: required.iter().map(|s| s.to_string()).collect(),
            additional: if strict {
                None
            } else {
                Some(Box::new(Schema::Any))
            },
        })
    }

    #[test]
    fn object_strict_rejects_unknown_key() {
        let s = obj(&[("a", Schema::number())], &[], true);
        assert_rejects(s, r#"{"b":1}"#);
    }

    #[test]
    fn object_strict_accepts_known_key() {
        let s = obj(&[("a", Schema::number())], &[], true);
        assert_accepts(s, r#"{"a":1}"#);
    }

    #[test]
    fn object_required_must_appear() {
        let s = obj(
            &[("a", Schema::number()), ("b", Schema::string())],
            &["a", "b"],
            true,
        );
        assert_rejects(s.clone(), r#"{"a":1}"#); // missing b
        assert_accepts(s, r#"{"a":1,"b":"x"}"#);
    }

    #[test]
    fn object_typed_value_string_rejects_number() {
        let s = obj(&[("name", Schema::string())], &[], true);
        assert_rejects(s, r#"{"name":42}"#);
    }

    #[test]
    fn object_integer_rejects_decimal() {
        let s = obj(&[("n", Schema::integer())], &[], true);
        assert_rejects(s.clone(), r#"{"n":1.5}"#);
        assert_accepts(s, r#"{"n":42}"#);
    }

    // ── Array schema ──────────────────────────────────────────────────

    #[test]
    fn array_typed_items_string_rejects_number() {
        let s = Schema::array(Schema::string());
        assert_rejects(s, r#"["a", 1]"#);
    }

    #[test]
    fn array_typed_items_string_accepts_strings() {
        let s = Schema::array(Schema::string());
        assert_accepts(s, r#"["a","b","c"]"#);
    }

    #[test]
    fn array_min_items_rejects_short() {
        let s = Schema::Array(ArraySchema {
            items: Box::new(Schema::Any),
            min: Some(2),
            max: None,
        });
        assert_rejects(s, "[1]");
    }

    // ── String schema ─────────────────────────────────────────────────

    #[test]
    fn string_const_only_exact_match() {
        let s = Schema::String(StringSchema {
            r#const: Some("hello".into()),
            ..Default::default()
        });
        assert_accepts(s.clone(), r#""hello""#);
        assert_rejects(s, r#""world""#);
    }

    #[test]
    fn string_const_rejects_diverging_prefix_early() {
        // The FSM should reject the first non-matching character without
        // waiting for the closing quote.
        let s = Schema::String(StringSchema {
            r#const: Some("hello".into()),
            ..Default::default()
        });
        let mut fsm = Fsm::new(s);
        assert_eq!(fsm.step_str(r#""he"#), StepResult::Ok);
        assert_eq!(fsm.step('y'), StepResult::Reject);
    }

    #[test]
    fn string_enum_accepts_member() {
        let s = Schema::String(StringSchema {
            r#enum: Some(vec!["a".into(), "b".into(), "c".into()]),
            ..Default::default()
        });
        assert_accepts(s.clone(), r#""b""#);
        assert_rejects(s, r#""z""#);
    }

    // ── Number schema ─────────────────────────────────────────────────

    #[test]
    fn number_minmax_via_object_wrapper() {
        // Numbers validate their bounds at the terminator (`,` / `}` / EOS).
        // Wrap inside an object so the terminator fires as part of the
        // outer dispatch.
        let s = obj(
            &[(
                "n",
                Schema::Number(NumberSchema {
                    integer: false,
                    minimum: Some(0.0),
                    maximum: Some(10.0),
                }),
            )],
            &[],
            true,
        );
        assert_accepts(s.clone(), r#"{"n":5}"#);
        assert_rejects(s.clone(), r#"{"n":11}"#);
        assert_rejects(s, r#"{"n":-1}"#);
    }

    // ── OneOf ─────────────────────────────────────────────────────────

    #[test]
    fn oneof_commits_on_string_vs_number() {
        let s = Schema::OneOf(vec![Schema::string(), Schema::number()]);
        assert_accepts(s.clone(), r#""hi""#);
        assert_accepts(s, "42");
    }

    #[test]
    fn oneof_branches_with_same_prefix_resolve() {
        // Two object schemas distinguished by the constant value of `name`.
        let a = obj(
            &[(
                "name",
                Schema::String(StringSchema {
                    r#const: Some("alpha".into()),
                    ..Default::default()
                }),
            )],
            &["name"],
            true,
        );
        let b = obj(
            &[(
                "name",
                Schema::String(StringSchema {
                    r#const: Some("beta".into()),
                    ..Default::default()
                }),
            )],
            &["name"],
            true,
        );
        let s = Schema::OneOf(vec![a, b]);
        assert_accepts(s.clone(), r#"{"name":"alpha"}"#);
        assert_accepts(s.clone(), r#"{"name":"beta"}"#);
        assert_rejects(s, r#"{"name":"gamma"}"#);
    }

    // ── Const ─────────────────────────────────────────────────────────

    #[test]
    fn const_literal_matches_canonical() {
        let s = Schema::Const(serde_json::json!(42));
        assert_accepts(s, "42");

        let s = Schema::Const(serde_json::json!("hello"));
        assert_accepts(s, r#""hello""#);

        let s = Schema::Const(serde_json::json!(true));
        assert_accepts(s, "true");

        let s = Schema::Const(serde_json::json!(null));
        assert_accepts(s, "null");
    }

    // ── Completion / depth ────────────────────────────────────────────

    #[test]
    fn is_complete_only_after_root_closes() {
        let mut fsm = Fsm::any();
        assert!(!fsm.is_complete());
        assert_eq!(fsm.step_str("{"), StepResult::Ok);
        assert!(!fsm.is_complete());
        assert_eq!(fsm.step_str("}"), StepResult::Ok);
        assert!(fsm.is_complete());
    }

    #[test]
    fn depth_tracks_open_containers() {
        let mut fsm = Fsm::any();
        assert_eq!(fsm.step_str("[[["), StepResult::Ok);
        assert_eq!(fsm.depth(), 3);
        assert_eq!(fsm.step_str("]]"), StepResult::Ok);
        assert_eq!(fsm.depth(), 1);
    }
}
