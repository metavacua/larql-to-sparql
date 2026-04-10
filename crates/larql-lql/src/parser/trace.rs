//! Parser for TRACE statements.
//!
//! Grammar:
//!
//! ```text
//! TRACE <prompt>
//!     [FOR <token>]
//!     [DECOMPOSE]
//!     [LAYERS <start>-<end>]
//!     [POSITIONS {LAST | ALL}]
//!     [SAVE <path>]
//! ```
//!
//! `FOR <token>` selects a target token to track through the residual stream
//! (rank, attn delta, ffn delta per layer).

use crate::ast::*;
use crate::lexer::{Keyword, Token};
use super::{Parser, ParseError};

impl Parser {
    pub(crate) fn parse_trace(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Trace)?;
        let prompt = self.expect_string()?;

        let mut answer = None;
        let mut decompose = false;
        let mut layers = None;
        let mut positions = None;
        let mut save = None;

        loop {
            match self.peek() {
                Token::Keyword(Keyword::For) => {
                    self.advance();
                    answer = Some(self.expect_string()?);
                }
                Token::Keyword(Keyword::Decompose) => {
                    self.advance();
                    decompose = true;
                }
                Token::Keyword(Keyword::Layers) => {
                    self.advance();
                    layers = Some(self.parse_range()?);
                }
                Token::Keyword(Keyword::Positions) => {
                    self.advance();
                    match self.peek() {
                        Token::Keyword(Keyword::All) => {
                            self.advance();
                            positions = Some(TracePositionMode::All);
                        }
                        _ => {
                            positions = Some(TracePositionMode::Last);
                        }
                    }
                }
                Token::Keyword(Keyword::Save) => {
                    self.advance();
                    save = Some(self.expect_string()?);
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Trace {
            prompt,
            answer,
            decompose,
            layers,
            positions,
            save,
        })
    }
}
