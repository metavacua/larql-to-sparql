/// Introspection statement parsers: SHOW (RELATIONS, LAYERS, FEATURES, MODELS), STATS.

use crate::ast::*;
use crate::lexer::{Keyword, Token};
use super::{Parser, ParseError};

impl Parser {
    pub(crate) fn parse_show(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Show)?;

        match self.peek() {
            Token::Keyword(Keyword::Relations) => {
                self.advance();
                let mut layer = None;
                let mut with_examples = false;
                let mut mode = DescribeMode::default();

                loop {
                    match self.peek() {
                        Token::Keyword(Keyword::At) => {
                            self.advance();
                            self.expect_keyword(Keyword::Layer)?;
                            layer = Some(self.expect_u32()?);
                        }
                        Token::Keyword(Keyword::With) => {
                            self.advance();
                            self.expect_keyword(Keyword::Examples)?;
                            with_examples = true;
                        }
                        Token::Keyword(Keyword::Verbose) => {
                            self.advance();
                            mode = DescribeMode::Verbose;
                        }
                        Token::Keyword(Keyword::Brief) => {
                            self.advance();
                            mode = DescribeMode::Brief;
                        }
                        Token::Keyword(Keyword::Raw) => {
                            self.advance();
                            mode = DescribeMode::Raw;
                        }
                        _ => break,
                    }
                }
                self.eat_semicolon();
                Ok(Statement::ShowRelations { layer, with_examples, mode })
            }
            Token::Keyword(Keyword::Layers) => {
                self.advance();
                let range = if self.check_keyword(Keyword::Range) {
                    self.advance();
                    Some(self.parse_range()?)
                } else if matches!(self.peek(), Token::IntegerLit(_)) {
                    Some(self.parse_range()?)
                } else {
                    None
                };
                self.eat_semicolon();
                Ok(Statement::ShowLayers { range })
            }
            Token::Keyword(Keyword::Features) => {
                self.advance();
                let layer = self.expect_u32()?;

                let conditions = if self.check_keyword(Keyword::Where) {
                    self.advance();
                    self.parse_conditions()?
                } else {
                    vec![]
                };

                let limit = if self.check_keyword(Keyword::Limit) {
                    self.advance();
                    Some(self.expect_u32()?)
                } else {
                    None
                };

                self.eat_semicolon();
                Ok(Statement::ShowFeatures { layer, conditions, limit })
            }
            Token::Keyword(Keyword::Models) => {
                self.advance();
                self.eat_semicolon();
                Ok(Statement::ShowModels)
            }
            Token::Keyword(Keyword::Patches) => {
                self.advance();
                self.eat_semicolon();
                Ok(Statement::ShowPatches)
            }
            _ => Err(ParseError(format!(
                "expected RELATIONS, LAYERS, FEATURES, MODELS, or PATCHES after SHOW, got {:?}",
                self.peek()
            ))),
        }
    }

    pub(crate) fn parse_stats(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Stats)?;
        let vindex = if let Token::StringLit(_) = self.peek() {
            Some(self.expect_string()?)
        } else {
            None
        };
        self.eat_semicolon();
        Ok(Statement::Stats { vindex })
    }
}
