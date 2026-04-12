//! Lifecycle statement parsers: EXTRACT, COMPILE, DIFF, USE

use crate::ast::*;
use crate::lexer::Keyword;
use super::{Parser, ParseError};

impl Parser {
    pub(crate) fn parse_extract(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Extract)?;
        self.expect_keyword(Keyword::Model)?;
        let model = self.expect_string()?;
        self.expect_keyword(Keyword::Into)?;
        let output = self.expect_string()?;

        let mut components = None;
        let mut layers = None;
        let mut extract_level = ExtractLevel::Browse;

        loop {
            match self.peek() {
                crate::lexer::Token::Keyword(Keyword::Components) => {
                    self.advance();
                    components = Some(self.parse_component_list()?);
                }
                crate::lexer::Token::Keyword(Keyword::Layers) => {
                    self.advance();
                    layers = Some(self.parse_range()?);
                }
                crate::lexer::Token::Keyword(Keyword::With) => {
                    self.advance();
                    // WITH INFERENCE | WITH ALL | WITH WEIGHTS (legacy)
                    if self.check_keyword(Keyword::Inference) {
                        self.advance();
                        extract_level = ExtractLevel::Inference;
                    } else if self.check_keyword(Keyword::All) {
                        self.advance();
                        extract_level = ExtractLevel::All;
                    } else {
                        // WITH WEIGHTS is legacy — maps to Inference
                        self.expect_keyword(Keyword::Weights)?;
                        extract_level = ExtractLevel::Inference;
                    }
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Extract { model, output, components, layers, extract_level })
    }

    pub(crate) fn parse_compile(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Compile)?;
        let vindex = self.parse_vindex_ref()?;
        self.expect_keyword(Keyword::Into)?;

        // COMPILE ... INTO MODEL or COMPILE ... INTO VINDEX
        let target = if self.check_keyword(Keyword::Model) {
            self.advance();
            CompileTarget::Model
        } else {
            // Accept "VINDEX" as an identifier (not a keyword)
            match self.peek() {
                crate::lexer::Token::Ident(ref s) if s.eq_ignore_ascii_case("vindex") => {
                    self.advance();
                    CompileTarget::Vindex
                }
                _ => {
                    self.expect_keyword(Keyword::Model)?; // will error with good message
                    CompileTarget::Model
                }
            }
        };

        let output = self.expect_string()?;

        let format = if self.check_keyword(Keyword::Format) {
            self.advance();
            Some(self.parse_output_format()?)
        } else {
            None
        };

        // Trailing clauses for COMPILE INTO VINDEX:
        //   ON CONFLICT {LAST_WINS | HIGHEST_CONFIDENCE | FAIL}
        //   WITH REFINE | WITHOUT REFINE              (default: refine = true)
        //   WITH DECOYS (<prompt>, <prompt>, ...)     (default: none)
        // All three are optional and may appear in any order. Each is
        // restricted to COMPILE INTO VINDEX — applying any of them to
        // COMPILE INTO MODEL is a parse error so users get a clear
        // message instead of silent acceptance.
        let mut on_conflict = None;
        let mut refine = true;
        let mut decoys: Option<Vec<String>> = None;

        loop {
            match self.peek() {
                crate::lexer::Token::Keyword(Keyword::On) => {
                    self.advance();
                    self.expect_keyword(Keyword::Conflict)?;
                    let strat = match self.peek() {
                        crate::lexer::Token::Keyword(Keyword::LastWins) => {
                            self.advance();
                            CompileConflict::LastWins
                        }
                        crate::lexer::Token::Keyword(Keyword::HighestConfidence) => {
                            self.advance();
                            CompileConflict::HighestConfidence
                        }
                        crate::lexer::Token::Keyword(Keyword::Fail) => {
                            self.advance();
                            CompileConflict::Fail
                        }
                        t => return Err(ParseError(format!(
                            "expected LAST_WINS | HIGHEST_CONFIDENCE | FAIL after ON CONFLICT, got {:?}",
                            t
                        ))),
                    };
                    if target != CompileTarget::Vindex {
                        return Err(ParseError(
                            "ON CONFLICT is only valid for COMPILE INTO VINDEX".into(),
                        ));
                    }
                    on_conflict = Some(strat);
                }
                crate::lexer::Token::Keyword(Keyword::With) => {
                    self.advance();
                    match self.peek() {
                        crate::lexer::Token::Keyword(Keyword::Refine) => {
                            self.advance();
                            if target != CompileTarget::Vindex {
                                return Err(ParseError(
                                    "WITH REFINE is only valid for COMPILE INTO VINDEX".into(),
                                ));
                            }
                            refine = true;
                        }
                        crate::lexer::Token::Keyword(Keyword::Decoys) => {
                            self.advance();
                            if target != CompileTarget::Vindex {
                                return Err(ParseError(
                                    "WITH DECOYS is only valid for COMPILE INTO VINDEX".into(),
                                ));
                            }
                            decoys = Some(self.parse_decoy_list()?);
                        }
                        t => return Err(ParseError(format!(
                            "expected REFINE or DECOYS after WITH on COMPILE, got {:?}",
                            t
                        ))),
                    }
                }
                crate::lexer::Token::Keyword(Keyword::Without) => {
                    self.advance();
                    self.expect_keyword(Keyword::Refine)?;
                    if target != CompileTarget::Vindex {
                        return Err(ParseError(
                            "WITHOUT REFINE is only valid for COMPILE INTO VINDEX".into(),
                        ));
                    }
                    refine = false;
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Compile {
            vindex, output, format, target, on_conflict, refine, decoys,
        })
    }

    /// Parse a parenthesised list of decoy prompt strings:
    /// `(<string>, <string>, ...)`. The opening `(` is the next token.
    fn parse_decoy_list(&mut self) -> Result<Vec<String>, ParseError> {
        self.expect_token(&crate::lexer::Token::LParen)?;
        let mut prompts = Vec::new();
        if !matches!(self.peek(), crate::lexer::Token::RParen) {
            loop {
                prompts.push(self.expect_string()?);
                if matches!(self.peek(), crate::lexer::Token::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect_token(&crate::lexer::Token::RParen)?;
        if prompts.is_empty() {
            return Err(ParseError("WITH DECOYS requires at least one prompt".into()));
        }
        Ok(prompts)
    }

    pub(crate) fn parse_diff(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Diff)?;
        let a = self.parse_vindex_ref()?;
        let b = self.parse_vindex_ref()?;

        let mut layer = None;
        let mut relation = None;
        let mut limit = None;

        loop {
            match self.peek() {
                crate::lexer::Token::Keyword(Keyword::Layer) => {
                    self.advance();
                    layer = Some(self.expect_u32()?);
                }
                crate::lexer::Token::Keyword(Keyword::Relation)
                | crate::lexer::Token::Keyword(Keyword::Relations) => {
                    self.advance();
                    relation = Some(self.expect_string()?);
                }
                crate::lexer::Token::Keyword(Keyword::Limit) => {
                    self.advance();
                    limit = Some(self.expect_u32()?);
                }
                crate::lexer::Token::Keyword(Keyword::Into) => {
                    self.advance();
                    self.expect_keyword(Keyword::Patch)?;
                    let path = self.expect_string()?;
                    self.eat_semicolon();
                    return Ok(Statement::Diff { a, b, layer, relation, limit, into_patch: Some(path) });
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Diff { a, b, layer, relation, limit, into_patch: None })
    }

    pub(crate) fn parse_use(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Use)?;

        let target = if self.check_keyword(Keyword::Model) {
            self.advance();
            let id = self.expect_string()?;
            let auto_extract = self.check_keyword(Keyword::AutoExtract);
            if auto_extract {
                self.advance();
            }
            UseTarget::Model { id, auto_extract }
        } else if self.check_keyword(Keyword::Remote) {
            self.advance();
            let url = self.expect_string()?;
            UseTarget::Remote(url)
        } else {
            let path = self.expect_string()?;
            UseTarget::Vindex(path)
        };

        self.eat_semicolon();
        Ok(Statement::Use { target })
    }
}
