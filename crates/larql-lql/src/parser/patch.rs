/// Patch statement parsers: BEGIN PATCH, SAVE PATCH, APPLY PATCH, SHOW PATCHES, REMOVE PATCH.

use crate::ast::*;
use crate::lexer::Keyword;
use super::{Parser, ParseError};

impl Parser {
    /// Parse a statement starting with BEGIN (BEGIN PATCH "file.vlp").
    pub(crate) fn parse_begin(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Begin)?;
        self.expect_keyword(Keyword::Patch)?;
        let path = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::BeginPatch { path })
    }

    /// Parse SAVE PATCH.
    pub(crate) fn parse_save(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Save)?;
        self.expect_keyword(Keyword::Patch)?;
        self.eat_semicolon();
        Ok(Statement::SavePatch)
    }

    /// Parse APPLY PATCH "file.vlp".
    pub(crate) fn parse_apply(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Apply)?;
        self.expect_keyword(Keyword::Patch)?;
        let path = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::ApplyPatch { path })
    }

    /// Parse REMOVE PATCH "file.vlp".
    pub(crate) fn parse_remove(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Remove)?;
        self.expect_keyword(Keyword::Patch)?;
        let path = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::RemovePatch { path })
    }
}
