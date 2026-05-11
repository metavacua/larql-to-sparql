//! Stage 4 — tokenizer.

use crate::error::VindexError;
use crate::extract::stage_labels::*;
use crate::extract::streaming::context::StreamingContext;
use crate::format::filenames::*;

impl<'a> StreamingContext<'a> {
    /// Stage 4 — tokenizer.
    pub(in crate::extract::streaming) fn write_tokenizer(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_TOKENIZER);
        let tokenizer_json = self
            .tokenizer
            .to_string(true)
            .map_err(|e| VindexError::Parse(format!("tokenizer serialize: {e}")))?;
        std::fs::write(self.output_dir.join(TOKENIZER_JSON), tokenizer_json)?;
        self.callbacks.on_stage_done(STAGE_TOKENIZER, 0.0);
        Ok(())
    }
}
