//! MOSS-TTS-Realtime prompt construction — the `[T, 1+rvq]` matrix the
//! generation loop prefls.
//!
//! Mirrors the reference processor exactly (`MossTTSRealtimeProcessor` in
//! the OpenMOSS repo): system prompt, optional voice-clone context whose
//! `<|audio_pad|>` run is overwritten channel-wise with reference codec
//! tokens, assistant header, then the text lead ([`DELAY_TOKENS_LEN`]
//! ids) with audio BOS on the last lead row. Byte-exactness of the
//! protocol strings is pinned by an ignored parity test against the
//! step-0 dump's prompt matrix — do not "tidy" the whitespace.
//!
//! Temporal policy stops here: this module builds state, the loop
//! consumes it, and nothing below the runtime knows about clocks.

use ndarray::Array2;

use larql_models::detect::ModelError;
use larql_models::speech::moss_tts_realtime::MossTtsRealtimeConfig;

/// How many text ids the prefill consumes before the first frame — the
/// text channel's lead over the audio channels. A protocol constant of
/// the checkpoint family (`delay_tokens_len` in the reference processor);
/// not present in `config.json`, so it is spelled here with that
/// provenance.
pub const DELAY_TOKENS_LEN: usize = 12;

/// The reference's system prompt, byte-exact (trailing space on the
/// second line included).
const SYSTEM_PROMPT: &str = "<|im_start|>system\nYou are a highly expressive text-to-speech (TTS) engine developed by Mosi Intelligence. \nYou possess natural language understanding, emotional modeling, and multi-style speech generation capabilities, allowing you to generate the corresponding speech based on the text given in the assistant.<|im_end|>\n";

const CONTEXT_PREFIX: &str =
    "<|im_start|>context\nThe assistant section should be synthesized using the following voice timbre:";
const CONTEXT_SUFFIX: &str = "<|im_end|>\n";
const ASSISTANT_HEADER: &str = "<|im_start|>assistant\n";
const AUDIO_PAD_TEXT: &str = "<|audio_pad|>";
const TEXT_PAD_TEXT: &str = "<|text_pad|>";

/// A built prompt: everything the batch generation loop needs.
#[derive(Debug)]
pub struct MossPrompt {
    /// `[T, 1+rvq]` — the prefill input, text lead and audio BOS placed.
    pub prefill_matrix: Array2<u32>,
    /// Text ids not consumed by the lead; fed one per frame.
    pub text_queue: Vec<u32>,
    /// Fed once `text_queue` is exhausted.
    pub text_pad_id: u32,
}

/// One turn's prompt *before* any text: system prompt, voice-clone
/// splice, assistant header. An incremental session
/// ([`crate::speech::moss_session::MossSession`]) holds this while the
/// text lead accumulates; [`append_text_lead`] completes it into a
/// prefill matrix.
#[derive(Debug)]
pub struct MossTurn {
    /// `[T, 1+rvq]` — every row up to (excluding) the text lead.
    pub matrix: Array2<u32>,
    /// Fed once the text stream is exhausted.
    pub text_pad_id: u32,
}

/// Build a voice-clone prompt for batch generation. `reference_codes` is
/// `[T, rvq]` (codec tokens of the reference audio); `None` builds the
/// no-clone prompt (the model's unconditioned voice).
pub fn build_prompt(
    tokenizer: &tokenizers::Tokenizer,
    config: &MossTtsRealtimeConfig,
    reference_codes: Option<&Array2<u32>>,
    text: &str,
) -> Result<MossPrompt, ModelError> {
    let turn = build_turn(tokenizer, config, reference_codes)?;
    let text_ids = encode(tokenizer, text)?;
    if text_ids.is_empty() {
        return Err(ModelError::Parse("empty text".into()));
    }
    let lead = text_ids.len().min(DELAY_TOKENS_LEN);
    Ok(MossPrompt {
        prefill_matrix: append_text_lead(&turn.matrix, &text_ids[..lead], config),
        text_queue: text_ids[lead..].to_vec(),
        text_pad_id: turn.text_pad_id,
    })
}

/// Build the turn prompt — everything except the text lead. The batch
/// path ([`build_prompt`]) and the incremental session both start here,
/// which is what makes their prefill matrices identical by construction.
pub fn build_turn(
    tokenizer: &tokenizers::Tokenizer,
    config: &MossTtsRealtimeConfig,
    reference_codes: Option<&Array2<u32>>,
) -> Result<MossTurn, ModelError> {
    let rvq = config.rvq;
    let pad = config.audio_pad_token as u32;
    let audio_pad_id = single_token_id(tokenizer, AUDIO_PAD_TEXT)?;
    let text_pad_id = single_token_id(tokenizer, TEXT_PAD_TEXT)?;

    // ── System (+ voice-clone context), tokenized as ONE string, as the
    // reference does — encoding across the concatenation boundary must
    // match its BPE exactly. ──
    let system_text = match reference_codes {
        Some(codes) => format!(
            "{SYSTEM_PROMPT}{CONTEXT_PREFIX}{}{CONTEXT_SUFFIX}",
            AUDIO_PAD_TEXT.repeat(codes.nrows())
        ),
        None => SYSTEM_PROMPT.to_string(),
    };
    let system_ids = encode(tokenizer, &system_text)?;
    let assistant_ids = encode(tokenizer, ASSISTANT_HEADER)?;

    let rows = system_ids.len() + assistant_ids.len();
    let mut matrix = Array2::<u32>::from_elem((rows, 1 + rvq), pad);
    for (row, &id) in system_ids.iter().chain(assistant_ids.iter()).enumerate() {
        matrix[[row, 0]] = id;
    }

    // ── Voice-clone splice: overwrite channels 1..=rvq of exactly the
    // contiguous <|audio_pad|> rows with the reference codes. ──
    if let Some(codes) = reference_codes {
        if codes.ncols() != rvq {
            return Err(ModelError::Parse(format!(
                "reference codes have {} channels, model expects {rvq}",
                codes.ncols()
            )));
        }
        let pad_rows: Vec<usize> = system_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == audio_pad_id)
            .map(|(row, _)| row)
            .collect();
        if pad_rows.len() != codes.nrows() {
            return Err(ModelError::Parse(format!(
                "tokenized prompt carries {} audio-pad rows for {} reference frames — \
                 the pad text did not tokenize one-to-one",
                pad_rows.len(),
                codes.nrows()
            )));
        }
        for (frame, &row) in pad_rows.iter().enumerate() {
            for channel in 0..rvq {
                matrix[[row, 1 + channel]] = codes[[frame, channel]];
            }
        }
    }

    Ok(MossTurn {
        matrix,
        text_pad_id,
    })
}

/// Complete a turn matrix into a prefill matrix: append one row per lead
/// id (text column set, audio channels pad) and place audio BOS on the
/// last row — §1.4's "BOS sits on channel 1 of the last prefill text
/// token".
pub fn append_text_lead(
    turn_matrix: &Array2<u32>,
    lead: &[u32],
    config: &MossTtsRealtimeConfig,
) -> Array2<u32> {
    assert!(
        !lead.is_empty(),
        "the text lead must carry at least one id — audio BOS has no row otherwise"
    );
    let rvq = config.rvq;
    assert_eq!(turn_matrix.ncols(), 1 + rvq);
    let rows = turn_matrix.nrows() + lead.len();
    let mut matrix = Array2::<u32>::from_elem((rows, 1 + rvq), config.audio_pad_token as u32);
    matrix
        .slice_mut(ndarray::s![..turn_matrix.nrows(), ..])
        .assign(turn_matrix);
    for (offset, &id) in lead.iter().enumerate() {
        matrix[[turn_matrix.nrows() + offset, 0]] = id;
    }
    matrix[[rows - 1, 1]] = config.audio_bos_token() as u32;
    matrix
}

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>, ModelError> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(|e| ModelError::Parse(format!("tokenize failed: {e}")))?
        .get_ids()
        .to_vec())
}

fn single_token_id(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<u32, ModelError> {
    let ids = encode(tokenizer, text)?;
    match ids.as_slice() {
        [id] => Ok(*id),
        other => Err(ModelError::Parse(format!(
            "{text:?} tokenizes to {} ids, expected a single special token",
            other.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speech::test_support::{synth_moss_tokenizer, synth_speech_model};

    /// Reference codes shaped `[frames, rvq]`, values distinct per cell.
    fn reference_codes(frames: usize, rvq: usize) -> Array2<u32> {
        Array2::from_shape_fn((frames, rvq), |(frame, channel)| {
            (frame * rvq + channel) as u32
        })
    }

    #[test]
    fn append_text_lead_places_lead_pad_and_bos() {
        let model = synth_speech_model();
        let config = &model.config;
        let turn = Array2::<u32>::from_elem((3, 1 + config.rvq), config.audio_pad_token as u32);
        let lead = [4u32, 5, 6];
        let matrix = append_text_lead(&turn, &lead, config);

        assert_eq!(matrix.nrows(), turn.nrows() + lead.len());
        // Turn rows carried through untouched.
        assert_eq!(matrix.slice(ndarray::s![..3, ..]), turn);
        // Lead ids on the text column, audio channels at pad.
        for (offset, &id) in lead.iter().enumerate() {
            assert_eq!(matrix[[3 + offset, 0]], id);
        }
        assert_eq!(matrix[[3, 1]], config.audio_pad_token as u32);
        // Audio BOS on channel 1 of the LAST lead row only.
        assert_eq!(matrix[[5, 1]], config.audio_bos_token() as u32);
        assert_eq!(matrix[[4, 1]], config.audio_pad_token as u32);
    }

    #[test]
    #[should_panic(expected = "at least one id")]
    fn append_text_lead_rejects_empty_lead() {
        let model = synth_speech_model();
        let turn = Array2::<u32>::from_elem(
            (2, 1 + model.config.rvq),
            model.config.audio_pad_token as u32,
        );
        append_text_lead(&turn, &[], &model.config);
    }

    #[test]
    fn build_prompt_is_build_turn_plus_lead() {
        // The identity the incremental session relies on: the batch
        // prefill matrix is exactly the turn matrix with the lead
        // appended.
        let model = synth_speech_model();
        let tokenizer = synth_moss_tokenizer();
        let codes = reference_codes(2, model.config.rvq);
        let text = "hello world one two three";

        let prompt = build_prompt(&tokenizer, &model.config, Some(&codes), text).expect("prompt");
        let turn = build_turn(&tokenizer, &model.config, Some(&codes)).expect("turn");
        let text_ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
        let lead = text_ids.len().min(DELAY_TOKENS_LEN);
        let rebuilt = append_text_lead(&turn.matrix, &text_ids[..lead], &model.config);

        assert_eq!(prompt.prefill_matrix, rebuilt);
        assert_eq!(prompt.text_queue, text_ids[lead..].to_vec());
        assert_eq!(prompt.text_pad_id, turn.text_pad_id);
    }

    #[test]
    fn build_turn_splices_reference_codes_channelwise() {
        let model = synth_speech_model();
        let tokenizer = synth_moss_tokenizer();
        let frames = 3;
        let codes = reference_codes(frames, model.config.rvq);
        let turn = build_turn(&tokenizer, &model.config, Some(&codes)).expect("turn");

        let audio_pad_id = tokenizer
            .encode(AUDIO_PAD_TEXT, false)
            .unwrap()
            .get_ids()
            .to_vec();
        let audio_pad_id = audio_pad_id[0];
        let splice_rows: Vec<usize> = (0..turn.matrix.nrows())
            .filter(|&row| turn.matrix[[row, 0]] == audio_pad_id)
            .collect();
        assert_eq!(
            splice_rows.len(),
            frames,
            "one splice row per reference frame"
        );
        for (frame, &row) in splice_rows.iter().enumerate() {
            for channel in 0..model.config.rvq {
                assert_eq!(turn.matrix[[row, 1 + channel]], codes[[frame, channel]]);
            }
        }
        // Non-splice rows keep every audio channel at pad.
        for row in 0..turn.matrix.nrows() {
            if !splice_rows.contains(&row) {
                for channel in 0..model.config.rvq {
                    assert_eq!(
                        turn.matrix[[row, 1 + channel]],
                        model.config.audio_pad_token as u32
                    );
                }
            }
        }
    }

    #[test]
    fn build_turn_rejects_channel_mismatch() {
        let model = synth_speech_model();
        let tokenizer = synth_moss_tokenizer();
        let codes = reference_codes(2, model.config.rvq + 1);
        let error = build_turn(&tokenizer, &model.config, Some(&codes)).unwrap_err();
        assert!(error.to_string().contains("channels"), "got: {error}");
    }

    #[test]
    fn build_prompt_rejects_empty_text() {
        let model = synth_speech_model();
        let tokenizer = synth_moss_tokenizer();
        let error = build_prompt(&tokenizer, &model.config, None, "").unwrap_err();
        assert!(error.to_string().contains("empty text"), "got: {error}");
    }
}
