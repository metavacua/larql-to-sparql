//! Arithmetic coding: encoder/decoder, bit IO, and the `.shannon` container.

use super::*;

pub(super) fn run_encode(args: EncodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.vindex.is_some() {
        return run_encode_vindex(args);
    }
    if args.context < 1 {
        return Err("--context must be at least 1".into());
    }
    let text = read_text(&args.input, args.bytes)?;
    let model = load_model(&args.model)?;
    let ids = encode_prompt(model.tokenizer(), &*model.weights().arch, &text)?;
    if ids.len() < 2 {
        return Err("input must tokenize to at least one encoded token".into());
    }

    eprintln!(
        "encoding {} bytes as {} target tokens...",
        text.len(),
        ids.len() - 1
    );
    let pb = progress_bar((ids.len() - 1) as u64, "encoding");
    let mut encoder = ArithmeticEncoder::new();
    for pos in 1..ids.len() {
        let prefix_start = pos.saturating_sub(args.context);
        let logits = logits_for_last_token(model.weights(), &ids[prefix_start..pos])?;
        let counts = quantized_counts(&logits)?;
        let (low, high) = interval_for_symbol(&counts, ids[pos])?;
        encoder.encode(low, high, FREQ_TOTAL);
        pb.inc(1);
    }
    pb.finish_and_clear();

    let payload = encoder.finish();
    let blob = ShannonFile {
        context: args.context as u32,
        first_token: ids[0],
        target_tokens: (ids.len() - 1) as u64,
        original_bytes: text.len() as u64,
        payload,
    };
    let bytes = blob.to_bytes();
    fs::write(&args.out, &bytes)?;

    let chars = text.chars().count().max(1) as f64;
    println!("original:        {:>10} bytes", text.len());
    println!("payload:         {:>10} bytes", blob.payload.len());
    println!("file:            {:>10} bytes", bytes.len());
    println!("tokens:          {:>10}", ids.len() - 1);
    println!(
        "ratio(payload):  {:>10.2}x",
        text.len() as f64 / blob.payload.len().max(1) as f64
    );
    println!(
        "bits/char:       {:>10.3}",
        blob.payload.len() as f64 * 8.0 / chars
    );
    println!("wrote: {}", args.out.display());
    Ok(())
}

pub(super) fn run_decode(args: DecodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.vindex.is_some() {
        return run_decode_vindex(args);
    }
    let mut raw = Vec::new();
    fs::File::open(&args.input)?.read_to_end(&mut raw)?;
    let blob = ShannonFile::from_bytes(&raw)?;
    if blob.context < 1 {
        return Err("compressed file has invalid context".into());
    }

    let model = load_model(&args.model)?;
    let mut decoder = ArithmeticDecoder::new(&blob.payload);
    let mut ids = Vec::with_capacity(blob.target_tokens as usize + 1);
    ids.push(blob.first_token);

    eprintln!("decoding {} target tokens...", blob.target_tokens);
    let pb = progress_bar(blob.target_tokens, "decoding");
    for _ in 0..blob.target_tokens {
        let prefix_start = ids.len().saturating_sub(blob.context as usize);
        let logits = logits_for_last_token(model.weights(), &ids[prefix_start..])?;
        let counts = quantized_counts(&logits)?;
        let value = decoder.scaled_value(FREQ_TOTAL);
        let (symbol, low, high) = symbol_for_value(&counts, value)?;
        decoder.decode(low, high, FREQ_TOTAL);
        ids.push(symbol);
        pb.inc(1);
    }
    pb.finish_and_clear();

    let text = model
        .tokenizer()
        .decode(&ids, true)
        .map_err(|e| format!("decode error: {e}"))?;
    fs::write(&args.out, text.as_bytes())?;
    println!("decoded:         {:>10} bytes", text.len());
    println!("expected:        {:>10} bytes", blob.original_bytes);
    println!("wrote: {}", args.out.display());
    Ok(())
}

pub(super) fn quantized_counts(logits: &[f32]) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    if logits.len() >= FREQ_TOTAL as usize {
        return Err("vocab is too large for arithmetic coder frequency total".into());
    }
    let max_logit = finite_max(logits)?;
    let exp_values: Vec<f64> = logits
        .iter()
        .map(|&v| {
            if v.is_finite() {
                ((v - max_logit) as f64).exp()
            } else {
                0.0
            }
        })
        .collect();
    let exp_sum: f64 = exp_values.iter().sum();
    if exp_sum <= 0.0 {
        return Err("invalid probability distribution".into());
    }
    let spare = FREQ_TOTAL as usize - logits.len();
    let mut max_idx = 0usize;
    let mut max_exp = f64::NEG_INFINITY;
    let mut sum = 0u32;
    let mut counts = Vec::with_capacity(logits.len());
    for (i, exp_v) in exp_values.iter().copied().enumerate() {
        if exp_v > max_exp {
            max_exp = exp_v;
            max_idx = i;
        }
        let count = 1 + (exp_v / exp_sum * spare as f64).floor() as u32;
        sum = sum.saturating_add(count);
        counts.push(count);
    }
    if sum > FREQ_TOTAL {
        return Err("frequency quantization overflowed".into());
    }
    counts[max_idx] += FREQ_TOTAL - sum;
    Ok(counts)
}

pub(super) fn interval_for_symbol(
    counts: &[u32],
    symbol: u32,
) -> Result<(u32, u32), Box<dyn std::error::Error>> {
    let symbol = symbol as usize;
    if symbol >= counts.len() {
        return Err(format!("symbol {symbol} out of frequency table").into());
    }
    let low: u32 = counts[..symbol].iter().sum();
    let high = low + counts[symbol];
    Ok((low, high))
}

pub(super) fn symbol_for_value(
    counts: &[u32],
    value: u32,
) -> Result<(u32, u32, u32), Box<dyn std::error::Error>> {
    let mut low = 0u32;
    for (symbol, &count) in counts.iter().enumerate() {
        let high = low + count;
        if value < high {
            return Ok((symbol as u32, low, high));
        }
        low = high;
    }
    Err("arithmetic decoder value outside frequency table".into())
}

pub(super) struct BitWriter {
    pub(super) bytes: Vec<u8>,
    pub(super) current: u8,
    pub(super) used: u8,
}

impl BitWriter {
    pub(super) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            used: 0,
        }
    }

    pub(super) fn write(&mut self, bit: bool) {
        self.current = (self.current << 1) | u8::from(bit);
        self.used += 1;
        if self.used == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.used = 0;
        }
    }

    pub(super) fn finish(mut self) -> Vec<u8> {
        if self.used > 0 {
            self.current <<= 8 - self.used;
            self.bytes.push(self.current);
        }
        self.bytes
    }
}

pub(super) struct BitReader<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) byte_idx: usize,
    pub(super) bit_idx: u8,
}

pub(super) struct ArithmeticEncoder {
    pub(super) low: u64,
    pub(super) high: u64,
    pub(super) pending: u64,
    pub(super) bits: BitWriter,
}

impl ArithmeticEncoder {
    pub(super) fn new() -> Self {
        Self {
            low: 0,
            high: TOP_VALUE,
            pending: 0,
            bits: BitWriter::new(),
        }
    }

    pub(super) fn encode(&mut self, cum_low: u32, cum_high: u32, total: u32) {
        let range = self.high - self.low + 1;
        self.high = self.low + (range * cum_high as u64) / total as u64 - 1;
        self.low += (range * cum_low as u64) / total as u64;

        loop {
            if self.high < HALF {
                self.output_bit_plus_follow(false);
            } else if self.low >= HALF {
                self.output_bit_plus_follow(true);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= FIRST_QTR && self.high < THIRD_QTR {
                self.pending += 1;
                self.low -= FIRST_QTR;
                self.high -= FIRST_QTR;
            } else {
                break;
            }
            self.low *= 2;
            self.high = self.high * 2 + 1;
        }
    }

    pub(super) fn finish(mut self) -> Vec<u8> {
        self.pending += 1;
        if self.low < FIRST_QTR {
            self.output_bit_plus_follow(false);
        } else {
            self.output_bit_plus_follow(true);
        }
        self.bits.finish()
    }

    pub(super) fn output_bit_plus_follow(&mut self, bit: bool) {
        self.bits.write(bit);
        for _ in 0..self.pending {
            self.bits.write(!bit);
        }
        self.pending = 0;
    }
}

pub(super) struct ArithmeticDecoder<'a> {
    pub(super) low: u64,
    pub(super) high: u64,
    pub(super) code: u64,
    pub(super) bits: BitReader<'a>,
}

pub(super) struct ShannonFile {
    pub(super) context: u32,
    pub(super) first_token: u32,
    pub(super) target_tokens: u64,
    pub(super) original_bytes: u64,
    pub(super) payload: Vec<u8>,
}

#[derive(Clone)]
pub(super) struct VindexShannonBlock {
    pub(super) first_token: u32,
    pub(super) target_tokens: u64,
    pub(super) payload: Vec<u8>,
}

impl ShannonFile {
    pub(super) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(36 + self.payload.len());
        out.extend_from_slice(b"LSC1");
        out.extend_from_slice(&self.context.to_le_bytes());
        out.extend_from_slice(&self.first_token.to_le_bytes());
        out.extend_from_slice(&self.target_tokens.to_le_bytes());
        out.extend_from_slice(&self.original_bytes.to_le_bytes());
        out.extend_from_slice(&(self.payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.payload);
        out
    }

    pub(super) fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        if bytes.len() < 36 || &bytes[..4] != b"LSC1" {
            return Err("not a LARQL Shannon compressed file".into());
        }
        let context = u32::from_le_bytes(bytes[4..8].try_into()?);
        let first_token = u32::from_le_bytes(bytes[8..12].try_into()?);
        let target_tokens = u64::from_le_bytes(bytes[12..20].try_into()?);
        let original_bytes = u64::from_le_bytes(bytes[20..28].try_into()?);
        let payload_len = u64::from_le_bytes(bytes[28..36].try_into()?) as usize;
        if bytes.len() != 36 + payload_len {
            return Err("compressed file payload length mismatch".into());
        }
        Ok(Self {
            context,
            first_token,
            target_tokens,
            original_bytes,
            payload: bytes[36..].to_vec(),
        })
    }
}

pub(super) fn encode_vindex_blocks(blocks: &[VindexShannonBlock]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"LSB1");
    out.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    for block in blocks {
        out.extend_from_slice(&block.first_token.to_le_bytes());
        out.extend_from_slice(&block.target_tokens.to_le_bytes());
        out.extend_from_slice(&(block.payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&block.payload);
    }
    out
}

pub(super) fn parse_vindex_blocks(
    bytes: &[u8],
) -> Result<Option<Vec<VindexShannonBlock>>, Box<dyn std::error::Error>> {
    if !bytes.starts_with(b"LSB1") {
        return Ok(None);
    }
    if bytes.len() < 8 {
        return Err("truncated vindex block payload".into());
    }
    let block_count = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    let mut offset = 8usize;
    let mut blocks = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        if bytes.len().saturating_sub(offset) < 20 {
            return Err("truncated vindex block header".into());
        }
        let first_token = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
        offset += 4;
        let target_tokens = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
        offset += 8;
        let payload_len = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?) as usize;
        offset += 8;
        if bytes.len().saturating_sub(offset) < payload_len {
            return Err("truncated vindex block payload".into());
        }
        blocks.push(VindexShannonBlock {
            first_token,
            target_tokens,
            payload: bytes[offset..offset + payload_len].to_vec(),
        });
        offset += payload_len;
    }
    if offset != bytes.len() {
        return Err("trailing bytes after vindex block payload".into());
    }
    Ok(Some(blocks))
}
