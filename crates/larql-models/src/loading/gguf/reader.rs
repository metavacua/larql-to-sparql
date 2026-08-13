//! Binary read helpers for the GGUF format.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
use std::io::Read;

use crate::detect::ModelError;

use super::constants::*;
use super::types::GgufValue;

pub(super) fn read_u8(r: &mut impl Read) -> Result<u8, ModelError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

pub(super) fn read_i8(r: &mut impl Read) -> Result<i8, ModelError> {
    Ok(read_u8(r)? as i8)
}

pub(super) fn read_u16(r: &mut impl Read) -> Result<u16, ModelError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

pub(super) fn read_i16(r: &mut impl Read) -> Result<i16, ModelError> {
    Ok(read_u16(r)? as i16)
}

pub(super) fn read_u32(r: &mut impl Read) -> Result<u32, ModelError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub(super) fn read_i32(r: &mut impl Read) -> Result<i32, ModelError> {
    Ok(read_u32(r)? as i32)
}

pub(super) fn read_u64(r: &mut impl Read) -> Result<u64, ModelError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

pub(super) fn read_i64(r: &mut impl Read) -> Result<i64, ModelError> {
    Ok(read_u64(r)? as i64)
}

pub(super) fn read_f32(r: &mut impl Read) -> Result<f32, ModelError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

pub(super) fn read_f64(r: &mut impl Read) -> Result<f64, ModelError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

pub(super) fn read_string(r: &mut impl Read) -> Result<String, ModelError> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| ModelError::Parse(e.to_string()))
}

pub(super) fn read_value(r: &mut impl Read) -> Result<GgufValue, ModelError> {
    let vtype = read_u32(r)?;
    match vtype {
        GGUF_TYPE_UINT8 => Ok(GgufValue::U8(read_u8(r)?)),
        GGUF_TYPE_INT8 => Ok(GgufValue::I8(read_i8(r)?)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::U16(read_u16(r)?)),
        GGUF_TYPE_INT16 => Ok(GgufValue::I16(read_i16(r)?)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::U32(read_u32(r)?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::I32(read_i32(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(read_f32(r)?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        GGUF_TYPE_STRING => Ok(GgufValue::String(read_string(r)?)),
        GGUF_TYPE_UINT64 => Ok(GgufValue::U64(read_u64(r)?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::I64(read_i64(r)?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(read_f64(r)?)),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(r)?;
            let len = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_array_element(r, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        _ => Err(ModelError::Parse(format!(
            "unknown GGUF metadata type: {vtype}"
        ))),
    }
}

pub(super) fn read_array_element(
    r: &mut impl Read,
    elem_type: u32,
) -> Result<GgufValue, ModelError> {
    match elem_type {
        GGUF_TYPE_UINT8 => Ok(GgufValue::U8(read_u8(r)?)),
        GGUF_TYPE_INT8 => Ok(GgufValue::I8(read_i8(r)?)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::U16(read_u16(r)?)),
        GGUF_TYPE_INT16 => Ok(GgufValue::I16(read_i16(r)?)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::U32(read_u32(r)?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::I32(read_i32(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(read_f32(r)?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        GGUF_TYPE_STRING => Ok(GgufValue::String(read_string(r)?)),
        GGUF_TYPE_UINT64 => Ok(GgufValue::U64(read_u64(r)?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::I64(read_i64(r)?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(read_f64(r)?)),
        _ => Err(ModelError::Parse(format!(
            "unknown GGUF array element type: {elem_type}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────
    // Byte-reading helpers + read_value/read_array_element variant
    // coverage. The existing GGUF-builder tests only emit STRING / U32 /
    // FLOAT32 metadata; the read-side dispatch arms for U8, I8, U16,
    // I16, I32, U64, I64, F64, Bool, ARRAY, and the unknown-type error
    // branch are exercised here directly.
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn read_value_dispatches_every_supported_variant() {
        use std::io::Cursor;
        // U8 (tag 0): tag(u32) + 1 byte payload.
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_UINT8.to_le_bytes());
        buf.push(0xAB);
        assert!(matches!(
            read_value(&mut Cursor::new(buf)).unwrap(),
            GgufValue::U8(0xAB)
        ));

        // I8 (tag 1).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_INT8.to_le_bytes());
        buf.push(0xFFu8); // -1 as i8
        assert!(matches!(
            read_value(&mut Cursor::new(buf)).unwrap(),
            GgufValue::I8(-1)
        ));

        // U16 (tag 2).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_UINT16.to_le_bytes());
        buf.extend_from_slice(&12345u16.to_le_bytes());
        assert!(matches!(
            read_value(&mut Cursor::new(buf)).unwrap(),
            GgufValue::U16(12345)
        ));

        // I16 (tag 3).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_INT16.to_le_bytes());
        buf.extend_from_slice(&(-7i16).to_le_bytes());
        assert!(matches!(
            read_value(&mut Cursor::new(buf)).unwrap(),
            GgufValue::I16(-7)
        ));

        // I32 (tag 5).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_INT32.to_le_bytes());
        buf.extend_from_slice(&(-65_536i32).to_le_bytes());
        assert!(matches!(
            read_value(&mut Cursor::new(buf)).unwrap(),
            GgufValue::I32(-65_536)
        ));

        // BOOL (tag 7): tag + 1 byte (0 = false, nonzero = true).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_BOOL.to_le_bytes());
        buf.push(1u8);
        assert!(matches!(
            read_value(&mut Cursor::new(buf)).unwrap(),
            GgufValue::Bool(true)
        ));
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_BOOL.to_le_bytes());
        buf.push(0u8);
        assert!(matches!(
            read_value(&mut Cursor::new(buf)).unwrap(),
            GgufValue::Bool(false)
        ));

        // U64 (tag 10).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_UINT64.to_le_bytes());
        buf.extend_from_slice(&(u64::MAX - 3).to_le_bytes());
        let v = read_value(&mut Cursor::new(buf)).unwrap();
        match v {
            GgufValue::U64(x) => assert_eq!(x, u64::MAX - 3),
            other => panic!("expected U64, got {other:?}"),
        }

        // I64 (tag 11).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_INT64.to_le_bytes());
        buf.extend_from_slice(&(-9_999_999i64).to_le_bytes());
        let v = read_value(&mut Cursor::new(buf)).unwrap();
        match v {
            GgufValue::I64(x) => assert_eq!(x, -9_999_999),
            other => panic!("expected I64, got {other:?}"),
        }

        // F64 (tag 12).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_FLOAT64.to_le_bytes());
        buf.extend_from_slice(&std::f64::consts::PI.to_le_bytes());
        let v = read_value(&mut Cursor::new(buf)).unwrap();
        match v {
            GgufValue::F64(x) => {
                assert!((x - std::f64::consts::PI).abs() < 1e-12);
            }
            other => panic!("expected F64, got {other:?}"),
        }
    }

    #[test]
    fn read_value_array_recurses_through_read_array_element() {
        use std::io::Cursor;
        // Array of 3 U32 values.
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        for v in [10u32, 20, 30] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        match read_value(&mut Cursor::new(buf)).unwrap() {
            GgufValue::Array(elems) => {
                assert_eq!(elems.len(), 3);
                assert!(matches!(elems[0], GgufValue::U32(10)));
                assert!(matches!(elems[1], GgufValue::U32(20)));
                assert!(matches!(elems[2], GgufValue::U32(30)));
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    #[test]
    fn read_array_element_dispatches_every_supported_variant() {
        use std::io::Cursor;

        type VariantCase = (u32, Vec<u8>, fn(GgufValue));
        let cases: &[VariantCase] = &[
            (GGUF_TYPE_UINT8, vec![0x42], |v| {
                assert!(matches!(v, GgufValue::U8(0x42)))
            }),
            (GGUF_TYPE_INT8, vec![0xFE], |v| {
                assert!(matches!(v, GgufValue::I8(-2)))
            }),
            (GGUF_TYPE_UINT16, 500u16.to_le_bytes().to_vec(), |v| {
                assert!(matches!(v, GgufValue::U16(500)))
            }),
            (GGUF_TYPE_INT16, (-9i16).to_le_bytes().to_vec(), |v| {
                assert!(matches!(v, GgufValue::I16(-9)))
            }),
            (GGUF_TYPE_UINT32, 7u32.to_le_bytes().to_vec(), |v| {
                assert!(matches!(v, GgufValue::U32(7)))
            }),
            (GGUF_TYPE_INT32, (-77_777i32).to_le_bytes().to_vec(), |v| {
                assert!(matches!(v, GgufValue::I32(-77_777)))
            }),
            (
                GGUF_TYPE_FLOAT32,
                2.5f32.to_le_bytes().to_vec(),
                |v| match v {
                    GgufValue::F32(x) => assert_eq!(x, 2.5),
                    other => panic!("expected F32, got {other:?}"),
                },
            ),
            (GGUF_TYPE_BOOL, vec![1u8], |v| {
                assert!(matches!(v, GgufValue::Bool(true)))
            }),
            (
                GGUF_TYPE_UINT64,
                12345u64.to_le_bytes().to_vec(),
                |v| match v {
                    GgufValue::U64(x) => assert_eq!(x, 12345),
                    other => panic!("expected U64, got {other:?}"),
                },
            ),
            (
                GGUF_TYPE_INT64,
                (-1234i64).to_le_bytes().to_vec(),
                |v| match v {
                    GgufValue::I64(x) => assert_eq!(x, -1234),
                    other => panic!("expected I64, got {other:?}"),
                },
            ),
            (
                GGUF_TYPE_FLOAT64,
                1.5f64.to_le_bytes().to_vec(),
                |v| match v {
                    GgufValue::F64(x) => assert_eq!(x, 1.5),
                    other => panic!("expected F64, got {other:?}"),
                },
            ),
        ];

        for (tag, bytes, check) in cases {
            let v = read_array_element(&mut Cursor::new(bytes.clone()), *tag).unwrap();
            check(v);
        }
    }

    #[test]
    fn read_array_element_string_variant() {
        use std::io::Cursor;
        let mut buf = Vec::new();
        buf.extend_from_slice(&6u64.to_le_bytes());
        buf.extend_from_slice(b"hello!");
        match read_array_element(&mut Cursor::new(buf), GGUF_TYPE_STRING).unwrap() {
            GgufValue::String(s) => assert_eq!(s, "hello!"),
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn read_value_unknown_metadata_type_errors() {
        use std::io::Cursor;
        let mut buf = Vec::new();
        buf.extend_from_slice(&999u32.to_le_bytes());
        match read_value(&mut Cursor::new(buf)) {
            Err(ModelError::Parse(msg)) => {
                assert!(msg.contains("unknown GGUF metadata type"), "got: {msg}");
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn read_array_element_unknown_type_errors() {
        use std::io::Cursor;
        match read_array_element(&mut Cursor::new(Vec::new()), 9999) {
            Err(ModelError::Parse(msg)) => {
                assert!(
                    msg.contains("unknown GGUF array element type"),
                    "got: {msg}"
                );
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn read_string_rejects_non_utf8() {
        use std::io::Cursor;
        let mut buf = Vec::new();
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&[0xFF, 0xFE, 0xFD]); // invalid UTF-8
        match read_string(&mut Cursor::new(buf)) {
            Err(ModelError::Parse(_)) => {}
            other => panic!("expected Parse error, got {other:?}"),
        }
    }
}
