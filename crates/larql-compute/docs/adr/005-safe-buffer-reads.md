# ADR-005: Consolidated Safe Buffer Read Function

**Status**: Accepted  
**Date**: 2026-04-06  
**Context**: 13 separate `unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }` sites across 8 Metal dispatch files. Each relied on command buffer completion for safety.

## Decision

Replace all 13 sites with one audited function:

```rust
pub fn read_buffer_f32(buf: &metal::Buffer, len: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    assert!(!ptr.is_null(), "Metal buffer contents pointer is null");
    assert!(buf.length() as usize >= len * std::mem::size_of::<f32>());
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}
```

## Origin

Original LARQL design. Standard Rust safety pattern for Metal buffer access.

## Consequences

- Single audit point for all GPU → CPU data transfer
- Null pointer check catches Metal allocation failures
- Size check prevents buffer overread
- Immediately copies to Vec — no dangling reference risk
- The `unsafe` keyword appears in exactly one place in the Metal backend
- Two `copy_nonoverlapping` calls (KV cache population) retained with `// SAFETY:` comments
