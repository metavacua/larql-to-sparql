# Trace Store Format Specification

**Version:** 0.1  
**Date:** 2026-04-02  
**Status:** Implemented  
**Implementation:** `larql-inference` crate, `trace` module (Rust)  
**Companion specs:** [Vindex Format](vindex-format-spec.md)

**Implementation coverage:** All three file formats (chain store, boundary store, context store) fully implemented with mmap'd reading, append-only writing, and zero-copy vector access.

---

## 1. Overview

Trace stores persist the internal activations captured during inference. Instead of discarding intermediate residuals, attention deltas, and FFN deltas after each forward pass, they are written to append-only binary files that can be mmap'd for later analysis.

Three formats exist at different granularity/size trade-offs:

| Format | Extension | Magic | Purpose | Per-token cost (gemma3-4b) |
|--------|-----------|-------|---------|---------------------------|
| Chain store | `.bin` | `TRAC` | Full layer-by-layer trace for every token | ~1.05 MB |
| Boundary store | `.bndx` | `BNDX` | One residual per window boundary | ~10 KB per boundary |
| Context store | `.ctxt` | `CTXT` | Tiered boundary data (residual + critical deltas) | 10-70 KB per boundary |

All three formats share common properties:

- **Append-only:** New data extends the file; existing data is never modified.
- **Mmap'd reads:** Files are memory-mapped with `MADV_RANDOM` on Unix. The OS pages in data on demand. RSS equals only the actively accessed regions.
- **Zero-copy:** Vector reads return `&[f32]` slices directly into mmap'd memory. No heap allocation, no copy.
- **Little-endian:** All multi-byte integers and floats are stored little-endian.
- **f32 only:** All vectors are stored as f32 (4 bytes per float). No f16 option for trace stores.

---

## 2. Chain Store (.bin, magic "TRAC")

The chain store records a complete activation trace for every token position. Each token produces a "chain" of vectors covering every layer waypoint (embedding + all transformer layers), with three vectors per waypoint: the residual stream, attention delta, and FFN delta.

### 2.1 File Layout

```
+--------------------------------------------------+
| Header                           64 bytes         |
+--------------------------------------------------+
| Token chain 0                    chain_size bytes  |
+--------------------------------------------------+
| Token chain 1                    chain_size bytes  |
+--------------------------------------------------+
| ...                                               |
+--------------------------------------------------+
| Token chain N-1                  chain_size bytes  |
+--------------------------------------------------+
```

Total file size = 64 + n_tokens * chain_size

### 2.2 Header (64 bytes)

```
Offset  Size  Type      Field          Description
------  ----  --------  ----------     -----------
0       4     [u8; 4]   magic          "TRAC" (0x54 0x52 0x41 0x43)
4       4     u32       version        Format version (currently 1)
8       4     u32       hidden_size    Dimension of each vector
12      4     u32       n_layers       Number of transformer layers (not counting embedding)
16      4     u32       n_tokens       Number of complete token chains stored
20      44    [u8; 44]  _reserved      Zero-padded, reserved for future use
```

The header is exactly 64 bytes. The `#[repr(C)]` layout ensures no padding surprises.

### 2.3 Token Chain Structure

Each token chain stores activations for (n_layers + 1) waypoints. The "+1" accounts for the embedding layer (layer -1 in logical indexing, stored as waypoint 0).

Each waypoint contains 3 vectors of `hidden_size` f32 floats:

| Vector | Index | Description |
|--------|-------|-------------|
| residual | 0 | Residual stream state at this waypoint |
| attn_delta | 1 | Change applied by the attention sublayer |
| ffn_delta | 2 | Change applied by the FFN sublayer |

**Chain size formula:**

```
n_waypoints = n_layers + 1
vectors_per_waypoint = 3
chain_size = n_waypoints * vectors_per_waypoint * hidden_size * 4  (bytes)
```

**Per-chain byte layout:**

```
Offset (relative to chain start)    Content
---------------------------------    -------
0                                    Waypoint 0 (embedding):
                                       [0 .. H*4)           residual  (H floats)
                                       [H*4 .. 2*H*4)       attn_delta (H floats)
                                       [2*H*4 .. 3*H*4)     ffn_delta  (H floats)

3*H*4                                Waypoint 1 (transformer layer 0):
                                       [3*H*4 .. 4*H*4)     residual
                                       [4*H*4 .. 5*H*4)     attn_delta
                                       [5*H*4 .. 6*H*4)     ffn_delta

...

(W-1)*3*H*4                         Waypoint W-1 (transformer layer n_layers-1):
                                       residual, attn_delta, ffn_delta
```

Where H = hidden_size, W = n_waypoints.

### 2.4 Random Access

To read a specific vector:

```
chain_offset  = HEADER_SIZE + token * chain_size
waypoint_off  = layer * 3 * hidden_size * 4
vector_off    = component * hidden_size * 4       // component: 0=residual, 1=attn, 2=ffn
byte_offset   = chain_offset + waypoint_off + vector_off
```

The result is a contiguous `hidden_size * 4` byte region, reinterpreted as `&[f32]` directly from mmap'd memory.

### 2.5 Layer Indexing Convention

The store uses physical waypoint indices (0-based). The logical layer mapping is:

| Waypoint index | Logical layer | Content |
|---------------|---------------|---------|
| 0 | -1 (embedding) | Post-embedding residual |
| 1 | 0 | After transformer layer 0 |
| 2 | 1 | After transformer layer 1 |
| ... | ... | ... |
| n_layers | n_layers - 1 | After final transformer layer |

When reading via `node()`, the store converts back: `logical_layer = waypoint_index - 1`.

---

## 3. Boundary Store (.bndx, magic "BNDX")

The boundary store exploits the Markov property of transformers: the residual at a window boundary is the complete compressed state of all preceding tokens. By storing one residual per window boundary instead of full traces, long sequences can be represented at dramatically lower cost.

### 3.1 File Layout

```
+--------------------------------------------------+
| Header                           64 bytes         |
+--------------------------------------------------+
| Index entries                    N_max * 16 bytes  |
|   (pre-allocated for max_boundaries entries)      |
+--------------------------------------------------+
| Residual data                    variable          |
|   (appended as boundaries are added)              |
+--------------------------------------------------+
```

The index region is pre-allocated at creation time to avoid shifting data when new boundaries are appended. Default pre-allocation: 10,000 entries (enough for ~2M tokens at window_size=200).

### 3.2 Header (64 bytes)

```
Offset  Size  Type      Field          Description
------  ----  --------  ----------     -----------
0       4     [u8; 4]   magic          "BNDX" (0x42 0x4E 0x44 0x58)
4       4     u32       version        Format version (currently 1)
8       4     u32       hidden_size    Dimension of residual vectors
12      4     u32       window_size    Tokens per window
16      4     u32       n_boundaries   Number of stored boundaries
20      4     u32       total_tokens   Total tokens processed so far
24      40    [u8; 40]  _reserved      Zero-padded, reserved for future use
```

### 3.3 Boundary Index Entry (16 bytes)

The index begins immediately after the header (offset 64). Each entry is 16 bytes:

```
Offset  Size  Type   Field           Description
------  ----  -----  ----------      -----------
0       4     u32    token_offset    Token position where this window starts
4       4     u32    window_tokens   Number of tokens in this window
8       4     u32    data_offset     Absolute byte offset to the residual data
12      4     u32    _reserved       Zero-padded
```

The `data_offset` field is an absolute byte offset into the file (not relative to the data section), pointing to the start of the residual vector for this boundary.

### 3.4 Residual Data

Each boundary stores one residual vector: `hidden_size` contiguous f32 values (little-endian).

```
Residual size = hidden_size * 4  bytes
```

Residual data is appended to the end of the file as boundaries are added. The pre-allocated index region sits between the header and the first residual, so:

```
First residual offset = HEADER_SIZE + max_boundaries * ENTRY_SIZE
                      = 64 + max_boundaries * 16
```

### 3.5 Append Protocol

When appending a new boundary:

1. Seek to end of file; record position as `data_pos`.
2. Write the residual vector (`hidden_size * 4` bytes).
3. Write the index entry at `HEADER_SIZE + boundary_idx * 16`.
4. Update `n_boundaries` and `total_tokens` in the header (seek to offset 0, rewrite 64 bytes).
5. Flush.

The header rewrite is the only mutation of existing data. Index entries and residual data are write-once.

---

## 4. Context Store (.ctxt, magic "CTXT")

The context store extends the boundary store with a tiered system. Instead of storing only the boundary residual, it can optionally store FFN and attention deltas at "critical layers" — the layers most important for knowledge retrieval and reconstruction. This enables partial or full reconstruction without replaying the forward pass.

### 4.1 Tiers

| Tier | Value | Stored per boundary | Vectors | Description |
|------|-------|---------------------|---------|-------------|
| Residual | 1 | Boundary residual | 1 | Needs forward pass replay for reconstruction |
| FfnDeltas | 2 | + FFN deltas at critical layers | 1 + n_critical | Partial reconstruction, no replay for knowledge queries |
| Full | 3 | + attention deltas at critical layers | 1 + 2 * n_critical | Full reconstruction, zero replay cost |

The tier is fixed at file creation time. All boundaries in a file use the same tier.

Critical layers are specified at creation time (up to 8, stored in the header). These are typically the knowledge-band layers identified during vindex extraction (e.g., layers 14-27 for gemma3-4b, subsampled to at most 8).

### 4.2 File Layout

```
+--------------------------------------------------+
| Header                          128 bytes         |
+--------------------------------------------------+
| Index entries                   N_max * 24 bytes   |
|   (pre-allocated for max_boundaries entries)      |
+--------------------------------------------------+
| Vector data                     variable           |
|   (appended as boundaries are added)              |
+--------------------------------------------------+
```

### 4.3 Header (128 bytes)

```
Offset  Size  Type              Field             Description
------  ----  ----------------  ---------------   -----------
0       4     [u8; 4]           magic             "CTXT" (0x43 0x54 0x58 0x54)
4       4     u32               version           Format version (currently 1)
8       4     u32               hidden_size       Dimension of each vector
12      4     u32               n_layers          Number of transformer layers
16      4     u32               window_size       Tokens per window
20      1     u8                tier              Storage tier (1, 2, or 3)
21      1     u8                n_critical        Number of critical layers (0-8)
22      2     [u8; 2]           _pad              Alignment padding
24      8     [u8; 8]           critical_layers   Critical layer indices (one byte each)
32      4     u32               n_boundaries      Number of stored boundaries
36      4     u32               total_tokens      Total tokens processed
40      88    [u8; 88]          _reserved         Zero-padded, reserved for future use
```

The `critical_layers` array stores up to 8 layer indices as individual bytes. Only the first `n_critical` entries are meaningful. This limits critical layer indices to 0-255, which covers all current architectures.

### 4.4 Context Index Entry (24 bytes)

The index begins at offset 128. Each entry is 24 bytes:

```
Offset  Size  Type   Field           Description
------  ----  -----  ----------      -----------
0       4     u32    token_offset    Token position where this window starts
4       4     u32    window_tokens   Number of tokens in this window
8       8     u64    data_offset     Absolute byte offset to this boundary's vectors
16      8     u64    _reserved       Zero-padded
```

Note: the context store uses `u64` for `data_offset` (vs `u32` in the boundary store), supporting files larger than 4 GB.

### 4.5 Per-Boundary Vector Layout

Within each boundary's data region, vectors are stored contiguously in this order:

**Tier 1 (Residual):**
```
[residual]                              1 vector
```

**Tier 2 (FfnDeltas):**
```
[residual]                              1 vector
[ffn_delta_0]                           critical layer 0 FFN delta
[ffn_delta_1]                           critical layer 1 FFN delta
...
[ffn_delta_{n_critical-1}]              last critical layer FFN delta
                                        ─────────────────────────────
                                        1 + n_critical vectors total
```

**Tier 3 (Full):**
```
[residual]                              1 vector
[ffn_delta_0]                           critical layer 0 FFN delta
[ffn_delta_1]                           critical layer 1 FFN delta
...
[ffn_delta_{n_critical-1}]              last critical layer FFN delta
[attn_delta_0]                          critical layer 0 attention delta
[attn_delta_1]                          critical layer 1 attention delta
...
[attn_delta_{n_critical-1}]             last critical layer attention delta
                                        ─────────────────────────────────
                                        1 + 2*n_critical vectors total
```

Each vector is `hidden_size * 4` bytes (f32, little-endian).

**Bytes per boundary:**

```
bytes_per_boundary = vectors_per_boundary * hidden_size * 4
```

### 4.6 Vector Access

To read a specific vector within a boundary:

```
residual:    data_offset + 0
ffn_delta i: data_offset + (1 + i) * hidden_size * 4
attn_delta i: data_offset + (1 + n_critical + i) * hidden_size * 4
```

All reads return `&[f32]` slices from mmap'd memory. The tier is checked before access: requesting an FFN delta from a Tier 1 file returns `None`.

---

## 5. Mmap Alignment and Access Patterns

### 5.1 Alignment

All headers are sized to be multiples of common alignment boundaries:
- Chain store header: 64 bytes (cache-line aligned)
- Boundary store header: 64 bytes
- Context store header: 128 bytes (two cache lines)

Vector data is f32-aligned (4-byte) by construction. Because headers are 64 or 128 bytes (both multiples of 4), and all size calculations multiply by `sizeof(f32)` = 4, vector data is always naturally aligned for f32 access.

### 5.2 madvise Strategy

All three stores use `MADV_RANDOM` on Unix systems. This is appropriate because:

- Trace analysis typically accesses arbitrary token positions (not sequential scans).
- Boundary lookups jump to specific windows based on token offset.
- The OS should not prefetch adjacent pages, as the next access is unpredictable.

### 5.3 Memory Residency

Only accessed pages consume physical RAM. For a chain store with 1000 tokens, accessing one token's chain pages in ~1 MB. The remaining ~1 GB stays on disk until needed.

### 5.4 Concurrent Access

The mmap is read-only (`Mmap`, not `MmapMut`). Multiple readers can safely share the same file. The writer uses file I/O (not mmap) and updates the header token count last, so a reader that opened before the write sees the old count and never reads partial data.

---

## 6. Version Compatibility

All three formats use version 1. The version field is checked on open; mismatched versions produce an error.

| Format | Current version | Magic | Header size |
|--------|----------------|-------|-------------|
| Chain store | 1 | `TRAC` | 64 bytes |
| Boundary store | 1 | `BNDX` | 64 bytes |
| Context store | 1 | `CTXT` | 128 bytes |

Each header includes reserved bytes (44, 40, and 88 bytes respectively) for forward-compatible additions. Future versions can use reserved space without changing the header size, allowing older readers to open newer files (they ignore the reserved region).

---

## 7. Example Sizes for Gemma3-4b (34 layers, 2560 hidden)

### 7.1 Chain Store

```
n_waypoints       = 34 + 1 = 35
vectors_per_chain = 35 * 3 = 105
chain_size        = 105 * 2560 * 4 = 1,075,200 bytes (~1.05 MB)
```

| Tokens | File size | Description |
|--------|-----------|-------------|
| 1 | 1.05 MB | Single token trace |
| 100 | 105 MB | Short prompt |
| 1,000 | 1.05 GB | Medium document |
| 10,000 | 10.5 GB | Long document |

The chain store grows linearly at ~1 MB per token. It is intended for detailed analysis of short sequences, not long-context storage.

### 7.2 Boundary Store

```
residual_size = 2560 * 4 = 10,240 bytes (~10 KB)
window_size   = 200 tokens (typical)
```

| Total tokens | Boundaries | Index size | Data size | Total file size |
|-------------|------------|-----------|-----------|-----------------|
| 10,000 | 50 | 800 B | 500 KB | ~661 KB |
| 100,000 | 500 | 8 KB | 5 MB | ~5.2 MB |
| 370,000 | 1,850 | 29 KB | 18.5 MB | ~18.7 MB |
| 1,000,000 | 5,000 | 78 KB | 50 MB | ~50.2 MB |

With default max_boundaries=10,000, pre-allocated index = 160 KB.

**Comparison:** KV cache for 370K tokens at gemma3-4b = ~56 GB. Boundary store = ~18.7 MB. Compression ratio: **~3,000x**.

### 7.3 Context Store

Assuming 4 critical layers (e.g., layers 14, 18, 22, 26):

| Tier | Vectors/boundary | Bytes/boundary | 370K tokens (1,850 boundaries) |
|------|-----------------|----------------|-------------------------------|
| 1 (Residual) | 1 | 10,240 (10 KB) | 18.5 MB |
| 2 (FfnDeltas) | 5 | 51,200 (50 KB) | 92.5 MB |
| 3 (Full) | 9 | 92,160 (90 KB) | 166.5 MB |

With 8 critical layers:

| Tier | Vectors/boundary | Bytes/boundary | 370K tokens (1,850 boundaries) |
|------|-----------------|----------------|-------------------------------|
| 1 (Residual) | 1 | 10 KB | 18.5 MB |
| 2 (FfnDeltas) | 9 | 90 KB | 163 MB |
| 3 (Full) | 17 | 170 KB | 308 MB |

**Comparison to KV cache (56 GB):**
- Tier 1: 18.5 MB = **3,000x** compression (needs replay)
- Tier 2 (4 critical): 92.5 MB = **600x** compression (partial reconstruction)
- Tier 3 (4 critical): 166.5 MB = **336x** compression (full, no replay)

With default max_boundaries=10,000, pre-allocated index = 240 KB.

---

## 8. Format Comparison

| Property | Chain Store | Boundary Store | Context Store |
|----------|------------|----------------|---------------|
| Extension | `.bin` | `.bndx` | `.ctxt` |
| Magic | `TRAC` | `BNDX` | `CTXT` |
| Header size | 64 B | 64 B | 128 B |
| Index | None (implicit) | 16 B/entry | 24 B/entry |
| Granularity | Every token, every layer | One residual per window | Tiered per window |
| Growth | ~1 MB/token | ~10 KB/boundary | 10-170 KB/boundary |
| Use case | Detailed analysis | Long-context compression | Tunable reconstruction |
| Max tokens (u32) | ~4B | ~4B | ~4B |
| Max file size | Limited by chain count (u32) | Limited by data_offset (u32 = 4 GB) | u64 data_offset = unlimited |

The chain store has no explicit index -- token chains are located by arithmetic from the header fields. The boundary and context stores use explicit index entries because residual data is appended at variable file positions (after the pre-allocated index region).

---

## 9. Write Protocol

All three formats follow the same append protocol:

1. **Seek to end** of file to find the data write position.
2. **Write vector data** (residual, deltas as applicable).
3. **Write index entry** (boundary/context stores only) at the pre-allocated slot.
4. **Update header** counters (seek to offset 0, rewrite header).
5. **Flush** to ensure durability.

The header update is always last. This means a crash between steps 2-3 leaves orphaned data but a consistent header. On next open, the token/boundary count in the header is authoritative.

---

## License

Apache-2.0
