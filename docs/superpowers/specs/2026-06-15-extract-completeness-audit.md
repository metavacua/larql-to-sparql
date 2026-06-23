# larql extract → vindex: completeness & resource audit (2026-06-15)

Goal (north star): every supported **architecture × source toolchain** must produce a
**complete** vindex — all declared weight classes populated + required auxiliary files —
**isomorphic** to the HF model (no *silent* drops; any intentional lossy transform documented).

Repo audited: fresh clone `/home/metavacua/larql-extract` @ `origin/main` d9b761f6 (chrishayuk).
Method: 5 parallel read-only subagent audits (safetensors loader, gguf/ggml loader, extract
pipeline completeness, resource/resumability, arch registry + source cmds). Every claim is
file:line-cited in the agent transcripts of session; this doc is the synthesis.

Existing fork issues that this confirms/extends: #147 (silent hollow safetensors), #148 (no
safetensors ternary decoder), #149 (no bitnet.rs arch), #150 (2B+ RAM/time → streaming/resumable),
#151 (clustering silently no-ops), #152 (feature_labels.json no extract writer).

NEW fork issues filed 2026-06-15 (metavacua/larql-to-sparql), each with upstream cross-refs:
- **#153** = A1+A2 GGUF fused 3-D MoE expert drop (intro chrishayuk@c54875db / PR#145,#139; precedent chrishayuk#49)
- **#154** = A3 silent GenericArch fallback (chrishayuk@dccf08d7 / #22)
- **#155** = A9 verify can't detect hollow (chrishayuk@ce3ef489)
- **#156** = B1+B2 resume no-op + dead Weights/Q4kWeights checkpoint (chrishayuk@571e139e, @a0d77d09)
- **#157** = B4+B5+B8 embedding RAM dup + O(layers²) down_meta rewrite (chrishayuk@fd6f0b43/PR#139)
- **#158** = A11+A12+A5 safetensors silent drops: I8 mis-decode, 3-D drop, skipped_tensors unsurfaced (chrishayuk@b8f4bcf4)
- **#159** = C3 phi/phi2/phi3 arch mapping (chrishayuk@dccf08d7)
- **#160** = C4 GGUF Q2_K half-wired (chrishayuk@c54875db)
- **#161** = C5+C6 BitNet ternary follow-ups: TQ1_0 unvalidated + I2_S scaling undoc'd (chrishayuk PR#148)

---

## 1. Real on-disk fixtures (resource & resumability testbed)

| artifact | state | use |
|---|---|---|
| `~/bitnet-gguf.vindex` | **PARTIAL** — `.extract_checkpoint.json completed:["gate"]`, gate_vectors.bin 1.06 GB + embeddings.bin 656 MB present; **died mid-`down_meta`**; no weights/index.json | **resumability + down_meta resource-cost fixture** (source `~/bitnet-gguf/ggml-model-i2_s.gguf` 1.19 GB present) |
| `~/bitnet-default.vindex` | **HOLLOW** — gate/up/down/attn = 0 bytes, embeddings+norms+index.json+tokenizer present | the #147 silent-hollow safetensors symptom, frozen |
| `~/larql-vindexes/smollm2-360m.vindex` | COMPLETE (32 layers, GQA) | known-good reference / regression baseline |

bitnet sources: `~/bitnet-gguf` (gguf i2_s) and `~/bitnet-default` (packed safetensors) — KEEP as fixtures.

---

## 2. Coverage matrix (architecture × source toolchain)

Toolchains: **ST** = safetensors load path; **GGUF** = gguf/ggml path; **HF** = `larql … hf`
(download/publish only, no extract); **CONV** = `larql convert` (gguf/st → build_vindex);
**BUILD** = `larql build` (Vindexfile, not model extract).

- **HF cmd** runs NO extraction (copy + checksum only) — not a completeness surface.
- **BUILD cmd** emits only gate_vectors+down_meta — by design, not a full extract.
- **CONV** = alias for extract (default level `browse` → partial by default).

Per-architecture (registry maps dense classes via trait defaults; MoE/MLA must be overridden):

| arch | dense | MoE experts | MLA | ST | GGUF |
|---|---|---|---|---|---|
| llama, mistral, gemma2/3/4*, granite, starcoder2, gpt2, tinymodel | ✔ | n/a | n/a | ✔ | ✔ (dense) |
| mixtral, qwen(-MoE), gpt_oss, deepseek(v2/v3) | ✔ | ✔ (per-expert keys) | ✔ (deepseek) | ✔ | **✘ fused 3D experts dropped** |
| deepseek_v4 | ✔ | ✔ | partial | browse-tier only | ✘ fused |
| **bitnet** | via *generic* (Llama-shaped) | n/a | n/a | **hollow (ternary undecoded)** | ✔ (TQ2_0/I2_S) |
| **phi/phi2/phi3** | via *generic* | n/a | n/a | generic fallback | generic fallback |
| any unknown model_type | via *generic*, **silently** | dropped if actually MoE | dropped | silent-incomplete | silent-incomplete |

\*gemma3/gemma4 large files sampled, not line-audited end-to-end (high confidence complete).

---

## 3. Consolidated gap list (deduped across agents)

### THEME A — silent incomplete/hollow output reported as success (exit 0)
| # | gap | sev | issue | branch |
|---|---|---|---|---|
| A1 | **GGUF fused 3D MoE expert tensors silently dropped** (`gguf/loader.rs:130 _ => {}`); modern Mixtral/Qwen3-MoE/DeepSeek GGUF lose ALL experts, load returns Ok | **HIGH** | NEW | `fix-gguf-fused-moe-expert-load` |
| A2 | No GGUF→HF key map for `*_exps`/`*_shexp`; arch layer only resolves un-fused per-expert names a GGUF never emits | HIGH | NEW | `gguf-moe-expert-key-normalization` |
| A3 | **Unknown `model_type` → silent `GenericArch` fallback** (`detect/mod.rs:138`); MoE/MLA models become dense-only, no warn/error | **HIGH** | NEW | `feat/strict-arch-fallback-warn` |
| A4 | **No post-extraction completeness validation**; extract returns Ok with 0-byte weights / 0 features | **HIGH** | #147 | `fix-extract-validate-nonempty-output` |
| A5 | `skipped_tensors` collected but never surfaced (safetensors) / hard-coded empty (gguf) — the silent-loss mechanism | HIGH | #147 | `feat/error-on-skipped-tensors` |
| A6 | **Streaming path (ALL safetensors) never runs clustering**; only in-memory/GGUF path clusters → relation_clusters absent silently | HIGH | #151 | `streaming-extract-run-clustering` |
| A7 | MoE models always collect **zero cluster directions** (gate_top gated on `!is_moe`) → clustering early-returns | MED | #151 | `moe-relation-cluster-directions` |
| A8 | `feature_labels.json` has **no extract-path writer** (separate probe pass only); undocumented | MED | #152 | `extract-emit-feature-labels-or-doc` |
| A9 | **`verify` cannot detect hollow/incomplete** — checksum-only; passes 0-byte files; no-ops (exit 0) when checksums absent | HIGH | NEW | `verify-completeness-and-nonempty` |
| A10 | `compute_checksums` has no non-empty assertion — legitimizes 0-byte outputs | MED | #147 | `checksums-reject-empty` |
| A11 | bare I8 weight (no `.scale`) silently mis-decoded as raw signed bytes (safetensors) | MED | NEW | `fix/i8-requires-scale-guard` |
| A12 | 3-D+ non-packed safetensors tensors dropped via `_ => {}` w/o even a skipped record | MED | NEW | `fix/record-dropped-nd-tensors` |

### THEME B — resource (RAM/time) & resumability  (#150)
| # | gap | sev | issue | branch |
|---|---|---|---|---|
| B1 | **`--resume` flag defined but NEVER read** (no-op; auto-resume always on; doc comment misdescribes) | **HIGH** | NEW | `fix-extract-resume-flag-noop` |
| B2 | **`Weights`/`Q4kWeights` checkpoint phases are dead code** — longest/heaviest stage can't resume; OOM/timeout restarts whole weight pass | **HIGH** | #150 | `wire-weights-checkpoint-resume` |
| B3 | GGUF at attention/inference/all or `--quant q4k` → **full in-memory `ModelWeights` load** (not streaming) → OOM on 2B+/6 GB host, ungated | HIGH | #150 | `gguf-streaming-or-memory-gate` |
| B4 | full embedding `[vocab,hidden]` f32 (~2 GB) resident across gate→down_meta; could release/f16/mmap | MED | #150 | `bound-embedding-residency` |
| B5 | `build_whole_word_vocab` allocates a **second embedding-sized array then discards it** (`_ww_*`) — ~2 GB wasted transient | MED | NEW | `drop-dead-whole-word-vocab-in-down-meta` |
| B6 | kquant MoE writer materializes whole layer's expert block + per-expert f32 copies at once | MED | #150 | `stream-kquant-moe-per-expert` |
| B7 | layer-level resume unimplemented (phase-level only) — kill mid-phase wastes hours | MED | #150 | `layer-level-resume-manifest` |
| B8 | down_meta re-serializes whole-model meta after every layer → O(layers²) write volume | LOW | NEW | `down-meta-append-not-rewrite` |

### THEME C — decoder/format coverage
| # | gap | sev | issue | branch |
|---|---|---|---|---|
| C1 | safetensors has **no ternary (BitNet 1.58) decoder**; U8-packed dropped, I8-packed mis-decoded; ternary lives only in `quant/ggml/tq.rs` (GGUF-only) | HIGH | #148 | `feat/safetensors-ternary-decoder` |
| C2 | no `bitnet.rs` arch mapping / no detect arm (falls to generic) | MED | #149 | `feat/bitnet-arch-mapping` |
| C3 | `phi/phi2/phi3` GGUF normalized to `phi` but no detect arm → generic (fused-QKV/partial-rotary unhandled) | MED | NEW | `feat/phi-arch-mapping` |
| C4 | GGUF `Q2_K` half-wired (type-id + name, no decoder/size arm) → whole load errors | MED | NEW | `add-q2k-dequant` |
| C5 | GGUF `TQ1_0` decoder unvalidated vs real BitNet GGUF (round-trip tests `#[ignore]`d) | MED | NEW | `verify-tq1-0-against-bitnet-gguf` |
| C6 | GGUF `I2_S` decodes at unit scale; per-channel `*_sub_norm` scale applied by undocumented downstream contract | LOW (doc) | NEW | `document-i2s-subnorm-scaling` |
| C7 | default extract level `browse` yields partial vindex by default — UX/doc | LOW (doc) | NEW | `docs/extract-level-completeness` |

---

## 4. Recommended branch ordering (independent → parallelizable)

Highest leverage, lowest blast radius first; each is its own branch off fresh `origin/main`,
pushed to `fork`, PR → chrishayuk/larql main (licensing block in PR body; no Co-Authored-By trailer).

1. **A4+A5+A9+A10 — "fail loud, never silent"** guardrail PR (post-extract completeness check +
   surface skipped_tensors + verify completeness). Architecture-agnostic; protects every model.
   This is the keystone — it would have turned both bitnet fixtures into loud errors.
2. **A1+A2 — GGUF fused MoE expert load.** Biggest *coverage* hole (all modern MoE GGUF).
3. **B1+B2 — resume flag + weights checkpoint.** Validate against `~/bitnet-gguf.vindex` partial.
4. **A3 — strict arch fallback** (warn/opt-in on generic for MoE/MLA).
5. **C1(+C2) — safetensors ternary decoder + bitnet arch.** Closes the bitnet-safetensors hollow.
6. **A6/A7, B4/B5, C3/C4 …** as follow-ons.

Open question for B-track: a memory-budget guard (B3) that refuses/streams instead of OOMing —
needs a design decision (hard cap? auto-downgrade level? `--max-ram`?).
