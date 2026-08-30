//! KV-cache construction and growth for the decode path.
//!
//! Split out of `decode/mod.rs`: these six functions answer one question —
//! what shape and capacity does this model's KV cache need — and none of
//! them encode a command buffer. Keeping them beside the token loop made
//! that file the place every KV question also had to be read through.
//!
//! The capacity rule that matters: [`MetalBackend::ensure_kv_cache_for_layers`]
//! sizes each layer by its *own* attention window, which is why it must not
//! route through [`MetalBackend::ensure_kv_cache_for_shapes`] — that one
//! grows every layer to a single `max_seq` and would re-inflate a sliding
//! layer on the first decode step.

use super::*;

impl MetalBackend {
    /// Create a KV cache for decode mode with uniform per-layer dims.
    ///
    /// Production decode/prefill should use [`Self::create_kv_cache_per_layer`]
    /// (or, on the trait surface, `preallocate_kv_cache_per_layer`) so models
    /// like Gemma 4 31B with sliding/global geometry alternation are sized
    /// correctly. This uniform helper is retained for synthetic-architecture
    /// tests and the lazy bootstrap inside `populate_kv_layer`, where the
    /// caller passes a single layer's `(num_kv_heads, head_dim)` and any
    /// subsequent layers are pushed via `kv.layers.push(...)` with their own
    /// per-layer dims.
    pub fn create_kv_cache(
        &self,
        num_layers: usize,
        max_seq: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> ops::kv_cache::KVCache {
        ops::kv_cache::KVCache::new(&self.bufs, num_layers, max_seq, num_kv_heads, head_dim)
    }

    /// Create a KV cache with per-layer shapes for models with asymmetric
    /// attention geometry (Gemma 4 31B sliding=16×256 / global=4×512).
    /// `shapes[i] = (num_kv_heads_i, head_dim_i)` for layer i.
    pub fn create_kv_cache_per_layer(
        &self,
        shapes: &[(usize, usize)],
        max_seq: usize,
    ) -> ops::kv_cache::KVCache {
        ops::kv_cache::KVCache::new_per_layer(&self.bufs, shapes, max_seq)
    }

    pub(crate) fn kv_shapes_for_layers(
        layers: &[larql_compute::FullPipelineLayer<'_>],
    ) -> Vec<(usize, usize)> {
        layers
            .iter()
            .map(|layer| (layer.num_kv_heads, layer.head_dim))
            .collect()
    }

    /// Per-layer capacities for the decode path: **every layer holds the
    /// full `default_capacity`**, windowed or not.
    ///
    /// The window-derived capacity (`kv_capacity_for_window`, i.e.
    /// `window x KV_COMPACTION_SLACK`) is a residency policy that is only
    /// sound where something compacts the cache back below it. On this
    /// path nothing does: `compact_kv_to_window` is called by the
    /// KV-engine `coarse_*` steps alone, while the decode path appends at
    /// row `current_len` and attends absolute rows `[T - window, T)`. A
    /// sliding layer sized to 256 rows was therefore written and read
    /// past its buffer from position 256 onward, by a margin growing with
    /// position — the memory corruption behind #229 (NaN mid-stack, router
    /// ids past `num_experts`, GPU page faults, hangs). Pinned by
    /// `decode_path_sizes_sliding_layers_for_the_full_max_seq_because_it_never_compacts`.
    ///
    /// `layers` is still the input so that a future decode-path compaction
    /// can reintroduce per-layer capacities here, at the one site that
    /// owns the fact.
    pub(crate) fn kv_capacities_for_layers(
        layers: &[larql_compute::FullPipelineLayer<'_>],
        default_capacity: usize,
    ) -> Vec<usize> {
        layers.iter().map(|_| default_capacity).collect()
    }

    /// Ensure a cache sized by each layer's *own* capacity.
    ///
    /// This must not route through [`Self::ensure_kv_cache_for_shapes`]:
    /// that grows every layer to a single `max_seq`, which would
    /// immediately re-inflate a sliding layer that was deliberately
    /// allocated at `SLACK * W` and undo the saving on the first decode
    /// step.
    pub(crate) fn ensure_kv_cache_for_layers<'a>(
        &self,
        cache: &'a mut Option<ops::kv_cache::KVCache>,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        max_seq: usize,
    ) -> &'a mut ops::kv_cache::KVCache {
        let shapes = Self::kv_shapes_for_layers(layers);
        let capacities = Self::kv_capacities_for_layers(layers, max_seq);

        let needs_rebuild = cache
            .as_ref()
            .is_none_or(|kv| kv.has_shape_mismatch(&shapes));
        if needs_rebuild {
            *cache = Some(ops::kv_cache::KVCache::new_per_layer_with_capacities(
                &self.bufs,
                &shapes,
                &capacities,
                max_seq,
            ));
        }
        let kv = cache.as_mut().expect("KV cache initialized above");
        kv.grow_to_capacities(&self.bufs, &shapes, &capacities, max_seq);
        kv
    }

    pub(crate) fn ensure_kv_cache_for_shapes<'a>(
        &self,
        cache: &'a mut Option<ops::kv_cache::KVCache>,
        shapes: &[(usize, usize)],
        max_seq: usize,
    ) -> &'a mut ops::kv_cache::KVCache {
        let needs_rebuild = cache
            .as_ref()
            .is_none_or(|kv| kv.has_shape_mismatch(shapes));

        if needs_rebuild {
            *cache = Some(self.create_kv_cache_per_layer(shapes, max_seq));
        }

        let kv = cache.as_mut().expect("KV cache initialized above");
        kv.grow_to_shapes(&self.bufs, shapes, max_seq);
        kv
    }
}
