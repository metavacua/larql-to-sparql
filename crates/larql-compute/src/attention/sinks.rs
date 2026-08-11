//! Resolving a layer's attention sinks from loaded weights.
//!
//! Kept separate from [`super::softmax`], which is pure maths with no
//! knowledge of weight storage, and separate from the four kernels that
//! need it — prefill, the two decode dispatch paths, and the cached
//! k-quant decode step. Those four would otherwise each carry their own
//! copy of the lookup and its length check, which is the duplication
//! that produced two of the bugs found on this rung.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Resolve a layer's per-head attention sinks, if the architecture has them.
///
/// Returns `None` when the architecture declares no sink key (every
/// architecture except GPT-OSS) or when the key is absent from the
/// loaded vectors — the latter happens with a vindex extracted before
/// sinks were written, which must degrade to an ordinary softmax rather
/// than fail to load.
///
/// `lookup` takes a key and returns the matching vector, if loaded. Takes a
/// closure rather than `&ModelWeights::vectors` directly so this function
/// doesn't need to name that map's concrete type — `larql-models`' own
/// portable `HashMap` alias uses a per-crate `FnvHasher` marker type on
/// wasm32 (duplicated per crate, since Rust modules don't cross crate
/// boundaries), which is a distinct type from this crate's own alias even
/// though both are structurally identical hashbrown maps.
///
/// # Panics
///
/// If the sink vector's length differs from the layer's query-head
/// count. That is not a recoverable condition: it means the extracted
/// tensor does not describe this model, and silently truncating or
/// zero-padding would produce a plausible-looking but wrong forward
/// pass — the exact failure mode this whole area was fixed to remove.
pub fn resolve<'a>(
    sinks_key: Option<String>,
    lookup: impl FnOnce(&str) -> Option<&'a [f32]>,
    num_q: usize,
    layer: usize,
) -> Option<&'a [f32]> {
    let key = sinks_key?;
    let v = lookup(&key)?;
    assert_eq!(
        v.len(),
        num_q,
        "layer {layer}: attention-sink vector `{key}` has {} entries but the layer has \
         {num_q} query heads — the extracted tensor does not match this model",
        v.len()
    );
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn vectors(entries: &[(&str, Vec<f32>)]) -> HashMap<String, Vec<f32>> {
        entries
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    fn lookup(v: &HashMap<String, Vec<f32>>) -> impl Fn(&str) -> Option<&[f32]> + '_ {
        move |k| v.get(k).map(|x| x.as_slice())
    }

    #[test]
    fn no_key_means_no_sinks() {
        let v = vectors(&[("layers.0.self_attn.sinks", vec![1.0, 2.0])]);
        assert!(resolve(None, lookup(&v), 2, 0).is_none());
    }

    #[test]
    fn missing_tensor_degrades_to_none() {
        // A vindex extracted before sinks were written: the architecture
        // names the key, the bytes are not there. Must load and run.
        let v = vectors(&[("layers.0.self_attn.q_proj.bias", vec![0.0; 4])]);
        assert!(resolve(Some("layers.0.self_attn.sinks".into()), lookup(&v), 4, 0).is_none());
    }

    #[test]
    fn present_tensor_is_returned() {
        let v = vectors(&[("layers.3.self_attn.sinks", vec![2.45, -1.0, 8.19, 0.0])]);
        let got = resolve(Some("layers.3.self_attn.sinks".into()), lookup(&v), 4, 3);
        assert_eq!(got, Some([2.45f32, -1.0, 8.19, 0.0].as_slice()));
    }

    #[test]
    #[should_panic(expected = "does not match this model")]
    fn wrong_length_panics_rather_than_truncating() {
        let v = vectors(&[("layers.0.self_attn.sinks", vec![1.0, 2.0])]);
        let _ = resolve(Some("layers.0.self_attn.sinks".into()), lookup(&v), 64, 0);
    }

    #[test]
    #[should_panic(expected = "layer 7")]
    fn panic_names_the_offending_layer() {
        let v = vectors(&[("layers.7.self_attn.sinks", vec![1.0; 8])]);
        let _ = resolve(Some("layers.7.self_attn.sinks".into()), lookup(&v), 64, 7);
    }
}
