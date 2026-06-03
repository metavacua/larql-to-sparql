/// Native adapter for larql-bridge traits.
/// Implements WeightProvider, KvStore, ExpertDispatch, HttpFetch using native APIs.
/// Phase B will fill in the implementations when the boundary is empirically defined.

pub struct NativeWeightProvider;
pub struct NativeKvStore;
pub struct NativeExpertDispatch;
pub struct NativeHttpFetch;
