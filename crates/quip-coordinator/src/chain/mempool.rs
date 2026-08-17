//! Mempool job-order types.
//!
//! > **CONFIRM crate API:** order discovery via block events + storage
//! > query through `quip-protocol-rs`. The decode of on-chain `JobOrder`
//! > is the CONFIRM point; conversion to wire `Job` is groundable.

/// A mempool job order ready for conversion to a wire `Job`.
#[derive(Debug, Clone)]
pub struct JobOrder {
    /// Opaque order id (typically LE-encoded `u64` from chain storage).
    pub order_id: Vec<u8>,
    /// Topology node ids.
    pub nodes: Vec<u32>,
    /// Topology undirected edges as `(u, v)` node-id pairs.
    pub edges: Vec<(u32, u32)>,
    /// Per-node linear fields in milli units (aligned with `nodes`).
    pub h_milli: Vec<i32>,
    /// Per-edge couplings in milli units (aligned with `edges`).
    pub j_milli: Vec<i32>,
    /// Optional energy gate (milli); `None` when the order has no min energy.
    pub min_energy_milli: Option<i64>,
    /// Optional diversity gate (milli).
    pub min_diversity_milli: Option<u32>,
    /// Optional minimum valid-solution count.
    pub min_solutions: Option<u32>,
    /// Soft deadline in milliseconds from a block-time estimate.
    pub deadline_ms: u64,
}
