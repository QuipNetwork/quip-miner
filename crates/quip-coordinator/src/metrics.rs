//! Live coordinator counters and identity for the `/api/v1` dashboard surface.
//!
//! The dashboard indexer reads seven controller counters, the chain account the
//! coordinator signs with, and the miner roster. None of those live in
//! [`crate::session::CoordinatorState`], and the HTTP handlers must not take the
//! coordinator mutex: an HTTP request would then queue behind the result path.
//!
//! So the counters are atomics and the two views are short-lived
//! `std::sync::Mutex` reads. A handler clones what it needs and drops the guard
//! before it builds any JSON, so no guard is ever held across an await.
//!
//! Counters are monotonic for the life of the process and reset to `0` on
//! restart, which is what the dashboard expects.

use crate::chain::MinerKind;
use crate::config::miner_kind_from_backend;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// The seven controller counters, as live atomics.
#[derive(Debug, Default)]
struct Counters {
    heads_observed: AtomicU64,
    contexts_dispatched: AtomicU64,
    results_received: AtomicU64,
    proofs_submitted: AtomicU64,
    stale_drops: AtomicU64,
    submission_errors: AtomicU64,
    duplicate_result_drops: AtomicU64,
}

impl Counters {
    fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            heads_observed: self.heads_observed.load(Ordering::Relaxed),
            contexts_dispatched: self.contexts_dispatched.load(Ordering::Relaxed),
            results_received: self.results_received.load(Ordering::Relaxed),
            proofs_submitted: self.proofs_submitted.load(Ordering::Relaxed),
            stale_drops: self.stale_drops.load(Ordering::Relaxed),
            submission_errors: self.submission_errors.load(Ordering::Relaxed),
            duplicate_result_drops: self.duplicate_result_drops.load(Ordering::Relaxed),
        }
    }
}

/// A point-in-time read of [`Counters`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CounterSnapshot {
    /// Chain heads the feeder advanced past.
    pub heads_observed: u64,
    /// Jobs handed to a miner session.
    pub contexts_dispatched: u64,
    /// Results a miner sent back.
    pub results_received: u64,
    /// Proofs the chain accepted.
    pub proofs_submitted: u64,
    /// Work abandoned because the round moved on.
    pub stale_drops: u64,
    /// Submit attempts that neither won nor lost a clean race.
    pub submission_errors: u64,
    /// Results for a job id the coordinator no longer has in flight.
    pub duplicate_result_drops: u64,
}

/// One advertised miner in the `/api/v1/status` roster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinerEntry {
    /// Session-wire miner id, for example `cpu-0`.
    pub id: String,
    /// Solving backend, starting `CPU`, `GPU`, or `QPU`.
    pub miner_type: String,
}

/// One `modes` entry: the counters and roster for one backend group.
#[derive(Debug, Clone)]
pub struct ModeView {
    /// Counters for the miners in this group.
    pub counters: CounterSnapshot,
    /// The miners in this group, in launch order.
    pub miners: Vec<MinerEntry>,
}

/// The chain account the coordinator signs proofs from.
///
/// Every field is empty when the process holds no usable signer key. The
/// dashboard reads an empty `ss58_address` as an absent identity and skips its
/// whole identity block, which is the correct reading for an unkeyed process.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Identity {
    /// SS58 form of the signing account.
    pub ss58_address: String,
    /// `0x`-prefixed hex of the same 32-byte account.
    pub account_id_hex: String,
    /// Node id filed in the chain descriptor.
    pub node_id: String,
}

/// The coordinator's own view of the chain, refreshed by the feeder each poll.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChainView {
    /// `0x`-prefixed hash of the block the last snapshot was taken at.
    pub head_hash: String,
    /// Number of that block.
    pub head_number: u64,
    /// The feeder has a snapshot and is staging work.
    pub is_mining: bool,
    /// `QuantumPow.Miners` holds the signing account.
    pub miner_registered: bool,
}

/// Per-backend counters and roster, plus the process-global counters.
#[derive(Debug)]
pub struct CoordinatorMetrics {
    started: Instant,
    global: Counters,
    /// Backend section name to its counters. `BTreeMap` so the `modes` object
    /// serializes in a stable order, which the golden fixture depends on.
    modes: BTreeMap<String, Counters>,
    rosters: BTreeMap<String, Vec<MinerEntry>>,
    miner_mode: BTreeMap<String, String>,
    identity: Mutex<Identity>,
    chain: Mutex<ChainView>,
}

impl CoordinatorMetrics {
    /// Build the metrics for a launch plan of `(miner_id, backend)` pairs.
    #[must_use]
    pub fn new(launch: &[(String, String)]) -> Self {
        let mut modes: BTreeMap<String, Counters> = BTreeMap::new();
        let mut rosters: BTreeMap<String, Vec<MinerEntry>> = BTreeMap::new();
        let mut miner_mode: BTreeMap<String, String> = BTreeMap::new();
        for (miner_id, backend) in launch {
            let _ = modes.entry(backend.clone()).or_default();
            rosters
                .entry(backend.clone())
                .or_default()
                .push(MinerEntry {
                    id: miner_id.clone(),
                    miner_type: miner_type_label(backend),
                });
            let _ = miner_mode.insert(miner_id.clone(), backend.clone());
        }
        Self {
            started: Instant::now(),
            global: Counters::default(),
            modes,
            rosters,
            miner_mode,
            identity: Mutex::new(Identity::default()),
            chain: Mutex::new(ChainView::default()),
        }
    }

    /// Seconds since the process started.
    #[must_use]
    pub fn uptime_seconds(&self) -> u64 {
        self.started.elapsed().as_secs()
    }

    /// Process-global counters. Equal to the sum across modes for every counter
    /// except `heads_observed`, which every mode shares.
    #[must_use]
    pub fn global(&self) -> CounterSnapshot {
        self.global.snapshot()
    }

    /// Per-backend counters and rosters, keyed by backend section name.
    #[must_use]
    pub fn mode_views(&self) -> BTreeMap<String, ModeView> {
        self.modes
            .iter()
            .map(|(backend, counters)| {
                (
                    backend.clone(),
                    ModeView {
                        counters: counters.snapshot(),
                        miners: self.rosters.get(backend).cloned().unwrap_or_default(),
                    },
                )
            })
            .collect()
    }

    /// Every advertised miner across every backend, in launch order.
    #[must_use]
    pub fn miners(&self) -> Vec<MinerEntry> {
        self.rosters.values().flatten().cloned().collect()
    }

    /// Current identity view. Empty until [`Self::set_identity`] runs.
    #[must_use]
    pub fn identity(&self) -> Identity {
        self.identity.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Replace the identity view.
    pub fn set_identity(&self, identity: Identity) {
        if let Ok(mut g) = self.identity.lock() {
            *g = identity;
        }
    }

    /// Current chain view. Zeroed until the first feeder poll succeeds.
    #[must_use]
    pub fn chain(&self) -> ChainView {
        self.chain.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Replace the chain view.
    pub fn set_chain(&self, chain: ChainView) {
        if let Ok(mut g) = self.chain.lock() {
            *g = chain;
        }
    }

    /// The feeder advanced past a new chain head.
    pub fn record_head_observed(&self) {
        let _ = self.global.heads_observed.fetch_add(1, Ordering::Relaxed);
        for counters in self.modes.values() {
            let _ = counters.heads_observed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// `n` jobs were handed to `miner_id`.
    pub fn record_contexts_dispatched(&self, miner_id: &str, n: u64) {
        self.bump(miner_id, n, |c| &c.contexts_dispatched);
    }

    /// `miner_id` sent a result back.
    pub fn record_result_received(&self, miner_id: &str) {
        self.bump(miner_id, 1, |c| &c.results_received);
    }

    /// The chain accepted a proof from `miner_id`.
    pub fn record_proof_submitted(&self, miner_id: &str) {
        self.bump(miner_id, 1, |c| &c.proofs_submitted);
    }

    /// Work from `miner_id` was dropped because the round moved on.
    pub fn record_stale_drop(&self, miner_id: &str) {
        self.bump(miner_id, 1, |c| &c.stale_drops);
    }

    /// A submit attempt for `miner_id` failed for a reason other than a lost
    /// race.
    pub fn record_submission_error(&self, miner_id: &str) {
        self.bump(miner_id, 1, |c| &c.submission_errors);
    }

    /// `miner_id` sent a result for a job the coordinator no longer holds.
    pub fn record_duplicate_result_drop(&self, miner_id: &str) {
        self.bump(miner_id, 1, |c| &c.duplicate_result_drops);
    }

    /// Add `n` to the global counter and to the counter of the mode that owns
    /// `miner_id`. A miner id outside the launch plan still moves the global
    /// counter, so the process totals stay right.
    fn bump(&self, miner_id: &str, n: u64, pick: fn(&Counters) -> &AtomicU64) {
        let _ = pick(&self.global).fetch_add(n, Ordering::Relaxed);
        if let Some(counters) = self
            .miner_mode
            .get(miner_id)
            .and_then(|backend| self.modes.get(backend))
        {
            let _ = pick(counters).fetch_add(n, Ordering::Relaxed);
        }
    }
}

/// Dashboard `miner_type` for a backend section.
///
/// The dashboard matches the leading `CPU`, `GPU`, or `QPU` and maps anything
/// else to `OTHER`, so the prefix comes from the pallet kind and the backend
/// name is appended as the suffix the UI displays.
#[must_use]
pub fn miner_type_label(backend: &str) -> String {
    let prefix = match miner_kind_from_backend(backend) {
        MinerKind::Cpu | MinerKind::Asic => "CPU",
        MinerKind::Gpu | MinerKind::Metal => "GPU",
        MinerKind::QpuDwave | MinerKind::QpuIbm | MinerKind::QpuIonq | MinerKind::QpuPasqal => {
            "QPU"
        }
    };
    let suffix = backend.to_uppercase();
    if suffix == prefix {
        prefix.to_string()
    } else {
        format!("{prefix}-{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics() -> CoordinatorMetrics {
        CoordinatorMetrics::new(&[
            ("cpu-0".to_string(), "cpu".to_string()),
            ("cuda-0".to_string(), "cuda".to_string()),
            ("cuda-1".to_string(), "cuda".to_string()),
        ])
    }

    #[test]
    fn a_fresh_process_has_zero_counters() {
        let m = metrics();
        assert_eq!(m.global(), CounterSnapshot::default());
    }

    #[test]
    fn per_miner_counters_land_on_the_global_set_and_the_owning_mode() {
        let m = metrics();
        m.record_contexts_dispatched("cpu-0", 4);
        m.record_contexts_dispatched("cuda-1", 6);

        assert_eq!(m.global().contexts_dispatched, 10);
        let modes = m.mode_views();
        #[expect(
            clippy::indexing_slicing,
            reason = "cpu and cuda are both in the launch plan built by metrics()"
        )]
        {
            assert_eq!(modes["cpu"].counters.contexts_dispatched, 4);
            assert_eq!(modes["cuda"].counters.contexts_dispatched, 6);
        }
    }

    /// Every backend observes the same chain heads, so this one counter is
    /// shared rather than partitioned. `/api/v1/stats` reports the global value.
    #[test]
    fn heads_observed_is_bumped_on_the_global_set_and_on_every_mode() {
        let m = metrics();
        m.record_head_observed();
        m.record_head_observed();

        assert_eq!(m.global().heads_observed, 2);
        let modes = m.mode_views();
        #[expect(
            clippy::indexing_slicing,
            reason = "cpu and cuda are both in the launch plan built by metrics()"
        )]
        {
            assert_eq!(modes["cpu"].counters.heads_observed, 2);
            assert_eq!(modes["cuda"].counters.heads_observed, 2);
        }
    }

    /// A result from a miner that is not in the launch plan still moves the
    /// global counter. Losing it would make the process totals wrong.
    #[test]
    fn an_unknown_miner_id_still_moves_the_global_counters() {
        let m = metrics();
        m.record_result_received("ghost-9");
        assert_eq!(m.global().results_received, 1);
        #[expect(
            clippy::indexing_slicing,
            reason = "cpu is in the launch plan built by metrics()"
        )]
        {
            assert_eq!(m.mode_views()["cpu"].counters.results_received, 0);
        }
    }

    #[test]
    fn mode_views_carry_the_miner_roster() {
        let modes = metrics().mode_views();
        #[expect(
            clippy::indexing_slicing,
            reason = "cuda is in the launch plan built by metrics()"
        )]
        {
            assert_eq!(
                modes["cuda"].miners,
                vec![
                    MinerEntry {
                        id: "cuda-0".into(),
                        miner_type: "GPU-CUDA".into()
                    },
                    MinerEntry {
                        id: "cuda-1".into(),
                        miner_type: "GPU-CUDA".into()
                    },
                ]
            );
        }
    }

    /// The dashboard matches on the `CPU`/`GPU`/`QPU` prefix and maps anything
    /// else to `OTHER`, so the prefix is load-bearing.
    #[test]
    fn miner_type_labels_start_with_a_category_the_dashboard_knows() {
        assert_eq!(miner_type_label("cpu"), "CPU");
        assert_eq!(miner_type_label("cuda"), "GPU-CUDA");
        assert_eq!(miner_type_label("metal"), "GPU-METAL");
        assert_eq!(miner_type_label("dwave"), "QPU-DWAVE");
        for backend in ["cpu", "cuda", "metal", "dwave", "unknown"] {
            let label = miner_type_label(backend);
            assert!(
                label.starts_with("CPU") || label.starts_with("GPU") || label.starts_with("QPU"),
                "{backend} produced {label}, which the dashboard maps to OTHER"
            );
        }
    }

    #[test]
    fn identity_and_chain_views_round_trip() {
        let m = metrics();
        assert_eq!(m.identity().ss58_address, "");

        m.set_identity(Identity {
            ss58_address: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".into(),
            account_id_hex: "0xd43593c715fdd31c61141abd04a99fd6822c8558854ccde39a5684e7a56da27d"
                .into(),
            node_id: "quip-miner-pow-CPU-1".into(),
        });
        assert_eq!(m.identity().node_id, "quip-miner-pow-CPU-1");

        m.set_chain(ChainView {
            head_hash: "0xabc".into(),
            head_number: 10_249,
            is_mining: true,
            miner_registered: true,
        });
        assert_eq!(m.chain().head_number, 10_249);
        assert!(m.chain().is_mining);
    }

    /// An unkeyed coordinator is a valid state. The empty roster and the empty
    /// identity are what `/api/v1/status` serves for it.
    #[test]
    fn an_empty_launch_plan_produces_no_modes() {
        let m = CoordinatorMetrics::new(&[]);
        assert!(m.mode_views().is_empty());
        assert_eq!(m.identity(), Identity::default());
    }
}
