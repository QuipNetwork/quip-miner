//! `download` subcommand: pull winning qblocks from chain, redraw + self-verify
//! their Ising problems, and emit a `hardest_models` nonce-ref dataset.

pub mod emit;
pub mod record;
pub mod timeline;

pub use emit::{bucket_and_rank, write_dataset, ManifestEntry};
pub use record::{
    build_instance_record, hex_plain, DifficultyJson, InstanceRecord, ProvenanceJson, VerifyError,
};
pub use timeline::{Era, Timeline};

use crate::chain::qblock::TopologyInputs;
use crate::chain::{ChainClient, ChainError};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Default h0 achievable energy floor (milli). Measured from the clean h0-era
/// corpus: the deepest genuine h0 qblock sits at −14,649,000 and none reach
/// −14,650,000, so any qblock below this could only have been mined on a deeper
/// (field-carrying) topology. Used by the transition-zone reclassification.
pub const DEFAULT_ENERGY_FLOOR_MILLI: i64 = -14_650_000;

/// Which qblock ids to pull.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selection {
    /// Inclusive `[from, to]` range of qblock ids.
    Range {
        /// First qblock id (inclusive).
        from: u64,
        /// Last qblock id (inclusive).
        to: u64,
    },
    /// Every qblock from `1` up to the chain's latest assigned id.
    All,
}

/// Inputs to [`run_download`].
#[derive(Debug)]
pub struct DownloadParams {
    /// Which qblock ids to pull.
    pub selection: Selection,
    /// Output dataset directory (created if absent).
    pub out_dir: PathBuf,
    /// Only keep qblocks whose resolved **mining-era** `topology_hash` matches
    /// this filter (the stored hash is unreliable, so the filter applies to the
    /// era-attributed hash).
    pub topology_filter: Option<[u8; 32]>,
    /// Max instances kept per topology bucket.
    pub cap: usize,
    /// Energy floor (milli) below which a qblock assigned by block to a shallow
    /// era is reclassified into the previous (deeper) era — see
    /// [`DEFAULT_ENERGY_FLOOR_MILLI`].
    pub energy_floor_milli: i64,
}

/// Outcome of a [`run_download`] run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadSummary {
    /// Total qblock ids scanned.
    pub scanned: u64,
    /// Instances that redrew and self-verified successfully.
    pub verified: usize,
    /// Qblock ids with no assigned qblock (gap or beyond the chain's range).
    pub skipped_missing: u64,
    /// Qblocks that decoded but failed era assignment or redraw self-verify.
    pub skipped_unverified: usize,
    /// Distinct topology buckets written.
    pub buckets: usize,
    /// Qblocks moved out of their block-assigned era into the previous (deeper)
    /// era by the transition-zone energy-floor guard.
    pub reclassified: usize,
}

/// Errors from the `download` orchestration.
#[derive(Debug)]
pub enum DownloadError {
    /// Underlying chain I/O failure.
    Chain(ChainError),
    /// Dataset write failure.
    Io(String),
    /// `Selection::All` requested but the chain has no qblocks yet.
    EmptyChain,
    /// The chain exposed no registered topologies, so no mining-era timeline can
    /// be built to attribute qblocks.
    NoTopologies,
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chain(e) => write!(f, "chain error: {e}"),
            Self::Io(s) => write!(f, "dataset write error: {s}"),
            Self::EmptyChain => write!(f, "chain has no qblocks yet"),
            Self::NoTopologies => {
                write!(
                    f,
                    "chain has no registered topologies; cannot build mining-era timeline"
                )
            }
        }
    }
}

impl std::error::Error for DownloadError {}

impl From<ChainError> for DownloadError {
    fn from(e: ChainError) -> Self {
        Self::Chain(e)
    }
}

/// Running state threaded through the per-id qblock collection loop.
#[derive(Default)]
struct Accumulator {
    records: Vec<InstanceRecord>,
    scanned: u64,
    skipped_missing: u64,
    skipped_unverified: usize,
    reclassified: usize,
}

/// Resolve the inclusive `[from, to]` qblock id range for `selection`.
async fn resolve_range(
    chain: &dyn ChainClient,
    selection: &Selection,
) -> Result<(u64, u64), DownloadError> {
    match selection {
        Selection::Range { from, to } => Ok((*from, *to)),
        Selection::All => {
            let latest = chain
                .fetch_latest_qblock_id()
                .await?
                .ok_or(DownloadError::EmptyChain)?;
            Ok((1, latest))
        }
    }
}

/// Fetch one qblock id, attribute it to its mining era (NOT its unreliable
/// stored `topology_hash`), redraw + self-verify against that era's topology,
/// and fold the outcome into `acc`. Never errors on a per-qblock content
/// problem (missing id, unassignable block, failed redraw) — those are counted,
/// not fatal; only chain I/O failures propagate.
async fn collect_qblock(
    chain: &dyn ChainClient,
    id: u64,
    filter: Option<[u8; 32]>,
    timeline: &Timeline,
    energy_floor_milli: i64,
    acc: &mut Accumulator,
) -> Result<(), DownloadError> {
    acc.scanned += 1;
    let Some(q) = chain.fetch_qblock_by_id(id).await? else {
        acc.skipped_missing += 1;
        return Ok(());
    };
    let Some((era_idx, reclassified)) =
        timeline.resolve_era(q.submitted_at, q.energy_milli, energy_floor_milli)
    else {
        tracing::warn!(
            qblock_id = id,
            submitted_at = q.submitted_at,
            "submitted_at precedes the earliest mining era; skipping"
        );
        acc.skipped_unverified += 1;
        return Ok(());
    };
    let Some(era) = timeline.era(era_idx) else {
        acc.skipped_unverified += 1;
        return Ok(());
    };
    if filter.is_some_and(|f| f != era.topology_hash) {
        return Ok(());
    }
    match build_instance_record(&q, &era.inputs, era.topology_hash) {
        Ok(r) => {
            if reclassified {
                acc.reclassified += 1;
                tracing::debug!(
                    qblock_id = id,
                    energy_milli = q.energy_milli,
                    "energy below floor; reclassified to previous era"
                );
            }
            acc.records.push(r);
        }
        Err(e) => {
            tracing::warn!(qblock_id = id, error = %e, "redraw self-verify failed; skipping");
            acc.skipped_unverified += 1;
        }
    }
    Ok(())
}

/// Best-effort chain "priority bucket": the default topology's hash, labeled
/// `chain-default` in the manifest. `None` if the chain has no default
/// topology configured or the read fails (non-fatal — the dataset still
/// writes, just without a `chain-default` alias).
async fn default_topology_hex(chain: &dyn ChainClient) -> Option<String> {
    let snap = chain
        .fetch_mining_snapshot(None, [0u8; 32], None)
        .await
        .ok()
        .flatten()?;
    (snap.topology_hash.len() == 32).then(|| {
        let mut h = [0u8; 32];
        #[expect(
            clippy::indexing_slicing,
            reason = "length checked to be exactly 32 above"
        )]
        h.copy_from_slice(&snap.topology_hash[..32]);
        hex_plain(&h)
    })
}

/// Pull winning qblocks over `params.selection`, redraw + self-verify each,
/// bucket by topology, rank hardest-first, and write the `hardest_models`
/// dataset to `params.out_dir`.
///
/// # Errors
/// Returns [`DownloadError`] on chain I/O failure, an empty chain under
/// `Selection::All`, or a dataset write failure.
pub async fn run_download(
    chain: &dyn ChainClient,
    params: &DownloadParams,
) -> Result<DownloadSummary, DownloadError> {
    let (from, to) = resolve_range(chain, &params.selection).await?;
    let timeline = Timeline::from_registered(chain.fetch_registered_topologies().await?);
    if timeline.is_empty() {
        return Err(DownloadError::NoTopologies);
    }
    let mut acc = Accumulator::default();
    for id in from..=to {
        collect_qblock(
            chain,
            id,
            params.topology_filter,
            &timeline,
            params.energy_floor_milli,
            &mut acc,
        )
        .await?;
    }
    let verified = acc.records.len();
    let buckets = bucket_and_rank(acc.records, params.cap);
    let default_hash_hex = default_topology_hex(chain).await;
    // Every era topology, keyed by hex, so each written bucket resolves its spec.
    let topo_by_hex: BTreeMap<String, TopologyInputs> = timeline
        .eras()
        .iter()
        .map(|e| (hex_plain(&e.topology_hash), e.inputs.clone()))
        .collect();
    write_dataset(
        &params.out_dir,
        &buckets,
        &topo_by_hex,
        default_hash_hex.as_deref(),
    )
    .map_err(|e| DownloadError::Io(e.to_string()))?;
    Ok(DownloadSummary {
        scanned: acc.scanned,
        verified,
        skipped_missing: acc.skipped_missing,
        skipped_unverified: acc.skipped_unverified,
        buckets: buckets.len(),
        reclassified: acc.reclassified,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::qblock::QBlockRecord;
    use crate::chain::FakeChain;
    use crate::topology::topology_hash_sets;

    fn topo() -> TopologyInputs {
        TopologyInputs {
            nodes: vec![0, 1, 2, 3],
            edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
            allowed_h_milli: vec![-1000, 0, 1000],
            allowed_j_milli: vec![-1000, 1000],
            allowed_spin_milli: vec![-1000, 1000],
        }
    }

    fn qblock_at(id: u64, hash: [u8; 32]) -> QBlockRecord {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "test ids are small; fit u8"
        )]
        let b = id as u8;
        QBlockRecord {
            qblock_id: id,
            miner: [1; 32],
            salt: [b; 32],
            energy_milli: -(100 * i64::try_from(id).unwrap_or(0)),
            reward: 0,
            submitted_at: u32::try_from(id).unwrap_or(0),
            last_proof_block_hash: [0; 32],
            topology_hash: hash,
            device_access_time_us: 0,
            max_energy_milli: 0,
            min_solutions: 1,
            min_diversity_milli: 0,
            nonce: [b; 32],
        }
    }

    #[tokio::test]
    async fn run_download_end_to_end_offline() {
        let t = topo();
        let hash = topology_hash_sets(
            &t.nodes,
            &t.edges,
            &t.allowed_h_milli,
            &t.allowed_j_milli,
            &t.allowed_spin_milli,
        );
        let fake = FakeChain::new(
            crate::chain::snapshot::MiningSnapshot {
                last_proof_block_hash: [0u8; 32],
                topology_hash: vec![],
                nodes: vec![],
                edges: vec![],
                allowed_h_milli: vec![],
                allowed_j_milli: vec![],
                allowed_spin_milli: vec![],
                min_solutions: 0,
                max_energy_milli: 0,
                min_diversity_milli: 0,
                block_number: 0,
            },
            None,
        );
        fake.set_registered_topology(hash, t.clone(), 0);
        for id in 1..=3u64 {
            fake.set_qblock(id, qblock_at(id, hash));
        }
        let dir = std::env::temp_dir().join(format!("hm-run-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let summary = run_download(
            &fake,
            &DownloadParams {
                selection: Selection::Range { from: 1, to: 3 },
                out_dir: dir.clone(),
                topology_filter: None,
                cap: 10_000,
                energy_floor_milli: DEFAULT_ENERGY_FLOOR_MILLI,
            },
        )
        .await
        .unwrap();
        assert_eq!(summary.scanned, 3);
        assert_eq!(summary.verified, 3);
        assert_eq!(summary.buckets, 1);
        assert_eq!(summary.skipped_missing, 0);
        assert_eq!(summary.skipped_unverified, 0);
        assert_eq!(summary.reclassified, 0);
        let jsonl =
            std::fs::read_to_string(dir.join(hex_plain(&hash)).join("instances.jsonl")).unwrap();
        assert_eq!(jsonl.lines().count(), 3);
        // hardest first: energy -300 (id 3) leads.
        assert!(jsonl.lines().next().unwrap().contains("-300"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn all_selection_uses_latest_qblock_id() {
        let t = topo();
        let hash = topology_hash_sets(
            &t.nodes,
            &t.edges,
            &t.allowed_h_milli,
            &t.allowed_j_milli,
            &t.allowed_spin_milli,
        );
        let fake = FakeChain::new(
            crate::chain::snapshot::MiningSnapshot {
                last_proof_block_hash: [0u8; 32],
                topology_hash: vec![],
                nodes: vec![],
                edges: vec![],
                allowed_h_milli: vec![],
                allowed_j_milli: vec![],
                allowed_spin_milli: vec![],
                min_solutions: 0,
                max_energy_milli: 0,
                min_diversity_milli: 0,
                block_number: 0,
            },
            None,
        );
        fake.set_registered_topology(hash, t, 0);
        fake.set_qblock(1, qblock_at(1, hash));
        // id 2 intentionally left unset — a gap counted as skipped_missing.
        fake.set_qblock_id(Some(2));
        let dir = std::env::temp_dir().join(format!("hm-all-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let summary = run_download(
            &fake,
            &DownloadParams {
                selection: Selection::All,
                out_dir: dir.clone(),
                topology_filter: None,
                cap: 10_000,
                energy_floor_milli: DEFAULT_ENERGY_FLOOR_MILLI,
            },
        )
        .await
        .unwrap();
        assert_eq!(summary.scanned, 2);
        assert_eq!(summary.verified, 1);
        assert_eq!(summary.skipped_missing, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn all_selection_on_empty_chain_errors() {
        let fake = FakeChain::new(
            crate::chain::snapshot::MiningSnapshot {
                last_proof_block_hash: [0u8; 32],
                topology_hash: vec![],
                nodes: vec![],
                edges: vec![],
                allowed_h_milli: vec![],
                allowed_j_milli: vec![],
                allowed_spin_milli: vec![],
                min_solutions: 0,
                max_energy_milli: 0,
                min_diversity_milli: 0,
                block_number: 0,
            },
            None,
        );
        let dir = std::env::temp_dir().join(format!("hm-empty-{}", std::process::id()));
        let err = run_download(
            &fake,
            &DownloadParams {
                selection: Selection::All,
                out_dir: dir,
                topology_filter: None,
                cap: 10_000,
                energy_floor_milli: DEFAULT_ENERGY_FLOOR_MILLI,
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, DownloadError::EmptyChain));
    }

    /// Two-era split mirroring the live chain: a ternary topology mined first
    /// (deep energies), an h0 topology mined after a switch block. Qblocks all
    /// carry the (clobbered) h0 stored hash, yet must split into two buckets by
    /// mining era, and an in-flight ternary proof that landed after the switch
    /// (energy below the h0 floor) must be pulled back into the ternary bucket.
    #[tokio::test]
    async fn two_era_split_by_mining_era_and_energy_floor() {
        // Ternary era: nonzero h field, reaches deep energies.
        let ternary = TopologyInputs {
            nodes: vec![0, 1, 2, 3],
            edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
            allowed_h_milli: vec![-1000, 0, 1000],
            allowed_j_milli: vec![-1000, 1000],
            allowed_spin_milli: vec![-1000, 1000],
        };
        // h0 era: same graph shape, h pinned to {0}. Distinct hash.
        let h0 = TopologyInputs {
            allowed_h_milli: vec![0],
            ..ternary.clone()
        };
        let tern_hash = topology_hash_sets(
            &ternary.nodes,
            &ternary.edges,
            &ternary.allowed_h_milli,
            &ternary.allowed_j_milli,
            &ternary.allowed_spin_milli,
        );
        let h0_hash = topology_hash_sets(
            &h0.nodes,
            &h0.edges,
            &h0.allowed_h_milli,
            &h0.allowed_j_milli,
            &h0.allowed_spin_milli,
        );
        assert_ne!(tern_hash, h0_hash);

        let fake = FakeChain::new(
            crate::chain::snapshot::MiningSnapshot {
                last_proof_block_hash: [0u8; 32],
                topology_hash: vec![],
                nodes: vec![],
                edges: vec![],
                allowed_h_milli: vec![],
                allowed_j_milli: vec![],
                allowed_spin_milli: vec![],
                min_solutions: 0,
                max_energy_milli: 0,
                min_diversity_milli: 0,
                block_number: 0,
            },
            None,
        );
        // Timeline: ternary [100, 200), h0 [200, ..).
        fake.set_registered_topology(tern_hash, ternary, 100);
        fake.set_registered_topology(h0_hash, h0, 200);

        // Every qblock carries the CLOBBERED h0 stored hash (as on the live chain).
        let mk = |id: u64, block: u32, energy: i64| {
            let mut q = qblock_at(id, h0_hash);
            q.submitted_at = block;
            q.energy_milli = energy;
            q
        };
        // id1: ternary era, deep.               → ternary bucket (by block)
        fake.set_qblock(1, mk(1, 150, -15_060_000));
        // id2: h0 era, shallow (genuine h0).     → h0 bucket
        fake.set_qblock(2, mk(2, 300, -14_540_000));
        // id3: h0 era by block, but deep (in-flight ternary proof past switch)
        //      → reclassified into ternary bucket by the energy-floor guard.
        fake.set_qblock(3, mk(3, 205, -14_984_000));

        let dir = std::env::temp_dir().join(format!("hm-2era-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let summary = run_download(
            &fake,
            &DownloadParams {
                selection: Selection::Range { from: 1, to: 3 },
                out_dir: dir.clone(),
                topology_filter: None,
                cap: 10_000,
                energy_floor_milli: DEFAULT_ENERGY_FLOOR_MILLI,
            },
        )
        .await
        .unwrap();
        assert_eq!(summary.verified, 3);
        assert_eq!(summary.buckets, 2, "ternary + h0");
        assert_eq!(summary.reclassified, 1, "id3 pulled back to ternary");

        let tern_dir = dir.join(hex_plain(&tern_hash));
        let h0_dir = dir.join(hex_plain(&h0_hash));
        let tern_lines = std::fs::read_to_string(tern_dir.join("instances.jsonl")).unwrap();
        let h0_lines = std::fs::read_to_string(h0_dir.join("instances.jsonl")).unwrap();
        assert_eq!(tern_lines.lines().count(), 2, "ids 1 and 3");
        assert_eq!(h0_lines.lines().count(), 1, "id 2 only");
        // The h0 bucket must hold nothing below the floor.
        for line in h0_lines.lines() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let energy = v
                .get("energy_milli")
                .and_then(serde_json::Value::as_i64)
                .unwrap();
            assert!(energy >= DEFAULT_ENERGY_FLOOR_MILLI);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
