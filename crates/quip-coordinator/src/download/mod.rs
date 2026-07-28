//! `download` subcommand: pull winning qblocks from chain, redraw + self-verify
//! their Ising problems, and emit a `hardest_models` nonce-ref dataset.

pub mod emit;
pub mod record;

pub use emit::{bucket_and_rank, write_dataset, ManifestEntry};
pub use record::{
    build_instance_record, hex_plain, DifficultyJson, InstanceRecord, ProvenanceJson, VerifyError,
};

use crate::chain::qblock::TopologyInputs;
use crate::chain::{ChainClient, ChainError};
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::path::PathBuf;

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
    /// Only keep qblocks whose `topology_hash` matches this filter.
    pub topology_filter: Option<[u8; 32]>,
    /// Max instances kept per topology bucket.
    pub cap: usize,
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
    /// Qblocks that decoded but failed topology lookup or redraw self-verify.
    pub skipped_unverified: usize,
    /// Distinct topology buckets written.
    pub buckets: usize,
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
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chain(e) => write!(f, "chain error: {e}"),
            Self::Io(s) => write!(f, "dataset write error: {s}"),
            Self::EmptyChain => write!(f, "chain has no qblocks yet"),
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
    topo_cache: BTreeMap<[u8; 32], TopologyInputs>,
    records: Vec<InstanceRecord>,
    scanned: u64,
    skipped_missing: u64,
    skipped_unverified: usize,
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

/// Fetch, filter, redraw, and self-verify one qblock id, folding the outcome
/// into `acc`. Never errors on a per-qblock content problem (missing id,
/// failed redraw) — those are counted, not fatal; only chain I/O failures
/// propagate.
async fn collect_qblock(
    chain: &dyn ChainClient,
    id: u64,
    filter: Option<[u8; 32]>,
    acc: &mut Accumulator,
) -> Result<(), DownloadError> {
    acc.scanned += 1;
    let Some(q) = chain.fetch_qblock_by_id(id).await? else {
        acc.skipped_missing += 1;
        return Ok(());
    };
    if filter.is_some_and(|f| f != q.topology_hash) {
        return Ok(());
    }
    let Some(topo) = topology_for(chain, &mut acc.topo_cache, q.topology_hash).await? else {
        tracing::warn!(qblock_id = id, "topology_meta missing; skipping");
        acc.skipped_unverified += 1;
        return Ok(());
    };
    match build_instance_record(&q, topo) {
        Ok(r) => acc.records.push(r),
        Err(e) => {
            tracing::warn!(qblock_id = id, error = %e, "redraw self-verify failed; skipping");
            acc.skipped_unverified += 1;
        }
    }
    Ok(())
}

/// Cache-or-fetch a topology's redraw inputs.
async fn topology_for<'a>(
    chain: &dyn ChainClient,
    cache: &'a mut BTreeMap<[u8; 32], TopologyInputs>,
    hash: [u8; 32],
) -> Result<Option<&'a TopologyInputs>, ChainError> {
    match cache.entry(hash) {
        Entry::Occupied(e) => Ok(Some(e.into_mut())),
        Entry::Vacant(v) => match chain.fetch_topology_meta(hash).await? {
            Some(t) => Ok(Some(v.insert(t))),
            None => Ok(None),
        },
    }
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
    let mut acc = Accumulator::default();
    for id in from..=to {
        collect_qblock(chain, id, params.topology_filter, &mut acc).await?;
    }
    let verified = acc.records.len();
    let buckets = bucket_and_rank(acc.records, params.cap);
    let default_hash_hex = default_topology_hex(chain).await;
    let topo_by_hex: BTreeMap<String, TopologyInputs> = acc
        .topo_cache
        .into_iter()
        .map(|(h, t)| (hex_plain(&h), t))
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
        fake.set_topology(hash, t.clone());
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
            },
        )
        .await
        .unwrap();
        assert_eq!(summary.scanned, 3);
        assert_eq!(summary.verified, 3);
        assert_eq!(summary.buckets, 1);
        assert_eq!(summary.skipped_missing, 0);
        assert_eq!(summary.skipped_unverified, 0);
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
        fake.set_topology(hash, t);
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
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, DownloadError::EmptyChain));
    }
}
