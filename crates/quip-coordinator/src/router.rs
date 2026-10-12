//! Capability index, per-miner staged queue, credit accounting, cancel, reject.

use crate::topology::Topology;
use quip_proto::v1::{ising_problem, Capabilities, Job, JobKind, RejectReason};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

/// Capability envelope from the latest accepted miner advertisement.
#[derive(Debug, Clone, Default)]
pub struct MinerCaps {
    /// Backend name.
    pub backend: String,
    /// Sampling algorithm name.
    pub algorithm: String,
    /// Accepted job kinds. Empty means all kinds.
    pub supported_kinds: Vec<i32>,
    /// Maximum nodes. Zero is unbounded.
    pub max_nodes: u32,
    /// Maximum edges. Zero is unbounded.
    pub max_edges: u32,
    /// Maximum degree. None makes no claim. Some(0) is unbounded.
    pub max_degree: Option<u32>,
    /// Fingerprint of the native graph, independent of chain topology hashes.
    pub native_topology_hash: Option<Vec<u8>>,
    /// Feature names advertised by the miner.
    pub features: Vec<String>,
    /// Problem generators supported for leases.
    pub generators: Vec<i32>,
    /// Models kept in flight. Zero means unknown.
    pub stream_width: u32,
}

impl MinerCaps {
    /// Whether this miner takes `ISING_GENERATE` leases with the generator
    /// this coordinator issues. The upgrade guide sends leases only to a peer
    /// that advertises both.
    #[must_use]
    pub fn accepts_leases(&self) -> bool {
        self.supported_kinds
            .contains(&(JobKind::IsingGenerate as i32))
            && self
                .generators
                .contains(&(quip_proto::v1::GeneratorAlgorithm::Blake3Chacha8V1 as i32))
    }

    /// Index a protocol-2 capability snapshot.
    #[must_use]
    pub fn from_capabilities(c: &Capabilities) -> Self {
        use quip_protocol::session::{algorithm_name, backend_name};
        Self {
            backend: backend_name(c.backend()).to_owned(),
            algorithm: algorithm_name(c.algorithm()).to_owned(),
            supported_kinds: c.supported_kinds.clone(),
            max_nodes: c.max_nodes,
            max_edges: c.max_edges,
            max_degree: c.max_degree,
            native_topology_hash: c.native_topology_hash.clone(),
            features: c.features.clone(),
            generators: c.generators.clone(),
            stream_width: c.stream_width,
        }
    }
}

#[derive(Debug)]
struct MinerQueue {
    caps: MinerCaps,
    staged: VecDeque<Job>,
    /// Consumable credit pool: each `JobRequest` adds credits, each dispatch
    /// spends one. The miner grants `width` up front and one per terminal
    /// event (Result or Reject), so dispatch stays 1:1 with completion and
    /// in-flight is bounded — never a cumulative ceiling that ack also frees
    /// (that double-count let in-flight grow without bound and deadlocked the
    /// miner's channels). In-flight itself is tracked by the coordinator's
    /// `inflight` map, so the router needs no separate outstanding counter.
    granted_credits: u32,
    /// Jobs dispatched (drained from `staged`) since the last `take_consumed`.
    /// The feeder reads-and-resets this each poll to size the adaptive staging
    /// window from the miner's observed drain rate (see `feeder_loop`).
    consumed_since_poll: u32,
    /// Lease salts the miner finished since the last `take_lease_salts`. The feeder turns this into a salts-per-second rate for lease sizing.
    salts_since_poll: u64,
    /// Lease salts the miner finished for the current round. Reset when a new
    /// round starts, not when the miner re-registers.
    round_salts: u64,
    /// Jobs this miner finished (Result or Reject) since it registered.
    /// The heartbeat reads this total. Polls do not reset it.
    completed: u64,
    unsupported_kinds: HashSet<i32>,
}

/// How often one miner's dropped-job total may reach the log at `warn`.
///
/// A miner that rejects everything it is offered (a QPU with an exhausted
/// budget, say) drops one job per dispatch, and at credit speed that is
/// thousands of lines a second. The count is what an operator needs, not the
/// individual jobs.
const DROP_WARN_INTERVAL: Duration = Duration::from_secs(30);

/// Dropped-job accounting for one miner, for the rate-limited warn.
#[derive(Debug)]
struct DropLog {
    /// Jobs dropped since the last line reached the log.
    since_warn: u64,
    /// When that line was emitted.
    last_warn: Instant,
}

#[derive(Debug)]
struct RoutingTopology {
    hash: Vec<u8>,
    n_nodes: u32,
    n_edges: u32,
    max_degree: u32,
    native_hash: [u8; 32],
    nodes: HashSet<u32>,
    edges: HashSet<(u32, u32)>,
}

impl RoutingTopology {
    fn new(topology: &Topology) -> Option<Self> {
        if topology.edges.0.len() != topology.edges.1.len() {
            return None;
        }
        let nodes: HashSet<u32> = topology.nodes.iter().copied().collect();
        if nodes.len() != topology.nodes.len()
            || topology
                .edges
                .0
                .iter()
                .zip(&topology.edges.1)
                .any(|(u, v)| u == v || !nodes.contains(u) || !nodes.contains(v))
        {
            return None;
        }
        let pairs = topology.edge_pairs();
        Some(Self {
            hash: topology.hash.clone(),
            n_nodes: u32::try_from(topology.nodes.len()).ok()?,
            n_edges: u32::try_from(pairs.len()).ok()?,
            max_degree: maximum_degree(pairs.iter().copied()),
            native_hash: crate::topology::native_topology_hash(&topology.nodes, &pairs)?,
            nodes,
            edges: pairs
                .into_iter()
                .map(|(u, v)| (u.min(v), u.max(v)))
                .collect(),
        })
    }
}

fn maximum_degree(edges: impl Iterator<Item = (u32, u32)>) -> u32 {
    let mut degrees: HashMap<u32, u32> = HashMap::new();
    for (u, v) in edges {
        for node in [u, v] {
            let degree = degrees.entry(node).or_default();
            *degree = degree.saturating_add(1);
        }
    }
    degrees.values().copied().max().unwrap_or(0)
}

/// Routes jobs to miners by capability, stages them, and gates dispatch on credits.
#[derive(Debug, Default)]
pub struct Router {
    miners: HashMap<String, MinerQueue>,
    topology: Option<RoutingTopology>,
    /// Jobs that could not be routed (no capable miner).
    pub unroutable: Vec<Job>,
    /// Per-miner drop counters behind [`DROP_WARN_INTERVAL`].
    drops: HashMap<String, DropLog>,
}

impl Router {
    /// Empty router with no registered miners.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cache the active session graph for hash-job and lease routing.
    pub fn set_topology(&mut self, topology: Option<&Topology>) {
        self.topology = topology.and_then(RoutingTopology::new);
    }

    /// Register (or re-register) a miner with the given capability envelope.
    pub fn register_miner(&mut self, miner_id: impl Into<String>, caps: MinerCaps) {
        let id = miner_id.into();
        match self.miners.get_mut(&id) {
            // Re-registration (e.g. a supervised restart of the same id): keep
            // any jobs re-staged onto this id while it was down — otherwise the
            // reclaim→re-route→re-handshake path silently drops them. Reset only
            // the volatile per-connection state and refresh caps.
            Some(q) => {
                q.caps = caps;
                q.granted_credits = 0;
                q.consumed_since_poll = 0;
                q.salts_since_poll = 0;
                q.unsupported_kinds.clear();
            }
            None => {
                let _ = self.miners.insert(
                    id,
                    MinerQueue {
                        caps,
                        staged: VecDeque::new(),
                        granted_credits: 0,
                        consumed_since_poll: 0,
                        salts_since_poll: 0,
                        round_salts: 0,
                        completed: 0,
                        unsupported_kinds: HashSet::new(),
                    },
                );
            }
        }
    }

    /// Stage `job` on a capable miner; returns the chosen `miner_id`, or `None`.
    pub fn route(&mut self, job: Job) -> Option<String> {
        self.route_excluding(job, None)
    }

    fn is_capable(&self, miner_id: &str, job: &Job) -> bool {
        self.miners
            .get(miner_id)
            .is_some_and(|q| capable(q, job, self.topology.as_ref()))
    }

    /// Like [`Router::route`], but never selects `exclude`. Used by
    /// [`Router::on_reject`] so a rejected job is not re-staged onto the miner
    /// that just rejected it.
    fn route_excluding(&mut self, job: Job, exclude: Option<&str>) -> Option<String> {
        let mut candidates: Vec<String> = self
            .miners
            .keys()
            .filter(|id| exclude != Some(id.as_str()) && self.is_capable(id, &job))
            .cloned()
            .collect();
        candidates.sort();
        if let Some(id) = candidates.first().cloned() {
            if let Some(q) = self.miners.get_mut(&id) {
                q.staged.push_back(job);
            }
            Some(id)
        } else {
            self.unroutable.push(job);
            None
        }
    }

    /// Replace a live envelope and re-route queued jobs that no longer fit.
    /// Credits, counters, and jobs already dispatched stay intact.
    pub fn update_capabilities(&mut self, miner_id: &str, caps: MinerCaps) -> bool {
        let Some(q) = self.miners.get_mut(miner_id) else {
            return false;
        };
        q.caps = caps;
        q.unsupported_kinds.clear();
        let staged = std::mem::take(&mut q.staged);
        for job in staged {
            if self.is_capable(miner_id, &job) {
                if let Some(q) = self.miners.get_mut(miner_id) {
                    q.staged.push_back(job);
                }
            } else {
                let _ = self.route_excluding(job, Some(miner_id));
            }
        }
        true
    }

    /// Add `credits` to the miner's consumable credit pool.
    pub fn grant_credits(&mut self, miner_id: &str, credits: u32) {
        if let Some(q) = self.miners.get_mut(miner_id) {
            q.granted_credits = q.granted_credits.saturating_add(credits);
        }
    }

    /// Add salts a miner reported finished in a `LeaseDone`.
    pub fn record_lease_salts(&mut self, miner_id: &str, salts: u64) {
        if let Some(q) = self.miners.get_mut(miner_id) {
            q.salts_since_poll = q.salts_since_poll.saturating_add(salts);
        }
    }

    /// Add salts a miner finished for the current round.
    pub fn record_round_salts(&mut self, miner_id: &str, salts: u64) {
        if let Some(q) = self.miners.get_mut(miner_id) {
            q.round_salts = q.round_salts.saturating_add(salts);
        }
    }

    /// Salts `miner_id` finished for the current round. 0 if unknown.
    #[must_use]
    pub fn round_salts(&self, miner_id: &str) -> u64 {
        self.miners.get(miner_id).map_or(0, |q| q.round_salts)
    }

    /// Start every miner's round salt count from zero.
    pub fn reset_round_salts(&mut self) {
        for q in self.miners.values_mut() {
            q.round_salts = 0;
        }
    }

    /// Read and reset the finished-salt count since the last call.
    pub fn take_lease_salts(&mut self, miner_id: &str) -> u64 {
        self.miners
            .get_mut(miner_id)
            .map_or(0, |q| std::mem::take(&mut q.salts_since_poll))
    }

    /// Dispatch a compatible staged job, spending one credit only on dispatch.
    pub fn next_job(&mut self, miner_id: &str) -> Option<Job> {
        loop {
            let job = {
                let q = self.miners.get_mut(miner_id)?;
                if q.granted_credits == 0 {
                    return None;
                }
                q.staged.pop_front()?
            };
            if !self.is_capable(miner_id, &job) {
                let _ = self.route_excluding(job, Some(miner_id));
                continue;
            }
            let q = self.miners.get_mut(miner_id)?;
            q.granted_credits -= 1;
            q.consumed_since_poll = q.consumed_since_poll.saturating_add(1);
            return Some(job);
        }
    }

    /// Read and reset the jobs-consumed counter for `miner_id` — the feeder's
    /// per-poll drain sample. Returns 0 for an unknown miner or an idle poll.
    pub fn take_consumed(&mut self, miner_id: &str) -> u32 {
        self.miners
            .get_mut(miner_id)
            .map_or(0, |q| std::mem::take(&mut q.consumed_since_poll))
    }

    /// Record that `miner_id` finished a job (Result or Reject).
    pub fn record_completion(&mut self, miner_id: &str) {
        if let Some(q) = self.miners.get_mut(miner_id) {
            q.completed = q.completed.saturating_add(1);
        }
    }

    /// Jobs `miner_id` has finished since it registered. 0 if unknown.
    ///
    /// The heartbeat prints this total and the delta since the last
    /// heartbeat. Feeder polls do not reset it.
    #[must_use]
    pub fn jobs_completed(&self, miner_id: &str) -> u64 {
        self.miners.get(miner_id).map_or(0, |q| q.completed)
    }

    /// Handle a miner reject: mark unsupported kinds, re-route to a *different*
    /// capable miner.
    ///
    /// Re-routing excludes the rejecting miner: for a deterministic reject
    /// (`Malformed`, `Expired`, `TopologyMissing`/`Mismatch`) the same job on the
    /// same miner fails again, and since the rejection grants its own replacement
    /// credit it would loop forever with a single miner. If no other miner can
    /// take it, the job lands in `unroutable` (dropped) rather than spinning.
    pub fn on_reject(&mut self, miner_id: &str, job: Job, reason: i32) {
        if reason == RejectReason::UnsupportedKind as i32 {
            if let Some(q) = self.miners.get_mut(miner_id) {
                let _ = q.unsupported_kinds.insert(job.kind);
            }
        }
        if self.route_excluding(job, Some(miner_id)).is_none() {
            tracing::debug!(
                miner = %miner_id,
                reason,
                "rejected job has no alternative capable miner; dropping"
            );
            self.note_drop(miner_id, reason);
        }
    }

    /// Count one dropped job, and let the total through at `warn` no more than
    /// once per [`DROP_WARN_INTERVAL`] per miner.
    ///
    /// The first drop after a quiet period reports immediately, so a one-off
    /// still surfaces; a flood collapses into one line carrying the count.
    fn note_drop(&mut self, miner_id: &str, reason: i32) {
        let now = Instant::now();
        match self.drops.get_mut(miner_id) {
            Some(log) if now.duration_since(log.last_warn) < DROP_WARN_INTERVAL => {
                log.since_warn += 1;
            }
            Some(log) => {
                tracing::warn!(
                    miner = %miner_id,
                    reason,
                    dropped = log.since_warn + 1,
                    window_s = DROP_WARN_INTERVAL.as_secs(),
                    "rejected jobs have no alternative capable miner; dropping"
                );
                log.since_warn = 0;
                log.last_warn = now;
            }
            None => {
                tracing::warn!(
                    miner = %miner_id,
                    reason,
                    dropped = 1,
                    "rejected job has no alternative capable miner; dropping"
                );
                let _ = self.drops.insert(
                    miner_id.to_string(),
                    DropLog {
                        since_warn: 0,
                        last_warn: now,
                    },
                );
            }
        }
    }

    /// Drop staged `PoW` jobs where `0 < generation <= max_generation`.
    /// Mempool jobs (`generation == 0`) are preserved. Returns how many jobs
    /// were dropped.
    pub fn cancel(&mut self, max_generation: u64) -> usize {
        let mut dropped = 0;
        for q in self.miners.values_mut() {
            let before = q.staged.len();
            q.staged
                .retain(|j| j.generation == 0 || j.generation > max_generation);
            dropped += before.saturating_sub(q.staged.len());
        }
        dropped
    }

    /// Return all outstanding + staged jobs for a miner (e.g. on crash re-queue).
    pub fn reclaim(&mut self, miner_id: &str) -> Vec<Job> {
        if let Some(q) = self.miners.get_mut(miner_id) {
            q.granted_credits = 0;
            return q.staged.drain(..).collect();
        }
        Vec::new()
    }

    /// Capability envelope for a registered miner, if any.
    #[must_use]
    pub fn caps(&self, miner_id: &str) -> Option<&MinerCaps> {
        self.miners.get(miner_id).map(|q| &q.caps)
    }

    /// Number of jobs currently staged for `miner_id` (0 if unknown).
    #[must_use]
    pub fn staged_len(&self, miner_id: &str) -> usize {
        self.miners.get(miner_id).map_or(0, |q| q.staged.len())
    }

    /// Registered miner ids, sorted for deterministic iteration (the feeder
    /// walks them to top each up to buffer depth).
    #[must_use]
    pub fn miner_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.miners.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Stage a job on a specific capable miner.
    pub fn stage_on(&mut self, miner_id: &str, job: Job) -> bool {
        if !self.is_capable(miner_id, &job) {
            return false;
        }
        if let Some(q) = self.miners.get_mut(miner_id) {
            q.staged.push_back(job);
            return true;
        }
        false
    }
}

fn capable(q: &MinerQueue, job: &Job, topology: Option<&RoutingTopology>) -> bool {
    let kind = job.kind;
    if q.unsupported_kinds.contains(&kind) {
        return false;
    }
    if !q.caps.supported_kinds.is_empty() && !q.caps.supported_kinds.contains(&kind) {
        return false;
    }
    if kind == JobKind::IsingGenerate as i32 && !q.caps.accepts_leases() {
        return false;
    }
    if q.caps.max_nodes > 0 && job_node_count(job, topology) > q.caps.max_nodes {
        return false;
    }
    if q.caps.max_edges > 0 && job_edge_count(job, topology) > q.caps.max_edges {
        return false;
    }
    if let Some(limit) = q.caps.max_degree.filter(|&limit| limit > 0) {
        if job_max_degree(job, topology).is_some_and(|degree| degree > limit) {
            return false;
        }
    }
    native_compatible(&q.caps, job, topology)
}

fn native_compatible(caps: &MinerCaps, job: &Job, topology: Option<&RoutingTopology>) -> bool {
    let Some(native_hash) = caps.native_topology_hash.as_ref() else {
        return true;
    };
    if native_hash.len() != 32 {
        return false;
    }
    if job.kind == JobKind::IsingGenerate as i32 {
        return job_topology(job, topology).is_some();
    }
    let Some(ising) = job.ising.as_ref() else {
        return false;
    };
    match ising.graph.as_ref() {
        Some(ising_problem::Graph::TopologyHash(hash)) => topology.is_some_and(|t| t.hash == *hash),
        Some(ising_problem::Graph::Edges(edges)) => {
            let Some(topology) = topology else {
                return false;
            };
            if native_hash.as_slice() != topology.native_hash.as_slice()
                || edges.u.len() != edges.v.len()
                || ising.encoding != quip_proto::v1::CoefficientEncoding::I32 as i32
                || ising.scale != crate::producer::problem::MILLI_SCALE
                || !ising.h.len().is_multiple_of(4)
                || ising.j.len() != edges.u.len().saturating_mul(4)
            {
                return false;
            }
            let n_nodes = job_node_count(job, Some(topology));
            if !(0..n_nodes).all(|node| topology.nodes.contains(&node)) {
                return false;
            }
            edges.u.iter().zip(&edges.v).all(|(&u, &v)| {
                u < n_nodes
                    && v < n_nodes
                    && u != v
                    && topology.edges.contains(&(u.min(v), u.max(v)))
            })
        }
        None => false,
    }
}

fn job_topology<'a>(
    job: &Job,
    topology: Option<&'a RoutingTopology>,
) -> Option<&'a RoutingTopology> {
    let hash = if job.kind == JobKind::IsingGenerate as i32 {
        job.generator.as_ref()?.topology_hash.as_slice()
    } else {
        let ising = job.ising.as_ref()?;
        match ising.graph.as_ref()? {
            ising_problem::Graph::TopologyHash(hash) => hash.as_slice(),
            ising_problem::Graph::Edges(_) => return None,
        }
    };
    topology.filter(|t| t.hash.as_slice() == hash)
}

fn job_node_count(job: &Job, topology: Option<&RoutingTopology>) -> u32 {
    if job.kind == JobKind::IsingGenerate as i32 {
        return job_topology(job, topology).map_or(0, |t| t.n_nodes);
    }
    job.ising.as_ref().map_or(0, |ising| {
        u32::try_from(ising.h.len() / 4).unwrap_or(u32::MAX)
    })
}

fn job_edge_count(job: &Job, topology: Option<&RoutingTopology>) -> u32 {
    if job.kind == JobKind::IsingGenerate as i32 {
        return job_topology(job, topology).map_or(0, |t| t.n_edges);
    }
    let Some(ising) = job.ising.as_ref() else {
        return 0;
    };
    let count = match ising.graph.as_ref() {
        Some(ising_problem::Graph::Edges(edges)) => edges.u.len(),
        Some(ising_problem::Graph::TopologyHash(_)) | None => ising.j.len() / 4,
    };
    u32::try_from(count).unwrap_or(u32::MAX)
}

fn job_max_degree(job: &Job, topology: Option<&RoutingTopology>) -> Option<u32> {
    if let Some(topology) = job_topology(job, topology) {
        return Some(topology.max_degree);
    }
    let ising = job.ising.as_ref()?;
    match ising.graph.as_ref() {
        Some(ising_problem::Graph::Edges(edges)) => {
            if edges.u.len() != edges.v.len() {
                return None;
            }
            Some(maximum_degree(
                edges.u.iter().copied().zip(edges.v.iter().copied()),
            ))
        }
        Some(ising_problem::Graph::TopologyHash(_)) => None,
        None => ising.j.is_empty().then_some(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_proto::v1::{JobKind, Provenance};

    fn caps_ising() -> MinerCaps {
        MinerCaps {
            backend: "mock".into(),
            algorithm: "sa".into(),
            supported_kinds: vec![JobKind::IsingSample as i32],
            max_nodes: 1000,
            max_edges: 10000,
            ..MinerCaps::default()
        }
    }

    fn caps_lease() -> MinerCaps {
        MinerCaps {
            supported_kinds: vec![JobKind::IsingSample as i32, JobKind::IsingGenerate as i32],
            generators: vec![quip_proto::v1::GeneratorAlgorithm::Blake3Chacha8V1 as i32],
            stream_width: 4,
            ..caps_ising()
        }
    }

    fn lease_job(start: u64) -> Job {
        let snap = crate::chain::snapshot::MiningSnapshot {
            head_hash: [0; 32],
            last_proof_block_hash: [7; 32],
            topology_hash: vec![9; 32],
            nodes: vec![0, 1],
            edges: vec![(0, 1)],
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![1000],
            allowed_spin_milli: vec![-1000, 1000],
            min_solutions: 1,
            max_energy_milli: 0,
            min_diversity_milli: 0,
            block_number: 1,
            spec_version: 117,
        };
        crate::lease::build_lease_job(&snap, [0; 32], start, 4, 1)
    }

    #[test]
    fn lease_routes_only_to_a_miner_with_the_generator() {
        let mut r = Router::new();
        r.register_miner("plain", caps_ising());
        assert_eq!(r.route(lease_job(1)), None);
        r.register_miner("lease", caps_lease());
        assert_eq!(r.route(lease_job(1)).as_deref(), Some("lease"));
    }

    #[test]
    fn kind_without_generator_does_not_accept_leases() {
        let caps = MinerCaps {
            generators: vec![],
            ..caps_lease()
        };
        assert!(!caps.accepts_leases());
        assert!(caps_lease().accepts_leases());
    }

    #[test]
    fn lease_salts_read_and_reset() {
        let mut r = Router::new();
        r.register_miner("lease", caps_lease());
        r.record_lease_salts("lease", 7);
        r.record_lease_salts("lease", 3);
        assert_eq!(r.take_lease_salts("lease"), 10);
        assert_eq!(r.take_lease_salts("lease"), 0);
        assert_eq!(r.take_lease_salts("unknown"), 0);
    }

    #[test]
    fn round_salts_survive_reregistration_and_reset_per_round() {
        let mut r = Router::new();
        r.register_miner("lease", caps_lease());
        r.record_round_salts("lease", 7);
        r.register_miner("lease", caps_lease());
        r.record_round_salts("lease", 3);
        assert_eq!(r.round_salts("lease"), 10);
        r.reset_round_salts();
        assert_eq!(r.round_salts("lease"), 0);
        assert_eq!(r.round_salts("unknown"), 0);
    }

    fn make_job(generation: u64, kind: JobKind) -> Job {
        Job {
            job_id: format!("g{generation}").into_bytes(),
            generator: None,
            kind: kind as i32,
            generation,
            deadline_ms: 9_999_999,
            ising: Some(crate::producer::problem::milli_problem(
                None, &[0; 2], &[0; 1],
            )),
            provenance: Some(Provenance {
                is_pow: generation != 0,
                order_id: vec![],
            }),
        }
    }

    #[test]
    fn route_picks_capable_miner_and_stages() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        let id = r.route(make_job(1, JobKind::IsingSample));
        assert_eq!(id.as_deref(), Some("cpu-0"));
        assert_eq!(r.staged_len("cpu-0"), 1);
    }

    #[test]
    fn next_job_consumes_credits() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        let _ = r.route(make_job(1, JobKind::IsingSample));
        let _ = r.route(make_job(2, JobKind::IsingSample));
        assert!(r.next_job("cpu-0").is_none()); // no credits
        r.grant_credits("cpu-0", 1);
        assert!(r.next_job("cpu-0").is_some()); // spends the one credit
        assert!(r.next_job("cpu-0").is_none()); // pool empty — a completion is
                                                // what refills it, not dispatch
        r.grant_credits("cpu-0", 1); // miner grants one per terminal event
        assert!(r.next_job("cpu-0").is_some()); // dispatches the replacement
    }

    #[test]
    fn never_exceeds_credits() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        let _ = r.route(make_job(1, JobKind::IsingSample));
        let _ = r.route(make_job(2, JobKind::IsingSample));
        r.grant_credits("cpu-0", 1);
        assert!(r.next_job("cpu-0").is_some());
        assert!(r.next_job("cpu-0").is_none());
    }

    #[test]
    fn take_consumed_counts_dispatches_then_resets() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        let _ = r.route(make_job(1, JobKind::IsingSample));
        let _ = r.route(make_job(2, JobKind::IsingSample));
        r.grant_credits("cpu-0", 2);
        assert!(r.next_job("cpu-0").is_some());
        assert!(r.next_job("cpu-0").is_some());
        assert_eq!(r.take_consumed("cpu-0"), 2); // two dispatched this interval
        assert_eq!(r.take_consumed("cpu-0"), 0); // reset after read
        assert_eq!(r.take_consumed("unknown"), 0); // unknown miner
    }

    /// The heartbeat must report completions, not the per-poll dispatch
    /// counter. Stage 8, dispatch 8, complete 3, then drain the poll
    /// counter twice. The reported number is 3, not 8 and not 0.
    #[test]
    fn heartbeat_reports_completions_not_poll_dispatches() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        for generation in 1..=8 {
            let _ = r.route(make_job(generation, JobKind::IsingSample));
        }
        r.grant_credits("cpu-0", 8);
        for _ in 0..8 {
            assert!(r.next_job("cpu-0").is_some());
        }
        for _ in 0..3 {
            r.record_completion("cpu-0");
        }
        // A feeder poll reads and resets the dispatch counter. The
        // heartbeat used to print whatever that single poll held.
        assert_eq!(r.take_consumed("cpu-0"), 8);
        assert_eq!(r.take_consumed("cpu-0"), 0);
        assert_eq!(r.jobs_completed("cpu-0"), 3);
        assert_eq!(r.jobs_completed("unknown"), 0);
    }

    #[test]
    fn cancel_drops_pow_keeps_mempool_and_newer() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        let _ = r.route(make_job(0, JobKind::IsingSample)); // mempool
        let _ = r.route(make_job(3, JobKind::IsingSample));
        let _ = r.route(make_job(5, JobKind::IsingSample));
        let _ = r.route(make_job(6, JobKind::IsingSample));
        assert_eq!(r.cancel(5), 2);
        // keep gen 0 and gen 6
        assert_eq!(r.staged_len("cpu-0"), 2);
        let mut gens: Vec<u64> = Vec::new();
        r.grant_credits("cpu-0", 10);
        while let Some(j) = r.next_job("cpu-0") {
            gens.push(j.generation);
        }
        assert_eq!(gens, vec![0, 6]);
    }

    #[test]
    fn miner_ids_are_sorted() {
        let mut r = Router::new();
        r.register_miner("cpu-1", caps_ising());
        r.register_miner("cpu-0", caps_ising());
        assert_eq!(
            r.miner_ids(),
            vec!["cpu-0".to_string(), "cpu-1".to_string()]
        );
    }

    #[test]
    fn stage_on_targets_specific_capable_miner() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        r.register_miner("cpu-1", caps_ising());
        // Directly stage on cpu-1 (first-fit `route` would have picked cpu-0).
        assert!(r.stage_on("cpu-1", make_job(1, JobKind::IsingSample)));
        assert_eq!(r.staged_len("cpu-0"), 0);
        assert_eq!(r.staged_len("cpu-1"), 1);
        // Unknown miner → not staged.
        assert!(!r.stage_on("nope", make_job(2, JobKind::IsingSample)));
    }

    #[test]
    fn on_reject_unsupported_stops_routing_kind() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        r.register_miner(
            "cpu-1",
            MinerCaps {
                backend: "mock".into(),
                algorithm: "sa".into(),
                supported_kinds: vec![JobKind::IsingSample as i32],
                max_nodes: 1000,
                max_edges: 10000,
                ..MinerCaps::default()
            },
        );
        let job = make_job(1, JobKind::IsingSample);
        let _ = r.route(job);
        r.grant_credits("cpu-0", 1);
        let dispatched = r.next_job("cpu-0").unwrap();
        r.on_reject("cpu-0", dispatched, RejectReason::UnsupportedKind as i32);
        // cpu-0 marked unsupported; re-route should land on cpu-1
        assert_eq!(r.staged_len("cpu-0"), 0);
        assert_eq!(r.staged_len("cpu-1"), 1);
        // further routes skip cpu-0
        let _ = r.route(make_job(2, JobKind::IsingSample));
        assert_eq!(r.staged_len("cpu-0"), 0);
        assert_eq!(r.staged_len("cpu-1"), 2);
    }

    /// A miner that rejects everything (an exhausted QPU budget) drops one job
    /// per dispatch. Those drops must collapse into one line per window, or the
    /// session log is unreadable for as long as the condition lasts.
    #[test]
    fn repeated_drops_collapse_into_one_warn_window() {
        let mut r = Router::new();
        r.register_miner("qpu-0", caps_ising());
        for g in 1..=50 {
            let job = make_job(g, JobKind::IsingSample);
            let _ = r.route(job);
            r.grant_credits("qpu-0", 1);
            let dispatched = r.next_job("qpu-0").expect("dispatched");
            r.on_reject("qpu-0", dispatched, RejectReason::Overloaded as i32);
        }
        let log = r.drops.get("qpu-0").expect("drop log");
        // The first drop reports at once and resets the counter; the other 49
        // are counted for the next line instead of being logged one by one.
        assert_eq!(log.since_warn, 49);
    }

    /// A one-off drop still reaches the log immediately: rate limiting must not
    /// swallow the first occurrence.
    #[test]
    fn first_drop_opens_a_window_immediately() {
        let mut r = Router::new();
        r.register_miner("qpu-0", caps_ising());
        let job = make_job(1, JobKind::IsingSample);
        let _ = r.route(job);
        r.grant_credits("qpu-0", 1);
        let dispatched = r.next_job("qpu-0").expect("dispatched");
        r.on_reject("qpu-0", dispatched, RejectReason::Overloaded as i32);
        assert_eq!(r.drops.get("qpu-0").expect("drop log").since_warn, 0);
    }

    #[test]
    fn on_reject_terminal_does_not_reroute_onto_same_single_miner() {
        // F6: with one capable miner, a deterministic reject must NOT re-stage
        // onto the rejecting miner (which would loop forever); it drops instead.
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        let job = make_job(1, JobKind::IsingSample);
        let _ = r.route(job);
        r.grant_credits("cpu-0", 1);
        let dispatched = r.next_job("cpu-0").unwrap();
        assert_eq!(r.staged_len("cpu-0"), 0);
        r.on_reject("cpu-0", dispatched, RejectReason::Malformed as i32);
        // Not re-staged onto the only (rejecting) miner.
        assert_eq!(r.staged_len("cpu-0"), 0);
    }

    #[test]
    fn reregister_preserves_staged_queue() {
        // F8: a supervised restart of the same id must keep jobs re-staged onto
        // it while it was down (reclaim → re-route → re-handshake path).
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        let _ = r.route(make_job(0, JobKind::IsingSample)); // mempool order (generation 0)
        assert_eq!(r.staged_len("cpu-0"), 1);
        // Re-handshake with the same id: queue survives, credits reset.
        r.register_miner("cpu-0", caps_ising());
        assert_eq!(r.staged_len("cpu-0"), 1);
    }

    #[test]
    fn indexes_native_hash_features_and_degree() {
        let wire = Capabilities {
            native_topology_hash: Some(vec![7; 32]),
            features: vec!["streaming".to_owned(), "initial-spins".to_owned()],
            max_degree: Some(20),
            ..Capabilities::default()
        };
        let caps = MinerCaps::from_capabilities(&wire);
        assert_eq!(caps.native_topology_hash, wire.native_topology_hash);
        assert_eq!(caps.features, wire.features);
        assert_eq!(caps.max_degree, Some(20));
    }

    #[test]
    fn capability_update_preserves_accounting_and_moves_incompatible_staged_jobs() {
        let mut r = Router::new();
        r.register_miner("a", caps_ising());
        r.register_miner("b", caps_ising());
        r.grant_credits("a", 2);
        assert!(r.stage_on("a", make_job(1, JobKind::IsingSample)));
        assert!(r.next_job("a").is_some());
        r.record_completion("a");
        r.record_lease_salts("a", 7);
        r.record_round_salts("a", 9);
        assert!(r.stage_on("a", make_job(2, JobKind::IsingSample)));

        assert!(r.update_capabilities(
            "a",
            MinerCaps {
                max_nodes: 1,
                ..caps_ising()
            }
        ));
        assert_eq!(r.staged_len("a"), 0);
        assert_eq!(r.staged_len("b"), 1);
        assert_eq!(r.jobs_completed("a"), 1);
        assert_eq!(r.take_consumed("a"), 1);
        assert_eq!(r.take_lease_salts("a"), 7);
        assert_eq!(r.round_salts("a"), 9);
        assert_eq!(r.miners.get("a").map(|q| q.granted_credits), Some(1));

        let mut small = make_job(3, JobKind::IsingSample);
        small.ising = Some(crate::producer::problem::milli_problem(None, &[0], &[]));
        assert!(r.stage_on("a", small));
        assert!(r.next_job("a").is_some());
        assert!(!r.update_capabilities("missing", caps_ising()));
        assert_eq!(r.miner_ids(), vec!["a".to_owned(), "b".to_owned()]);
    }

    #[test]
    fn fresh_snapshot_replaces_fields_and_clears_learned_unsupported_kinds() {
        let mut r = Router::new();
        r.register_miner(
            "a",
            MinerCaps {
                native_topology_hash: Some(vec![7; 32]),
                features: vec!["initial-spins".to_owned()],
                max_degree: Some(20),
                ..caps_ising()
            },
        );
        r.on_reject(
            "a",
            make_job(1, JobKind::IsingSample),
            RejectReason::UnsupportedKind as i32,
        );
        assert!(!r.stage_on("a", make_job(2, JobKind::IsingSample)));
        assert!(r.update_capabilities("a", caps_ising()));
        let caps = r.caps("a");
        assert_eq!(caps.and_then(|c| c.native_topology_hash.as_ref()), None);
        assert_eq!(caps.map(|c| c.features.as_slice()), Some(&[][..]));
        assert_eq!(caps.and_then(|c| c.max_degree), None);
        assert!(r.stage_on("a", make_job(3, JobKind::IsingSample)));
    }

    #[test]
    fn dispatch_rechecks_a_queue_preserved_across_reregistration() {
        let mut r = Router::new();
        r.register_miner("a", caps_ising());
        r.register_miner("b", caps_ising());
        assert!(r.stage_on("a", make_job(1, JobKind::IsingSample)));
        r.register_miner(
            "a",
            MinerCaps {
                max_nodes: 1,
                ..caps_ising()
            },
        );
        r.grant_credits("a", 1);
        assert!(r.next_job("a").is_none());
        assert_eq!(r.staged_len("b"), 1);
        assert_eq!(r.miners.get("a").map(|q| q.granted_credits), Some(1));
    }

    fn degree_job(degree: u32) -> Job {
        let h: Vec<i32> = (0..=degree).map(|_| 0).collect();
        let j: Vec<i32> = (0..degree).map(|_| 0).collect();
        let edges = quip_proto::v1::EdgeList {
            u: (0..degree).map(|_| 0).collect(),
            v: (1..=degree).collect(),
        };
        Job {
            job_id: degree.to_le_bytes().to_vec(),
            kind: JobKind::IsingSample as i32,
            ising: Some(crate::producer::problem::milli_problem(
                Some(ising_problem::Graph::Edges(edges)),
                &h,
                &j,
            )),
            ..Job::default()
        }
    }

    fn hash_job(topology: &Topology) -> Job {
        let h: Vec<i32> = topology.nodes.iter().map(|_| 0).collect();
        let j: Vec<i32> = topology.edges.0.iter().map(|_| 0).collect();
        Job {
            job_id: topology.hash.clone(),
            kind: JobKind::IsingSample as i32,
            ising: Some(crate::producer::problem::milli_problem(
                Some(ising_problem::Graph::TopologyHash(topology.hash.clone())),
                &h,
                &j,
            )),
            ..Job::default()
        }
    }

    #[test]
    fn degree_boundary_routes_and_stages_before_dispatch() {
        let mut r = Router::new();
        r.register_miner(
            "a-msa",
            MinerCaps {
                max_degree: Some(20),
                ..caps_ising()
            },
        );
        r.register_miner("z-cpu", caps_ising());
        assert_eq!(r.route(degree_job(20)).as_deref(), Some("a-msa"));
        assert_eq!(r.route(degree_job(21)).as_deref(), Some("z-cpu"));
        assert!(!r.stage_on("a-msa", degree_job(21)));
        assert!(r.stage_on("a-msa", degree_job(20)));
    }

    #[test]
    fn cached_sparse_graph_bounds_hash_jobs_and_leases() {
        let topology = Topology::from_nodes_edges(
            (0..=21).map(|n| 10 + n * 10).collect(),
            (1..=21).map(|n| (10, 10 + n * 10)).collect(),
            &[0],
            &[1000],
            &[-1000, 1000],
        );
        let mut r = Router::new();
        r.set_topology(Some(&topology));
        r.register_miner(
            "msa",
            MinerCaps {
                max_degree: Some(20),
                ..caps_lease()
            },
        );
        assert!(!r.stage_on("msa", hash_job(&topology)));
        let mut lease = lease_job(100);
        if let Some(generator) = lease.generator.as_mut() {
            generator.topology_hash.clone_from(&topology.hash);
        }
        assert!(!r.stage_on("msa", lease.clone()));
        assert!(r.update_capabilities(
            "msa",
            MinerCaps {
                max_degree: Some(21),
                max_nodes: 21,
                ..caps_lease()
            }
        ));
        assert!(!r.stage_on("msa", lease.clone()));
        assert!(r.update_capabilities(
            "msa",
            MinerCaps {
                max_degree: Some(21),
                max_edges: 20,
                ..caps_lease()
            }
        ));
        assert!(!r.stage_on("msa", lease.clone()));
        assert!(r.update_capabilities(
            "msa",
            MinerCaps {
                max_degree: Some(21),
                ..caps_lease()
            }
        ));
        assert!(r.stage_on("msa", lease));
        assert!(r.stage_on("msa", hash_job(&topology)));
    }

    #[test]
    fn degree_snapshot_rechecks_staged_jobs_without_spending_credits() {
        let mut r = Router::new();
        r.register_miner("a", caps_ising());
        r.register_miner("b", caps_ising());
        r.grant_credits("a", 2);
        r.grant_credits("b", 3);
        for degree in [21, 20, 22, 19, 23, 18] {
            assert!(r.stage_on("a", degree_job(degree)));
        }
        assert!(r.update_capabilities(
            "a",
            MinerCaps {
                max_degree: Some(20),
                ..caps_ising()
            }
        ));
        assert_eq!(
            r.miners.get("a").map(|q| q
                .staged
                .iter()
                .map(|j| j.job_id.clone())
                .collect::<Vec<_>>()),
            Some(vec![
                20_u32.to_le_bytes().to_vec(),
                19_u32.to_le_bytes().to_vec(),
                18_u32.to_le_bytes().to_vec(),
            ])
        );
        assert_eq!(
            r.miners.get("b").map(|q| q
                .staged
                .iter()
                .map(|j| j.job_id.clone())
                .collect::<Vec<_>>()),
            Some(vec![
                21_u32.to_le_bytes().to_vec(),
                22_u32.to_le_bytes().to_vec(),
                23_u32.to_le_bytes().to_vec(),
            ])
        );
        assert_eq!(r.miners.get("a").map(|q| q.granted_credits), Some(2));
        assert_eq!(r.miners.get("b").map(|q| q.granted_credits), Some(3));
        assert_eq!(r.take_consumed("a"), 0);
        assert_eq!(r.take_consumed("b"), 0);
        assert_eq!(
            r.next_job("a").map(|j| j.job_id),
            Some(20_u32.to_le_bytes().to_vec())
        );
    }

    #[test]
    fn unspecified_and_zero_degree_keep_the_fallback() {
        let mut r = Router::new();
        r.register_miner("m", caps_ising());
        assert!(r.stage_on("m", degree_job(21)));
        assert!(r.update_capabilities(
            "m",
            MinerCaps {
                max_degree: Some(0),
                ..caps_ising()
            }
        ));
        assert!(r.stage_on("m", degree_job(21)));
        assert!(r.update_capabilities(
            "m",
            MinerCaps {
                max_degree: Some(20),
                ..caps_ising()
            }
        ));
        let topology =
            Topology::from_nodes_edges(vec![0, 1], vec![(0, 1)], &[0], &[1000], &[-1000, 1000]);
        assert!(r.stage_on("m", hash_job(&topology)));
        r.set_topology(Some(&topology));
        assert!(r.stage_on("m", hash_job(&topology)));
        r.set_topology(None);
        assert!(r.stage_on("m", hash_job(&topology)));
    }

    #[test]
    fn degree_counts_duplicate_and_zero_coupling_edges() {
        let mut r = Router::new();
        r.register_miner(
            "m",
            MinerCaps {
                max_degree: Some(1),
                ..caps_ising()
            },
        );
        let duplicate = Job {
            kind: JobKind::IsingSample as i32,
            ising: Some(crate::producer::problem::milli_problem(
                Some(ising_problem::Graph::Edges(quip_proto::v1::EdgeList {
                    u: vec![0, 0],
                    v: vec![1, 1],
                })),
                &[0, 0],
                &[0, 0],
            )),
            ..Job::default()
        };
        assert!(!r.stage_on("m", duplicate));
        assert!(r.stage_on("m", degree_job(1)));
        assert!(r.stage_on("m", degree_job(0)));
    }

    fn assert_graph_job_staging(r: &mut Router, topology: &Topology, expected: bool) {
        assert_eq!(r.stage_on("m", hash_job(topology)), expected);
        let mut lease = lease_job(100);
        if let Some(generator) = lease.generator.as_mut() {
            generator.topology_hash.clone_from(&topology.hash);
        }
        assert_eq!(r.stage_on("m", lease), expected);
    }

    #[test]
    fn unmatched_and_cleared_topology_keep_the_fallback() {
        let topology = Topology::from_nodes_edges(
            vec![10, 20, 30],
            vec![(10, 20), (10, 30)],
            &[0],
            &[1000],
            &[-1000, 1000],
        );
        let mut r = Router::new();
        r.register_miner(
            "m",
            MinerCaps {
                max_degree: Some(1),
                ..caps_lease()
            },
        );
        r.set_topology(Some(&topology));
        assert_graph_job_staging(&mut r, &topology, false);

        let mut unmatched = topology.clone();
        unmatched.hash = vec![99; 32];
        assert_graph_job_staging(&mut r, &unmatched, true);
        r.set_topology(None);
        assert_graph_job_staging(&mut r, &topology, true);
    }

    #[test]
    fn invalid_topologies_clear_cached_graph_metrics() {
        let topology = Topology::from_nodes_edges(
            vec![10, 20, 30],
            vec![(10, 20), (10, 30)],
            &[0],
            &[1000],
            &[-1000, 1000],
        );
        let invalid = [
            Topology {
                edges: (vec![10, 10], vec![20]),
                ..topology.clone()
            },
            Topology {
                nodes: vec![10, 20, 30, 10],
                ..topology.clone()
            },
            Topology {
                edges: (vec![10], vec![10]),
                ..topology.clone()
            },
            Topology {
                edges: (vec![10], vec![40]),
                ..topology.clone()
            },
        ];
        let mut r = Router::new();
        r.register_miner(
            "m",
            MinerCaps {
                max_degree: Some(1),
                ..caps_lease()
            },
        );
        for invalid in invalid {
            r.set_topology(Some(&topology));
            assert_graph_job_staging(&mut r, &topology, false);
            r.set_topology(Some(&invalid));
            assert_graph_job_staging(&mut r, &topology, true);
        }
    }

    fn native_caps(topology: &Topology) -> MinerCaps {
        MinerCaps {
            native_topology_hash: crate::topology::native_topology_hash(
                &topology.nodes,
                &topology.edge_pairs(),
            )
            .map(|hash| hash.to_vec()),
            ..caps_ising()
        }
    }

    fn inline_job(nodes: usize, edges: &[(u32, u32)]) -> Job {
        let (u, v) = edges.iter().copied().unzip();
        Job {
            kind: JobKind::IsingSample as i32,
            ising: Some(crate::producer::problem::milli_problem(
                Some(ising_problem::Graph::Edges(quip_proto::v1::EdgeList {
                    u,
                    v,
                })),
                &vec![0; nodes],
                &vec![0; edges.len()],
            )),
            ..Job::default()
        }
    }

    #[test]
    fn native_router_accepts_hash_jobs_and_only_proven_inline_subgraphs() {
        let topology = Topology::from_nodes_edges(
            vec![0, 1, 2, 3],
            vec![(0, 1), (1, 2), (2, 3)],
            &[0],
            &[1000],
            &[-1000, 1000],
        );
        let caps = native_caps(&topology);
        assert!(caps.native_topology_hash.is_some());
        assert_ne!(caps.native_topology_hash.as_ref(), Some(&topology.hash));
        let mut r = Router::new();
        r.set_topology(Some(&topology));
        r.register_miner("a-qpu", caps);
        r.register_miner("z-cpu", caps_ising());
        assert_eq!(r.route(hash_job(&topology)).as_deref(), Some("a-qpu"));
        assert_eq!(r.route(inline_job(2, &[(1, 0)])).as_deref(), Some("a-qpu"));
        assert_eq!(r.route(inline_job(3, &[(0, 2)])).as_deref(), Some("z-cpu"));
        assert!(!r.stage_on("a-qpu", inline_job(3, &[(0, 2)])));
        assert!(!r.stage_on("a-qpu", inline_job(5, &[])));
        assert!(!r.stage_on("a-qpu", inline_job(2, &[(1, 2)])));
        assert!(r.stage_on("a-qpu", inline_job(2, &[])));
    }

    #[test]
    fn unknown_native_graph_keeps_hash_fallback_and_skips_inline_work() {
        let topology =
            Topology::from_nodes_edges(vec![0, 1], vec![(0, 1)], &[0], &[1000], &[-1000, 1000]);
        let mut r = Router::new();
        r.set_topology(Some(&topology));
        r.register_miner(
            "qpu",
            MinerCaps {
                native_topology_hash: Some(vec![8; 32]),
                ..caps_ising()
            },
        );
        assert!(r.stage_on("qpu", hash_job(&topology)));
        assert!(!r.stage_on("qpu", inline_job(2, &[(0, 1)])));
        let mut wrong_hash_job = hash_job(&topology);
        if let Some(ising) = wrong_hash_job.ising.as_mut() {
            ising.graph = Some(ising_problem::Graph::TopologyHash(vec![8; 32]));
        }
        assert!(!r.stage_on("qpu", wrong_hash_job));
        r.set_topology(None);
        assert!(!r.stage_on("qpu", hash_job(&topology)));
        assert!(!r.stage_on("qpu", inline_job(2, &[(0, 1)])));
    }

    #[test]
    fn inline_positions_are_never_relabelled_through_sparse_native_nodes() {
        let topology = Topology::from_nodes_edges(
            vec![10, 20, 30],
            vec![(10, 20), (20, 30)],
            &[0],
            &[1000],
            &[-1000, 1000],
        );
        let mut r = Router::new();
        r.set_topology(Some(&topology));
        r.register_miner("qpu", native_caps(&topology));
        assert!(r.stage_on("qpu", hash_job(&topology)));
        assert!(!r.stage_on("qpu", inline_job(2, &[(0, 1)])));
        assert!(!r.stage_on("qpu", inline_job(2, &[])));
    }

    #[test]
    fn native_snapshot_and_topology_changes_recheck_queued_jobs() {
        let topology =
            Topology::from_nodes_edges(vec![0, 1], vec![(0, 1)], &[0], &[1000], &[-1000, 1000]);
        let mut r = Router::new();
        r.set_topology(Some(&topology));
        r.register_miner("a", native_caps(&topology));
        r.register_miner("b", caps_ising());
        r.grant_credits("a", 2);
        assert!(r.stage_on("a", inline_job(2, &[(0, 1)])));
        assert!(r.update_capabilities(
            "a",
            MinerCaps {
                native_topology_hash: Some(vec![8; 32]),
                ..caps_ising()
            }
        ));
        assert_eq!(r.staged_len("a"), 0);
        assert_eq!(r.staged_len("b"), 1);
        assert_eq!(r.miners.get("a").map(|q| q.granted_credits), Some(2));
        assert!(r.update_capabilities("a", native_caps(&topology)));
        assert!(r.stage_on("a", inline_job(2, &[(0, 1)])));
        let replacement =
            Topology::from_nodes_edges(vec![0, 1], vec![], &[0], &[1000], &[-1000, 1000]);
        r.set_topology(Some(&replacement));
        assert!(r.next_job("a").is_none());
        assert_eq!(r.staged_len("b"), 2);
        assert_eq!(r.miners.get("a").map(|q| q.granted_credits), Some(2));
    }

    #[test]
    fn malformed_native_evidence_never_admits_inline_work() {
        let topology =
            Topology::from_nodes_edges(vec![0, 1], vec![(0, 1)], &[0], &[1000], &[-1000, 1000]);
        let mut r = Router::new();
        r.set_topology(Some(&topology));
        r.register_miner(
            "qpu",
            MinerCaps {
                native_topology_hash: Some(vec![0; 31]),
                ..caps_ising()
            },
        );
        assert!(!r.stage_on("qpu", inline_job(2, &[(0, 1)])));
        assert!(!r.stage_on("qpu", hash_job(&topology)));
        assert!(r.update_capabilities("qpu", native_caps(&topology)));
        let mut malformed = inline_job(2, &[(0, 1)]);
        if let Some(ising) = malformed.ising.as_mut() {
            ising.graph = Some(ising_problem::Graph::Edges(quip_proto::v1::EdgeList {
                u: vec![0],
                v: vec![],
            }));
        }
        assert!(!r.stage_on("qpu", malformed));
        let mut invalid_topology = topology;
        invalid_topology.edges.1.clear();
        r.set_topology(Some(&invalid_topology));
        assert!(!r.stage_on("qpu", inline_job(2, &[(0, 1)])));
    }

    fn assert_native_inline_rejected(
        r: &mut Router,
        mutate: impl FnOnce(&mut quip_proto::v1::IsingProblem),
    ) {
        let mut job = inline_job(2, &[(0, 1)]);
        if let Some(ising) = job.ising.as_mut() {
            mutate(ising);
        }
        assert!(!r.stage_on("qpu", job));
    }

    #[test]
    fn native_inline_proof_rejects_invalid_coefficients_and_missing_graphs() {
        let topology =
            Topology::from_nodes_edges(vec![0, 1], vec![(0, 1)], &[0], &[1000], &[-1000, 1000]);
        let mut r = Router::new();
        r.set_topology(Some(&topology));
        r.register_miner("qpu", native_caps(&topology));
        assert!(r.stage_on("qpu", inline_job(2, &[(0, 1)])));
        assert_native_inline_rejected(&mut r, |ising| {
            ising.encoding = quip_proto::v1::CoefficientEncoding::I8 as i32;
        });
        assert_native_inline_rejected(&mut r, |ising| ising.scale = 1);
        assert_native_inline_rejected(&mut r, |ising| ising.h.push(0));
        assert_native_inline_rejected(&mut r, |ising| ising.j.clear());
        assert_native_inline_rejected(&mut r, |ising| ising.j.push(0));
        assert_native_inline_rejected(&mut r, |ising| ising.graph = None);
        assert!(!r.stage_on("qpu", inline_job(2, &[(0, 0)])));
        assert!(!r.stage_on(
            "qpu",
            Job {
                kind: JobKind::IsingSample as i32,
                ..Job::default()
            }
        ));
    }

    #[test]
    fn native_hash_jobs_and_leases_use_only_active_chain_hash() {
        let topology =
            Topology::from_nodes_edges(vec![0, 1], vec![(0, 1)], &[0], &[1000], &[-1000, 1000]);
        let mut r = Router::new();
        r.register_miner(
            "m",
            MinerCaps {
                native_topology_hash: Some(vec![8; 32]),
                ..caps_lease()
            },
        );
        r.set_topology(Some(&topology));
        assert_graph_job_staging(&mut r, &topology, true);
        let mut wrong = topology.clone();
        wrong.hash = vec![8; 32];
        assert_graph_job_staging(&mut r, &wrong, false);
        r.set_topology(None);
        assert_graph_job_staging(&mut r, &topology, false);
        r.set_topology(Some(&topology));
        assert!(r.update_capabilities(
            "m",
            MinerCaps {
                native_topology_hash: Some(vec![0; 31]),
                ..caps_lease()
            }
        ));
        assert_graph_job_staging(&mut r, &topology, false);
    }

    #[test]
    fn refusal_fallback_still_excludes_the_rejecting_miner_and_drops_without_alternative() {
        for reason in [RejectReason::TopologyMismatch, RejectReason::TooLarge] {
            let mut r = Router::new();
            r.register_miner("a", caps_ising());
            r.register_miner("b", caps_ising());
            r.on_reject("a", make_job(1, JobKind::IsingSample), reason as i32);
            assert_eq!(r.staged_len("a"), 0);
            assert_eq!(r.staged_len("b"), 1);
            let mut single = Router::new();
            single.register_miner("a", caps_ising());
            single.on_reject("a", make_job(1, JobKind::IsingSample), reason as i32);
            assert_eq!(single.staged_len("a"), 0);
            assert_eq!(single.unroutable.len(), 1);
        }
    }
}
