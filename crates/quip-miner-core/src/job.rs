//! Job validation and dispatch: wire fields → [`IsingGraph`] → [`Sampler`] →
//! `Result`/`Reject`.

use crate::adapt::adapt_params;
use crate::ising::{IsingGraph, SampleParams};
use crate::session::BackendIdentity;
use crate::Sampler;
use crate::{StreamJob, StreamOutcome, StreamResult};
use quip_proto::v1::{
    ising_problem, miner_msg, IsingProblem, Job, JobKind, JobRequest, MinerMsg, Reject,
    RejectReason, Result as JobResult, SamplerMeta, Solution, Status, Topology,
};
use quip_protocol::wire::{decode_i32_le, encode_spins, WireError};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Session-cached topology, used to resolve `TopologyHash` jobs to dense edges.
///
/// Built once from the `Topology` message at Configure. `pos` maps each native
/// node id to its dense position (id → index in received order), so sparse
/// D-Wave qubit ids resolve without a per-job remap.
pub(crate) struct TopologyCache {
    hash: Vec<u8>,
    edges: Vec<(u32, u32)>,
    pos: HashMap<u32, usize>,
    allowed_h: Vec<i32>,
}

impl TopologyCache {
    /// Build from a `Topology` message. Node ids map to their received-order
    /// position; edges keep received order (the consensus `j`-zip invariant).
    pub(crate) fn from_proto(t: &Topology) -> Self {
        let mut pos = HashMap::with_capacity(t.nodes.len());
        for (i, &node) in t.nodes.iter().enumerate() {
            let _ = pos.insert(node, i);
        }
        let edges = t.edges.as_ref().map_or_else(Vec::new, |e| {
            e.u.iter().zip(&e.v).map(|(&u, &v)| (u, v)).collect()
        });
        Self {
            hash: t.hash.clone(),
            edges,
            pos,
            allowed_h: t.allowed_h_milli.clone(),
        }
    }

    pub(crate) fn allowed_h(&self) -> &[i32] {
        &self.allowed_h
    }
}

/// Session difficulty target from `SetTarget`. The miner adapts its sampling
/// budget from `max_energy_milli`; the `num_*` fields are optional overrides.
pub(crate) struct SessionTarget {
    pub(crate) max_energy_milli: i64,
    pub(crate) min_solutions: u32,
    pub(crate) num_reads: u32,
    pub(crate) num_sweeps: u32,
}

impl SessionTarget {
    // anneal_time_us is ignored on the SA/GPU Rust path (QPU adapt lives in the
    // Python dwave miner).
    pub(crate) fn from_proto(s: &quip_proto::v1::SetTarget) -> Self {
        Self {
            max_energy_milli: s.max_energy_milli,
            min_solutions: s.min_solutions,
            num_reads: s.num_reads,
            num_sweeps: s.num_sweeps,
        }
    }
}

/// Resolve one sampling param: per-job override, else `SetTarget` override,
/// else the adapted value, else the fallback. `0` means "unset".
fn pick_param(job: u32, target: u32, adapt: Option<u32>, fallback: u32) -> u32 {
    if job != 0 {
        job
    } else if target != 0 {
        target
    } else {
        adapt.unwrap_or(fallback)
    }
}

pub(crate) const DEFAULT_NUM_SWEEPS: usize = 64;

// v0.2 parity: the Python GPU miners ran Gibbs at 2x the SA sweep budget
// (`GIBBS_SWEEP_MULTIPLIER` in GPU/cuda_miner.py, GPU/metal_miner.py) because
// Gibbs converges slower per sweep than SA. The v0.3 adapt path resolves an
// algorithm-agnostic `num_sweeps`, so this gate restores that parity for all
// backends (cpu/cuda/metal) sharing this code path.
const GIBBS_SWEEP_MULTIPLIER: u32 = 2;

pub(crate) fn now_unix_ms() -> u64 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "unix ms since epoch fits u64 for the lifetime of this protocol"
    )]
    {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// Fresh per-job seed from OS entropy. Two jobs issued in the same
/// millisecond must not sample identically, which a wall-clock-derived seed
/// would cause (miner-core is shared by cpu/cuda/metal).
fn os_seed() -> u64 {
    let mut bytes = [0u8; 8];
    #[expect(
        clippy::expect_used,
        reason = "OS CSPRNG failure is unrecoverable at process level; seed is required"
    )]
    {
        getrandom::getrandom(&mut bytes).expect("os rng");
    }
    u64::from_le_bytes(bytes)
}

pub(crate) fn miner(msg: miner_msg::Msg) -> MinerMsg {
    MinerMsg { msg: Some(msg) }
}

pub(crate) fn status_msg(miner_id: &str, jobs_done: u64, utilization: f64) -> MinerMsg {
    miner(miner_msg::Msg::Status(Status {
        miner_id: miner_id.into(),
        utilization,
        jobs_done,
        abandoned_generation: 0,
        sampler_stats: HashMap::default(),
    }))
}

pub(crate) fn reject(job_id: Vec<u8>, reason: RejectReason) -> MinerMsg {
    miner(miner_msg::Msg::Reject(Reject {
        job_id,
        reason: reason as i32,
    }))
}

/// Milli-int-encoded field → float vector.
fn decode_milli_f64(bytes: &[u8]) -> Result<Vec<f64>, WireError> {
    Ok(decode_i32_le(bytes)?
        .iter()
        .map(|&v| f64::from(v) / 1000.0)
        .collect())
}

fn resolve_edges(
    ising: &IsingProblem,
    cache: Option<&TopologyCache>,
) -> Result<Vec<(usize, usize)>, RejectReason> {
    match &ising.graph {
        // Inline edges use dense 0..n-1 ids straight off the wire.
        Some(ising_problem::Graph::Edges(e)) => {
            if e.u.len() != e.v.len() {
                return Err(RejectReason::Malformed);
            }
            Ok(e.u
                .iter()
                .zip(&e.v)
                .map(|(&u, &v)| (u as usize, v as usize))
                .collect())
        }
        // Topology-hash jobs resolve against the session-cached topology,
        // mapping native (possibly sparse) node ids to dense positions.
        Some(ising_problem::Graph::TopologyHash(h)) => {
            let cache = cache.ok_or(RejectReason::TopologyMissing)?;
            if *h != cache.hash {
                return Err(RejectReason::TopologyMismatch);
            }
            let mut out = Vec::with_capacity(cache.edges.len());
            for &(u, v) in &cache.edges {
                let pu = *cache.pos.get(&u).ok_or(RejectReason::Malformed)?;
                let pv = *cache.pos.get(&v).ok_or(RejectReason::Malformed)?;
                out.push((pu, pv));
            }
            Ok(out)
        }
        None => Ok(Vec::new()),
    }
}

/// Validate wire fields and build the base Ising graph, or a reject reason.
fn parse_ising(
    ising: &IsingProblem,
    max_nodes: u32,
    max_edges: u32,
    cache: Option<&TopologyCache>,
) -> Result<IsingGraph, RejectReason> {
    let h = decode_milli_f64(&ising.h_milli_le32).map_err(|_| RejectReason::Malformed)?;
    let j = decode_milli_f64(&ising.j_milli_le32).map_err(|_| RejectReason::Malformed)?;
    let edges = resolve_edges(ising, cache)?;
    if !edges.is_empty() && j.len() != edges.len() {
        return Err(RejectReason::Malformed);
    }
    let n = h.len();
    if n > max_nodes as usize || edges.len() > max_edges as usize {
        return Err(RejectReason::TooLarge);
    }
    for &(u, v) in &edges {
        if u >= n || v >= n {
            return Err(RejectReason::Malformed);
        }
    }
    Ok(IsingGraph::new(h, j, edges))
}

/// Best-effort parse of `num_sweeps = N` from `Configure.backend_toml`.
pub(crate) fn num_sweeps_from_toml(backend_toml: &str) -> usize {
    for line in backend_toml.lines() {
        let line = line.split('#').next().unwrap_or(line).trim();
        if let Some(rest) = line.strip_prefix("num_sweeps") {
            let rest = rest.trim().trim_start_matches('=').trim();
            if let Ok(n) = rest.parse::<usize>() {
                if n > 0 {
                    return n;
                }
            }
        }
    }
    DEFAULT_NUM_SWEEPS
}

/// A validated job ready to sample, or an immediate reject reply.
pub(crate) enum Prepared {
    /// Reject reply to send now (no sampling).
    Reject(MinerMsg),
    /// Hand to the streaming sampler; `num_reads`/`num_sweeps` are the resolved
    /// values, carried so [`finalize_result`] can build `SamplerMeta`.
    Sample {
        job: StreamJob,
        num_reads: u32,
        num_sweeps: u32,
    },
}

/// Validate + resolve one job: reject reasons short-circuit; otherwise resolve
/// the sampling budget (per-job override > `SetTarget` override > adapt > default)
/// and return a [`StreamJob`] for the streaming sampler.
pub(crate) fn prepare_job<S: Sampler>(
    job: Job,
    sampler: &S,
    id: &BackendIdentity,
    default_sweeps: usize,
    sweeps_per_beta: Option<usize>,
    cache: Option<&TopologyCache>,
    target: Option<&SessionTarget>,
) -> Prepared {
    let job_id = job.job_id.clone();
    // Capture before `job.ising` is moved out below; threads into StreamJob so
    // the sampler knows what a Cancel invalidates.
    let generation = job.generation;

    if job.kind != JobKind::IsingSample as i32 {
        return Prepared::Reject(reject(job_id, RejectReason::UnsupportedKind));
    }
    // deadline_ms == 0 means "no deadline" (only mempool/chain jobs carry one).
    if job.deadline_ms != 0 && job.deadline_ms < now_unix_ms() {
        return Prepared::Reject(reject(job_id, RejectReason::Expired));
    }
    let Some(ising) = job.ising else {
        return Prepared::Reject(reject(job_id, RejectReason::Malformed));
    };

    let graph = match parse_ising(&ising, id.max_nodes, id.max_edges, cache) {
        Ok(g) => g,
        Err(reason) => return Prepared::Reject(reject(job_id, reason)),
    };

    // adapt runs only when a target is set (uses the parsed problem's node/edge
    // counts and the topology's allowed_h).
    let adapt = target.map(|t| {
        let allowed_h = cache.map_or(&[][..], TopologyCache::allowed_h);
        adapt_params(
            t.max_energy_milli,
            t.min_solutions.max(1),
            graph.h.len(),
            graph.edges.len(),
            allowed_h,
            &id.adapt,
        )
    });
    let t_reads = target.map_or(0, |t| t.num_reads);
    let t_sweeps = target.map_or(0, |t| t.num_sweeps);
    let num_reads = pick_param(ising.num_reads, t_reads, adapt.map(|a| a.num_reads), 1);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "default_sweeps comes from config/CLI and is a small sweep budget well under u32::MAX"
    )]
    let num_sweeps = pick_param(
        ising.num_sweeps,
        t_sweeps,
        adapt.map(|a| a.num_sweeps),
        default_sweeps as u32,
    );
    let num_sweeps = if id.algorithm == "gibbs" {
        num_sweeps.saturating_mul(GIBBS_SWEEP_MULTIPLIER)
    } else {
        num_sweeps
    };

    if num_reads > sampler.max_reads() {
        return Prepared::Reject(reject(job_id, RejectReason::TooLarge));
    }

    let params = SampleParams {
        num_reads: num_reads as usize,
        num_sweeps: num_sweeps as usize,
        seed: os_seed(),
        sweeps_per_beta: sweeps_per_beta.unwrap_or(1),
        ..Default::default()
    };
    Prepared::Sample {
        job: StreamJob {
            job_id,
            graph,
            params,
            generation,
        },
        num_reads,
        num_sweeps,
    }
}

/// Turn a [`StreamResult`] into the reply(s): a `Result` on completion, a
/// `Reject` on sampler error, or nothing upstream when the job's generation was
/// cancelled. Every path refunds one credit (`JobRequest{1}`) so the
/// coordinator's consume-on-dispatch pool never leaks a slot. `num_reads`/
/// `num_sweeps` are the resolved values from [`prepare_job`], echoed into
/// `SamplerMeta`.
pub(crate) fn finalize_result(
    sr: StreamResult,
    num_reads: u32,
    num_sweeps: u32,
    jobs_done: &mut u64,
) -> Vec<MinerMsg> {
    let samples = match sr.outcome {
        StreamOutcome::Completed(Ok(s)) => s,
        // A reject is terminal for this job too, so replace its credit like a
        // completion does — otherwise the coordinator's consume-on-dispatch
        // pool leaks one slot per reject and the pipeline slowly starves.
        StreamOutcome::Completed(Err(reason)) => {
            return vec![
                reject(sr.job_id, reason),
                miner(miner_msg::Msg::JobRequest(JobRequest { credits: 1 })),
            ]
        }
        // Generation abandoned on reseed: the coordinator has moved on, so send
        // no stale Result/Reject — only refund the credit to keep pipeline depth.
        StreamOutcome::Cancelled => {
            return vec![miner(miner_msg::Msg::JobRequest(JobRequest { credits: 1 }))]
        }
    };
    let solutions: Vec<Solution> = samples
        .into_iter()
        .map(|r| Solution {
            spins_bytes: encode_spins(&r.spins),
            energy_milli: r.energy_milli,
        })
        .collect();

    *jobs_done = jobs_done.saturating_add(1);

    let result = JobResult {
        job_id: sr.job_id,
        solutions,
        meta: Some(SamplerMeta {
            reads: num_reads,
            sweeps: num_sweeps,
            device_access_time_us: sr.device_access_time_us,
            qpu_access_us: 0,
            extra: HashMap::default(),
        }),
    };

    vec![
        miner(miner_msg::Msg::Result(result)),
        miner(miner_msg::Msg::JobRequest(JobRequest { credits: 1 })),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapt::AdaptBounds;
    use crate::ising::{IsingGraph, SamplerResult};
    use quip_proto::v1::{ising_problem::Graph, EdgeList, Topology};
    use quip_protocol::wire::encode_i32_le;

    struct StubSampler;

    impl Sampler for StubSampler {
        fn sample(
            &self,
            _graph: &IsingGraph,
            _params: &SampleParams,
        ) -> Result<Vec<SamplerResult>, RejectReason> {
            Ok(vec![])
        }
    }

    const TEST_ADAPT: AdaptBounds = AdaptBounds {
        min_sweeps: 1,
        max_sweeps: u32::MAX,
        min_reads: 1,
        max_reads: u32::MAX,
        reads_solution_min_factor: 1,
        reads_solution_max_factor: 1,
        reads_solution_floor_factor: 0,
    };

    fn identity(algorithm: &'static str) -> BackendIdentity {
        BackendIdentity {
            backend: "test",
            algorithm,
            max_nodes: 100_000,
            max_edges: 1_000_000,
            adapt: TEST_ADAPT,
        }
    }

    /// A minimal valid `IsingSample` job with inline edges (no topology cache
    /// needed) and `num_sweeps` left unset (0), so `prepare_job` falls back to
    /// `default_sweeps`.
    fn edges_job(job_id: u8) -> Job {
        Job {
            job_id: vec![job_id],
            kind: JobKind::IsingSample as i32,
            generation: 0,
            deadline_ms: now_unix_ms() + 60_000,
            ising: Some(IsingProblem {
                graph: Some(Graph::Edges(EdgeList {
                    u: vec![0],
                    v: vec![1],
                })),
                h_milli_le32: encode_i32_le(&[1000, 1000]),
                j_milli_le32: encode_i32_le(&[1000]),
                num_reads: 0,
                num_sweeps: 0,
                anneal_time_us: 0,
            }),
            provenance: None,
        }
    }

    fn hash_job(hash: Vec<u8>, h_milli: &[i32], j_milli: &[i32]) -> IsingProblem {
        IsingProblem {
            graph: Some(Graph::TopologyHash(hash)),
            h_milli_le32: encode_i32_le(h_milli),
            j_milli_le32: encode_i32_le(j_milli),
            num_reads: 0,
            num_sweeps: 0,
            anneal_time_us: 0,
        }
    }

    #[test]
    fn topology_cache_maps_sparse_ids_to_positions() {
        let topo = Topology {
            hash: vec![0xAB; 32],
            nodes: vec![0, 12, 2400],
            allowed_h_milli: vec![],
            edges: Some(EdgeList {
                u: vec![0, 12],
                v: vec![12, 2400],
            }),
        };
        let c = TopologyCache::from_proto(&topo);
        assert_eq!(c.pos.get(&0), Some(&0));
        assert_eq!(c.pos.get(&12), Some(&1));
        assert_eq!(c.pos.get(&2400), Some(&2));
        // received order, unsorted
        assert_eq!(c.edges, vec![(0, 12), (12, 2400)]);
        assert_eq!(c.hash, vec![0xAB; 32]);
    }

    #[test]
    fn hash_job_resolves_sparse_edges_to_positions() {
        let topo = Topology {
            hash: vec![7; 32],
            nodes: vec![0, 12, 2400],
            allowed_h_milli: vec![],
            edges: Some(EdgeList {
                u: vec![0, 12],
                v: vec![12, 2400],
            }),
        };
        let cache = TopologyCache::from_proto(&topo);
        let ising = hash_job(vec![7; 32], &[1000, -1000, 1000], &[1000, -1000]);
        let g = parse_ising(&ising, 100_000, 1_000_000, Some(&cache)).unwrap();
        assert_eq!(g.edges, vec![(0, 1), (1, 2)]);
        assert_eq!(g.h, vec![1.0, -1.0, 1.0]);
        assert_eq!(g.j, vec![1.0, -1.0]);
    }

    #[test]
    fn hash_job_without_cache_rejects_missing() {
        let ising = hash_job(vec![7; 32], &[1000], &[]);
        let err = parse_ising(&ising, 100_000, 1_000_000, None).unwrap_err();
        assert_eq!(err, RejectReason::TopologyMissing);
    }

    #[test]
    fn hash_job_wrong_hash_rejects_mismatch() {
        let topo = Topology {
            hash: vec![1; 32],
            nodes: vec![0],
            allowed_h_milli: vec![],
            edges: None,
        };
        let cache = TopologyCache::from_proto(&topo);
        let ising = hash_job(vec![2; 32], &[1000], &[]);
        let err = parse_ising(&ising, 100_000, 1_000_000, Some(&cache)).unwrap_err();
        assert_eq!(err, RejectReason::TopologyMismatch);
    }

    #[test]
    fn param_precedence_job_over_target_over_adapt_over_fallback() {
        // per-job override wins over everything
        assert_eq!(pick_param(5, 7, Some(268), 1), 5);
        // SetTarget override wins when no per-job override
        assert_eq!(pick_param(0, 7, Some(268), 1), 7);
        // adapt value when neither override set
        assert_eq!(pick_param(0, 0, Some(268), 1), 268);
        // fallback when no target/adapt (e.g. no SetTarget cached)
        assert_eq!(pick_param(0, 0, None, 64), 64);
    }

    #[test]
    fn session_target_from_proto_carries_target_and_overrides() {
        let s = quip_proto::v1::SetTarget {
            max_energy_milli: -14_700_000,
            min_solutions: 5,
            min_diversity_milli: 200,
            num_reads: 0,
            num_sweeps: 99,
            anneal_time_us: 0,
        };
        let t = SessionTarget::from_proto(&s);
        assert_eq!(t.max_energy_milli, -14_700_000);
        assert_eq!(t.min_solutions, 5);
        assert_eq!(t.num_sweeps, 99);
        assert_eq!(t.num_reads, 0);
    }

    #[test]
    fn hash_job_edge_id_absent_from_map_rejects_malformed() {
        // edge references id 999, which is not in nodes → no position.
        let topo = Topology {
            hash: vec![7; 32],
            nodes: vec![0, 1],
            allowed_h_milli: vec![],
            edges: Some(EdgeList {
                u: vec![0],
                v: vec![999],
            }),
        };
        let cache = TopologyCache::from_proto(&topo);
        let ising = hash_job(vec![7; 32], &[1000, 1000], &[1000]);
        let err = parse_ising(&ising, 100_000, 1_000_000, Some(&cache)).unwrap_err();
        assert_eq!(err, RejectReason::Malformed);
    }

    #[test]
    fn gibbs_job_resolves_2x_the_sweeps_of_the_same_sa_job() {
        let sampler = StubSampler;
        let default_sweeps = 64;

        let sa_id = identity("sa");
        let Prepared::Sample {
            num_sweeps: sa_sweeps,
            ..
        } = prepare_job(
            edges_job(1),
            &sampler,
            &sa_id,
            default_sweeps,
            None,
            None,
            None,
        )
        else {
            panic!("expected Sample");
        };

        let gibbs_id = identity("gibbs");
        let Prepared::Sample {
            num_sweeps: gibbs_sweeps,
            ..
        } = prepare_job(
            edges_job(2),
            &sampler,
            &gibbs_id,
            default_sweeps,
            None,
            None,
            None,
        )
        else {
            panic!("expected Sample");
        };

        #[expect(
            clippy::cast_possible_truncation,
            reason = "test default_sweeps is a small constant well under u32::MAX"
        )]
        {
            assert_eq!(sa_sweeps, default_sweeps as u32);
            assert_eq!(gibbs_sweeps, sa_sweeps * GIBBS_SWEEP_MULTIPLIER);
        }
    }

    #[test]
    fn back_to_back_jobs_get_different_os_entropy_seeds() {
        let sampler = StubSampler;
        let id = identity("sa");

        let Prepared::Sample { job: job1, .. } =
            prepare_job(edges_job(1), &sampler, &id, 64, None, None, None)
        else {
            panic!("expected Sample");
        };
        let Prepared::Sample { job: job2, .. } =
            prepare_job(edges_job(2), &sampler, &id, 64, None, None, None)
        else {
            panic!("expected Sample");
        };

        assert_ne!(job1.params.seed, job2.params.seed);
    }

    #[test]
    fn sweeps_per_beta_flag_flows_into_sample_params() {
        let sampler = StubSampler;
        let id = identity("sa");

        // The miner-local CLI value threads through to SampleParams.
        let Prepared::Sample { job, .. } =
            prepare_job(edges_job(1), &sampler, &id, 64, Some(3), None, None)
        else {
            panic!("expected Sample");
        };
        assert_eq!(job.params.sweeps_per_beta, 3);

        // Unset falls back to the default of 1.
        let Prepared::Sample { job, .. } =
            prepare_job(edges_job(2), &sampler, &id, 64, None, None, None)
        else {
            panic!("expected Sample");
        };
        assert_eq!(job.params.sweeps_per_beta, 1);
    }
}
