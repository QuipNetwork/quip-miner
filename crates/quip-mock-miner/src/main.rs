//! Reference mock miner binary for protocol conformance testing.
//!
//! Implements the gRPC miner session loop against a coordinator, with a trivial
//! all-+1 sampler used only to exercise handshake, rejects, and result shape.

use clap::Parser;
use quip_proto::v1::miner_service_client::MinerServiceClient;
use quip_proto::v1::{
    coord_msg, ising_problem, miner_msg, CoordMsg, Fatal, IsingProblem, Job, JobKind, JobRequest,
    MinerMsg, Ready, Reject, RejectReason, Result as JobResult, SamplerMeta, Solution, Status,
    Topology,
};
use quip_protocol::scoring::energy_milli;
use quip_protocol::session::{
    build_hello, check_welcome, BackendCaps, ExitCode, SessionConfig, SessionError,
};
use quip_protocol::wire::decode_i32_le;
use std::process::ExitCode as ProcessExitCode;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Endpoint, Uri};

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[arg(long)]
    quip_coordinator: Option<String>,
    #[arg(long)]
    miner_id: Option<String>,
    #[arg(long)]
    capabilities: bool,
    #[arg(long)]
    check: bool,
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn print_capabilities() {
    // schema lives in quip-protocol; mock advertises a permissive envelope
    #[expect(
        clippy::print_stdout,
        reason = "CLI --capabilities prints JSON to stdout for the coordinator"
    )]
    {
        println!(
            r#"{{"backend":"mock","algorithm":"sa","supported_kinds":["ISING_SAMPLE"],"max_nodes":100000,"max_edges":1000000}}"#
        );
    }
}

/// Map a session/CLI failure to a documented exit code (64/69/70/77).
fn coded_exit(code: ExitCode) -> ProcessExitCode {
    // ExitCode discriminants are 0..=77; all fit in u8.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "ExitCode values are documented 0/64/69/70/77 and fit in u8"
    )]
    {
        ProcessExitCode::from(code.as_i32() as u8)
    }
}

#[tokio::main]
async fn main() -> ProcessExitCode {
    match run().await {
        Ok(()) => ProcessExitCode::SUCCESS,
        Err(code) => coded_exit(code),
    }
}

async fn run() -> Result<(), ExitCode> {
    let cli = Cli::parse();
    if cli.capabilities {
        print_capabilities();
        return Ok(());
    }
    if cli.check {
        // Mock is always runnable on this host; a real miner would probe CUDA/Metal
        // and return EnvIncompatible (69) when the backend cannot start.
        return Ok(());
    }
    let uri = cli.quip_coordinator.ok_or(ExitCode::ConfigInvalid)?;
    let miner_id = cli.miner_id.unwrap_or_else(|| "mock-0".into());
    run_session(&uri, &miner_id).await
}

fn now_unix_ms() -> u64 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "unix epoch millis fit in u64 for practical wall-clock times"
    )]
    {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

fn miner(msg: miner_msg::Msg) -> MinerMsg {
    MinerMsg { msg: Some(msg) }
}

fn status_msg(miner_id: &str) -> MinerMsg {
    miner(miner_msg::Msg::Status(Status {
        miner_id: miner_id.into(),
        utilization: 0.0,
        jobs_done: 0,
        abandoned_generation: 0,
        sampler_stats: std::collections::HashMap::default(),
    }))
}

fn fatal_msg(code: ExitCode, reason: impl Into<String>) -> MinerMsg {
    #[expect(
        clippy::cast_sign_loss,
        reason = "ExitCode values are non-negative documented codes 0/64/69/70/77"
    )]
    let exit_code = code.as_i32() as u32;
    miner(miner_msg::Msg::Fatal(Fatal {
        exit_code,
        reason: reason.into(),
        restart_required: true,
    }))
}

/// Milli-int-encoded field -> float vector; an empty field decodes to an empty vec.
fn decode_milli_f64(bytes: &[u8]) -> Result<Vec<f64>, quip_protocol::wire::WireError> {
    Ok(decode_i32_le(bytes)?
        .iter()
        .map(|&v| f64::from(v) / 1000.0)
        .collect())
}

/// Session-cached topology (mirrors quip-miner-core): resolves `TopologyHash`
/// jobs to dense edges, mapping native (possibly sparse) node ids to positions.
struct SessionTopo {
    hash: Vec<u8>,
    edges: Vec<(u32, u32)>,
    pos: std::collections::HashMap<u32, usize>,
}

impl SessionTopo {
    fn from_proto(t: &Topology) -> Self {
        let mut pos = std::collections::HashMap::with_capacity(t.nodes.len());
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
        }
    }
}

fn resolve_edges(
    ising: &IsingProblem,
    topo: Option<&SessionTopo>,
) -> Result<Vec<(usize, usize)>, RejectReason> {
    match &ising.graph {
        Some(ising_problem::Graph::Edges(e)) => Ok(e
            .u
            .iter()
            .zip(&e.v)
            .map(|(&u, &v)| (u as usize, v as usize))
            .collect()),
        Some(ising_problem::Graph::TopologyHash(h)) => {
            let topo = topo.ok_or(RejectReason::TopologyMissing)?;
            if *h != topo.hash {
                return Err(RejectReason::TopologyMismatch);
            }
            let mut out = Vec::with_capacity(topo.edges.len());
            for &(u, v) in &topo.edges {
                let pu = *topo.pos.get(&u).ok_or(RejectReason::Malformed)?;
                let pv = *topo.pos.get(&v).ok_or(RejectReason::Malformed)?;
                out.push((pu, pv));
            }
            Ok(out)
        }
        None => Ok(Vec::new()),
    }
}

fn reject(job_id: Vec<u8>, reason: RejectReason) -> MinerMsg {
    miner(miner_msg::Msg::Reject(Reject {
        job_id,
        reason: reason as i32,
    }))
}

/// A reject plus a `JobRequest` for one replacement credit. A reject is
/// terminal for the job just like a completion, so it must refund a credit to
/// the coordinator's consume-on-dispatch pool — otherwise every reject leaks a
/// slot and the pipeline starves. Mirrors the `Result` + `JobRequest` pair on
/// the success path.
fn reject_and_replace(job_id: Vec<u8>, reason: RejectReason) -> Vec<MinerMsg> {
    vec![
        reject(job_id, reason),
        miner(miner_msg::Msg::JobRequest(JobRequest { credits: 1 })),
    ]
}

/// Validate a job and produce the reply(s): a `Reject` on bad input, otherwise a
/// `Result` (all-+1 spins) followed by a `JobRequest` for one more credit.
fn handle_job(job: Job, topo: Option<&SessionTopo>) -> Vec<MinerMsg> {
    let job_id = job.job_id.clone();

    // UNSUPPORTED_KIND: only ISING_SAMPLE is implemented by this reference miner.
    if job.kind != JobKind::IsingSample as i32 {
        return reject_and_replace(job_id, RejectReason::UnsupportedKind);
    }

    let ising = job.ising.unwrap_or_default();

    // MALFORMED: h field is not a valid i32-LE array (length not a multiple of 4).
    let Ok(h) = decode_milli_f64(&ising.h_milli_le32) else {
        return reject_and_replace(job_id, RejectReason::Malformed);
    };

    // MALFORMED: j field must also be a valid i32-LE array (mirror h handling).
    let Ok(j) = decode_milli_f64(&ising.j_milli_le32) else {
        return reject_and_replace(job_id, RejectReason::Malformed);
    };

    // EXPIRED: the deadline has already passed.
    // deadline_ms == 0 means "no deadline" (only mempool/chain jobs carry one).
    if job.deadline_ms != 0 && job.deadline_ms < now_unix_ms() {
        return reject_and_replace(job_id, RejectReason::Expired);
    }

    let edges = match resolve_edges(&ising, topo) {
        Ok(e) => e,
        Err(reason) => return reject_and_replace(job_id, reason),
    };
    let n = h.len();

    // MALFORMED: an edge endpoint indexes past the node count (mirrors
    // quip-miner-core::parse_ising, which rejects out-of-bounds inline edges
    // instead of silently skipping them during scoring).
    if edges.iter().any(|&(u, v)| u >= n || v >= n) {
        return reject_and_replace(job_id, RejectReason::Malformed);
    }

    let spins = vec![1i8; n];
    let energy = energy_milli(&spins, &h, &j, &edges);
    let result = JobResult {
        job_id,
        solutions: vec![Solution {
            spins_bytes: vec![0x01u8; n],
            energy_milli: energy,
        }],
        // The reference miner produces a single trivial read with no annealing;
        // report that faithfully so results carry SamplerMeta like real miners.
        meta: Some(SamplerMeta {
            reads: 1,
            sweeps: 0,
            device_access_time_us: 0,
            ..Default::default()
        }),
    };
    vec![
        miner(miner_msg::Msg::Result(result)),
        miner(miner_msg::Msg::JobRequest(JobRequest { credits: 1 })),
    ]
}

#[expect(
    clippy::too_many_lines,
    reason = "session loop kept as one linear match over CoordMsg for clarity"
)]
async fn run_session(uri: &str, miner_id: &str) -> Result<(), ExitCode> {
    // Resolve token before any network I/O so a missing QUIP_SESSION_TOKEN
    // always maps to exit 77 (never InternalFatal from a connect failure).
    let hello = build_hello(
        miner_id,
        "mock",
        "sa",
        &[JobKind::IsingSample],
        BackendCaps {
            max_nodes: 0,
            max_edges: 0,
        },
    )
    .map_err(|e: SessionError| ExitCode::from(e))?;

    let path = uri.strip_prefix("unix://").unwrap_or(uri).to_string();
    let channel = Endpoint::try_from("http://[::]:50051") // dummy authority, unused for UDS
        .map_err(|_| ExitCode::InternalFatal)?
        .connect_with_connector(tower::service_fn(move |_: Uri| {
            let p = path.clone();
            async move {
                let s = tokio::net::UnixStream::connect(p).await?;
                Ok::<_, std::io::Error>(hyper_util::rt::TokioIo::new(s))
            }
        }))
        .await
        .map_err(|_| ExitCode::InternalFatal)?;
    let mut client = MinerServiceClient::new(channel);

    let (tx, rx) = mpsc::channel::<MinerMsg>(16);
    // Send Hello before opening the inbound stream so the coordinator's handshake
    // has something to read immediately.
    tx.send(miner(miner_msg::Msg::Hello(hello)))
        .await
        .map_err(|_| ExitCode::InternalFatal)?;

    let mut inbound = client
        .session(ReceiverStream::new(rx))
        .await
        .map_err(|_| ExitCode::InternalFatal)?
        .into_inner();

    let mut config: Option<SessionConfig> = None;
    let mut grace_ms: u64 = 5000;
    let mut session_err: Option<ExitCode> = None;
    let mut session_topo: Option<SessionTopo> = None;
    loop {
        let idle = u64::from(config.as_ref().map_or(300, |c| c.idle_timeout_s));
        let next = tokio::time::timeout(Duration::from_secs(idle), inbound.message()).await;
        let cm: CoordMsg = match next {
            // idle timeout with no job, or coordinator closed the stream -> clean exit
            Err(_) | Ok(Ok(None)) => break,
            Ok(Ok(Some(cm))) => cm,
            Ok(Err(_)) => return Err(ExitCode::InternalFatal),
        };
        match cm.msg {
            Some(coord_msg::Msg::Welcome(w)) => {
                if let Err(e) = check_welcome(&w) {
                    let reason = e.to_string();
                    let code = ExitCode::from(e);
                    let _ = tx.send(fatal_msg(code, reason)).await;
                    session_err = Some(code);
                    break;
                }
            }
            Some(coord_msg::Msg::Configure(c)) => {
                config = Some(SessionConfig::from_configure(miner_id.into(), &c));
                tx.send(miner(miner_msg::Msg::Ready(Ready {})))
                    .await
                    .map_err(|_| ExitCode::InternalFatal)?;
            }
            Some(coord_msg::Msg::Topology(t)) => {
                session_topo = Some(SessionTopo::from_proto(&t));
            }
            Some(coord_msg::Msg::Job(job)) => {
                for reply in handle_job(job, session_topo.as_ref()) {
                    tx.send(reply).await.map_err(|_| ExitCode::InternalFatal)?;
                }
            }
            Some(coord_msg::Msg::Cancel(_)) => {
                // No jobs buffered in this mock; acknowledge via Status.
                tx.send(status_msg(miner_id))
                    .await
                    .map_err(|_| ExitCode::InternalFatal)?;
            }
            Some(coord_msg::Msg::Ping(_)) => {
                tx.send(status_msg(miner_id))
                    .await
                    .map_err(|_| ExitCode::InternalFatal)?;
            }
            Some(coord_msg::Msg::Shutdown(s)) => {
                grace_ms = if s.grace_ms == 0 {
                    5000
                } else {
                    u64::from(s.grace_ms)
                };
                break;
            }
            // SetTarget and empty oneof: nothing to do in this reference miner.
            Some(coord_msg::Msg::SetTarget(_)) | None => {}
        }
    }
    // Signal end-of-outbound so tonic flushes every buffered reply, then drain
    // inbound until the coordinator closes its side. Without this the process
    // (and its runtime) can tear down before queued Results reach the socket.
    // Bounded by grace_ms: a coordinator that never closes its side (or dies)
    // must not hang this drain forever.
    drop(tx);
    let drain = async {
        while inbound.message().await?.is_some() {}
        Ok::<(), tonic::Status>(())
    };
    let _ = tokio::time::timeout(Duration::from_millis(grace_ms), drain).await;
    match session_err {
        Some(code) => Err(code),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_protocol::wire::encode_i32_le;

    fn sample_job(job_id: &[u8], kind: JobKind, j_bytes: Vec<u8>) -> Job {
        Job {
            job_id: job_id.to_vec(),
            kind: kind as i32,
            generation: 1,
            deadline_ms: now_unix_ms() + 60_000,
            ising: Some(IsingProblem {
                graph: Some(ising_problem::Graph::Edges(quip_proto::v1::EdgeList {
                    u: vec![0],
                    v: vec![1],
                })),
                h_milli_le32: encode_i32_le(&[1000, -1000]),
                j_milli_le32: j_bytes,
                num_reads: 1,
                num_sweeps: 0,
                anneal_time_us: 0,
            }),
            provenance: None,
        }
    }

    fn first_reject_reason(msgs: &[MinerMsg]) -> Option<i32> {
        msgs.iter().find_map(|m| match &m.msg {
            Some(miner_msg::Msg::Reject(r)) => Some(r.reason),
            _ => None,
        })
    }

    fn first_msg_job_id(msgs: &[MinerMsg]) -> &[u8] {
        msgs.first()
            .and_then(|m| match &m.msg {
                Some(miner_msg::Msg::Reject(r)) => Some(r.job_id.as_slice()),
                _ => None,
            })
            .unwrap_or(b"")
    }

    #[test]
    fn gate_circuit_rejects_unsupported_kind() {
        let msgs = handle_job(
            sample_job(b"gate", JobKind::GateCircuit, encode_i32_le(&[500])),
            None,
        );
        assert_eq!(
            first_reject_reason(&msgs),
            Some(RejectReason::UnsupportedKind as i32)
        );
        assert_eq!(first_msg_job_id(&msgs), b"gate");
    }

    #[test]
    fn malformed_j_rejects_malformed() {
        let msgs = handle_job(
            sample_job(
                b"bad-j",
                JobKind::IsingSample,
                vec![0x01, 0x02, 0x03], // len 3, not a multiple of 4
            ),
            None,
        );
        assert_eq!(
            first_reject_reason(&msgs),
            Some(RejectReason::Malformed as i32)
        );
        assert_eq!(first_msg_job_id(&msgs), b"bad-j");
    }

    #[test]
    fn out_of_bounds_edge_rejects_malformed() {
        // h has 2 nodes (indices 0, 1); the inline edge references node 2,
        // which is out of bounds and must be rejected rather than silently
        // skipped during scoring.
        let mut job = sample_job(b"oob-edge", JobKind::IsingSample, encode_i32_le(&[500]));
        job.ising.as_mut().unwrap().graph =
            Some(ising_problem::Graph::Edges(quip_proto::v1::EdgeList {
                u: vec![0],
                v: vec![2],
            }));
        let msgs = handle_job(job, None);
        assert_eq!(
            first_reject_reason(&msgs),
            Some(RejectReason::Malformed as i32)
        );
        assert_eq!(first_msg_job_id(&msgs), b"oob-edge");
    }
}
