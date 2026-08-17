//! Generalized drive harness: spawn a miner, stage every job pulled from a
//! `JobSource` up front, validate each `Result` against its gates, and send a
//! clean `Shutdown` once every job has reached a terminal outcome.
//!
//! v1 is single-miner (scope guardrail), so a `Reject` is recorded as a
//! failing row rather than re-queued to the same miner: re-routing to the
//! only capable miner cannot resolve a structural reject (malformed/too
//! large) and risks an infinite reject loop. Production `run_session`
//! (`session.rs`) keeps the multi-miner reroute behavior; this is a
//! deliberate drive-mode-only deviation.

use crate::config::LaunchEntry;
use crate::drive::report::JobRow;
use crate::router::MinerCaps;
use crate::session::{coord, shutdown_msg, CoordinatorState};
use crate::topology::Topology;
use crate::validate::validate_result;
use quip_proto::v1::miner_service_server::{MinerService, MinerServiceServer};
use quip_proto::v1::{
    coord_msg, miner_msg, Configure, CoordMsg, Job, MinerMsg, Reject, Result as JobResult, Welcome,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;
use tokio::net::UnixListener;
use tokio::process::Command;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

/// Inputs for a full drive-many run.
pub struct DriveManyParams<'a> {
    /// Absolute or `PATH`-resolved miner binary to spawn.
    pub miner_bin: &'a str,
    /// Unix-domain socket path the coordinator binds and the miner dials.
    pub sock_path: &'a str,
    /// Miner id registered on the coordinator and passed as `--miner-id`.
    pub miner_id: &'a str,
    /// Session token the miner must present in `Hello`.
    pub token: &'a str,
    /// Launch plan (configure payload) for this miner.
    pub entry: &'a LaunchEntry,
    /// Optional topology wire message staged after handshake.
    pub topology: Option<Topology>,
    /// Optional quality-gate target staged after handshake.
    pub target: Option<quip_proto::v1::SetTarget>,
    /// Jobs to stage and drive to terminal outcomes.
    pub jobs: Vec<Job>,
    /// Forwarded to the spawned miner's `--utilization` when set (GPU backends).
    pub utilization: Option<u32>,
    /// Forwarded as `--yielding` to the spawned miner (GPU backends).
    pub yielding: bool,
    /// Forwarded as `--log-level` so the miner's own subscriber matches the
    /// verbosity the operator asked `drive` for. The miner inherits this
    /// process's stderr, so without this its per-job lines stay at the miner
    /// default no matter what `drive --log-level` says.
    pub log_level: crate::logging::LogLevel,
}

/// Result of a drive-many run: per-job rows plus handshake/exit status.
#[derive(Debug, Default)]
pub struct DriveManyReport {
    /// Whether the miner completed a successful Hello/Welcome handshake.
    pub handshake_ok: bool,
    /// Per-job scoring rows recorded during the run.
    pub rows: Vec<JobRow>,
    /// Number of jobs handed to the run. `rows.len() < total` means the run was
    /// truncated (miner crash / dropped Result / timeout) — an incomplete run,
    /// not per-job data.
    pub total: usize,
    /// Miner process exit code (`-1` if killed or wait failed).
    pub miner_exit_code: i32,
    /// Real wall-clock span first-dispatch → last-result (ms). Throughput is
    /// `jobs / this`, not the sum of per-job `wall_ms` (which overcounts the
    /// concurrent, overlapping jobs the streaming backends run).
    pub run_wall_ms: u64,
}

/// A failing `JobRow` for a job that produced no valid `Result` — rejected,
/// unroutable (no capable miner), or cut off by a miner `Fatal`.
fn failing_row(job_id: Vec<u8>, is_pow: bool, wall_ms: u64) -> JobRow {
    JobRow {
        job_id,
        is_pow,
        n_solutions: 0,
        best_energy_milli: i64::MAX,
        diversity_milli: 0,
        passed: false,
        device_access_time_us: 0,
        reads: 0,
        sweeps: 0,
        wall_ms,
        rejected: true,
    }
}

fn job_is_pow(job: &Job) -> bool {
    job.provenance.as_ref().is_some_and(|p| p.is_pow)
}

/// Shared per-run bookkeeping the session task mutates as jobs complete.
struct RunState {
    rows: StdMutex<Vec<JobRow>>,
    dispatch_at: StdMutex<HashMap<Vec<u8>, Instant>>,
    /// First job dispatch and last result — the real wall-clock span of the
    /// run. Throughput is `jobs / span`; summing per-job `wall_ms` would
    /// overcount the streaming backends' concurrent (overlapping) jobs.
    span: StdMutex<(Option<Instant>, Option<Instant>)>,
    total: usize,
}

impl RunState {
    fn new(total: usize) -> Self {
        Self {
            rows: StdMutex::new(Vec::new()),
            dispatch_at: StdMutex::new(HashMap::new()),
            span: StdMutex::new((None, None)),
            total,
        }
    }

    #[expect(
        clippy::unwrap_used,
        reason = "StdMutex poison only if a prior holder panicked"
    )]
    fn note_dispatch(&self, job_id: Vec<u8>) {
        let now = Instant::now();
        let _ = self.dispatch_at.lock().unwrap().insert(job_id, now);
        let mut span = self.span.lock().unwrap();
        if span.0.is_none() {
            span.0 = Some(now);
        }
    }

    #[expect(
        clippy::unwrap_used,
        clippy::cast_possible_truncation,
        reason = "StdMutex poison only if a prior holder panicked; wall-clock ms fit in u64 for drive runs"
    )]
    fn wall_ms_since_dispatch(&self, job_id: &[u8]) -> u64 {
        self.dispatch_at
            .lock()
            .unwrap()
            .remove(job_id)
            .map_or(0, |t| t.elapsed().as_millis() as u64)
    }

    #[expect(
        clippy::unwrap_used,
        reason = "StdMutex poison only if a prior holder panicked"
    )]
    fn record(&self, row: JobRow) {
        self.rows.lock().unwrap().push(row);
        self.span.lock().unwrap().1 = Some(Instant::now());
    }

    /// Real wall-clock span first-dispatch → last-result, in ms.
    #[expect(
        clippy::unwrap_used,
        clippy::cast_possible_truncation,
        reason = "StdMutex poison only if a prior holder panicked; wall-clock ms fit in u64 for drive runs"
    )]
    fn wall_span_ms(&self) -> u64 {
        let span = self.span.lock().unwrap();
        match (span.0, span.1) {
            (Some(start), Some(end)) => end.saturating_duration_since(start).as_millis() as u64,
            _ => 0,
        }
    }

    #[expect(
        clippy::unwrap_used,
        reason = "StdMutex poison only if a prior holder panicked"
    )]
    fn is_complete(&self) -> bool {
        self.rows.lock().unwrap().len() >= self.total
    }
}

/// tonic service used by `run_drive`: stages every job in `jobs` on
/// handshake, then drives the same Ready/JobRequest/Result/Reject flow as
/// production, recording a `JobRow` per terminal outcome.
struct DriveManyService {
    state: Arc<Mutex<CoordinatorState>>,
    jobs: Arc<Mutex<Vec<Job>>>,
    run: Arc<RunState>,
    miner_id: String,
}

#[tonic::async_trait]
impl MinerService for DriveManyService {
    type SessionStream = ReceiverStream<Result<CoordMsg, Status>>;

    async fn session(
        &self,
        request: Request<Streaming<MinerMsg>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let mut inbound = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<CoordMsg, Status>>(64);
        let state = Arc::clone(&self.state);
        let jobs = Arc::clone(&self.jobs);
        let run = Arc::clone(&self.run);
        let miner_id = self.miner_id.clone();

        drop(tokio::spawn(async move {
            run_drive_session(&mut inbound, &tx, state, jobs, run, miner_id).await;
        }));

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

/// Hello -> token/version check -> register + stage every job -> Welcome/
/// Configure/Topology. Returns the effective `Configure`, or `None` on a
/// failed handshake (stream is dropped; a well-behaved miner exits 77).
async fn handshake(
    inbound: &mut Streaming<MinerMsg>,
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
    state: &Arc<Mutex<CoordinatorState>>,
    jobs: &Arc<Mutex<Vec<Job>>>,
    run: &Arc<RunState>,
    miner_id: &str,
) -> Option<Configure> {
    let Ok(Some(MinerMsg {
        msg: Some(miner_msg::Msg::Hello(hello)),
    })) = inbound.message().await
    else {
        return None;
    };
    {
        let st = state.lock().await;
        let expected = st
            .expected_tokens
            .get(miner_id)
            .cloned()
            .unwrap_or_default();
        // Mirror production `run_session`: an empty expected token must never
        // authenticate an empty-token miner.
        if expected.is_empty() || hello.session_token != expected || hello.protocol_version != 1 {
            return None;
        }
    }
    let configure = {
        let mut st = state.lock().await;
        st.router.register_miner(
            miner_id.to_string(),
            MinerCaps {
                backend: hello.backend,
                algorithm: hello.algorithm,
                supported_kinds: hello.supported_kinds,
                max_nodes: hello.max_nodes,
                max_edges: hello.max_edges,
            },
        );
        let staged = jobs.lock().await.drain(..).collect::<Vec<_>>();
        for j in staged {
            let _ = st.router.route(j);
        }
        // Reconcile jobs no registered miner can serve: record failing rows now
        // so the completion gate accounts for them instead of stalling to the
        // overall timeout.
        for j in std::mem::take(&mut st.router.unroutable) {
            run.record(failing_row(j.job_id.clone(), job_is_pow(&j), 0));
        }
        st.configure.get(miner_id).cloned().unwrap_or(Configure {
            queue_depth: 3,
            idle_timeout_s: 300,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: String::new(),
        })
    };
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Welcome(Welcome {
            protocol_version: 1,
        }))))
        .await;
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Configure(configure.clone()))))
        .await;
    {
        let st = state.lock().await;
        if let Some(topo) = st.topology.as_ref() {
            let _ = tx
                .send(Ok(coord(coord_msg::Msg::Topology(topo.to_proto()))))
                .await;
        }
        if let Some(target) = st.target.as_ref() {
            let _ = tx.send(Ok(coord(coord_msg::Msg::SetTarget(*target)))).await;
        }
    }
    Some(configure)
}

/// Depth of the read loop's queue of credit grants to the dispatcher task.
/// Grants are single `u32`s, so this only has to absorb a burst of
/// `JobRequest`s arriving while the dispatcher is mid-send.
const GRANT_CHANNEL_DEPTH: usize = 1024;

/// Grant `credits` and dispatch every staged job the router allows.
/// Returns `false` when the outbound channel is closed (session should end).
async fn dispatch_jobs(
    state: &Arc<Mutex<CoordinatorState>>,
    miner_id: &str,
    credits: u32,
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
    run: &Arc<RunState>,
) -> bool {
    // Collect everything to dispatch under the lock, then release it before
    // sending. `tx.send().await` backpressures when the miner is busy, and
    // holding `state` across it serializes result write-back — `handle_result`
    // needs the same lock — which starves the miner's streaming pipeline.
    let jobs: Vec<Job> = {
        let mut st = state.lock().await;
        st.router.grant_credits(miner_id, credits);
        let mut jobs = Vec::new();
        while let Some(job) = st.router.next_job(miner_id) {
            run.note_dispatch(job.job_id.clone());
            let _ = st.inflight.insert(job.job_id.clone(), job.clone());
            jobs.push(job);
        }
        jobs
    };
    for job in jobs {
        if tx.send(Ok(coord(coord_msg::Msg::Job(job)))).await.is_err() {
            return false;
        }
    }
    true
}

/// Validate a `Result` against its job's gates and record a `JobRow`. Drive
/// mode scores and reports only — it never submits to a chain.
///
/// Spawned per result rather than awaited inline (see [`handle_message`]): the
/// state lock is held only long enough to ack the router, take the inflight
/// job, and clone the run-resolved topology `Arc` + gates, then the CPU-bound
/// `validate_result` runs on the blocking pool. The topology is a run constant
/// (`CoordinatorState::resolved_topo`), so the `Arc::clone` is a refcount bump,
/// never a data copy.
async fn handle_result(
    state: &Arc<Mutex<CoordinatorState>>,
    result: JobResult,
    run: &Arc<RunState>,
) {
    let (job, topo, gates) = {
        let mut st = state.lock().await;
        let job = st.inflight.remove(&result.job_id);
        let topo = Arc::clone(&st.resolved_topo);
        let gates = crate::validate::gates_from_target(st.target.as_ref());
        (job, topo, gates)
    };
    let Some(job) = job else { return };
    let is_pow = job.provenance.as_ref().is_some_and(|p| p.is_pow);
    let Some(ising) = job.ising else { return };
    let job_id = result.job_id;
    let solutions = result.solutions;
    let n_solutions = solutions.len();
    let (device_us, reads, sweeps) = result
        .meta
        .as_ref()
        .map_or((0, 0, 0), |m| (m.device_access_time_us, m.reads, m.sweeps));
    let Ok(validated) =
        tokio::task::spawn_blocking(move || validate_result(&ising, &solutions, &gates, &topo))
            .await
    else {
        // validation task panicked; leave the run to time out
        return;
    };
    let wall_ms = run.wall_ms_since_dispatch(&job_id);
    run.record(JobRow {
        job_id,
        is_pow,
        n_solutions,
        best_energy_milli: validated.best_energy_milli,
        diversity_milli: validated.diversity_milli,
        passed: validated.accepted,
        device_access_time_us: device_us,
        reads,
        sweeps,
        wall_ms,
        rejected: false,
    });
}

/// Record a `Reject` as a failing row (see module docs for the no-reroute
/// deviation) and ack the router so its credit bookkeeping stays consistent.
async fn handle_reject(state: &Arc<Mutex<CoordinatorState>>, rej: Reject, run: &Arc<RunState>) {
    let is_pow = {
        let mut st = state.lock().await;
        st.inflight
            .remove(&rej.job_id)
            .is_some_and(|j| job_is_pow(&j))
    };
    let wall_ms = run.wall_ms_since_dispatch(&rej.job_id);
    run.record(failing_row(rej.job_id, is_pow, wall_ms));
}

/// A miner `Fatal` ends the session; record every still-inflight job as a
/// failing row so those jobs are reconciled rather than left to stall the run
/// to the overall timeout. Jobs never dispatched stay unrecorded on purpose so
/// the caller can see the run was truncated (`rows.len() < total`).
async fn handle_fatal(state: &Arc<Mutex<CoordinatorState>>, run: &Arc<RunState>) {
    let inflight: Vec<Job> = {
        let mut st = state.lock().await;
        st.inflight.drain().map(|(_, j)| j).collect()
    };
    for j in inflight {
        let wall_ms = run.wall_ms_since_dispatch(&j.job_id);
        run.record(failing_row(j.job_id.clone(), job_is_pow(&j), wall_ms));
    }
}

async fn handle_message(
    msg: MinerMsg,
    state: &Arc<Mutex<CoordinatorState>>,
    _miner_id: &str,
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
    run: &Arc<RunState>,
    grants: &mpsc::Sender<u32>,
    seed_credits: u32,
) -> bool {
    match msg.msg {
        // Seed credits from Configure so the first job can dispatch before the
        // miner sends JobRequest: quip-mock-miner only requests after
        // completing a job. This makes the pool exceed the miner's stated depth
        // by `queue_depth` — harmless now that neither side couples reads to
        // writes, but see the note in MINER_PROTOCOL.md about making the miner
        // the sole credit source once every backend sends an initial request.
        Some(miner_msg::Msg::Ready(_)) => grants.send(seed_credits).await.is_ok(),
        // Hand the grant to the dispatcher task rather than dispatching here:
        // sending jobs blocks once the outbound channel fills, and this is the
        // only task reading results. Blocking here stops us reading the very
        // results that would free credits — the deadlock this split fixes.
        Some(miner_msg::Msg::JobRequest(req)) => grants.send(req.credits).await.is_ok(),
        Some(miner_msg::Msg::Result(result)) => {
            // Validate concurrently so the session task keeps draining the
            // stream and dispatching replacement jobs instead of blocking
            // ~300ms per result. The final row can land after the loop's own
            // completion check, so signal shutdown from whichever spawned task
            // completes the run.
            let state = Arc::clone(state);
            let run = Arc::clone(run);
            let tx = tx.clone();
            drop(tokio::spawn(async move {
                handle_result(&state, result, &run).await;
                if run.is_complete() {
                    let _ = tx.send(Ok(shutdown_msg(500))).await;
                }
            }));
            true
        }
        Some(miner_msg::Msg::Reject(rej)) => {
            handle_reject(state, rej, run).await;
            true
        }
        Some(miner_msg::Msg::Fatal(_)) => {
            handle_fatal(state, run).await;
            false
        }
        _ => true,
    }
}

async fn run_drive_session(
    inbound: &mut Streaming<MinerMsg>,
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
    state: Arc<Mutex<CoordinatorState>>,
    jobs: Arc<Mutex<Vec<Job>>>,
    run: Arc<RunState>,
    miner_id: String,
) {
    let Some(configure) = handshake(inbound, tx, &state, &jobs, &run, &miner_id).await else {
        return;
    };
    // Nothing left to run — either no jobs, or every job was unroutable and
    // already reconciled during handshake. Shut down without waiting for Ready.
    if run.total == 0 || run.is_complete() {
        let _ = tx.send(Ok(shutdown_msg(500))).await;
        return;
    }
    let seed_credits = configure.queue_depth.max(1);
    // Dispatcher task: the only place that writes `Job`s. Kept off the read
    // path so a full outbound channel can never stop us draining results.
    let (grants, mut grant_rx) = mpsc::channel::<u32>(GRANT_CHANNEL_DEPTH);
    let dispatcher = {
        let state = Arc::clone(&state);
        let run = Arc::clone(&run);
        let tx = tx.clone();
        let miner_id = miner_id.clone();
        tokio::spawn(async move {
            while let Some(credits) = grant_rx.recv().await {
                if !dispatch_jobs(&state, &miner_id, credits, &tx, &run).await {
                    return;
                }
            }
        })
    };

    loop {
        let Ok(Some(msg)) = inbound.message().await else {
            break;
        };
        let keep_going =
            handle_message(msg, &state, &miner_id, tx, &run, &grants, seed_credits).await;
        if !keep_going {
            break;
        }
        if run.is_complete() {
            let _ = tx.send(Ok(shutdown_msg(500))).await;
            break;
        }
    }
    drop(grants);
    dispatcher.abort();
}

/// Full drive-many run: spawn the miner over UDS, stage every job, validate
/// every `Result`, and send a clean `Shutdown` once all jobs are terminal.
///
/// # Panics
///
/// Panics if the Unix socket cannot be bound or the miner process cannot be
/// spawned (drive CLI setup failures are fatal).
#[expect(
    clippy::expect_used,
    clippy::print_stderr,
    clippy::unwrap_used,
    reason = "drive CLI: bind/spawn failures are fatal setup errors; StdMutex poison only if a prior holder panicked; eprintln is drive CLI interrupt notice"
)]
pub async fn run_drive(p: DriveManyParams<'_>) -> DriveManyReport {
    let sock_path = p.sock_path;
    let miner_id = p.miner_id;
    let _ = std::fs::remove_file(sock_path);
    if let Some(parent) = Path::new(sock_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let uds = UnixListener::bind(sock_path).expect("bind unix socket");
    let incoming = UnixListenerStream::new(uds);

    let mut st = CoordinatorState::new();
    let _ = st.expected_tokens.insert(miner_id.into(), p.token.into());
    let _ = st
        .configure
        .insert(miner_id.into(), p.entry.configure.clone());
    // Resolves the position-indexed scoring topology once (a run constant);
    // every result handler borrows it by `Arc`, never rebuilding the graph.
    st.set_topology(p.topology);
    st.target = p.target;
    let total = p.jobs.len();
    let state = Arc::new(Mutex::new(st));
    let jobs = Arc::new(Mutex::new(p.jobs));
    let run = Arc::new(RunState::new(total));

    let svc = DriveManyService {
        state: Arc::clone(&state),
        jobs,
        run: Arc::clone(&run),
        miner_id: miner_id.to_string(),
    };

    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(MinerServiceServer::new(svc))
            .serve_with_incoming(incoming)
            .await
    });

    let socket_uri = format!("unix://{sock_path}");
    let mut cmd = Command::new(p.miner_bin);
    let _ = cmd
        .arg("--quip-coordinator")
        .arg(&socket_uri)
        .arg("--miner-id")
        .arg(miner_id)
        .arg("--log-level")
        .arg(p.log_level.to_string())
        .env("QUIP_SESSION_TOKEN", p.token);
    // Mechanism A: forward governor flags to the spawned miner's own CLI
    // (cuda/metal). config.toml still overrides these via backend_toml.
    if let Some(util) = p.utilization {
        let _ = cmd.arg("--utilization").arg(util.to_string());
    }
    if p.yielding {
        let _ = cmd.arg("--yielding");
    }
    let mut child = cmd.spawn().expect("spawn miner");

    // No artificial time limit: the run ends when the miner has serviced every
    // job (it exits after the session sends `Shutdown` on completion), however
    // long that takes — mining a hard problem can legitimately run for hours.
    // The only early stop is the operator: Ctrl-C kills the miner and reports
    // whatever completed so far (`rows.len() < total` marks it truncated).
    let exit_code = tokio::select! {
        status = child.wait() => status.map_or(-1, |s| s.code().unwrap_or(-1)),
        _ = tokio::signal::ctrl_c() => {
            eprintln!("drive: interrupted; stopping and reporting completed jobs");
            let _ = child.kill().await;
            let _ = child.wait().await;
            -1
        }
    };

    server.abort();
    let _ = std::fs::remove_file(sock_path);

    let st = state.lock().await;
    let handshake_ok = st.router.caps(miner_id).is_some();
    let run_wall_ms = run.wall_span_ms();
    let rows = run.rows.lock().unwrap().clone();
    DriveManyReport {
        handshake_ok,
        rows,
        total,
        miner_exit_code: exit_code,
        run_wall_ms,
    }
}
