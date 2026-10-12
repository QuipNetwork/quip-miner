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
use crate::router::{MinerCaps, Router};
use crate::session::{coord, shutdown_msg, CoordinatorState};
use crate::topology::Topology;
use crate::validate::validate_result;
use quip_proto::v1::miner_service_server::{MinerService, MinerServiceServer};
use quip_proto::v1::{
    coord_msg, miner_msg, Configure, CoordMsg, Job, MinerMsg, Reject, Result as JobResult, Welcome,
};
use quip_protocol::session::PROTOCOL_VERSION;
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
        raw_best_energy_milli: i64::MAX,
        diversity_milli: 0,
        passed: false,
        device_access_time_us: 0,
        reads: 0,
        sweeps: 0,
        wall_ms,
        rejected: true,
        salts_done: 0,
        winners: 0,
    }
}

fn job_is_pow(job: &Job) -> bool {
    job.provenance.as_ref().is_some_and(|p| p.is_pow)
}

/// Shared per-run bookkeeping the session task mutates as jobs complete.
struct RunState {
    rows: StdMutex<Vec<JobRow>>,
    dispatch_at: StdMutex<HashMap<Vec<u8>, Instant>>,
    lease_winners: StdMutex<HashMap<Vec<u8>, u32>>,
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
            lease_winners: StdMutex::new(HashMap::new()),
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

    /// Count jobs no miner can serve as terminal failures, including live updates.
    fn record_unroutable(&self, router: &mut Router) {
        for job in std::mem::take(&mut router.unroutable) {
            let is_pow = job_is_pow(&job);
            self.record(failing_row(job.job_id, is_pow, 0));
        }
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
    let caps = hello.capabilities.clone().unwrap_or_default();
    {
        let st = state.lock().await;
        let expected = st
            .expected_tokens
            .get(miner_id)
            .cloned()
            .unwrap_or_default();
        // Mirror production `run_session`: an empty expected token must never
        // authenticate an empty-token miner.
        if expected.is_empty()
            || hello.session_token != expected
            || hello.capabilities.is_none()
            || caps.protocol_version != PROTOCOL_VERSION
        {
            return None;
        }
    }
    let configure = {
        let mut st = state.lock().await;
        st.router
            .register_miner(miner_id.to_string(), MinerCaps::from_capabilities(&caps));
        let staged = jobs.lock().await.drain(..).collect::<Vec<_>>();
        for j in staged {
            let _ = st.router.route(j);
        }
        // Reconcile jobs no registered miner can serve: record failing rows now
        // so the completion gate accounts for them instead of stalling to the
        // overall timeout.
        run.record_unroutable(&mut st.router);
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
            protocol_version: PROTOCOL_VERSION,
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
        raw_best_energy_milli: validated.raw_best_energy_milli,
        diversity_milli: validated.diversity_milli,
        passed: validated.accepted,
        device_access_time_us: device_us,
        reads,
        sweeps,
        wall_ms,
        rejected: false,
        salts_done: 1,
        winners: u32::from(validated.accepted),
    });
}

/// Verify a reported lease salt inline and count it as a winner when it clears
/// the target, so the count is updated before the following `LeaseDone` on the
/// same miner stream is handled.
async fn handle_lease_result(
    state: &Arc<Mutex<CoordinatorState>>,
    result: &JobResult,
    run: &RunState,
) {
    let (generator, view, target) = {
        let st = state.lock().await;
        let Some(job) = st.inflight.get(&result.job_id) else {
            return;
        };
        let Some(generator) = job.generator.as_ref() else {
            return;
        };
        let (Some(view), Some(target)) = (st.lease_topology.clone(), st.target) else {
            return;
        };
        (generator.clone(), view, target)
    };
    let winner = crate::lease::verify_and_select(&generator, &view, &target, result)
        .is_ok_and(|(_, validated)| validated.accepted);
    if winner {
        #[expect(
            clippy::unwrap_used,
            reason = "StdMutex poison only if a prior holder panicked"
        )]
        {
            let mut winners = run.lease_winners.lock().unwrap();
            let count = winners.entry(result.job_id.clone()).or_default();
            *count += 1;
        }
    }
}

async fn handle_lease_done(
    state: &Arc<Mutex<CoordinatorState>>,
    done: quip_proto::v1::LeaseDone,
    run: &RunState,
) {
    let is_lease = state.lock().await.inflight.remove(&done.job_id).is_some();
    if !is_lease {
        return;
    }
    #[expect(
        clippy::unwrap_used,
        reason = "StdMutex poison only if a prior holder panicked"
    )]
    let winners = run
        .lease_winners
        .lock()
        .unwrap()
        .remove(&done.job_id)
        .unwrap_or(0);
    let wall_ms = run.wall_ms_since_dispatch(&done.job_id);
    run.record(JobRow {
        job_id: done.job_id,
        is_pow: true,
        n_solutions: 0,
        best_energy_milli: done.best_energy_milli,
        raw_best_energy_milli: done.best_energy_milli,
        diversity_milli: 0,
        passed: winners > 0,
        device_access_time_us: 0,
        wall_ms,
        reads: 0,
        sweeps: 0,
        rejected: false,
        salts_done: done.salts_done,
        winners,
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
    miner_id: &str,
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
        // writes, but see the note in quip-solver-core's SPEC.md about making the miner
        // the sole credit source once every backend sends an initial request.
        Some(miner_msg::Msg::Ready(_)) => grants.send(seed_credits).await.is_ok(),
        // Hand the grant to the dispatcher task rather than dispatching here:
        // sending jobs blocks once the outbound channel fills, and this is the
        // only task reading results. Blocking here stops us reading the very
        // results that would free credits — the deadlock this split fixes.
        Some(miner_msg::Msg::JobRequest(req)) => grants.send(req.credits).await.is_ok(),
        Some(miner_msg::Msg::Capabilities(caps)) => {
            let mut st = state.lock().await;
            if st.apply_capabilities(miner_id, &caps) {
                run.record_unroutable(&mut st.router);
            }
            true
        }
        Some(miner_msg::Msg::Result(result)) => {
            let is_lease = {
                let st = state.lock().await;
                st.inflight
                    .get(&result.job_id)
                    .is_some_and(crate::lease::is_lease)
            };
            if is_lease {
                handle_lease_result(state, &result, run).await;
                return true;
            }
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
        Some(miner_msg::Msg::LeaseDone(done)) => {
            handle_lease_done(state, done, run).await;
            if run.is_complete() {
                let _ = tx.send(Ok(shutdown_msg(500))).await;
            }
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
    state
        .lock()
        .await
        .register_wakeup(&miner_id, grants.clone());
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
    state.lock().await.deregister_outbound(&miner_id);
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
    // The session flags — including `--device` when the entry carries one —
    // come from the supervised path's builder, so drive cannot drift from the
    // argv a real run produces. Only the governor flags below are drive's own.
    let _ = cmd
        .args(crate::supervisor::child_args(
            p.entry,
            &socket_uri,
            p.log_level,
        ))
        .env("QUIP_SESSION_TOKEN", p.token);
    // Mechanism A: forward governor flags to the spawned miner's own CLI
    // (cuda/metal). Drive has no `backend_toml` to carry them, which is why
    // these are argv here and are not in the supervised path.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drive::{
        aggregate, drain_all, parse_topology_spec, write_jsonl, LeaseSource, ListSource,
        TopologySpec, TopologySpecError,
    };
    use crate::validate::MAX_PROOF_SOLUTIONS_WIRE;
    use quip_proto::v1::miner_service_client::MinerServiceClient;
    use quip_proto::v1::{
        ising_problem, Capabilities, EdgeList, Hello, JobKind, LeaseDone, Provenance, Ready,
        SetTarget, Solution,
    };
    use quip_protocol::chacha8::draw_ising_milli;
    use quip_protocol::wire::encode_spins_packed;

    struct DriveTestSession {
        _directory: tempfile::TempDir,
        server: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
        inbound: mpsc::Sender<MinerMsg>,
        outbound: Streaming<CoordMsg>,
        state: Arc<Mutex<CoordinatorState>>,
        run: Arc<RunState>,
    }

    impl DriveTestSession {
        async fn new(jobs: Vec<Job>, topology: Option<Topology>) -> Self {
            let directory = tempfile::tempdir().expect("test directory is created");
            let socket = directory.path().join("miner.sock");
            let listener = UnixListener::bind(&socket).expect("test socket is bound");
            let mut st = CoordinatorState::new();
            let _ = st
                .expected_tokens
                .insert("m".to_owned(), "token".to_owned());
            let _ = st.configure.insert(
                "m".to_owned(),
                Configure {
                    queue_depth: 3,
                    ..Configure::default()
                },
            );
            let has_topology = topology.is_some();
            st.set_topology(topology);
            let state = Arc::new(Mutex::new(st));
            let run = Arc::new(RunState::new(jobs.len()));
            let service = DriveManyService {
                state: Arc::clone(&state),
                jobs: Arc::new(Mutex::new(jobs)),
                run: Arc::clone(&run),
                miner_id: "m".to_owned(),
            };
            let server = tokio::spawn(async move {
                Server::builder()
                    .add_service(MinerServiceServer::new(service))
                    .serve_with_incoming(UnixListenerStream::new(listener))
                    .await
            });
            let channel = tonic::transport::Endpoint::from_static("http://[::]:50051")
                .connect_with_connector(tower::service_fn(move |_: tonic::transport::Uri| {
                    let socket = socket.clone();
                    async move {
                        tokio::net::UnixStream::connect(socket)
                            .await
                            .map(hyper_util::rt::TokioIo::new)
                    }
                }))
                .await
                .expect("test miner connects");
            let (inbound, inbound_rx) = mpsc::channel(16);
            inbound
                .send(MinerMsg {
                    msg: Some(miner_msg::Msg::Hello(Hello {
                        miner_id: "m".to_owned(),
                        session_token: "token".to_owned(),
                        capabilities: Some(Capabilities {
                            protocol_version: PROTOCOL_VERSION,
                            ..Capabilities::default()
                        }),
                    })),
                })
                .await
                .expect("test miner sends Hello");
            let outbound = MinerServiceClient::new(channel)
                .session(ReceiverStream::new(inbound_rx))
                .await
                .expect("drive session starts")
                .into_inner();
            let mut session = Self {
                _directory: directory,
                server,
                inbound,
                outbound,
                state,
                run,
            };
            let coord_msg::Msg::Welcome(_) = session.receive().await else {
                panic!("drive must welcome the miner");
            };
            let coord_msg::Msg::Configure(configure) = session.receive().await else {
                panic!("drive must configure the miner");
            };
            assert_eq!(configure.queue_depth, 3);
            if has_topology {
                let coord_msg::Msg::Topology(_) = session.receive().await else {
                    panic!("drive must send the configured topology");
                };
            }
            session
        }

        async fn send(&self, msg: miner_msg::Msg) {
            self.inbound
                .send(MinerMsg { msg: Some(msg) })
                .await
                .expect("test miner sends a frame");
        }

        async fn receive(&mut self) -> coord_msg::Msg {
            tokio::time::timeout(std::time::Duration::from_secs(5), self.outbound.message())
                .await
                .expect("drive responds without stalling")
                .expect("drive response is valid")
                .expect("drive session remains open")
                .msg
                .expect("drive response has a payload")
        }

        async fn finish(&mut self, jobs: &[Job]) {
            for job in jobs {
                let nodes = job.ising.as_ref().expect("test job is Ising").h.len() / 4;
                self.send(miner_msg::Msg::Result(JobResult {
                    job_id: job.job_id.clone(),
                    solutions: vec![Solution {
                        spins: encode_spins_packed(&vec![1; nodes]),
                        energy_milli: 0,
                    }],
                    ..JobResult::default()
                }))
                .await;
            }
            let coord_msg::Msg::Shutdown(_) = self.receive().await else {
                panic!("drive must shut down after every job is terminal");
            };
            assert!(self.run.is_complete());
        }
    }

    impl Drop for DriveTestSession {
        fn drop(&mut self) {
            self.server.abort();
        }
    }

    fn inline_job(id: u8, nodes: usize, edges: &[(u32, u32)]) -> Job {
        let (u, v) = edges.iter().copied().unzip();
        Job {
            job_id: vec![id],
            kind: JobKind::IsingSample as i32,
            ising: Some(crate::producer::problem::milli_problem(
                Some(ising_problem::Graph::Edges(EdgeList { u, v })),
                &vec![0; nodes],
                &vec![0; edges.len()],
            )),
            provenance: Some(Provenance {
                is_pow: true,
                ..Provenance::default()
            }),
            ..Job::default()
        }
    }

    #[tokio::test]
    async fn live_capability_degree_reduction_preserves_order_accounting_and_completion() {
        let degree_twenty = (1..=20).map(|node| (0, node)).collect::<Vec<_>>();
        let degree_twenty_one = (1..=21).map(|node| (0, node)).collect::<Vec<_>>();
        let mut session = DriveTestSession::new(
            vec![
                inline_job(1, 22, &degree_twenty),
                inline_job(2, 22, &degree_twenty_one),
                inline_job(3, 22, &[(0, 1)]),
            ],
            None,
        )
        .await;
        {
            let mut st = session.state.lock().await;
            assert_eq!(st.router.caps("m").and_then(|caps| caps.max_degree), None);
            assert_eq!(st.router.staged_len("m"), 3);
            st.router.grant_credits("m", 4);
            st.router.record_completion("m");
            st.router.record_lease_salts("m", 7);
            st.router.record_round_salts("m", 9);
        }
        session
            .send(miner_msg::Msg::Capabilities(Capabilities {
                protocol_version: PROTOCOL_VERSION,
                max_degree: Some(20),
                ..Capabilities::default()
            }))
            .await;
        // The snapshot wake must use existing credits before Ready grants more.
        let mut dispatched = Vec::new();
        for id in [1, 3] {
            let coord_msg::Msg::Job(job) = session.receive().await else {
                panic!("compatible queued jobs must dispatch");
            };
            assert_eq!(job.job_id, vec![id]);
            dispatched.push(job);
        }
        {
            let rows = session
                .run
                .rows
                .lock()
                .expect("report lock is not poisoned");
            assert_eq!(rows.len(), 1);
            let row = rows.first().expect("one terminal failure is recorded");
            assert_eq!(row.job_id, vec![2]);
            assert!(row.is_pow);
            assert!(row.rejected);
            assert!(!row.passed);
            assert_eq!(row.n_solutions, 0);
            assert_eq!(row.best_energy_milli, i64::MAX);
            assert_eq!(row.raw_best_energy_milli, i64::MAX);
            assert_eq!(row.wall_ms, 0);
        }
        session.send(miner_msg::Msg::Ready(Ready {})).await;
        session.finish(&dispatched).await;
        assert_eq!(
            session
                .run
                .rows
                .lock()
                .expect("report lock is not poisoned")
                .len(),
            3
        );
        let mut st = session.state.lock().await;
        assert_eq!(st.router.jobs_completed("m"), 1);
        assert_eq!(st.router.take_consumed("m"), 2);
        assert_eq!(st.router.take_lease_salts("m"), 7);
        assert_eq!(st.router.round_salts("m"), 9);
        // Four existing credits plus three from Ready, minus two dispatched jobs.
        for id in 10..15 {
            assert!(st.router.stage_on("m", inline_job(id, 22, &[(0, 1)])));
            assert_eq!(
                st.router.next_job("m").map(|job| job.job_id),
                Some(vec![id])
            );
        }
        assert!(st.router.stage_on("m", inline_job(15, 22, &[(0, 1)])));
        assert!(st.router.next_job("m").is_none());
    }

    #[tokio::test]
    async fn live_capability_native_fingerprint_after_configure_requires_inline_proof() {
        let topology = Topology::from_nodes_edges(
            vec![0, 1, 2],
            vec![(0, 1), (1, 2)],
            &[0],
            &[1000],
            &[-1000, 1000],
        );
        let native_hash =
            crate::topology::native_topology_hash(&topology.nodes, &topology.edge_pairs())
                .expect("native graph is valid")
                .to_vec();
        let mut session = DriveTestSession::new(
            vec![
                inline_job(1, 3, &[(0, 1)]),
                inline_job(2, 3, &[(0, 2)]),
                inline_job(3, 4, &[]),
            ],
            Some(topology.clone()),
        )
        .await;
        session
            .send(miner_msg::Msg::Capabilities(Capabilities {
                protocol_version: PROTOCOL_VERSION,
                native_topology_hash: Some(native_hash.clone()),
                ..Capabilities::default()
            }))
            .await;
        session.send(miner_msg::Msg::Ready(Ready {})).await;
        let coord_msg::Msg::Job(job) = session.receive().await else {
            panic!("the proven inline subgraph must dispatch");
        };
        assert_eq!(job.job_id, vec![1]);
        {
            let rows = session
                .run
                .rows
                .lock()
                .expect("report lock is not poisoned");
            assert_eq!(
                rows.iter()
                    .map(|row| row.job_id.clone())
                    .collect::<Vec<_>>(),
                vec![vec![2], vec![3]]
            );
            assert!(rows.iter().all(|row| row.rejected && !row.passed));
        }
        session.finish(&[job]).await;
        {
            let st = session.state.lock().await;
            assert_eq!(
                st.router
                    .caps("m")
                    .and_then(|caps| caps.native_topology_hash.as_ref()),
                Some(&native_hash)
            );
            assert_eq!(st.router.staged_len("m"), 0);
            assert!(st.inflight.is_empty());
        }
        let mut all_unroutable =
            DriveTestSession::new(vec![inline_job(4, 3, &[(0, 2)])], Some(topology)).await;
        all_unroutable
            .send(miner_msg::Msg::Capabilities(Capabilities {
                protocol_version: PROTOCOL_VERSION,
                native_topology_hash: Some(native_hash),
                ..Capabilities::default()
            }))
            .await;
        all_unroutable.finish(&[]).await;
        let rows = all_unroutable
            .run
            .rows
            .lock()
            .expect("report lock is not poisoned");
        assert_eq!(rows.len(), 1);
        let row = rows.first().expect("one terminal failure is recorded");
        assert_eq!(row.job_id, vec![4]);
        assert!(row.rejected);
        assert!(!row.passed);
    }

    #[tokio::test]
    async fn live_capability_incompatible_protocol_keeps_the_last_accepted_snapshot() {
        let edges = (1..=21).map(|node| (0, node)).collect::<Vec<_>>();
        let mut session = DriveTestSession::new(
            vec![inline_job(1, 22, &edges), inline_job(2, 22, &[(0, 1)])],
            None,
        )
        .await;
        session
            .send(miner_msg::Msg::Capabilities(Capabilities {
                protocol_version: PROTOCOL_VERSION,
                max_nodes: 64,
                max_edges: 128,
                max_degree: Some(21),
                stream_width: 4,
                features: vec!["streaming".to_owned()],
                ..Capabilities::default()
            }))
            .await;
        session
            .send(miner_msg::Msg::Capabilities(Capabilities {
                protocol_version: 1,
                max_nodes: 1,
                max_degree: Some(20),
                native_topology_hash: Some(vec![9; 32]),
                ..Capabilities::default()
            }))
            .await;
        session.send(miner_msg::Msg::Ready(Ready {})).await;
        let mut dispatched = Vec::new();
        for id in [1, 2] {
            let coord_msg::Msg::Job(job) = session.receive().await else {
                panic!("the invalid snapshot must not remove compatible jobs");
            };
            assert_eq!(job.job_id, vec![id]);
            dispatched.push(job);
        }
        {
            let st = session.state.lock().await;
            let caps = st.router.caps("m").expect("miner remains registered");
            assert_eq!(caps.max_nodes, 64);
            assert_eq!(caps.max_edges, 128);
            assert_eq!(caps.max_degree, Some(21));
            assert_eq!(caps.stream_width, 4);
            assert_eq!(caps.features, vec!["streaming".to_owned()]);
            assert_eq!(caps.native_topology_hash, None);
            assert!(st.router.unroutable.is_empty());
        }
        session.finish(&dispatched).await;
        let rows = session
            .run
            .rows
            .lock()
            .expect("report lock is not poisoned");
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| !row.rejected));
    }

    fn report_spec() -> Result<TopologySpec, TopologySpecError> {
        parse_topology_spec(
            r#"{
                "nodes": [40, 7, 81, 2],
                "edges": [],
                "allowed_h_milli": [-1000],
                "allowed_j_milli": [1000],
                "gates": {
                    "min_solutions": 2,
                    "max_energy_milli": 0,
                    "min_diversity_milli": 400
                }
            }"#,
        )
    }

    fn packed_solution(spins: &[i8]) -> Solution {
        Solution {
            spins: encode_spins_packed(spins),
            // A wire energy claim must not determine either report metric.
            energy_milli: i64::MIN,
        }
    }

    async fn score_nonce_result(
        spec: &TopologySpec,
        solutions: Vec<Solution>,
    ) -> Result<JobRow, Box<dyn std::error::Error>> {
        let text = format!(r#"{{"nonce":"{}"}}"#, "11".repeat(32));
        let snapshot = spec.to_snapshot();
        let mut source = ListSource::parse(&text, Some(&snapshot), 0)?;
        let job = drain_all(&mut source)
            .into_iter()
            .next()
            .ok_or("nonce replay produced no job")?;
        let job_id = job.job_id.clone();
        let mut state = CoordinatorState::new();
        state.set_topology(Some(spec.topology.clone()));
        state.target = Some(SetTarget {
            max_energy_milli: spec.max_energy_milli,
            min_solutions: spec.min_solutions,
            min_diversity_milli: spec.min_diversity_milli,
            max_proof_solutions: MAX_PROOF_SOLUTIONS_WIRE,
            num_reads: 0,
            num_sweeps: 0,
            anneal_time_us: 0,
        });
        let _ = state.inflight.insert(job_id.clone(), job);
        let state = Arc::new(Mutex::new(state));
        let run = Arc::new(RunState::new(1));
        run.note_dispatch(job_id.clone());
        handle_result(
            &state,
            JobResult {
                job_id,
                solutions,
                meta: None,
                salt: Vec::new(),
                nonce: Vec::new(),
            },
            &run,
        )
        .await;
        let mut rows = run.rows.lock().map_err(|error| error.to_string())?;
        if rows.len() != 1 {
            return Err(
                format!("result handler recorded {} rows instead of one", rows.len()).into(),
            );
        }
        rows.pop()
            .ok_or_else(|| "result handler recorded no row".into())
    }

    fn assert_selected_and_raw_energy(row: &JobRow) {
        assert_eq!(row.best_energy_milli, -2000);
        assert_eq!(row.raw_best_energy_milli, -4000);
    }

    #[tokio::test]
    async fn qui_1540_report_distinguishes_selected_and_raw_energy() {
        let spec = report_spec().expect("report topology is valid");
        let row = score_nonce_result(
            &spec,
            vec![
                packed_solution(&[1, 1, 1, 1]),
                packed_solution(&[1, -1, 1, 1]),
                packed_solution(&[1, 1, -1, 1]),
            ],
        )
        .await
        .expect("nonce result produces one report row");
        assert_selected_and_raw_energy(&row);
        assert_eq!(row.diversity_milli, 500);
        assert_eq!(row.n_solutions, 3);
        assert!(row.is_pow);
        assert!(row.passed);

        let directory = tempfile::tempdir().expect("temporary report directory is created");
        let path = directory.path().join("report.jsonl");
        let rows = std::slice::from_ref(&row);
        write_jsonl(&path, rows, &aggregate(rows, 1)).expect("JSONL report is written");
        let text = std::fs::read_to_string(path).expect("JSONL report is readable");
        let first = text.lines().next().expect("JSONL report contains a row");
        let json: serde_json::Value =
            serde_json::from_str(first).expect("report row is valid JSON");
        assert_eq!(
            json.get("best_energy_milli"),
            Some(&serde_json::json!(-2000))
        );
        assert_eq!(
            json.get("raw_best_energy_milli"),
            Some(&serde_json::json!(-4000))
        );
    }

    #[tokio::test]
    async fn qui_1540_failed_diversity_gate_keeps_selected_energy() {
        let mut spec = report_spec().expect("report topology is valid");
        spec.min_diversity_milli = 600;
        let row = score_nonce_result(
            &spec,
            vec![
                packed_solution(&[1, 1, 1, 1]),
                packed_solution(&[1, -1, 1, 1]),
                packed_solution(&[1, 1, -1, 1]),
            ],
        )
        .await
        .expect("nonce result produces one report row");
        assert!(!row.passed);
        assert_selected_and_raw_energy(&row);
    }

    #[tokio::test]
    async fn qui_1540_report_handles_empty_invalid_and_ceiling_equal_rows() {
        let mut spec = report_spec().expect("report topology is valid");
        spec.max_energy_milli = -4000;
        let invalid = Solution {
            // Bits 4 through 7 are padding for this four-node topology.
            spins: vec![0xff],
            energy_milli: i64::MIN,
        };
        for (solutions, expected_raw) in [
            (vec![packed_solution(&[1, 1, 1, 1]), invalid.clone()], -4000),
            (vec![invalid], i64::MAX),
            (Vec::new(), i64::MAX),
            (
                vec![Solution {
                    spins: Vec::new(),
                    energy_milli: i64::MIN,
                }],
                i64::MAX,
            ),
        ] {
            let count = solutions.len();
            let row = score_nonce_result(&spec, solutions)
                .await
                .expect("nonce result produces one report row");
            assert_eq!(row.n_solutions, count);
            assert_eq!(row.best_energy_milli, i64::MAX);
            assert_eq!(row.raw_best_energy_milli, expected_raw);
            assert!(!row.passed);
            assert!(!row.rejected);
        }
    }

    #[tokio::test]
    async fn qui_1540_nonce_replay_preserves_sparse_node_and_edge_order() {
        let spec = parse_topology_spec(
            r#"{
                "nodes": [40, 7, 81, 2],
                "edges": [[81, 2], [40, 81], [7, 2]],
                "allowed_h_milli": [0],
                "allowed_j_milli": [700, 1700, 2900],
                "gates": {"min_solutions": 1}
            }"#,
        )
        .expect("sparse report topology is valid");
        assert_eq!(spec.topology.nodes, vec![40, 7, 81, 2]);
        assert_eq!(spec.topology.edge_pairs(), vec![(81, 2), (40, 81), (7, 2)]);
        let (h, j) = draw_ising_milli(
            [0x11; 32],
            4,
            3,
            &spec.allowed_h_milli,
            &spec.allowed_j_milli,
        )
        .expect("nonce draws valid coefficients");
        assert_eq!(h, vec![0, 0, 0, 0]);
        let [first, second, third] = j.as_slice() else {
            panic!("expected three replay couplings");
        };
        // Dense edges are (2,3), (0,2), (1,3), in that received order.
        let expected = -i64::from(*first) + i64::from(*second) + i64::from(*third);
        let row = score_nonce_result(&spec, vec![packed_solution(&[1, -1, 1, -1])])
            .await
            .expect("nonce result produces one report row");
        assert_eq!(row.best_energy_milli, expected);
        assert_eq!(row.raw_best_energy_milli, expected);
        assert!(row.passed);
    }

    #[tokio::test]
    async fn qui_1540_lease_summary_copies_the_miner_minimum() {
        let spec = report_spec().expect("report topology is valid");
        let mut source = LeaseSource::new(&spec, [0; 32], 7, 1, 1);
        let job = drain_all(&mut source)
            .into_iter()
            .next()
            .expect("lease source produces one job");
        let job_id = job.job_id.clone();
        let mut state = CoordinatorState::new();
        let _ = state.inflight.insert(job_id.clone(), job);
        let state = Arc::new(Mutex::new(state));
        let run = RunState::new(1);
        handle_lease_done(
            &state,
            LeaseDone {
                job_id,
                salts_done: 1,
                best_energy_milli: -700,
            },
            &run,
        )
        .await;
        let mut rows = run.rows.lock().expect("report rows lock is not poisoned");
        let row = rows.pop().expect("lease handler records one row");
        assert_eq!(row.best_energy_milli, -700);
        assert_eq!(row.raw_best_energy_milli, -700);
        assert_eq!(row.diversity_milli, 0);
        assert_eq!(row.salts_done, 1);
        assert!(!row.passed);
    }

    #[test]
    fn qui_1540_terminal_failure_uses_missing_energy_sentinels() {
        let row = failing_row(vec![1], true, 0);
        assert_eq!(row.best_energy_milli, i64::MAX);
        assert_eq!(row.raw_best_energy_milli, i64::MAX);
        assert!(row.rejected);
        assert!(!row.passed);
    }
}
