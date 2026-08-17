//! Generic gRPC session loop over a UDS bidi stream.
//!
//! Modeled on `quip-mock-miner`: Hello → Welcome → Configure → Ready, job
//! handling with Reject reasons, Status on Ping/Cancel, clean drain on
//! Shutdown / idle timeout. Backends supply only a [`Sampler`].

use crate::cli::CommonArgs;
use crate::display::{energy_units, format_duration_ms};
use crate::job::{
    finalize_result, miner, num_sweeps_from_toml, prepare_job, status_msg, Prepared, SessionTarget,
    TopologyCache, DEFAULT_NUM_SWEEPS,
};
use crate::{CancelGuard, Sampler, StreamJob, StreamOutcome, StreamResult};
use quip_proto::v1::miner_service_client::MinerServiceClient;
use quip_proto::v1::{coord_msg, miner_msg, CoordMsg, JobKind, JobRequest, MinerMsg, Ready};
use quip_protocol::session::{build_hello, BackendCaps, ExitCode, SessionConfig, SessionError};
use std::collections::HashMap;
use std::process::ExitCode as StdExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Endpoint, Uri};

/// Static backend metadata advertised in capabilities and Hello.
#[derive(Clone, Copy, Debug)]
pub struct BackendIdentity {
    /// Backend name advertised in Hello / capabilities (e.g. `"cpu"`, `"cuda"`).
    pub backend: &'static str,
    /// Algorithm name advertised in Hello / capabilities (e.g. `"sa"`, `"gibbs"`).
    pub algorithm: &'static str,
    /// Hard cap on variables accepted for a job.
    pub max_nodes: u32,
    /// Hard cap on edges accepted for a job.
    pub max_edges: u32,
    /// Sampling-parameter envelope used by `adapt::adapt_params`.
    pub adapt: crate::adapt::AdaptBounds,
}

/// Failure to open a device / build a sampler, surfaced as `EnvIncompatible`.
#[derive(Debug)]
pub struct OpenError(pub String);

#[expect(
    clippy::print_stdout,
    reason = "user-facing CLI capabilities JSON for --capabilities"
)]
fn print_capabilities(id: &BackendIdentity) {
    println!(
        r#"{{"backend":"{}","algorithm":"{}","supported_kinds":["ISING_SAMPLE"],"max_nodes":{},"max_edges":{}}}"#,
        id.backend, id.algorithm, id.max_nodes, id.max_edges
    );
}

/// Emit a miner progress line every N completed jobs (v0.2 `mine_work_item`
/// parity).
const PROGRESS_LOG_INTERVAL: u64 = 10;

/// Depth of the read loop's queue to the outbound writer task.
///
/// Carries only control replies — `Ready`, `JobRequest`, `Status`, and
/// prepare-time `Reject`s — which are small and, apart from a burst of rejects
/// from a malformed job run, rare. Sized so the read loop never waits on it in
/// practice while still bounding memory if the peer stops reading entirely.
const CTRL_CHANNEL_DEPTH: usize = 256;

/// How often to send an HTTP/2 PING on an otherwise quiet session.
///
/// A miner grinding one hard nonce sends nothing for minutes at a time, so the
/// PINGs have to keep going while idle (`keep_alive_while_idle`) or the exact
/// case they exist for is the case they skip.
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(20);

/// How long to wait for a PING ack before treating the coordinator as gone.
const KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(20);

/// How often to re-check whether the sampler worker has finished, while waiting
/// out the shutdown grace window.
const SAMPLER_JOIN_POLL: Duration = Duration::from_millis(20);

/// What the read loop knew about a job at prepare time, held until the writer
/// finalizes it.
///
/// The sampling parameters build the outbound `SamplerMeta`. The rest exists so
/// the completion log can report an attempt the way the v0.2.1 miner did:
/// elapsed wall time, and the requirement the attempt was measured against.
/// Only the read loop sees the session `SetTarget`, so it records the
/// thresholds here rather than sharing the target with the writer.
struct PendingJob {
    num_reads: u32,
    num_sweeps: u32,
    started: std::time::Instant,
    /// Session energy threshold, or `None` when no `SetTarget` has arrived.
    max_energy_milli: Option<i64>,
    min_solutions: u32,
}

/// `job_id` → its [`PendingJob`], shared between the session's read loop (which
/// inserts) and its outbound writer (which removes).
type PendingParams = Arc<StdMutex<HashMap<Vec<u8>, PendingJob>>>;

/// Render the leading bytes of a job id for logs.
///
/// Job ids are opaque and long. Eight bytes is enough to correlate a completion
/// with its dispatch in a coordinator log without wrapping the line.
fn short_job_id(job_id: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(18);
    for b in job_id.iter().take(8) {
        let _ = write!(s, "{b:02x}");
    }
    if job_id.len() > 8 {
        s.push_str("..");
    }
    s
}

#[expect(
    clippy::cast_precision_loss,
    reason = "job count and elapsed seconds are small; the f64 rate is display-only"
)]
fn log_progress(
    backend: &str,
    jobs_done: u64,
    elapsed: Duration,
    best_energy_milli: i64,
    pending: Option<&PendingJob>,
) {
    let secs = elapsed.as_secs_f64();
    let rate = if secs > 0.0 {
        jobs_done as f64 / secs
    } else {
        0.0
    };
    let best = if best_energy_milli == i64::MAX {
        "n/a".to_owned()
    } else {
        energy_units(best_energy_milli).to_string()
    };
    let (reads, sweeps) = pending.map_or((0, 0), |p| (p.num_reads, p.num_sweeps));
    let requirement = format_requirement(pending);
    tracing::info!(
        "[quip-miner-{backend}] progress: {jobs_done} jobs | {rate:.1} jobs/s | reads={reads} sweeps={sweeps} | best={best} | {requirement}"
    );
}

/// Requirement text shared by the progress line.
///
/// The per-attempt line no longer carries this. Sampling parameters stay
/// constant for a round, so the progress interval is enough.
fn format_requirement(pending: Option<&PendingJob>) -> String {
    match pending.and_then(|p| p.max_energy_milli) {
        Some(max) => format!(
            "requires energy<={}, solutions>={}",
            energy_units(max),
            pending.map_or(0, |p| p.min_solutions)
        ),
        None => "no target set".to_owned(),
    }
}

/// Log one finished attempt, mirroring the v0.2.1 miner's per-attempt line.
///
/// Every terminal outcome is logged, not only the ones that clear the target.
/// A miner that is working correctly but not winning looks identical to a
/// wedged one unless the losing attempts are visible, which is the gap this
/// closes: the only per-job signal before this was [`log_progress`], and it
/// fires once per ten jobs, so a miner taking minutes per job showed nothing
/// for the better part of an hour.
///
/// `pending` is `None` when the writer sees a result for a job the read loop
/// never recorded. That is not reachable through the normal path, so the line
/// still goes out, with the parameters and elapsed time reported as unknown.
fn log_attempt(backend: &str, sr: &StreamResult, pending: Option<&PendingJob>) {
    let job = short_job_id(&sr.job_id);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "attempt wall time in ms fits u64 for any realistic job"
    )]
    let elapsed_ms = pending.map_or(0, |p| p.started.elapsed().as_millis() as u64);
    let device_ms = sr.device_access_time_us / 1000;
    let wall = format_duration_ms(elapsed_ms);
    let device = format_duration_ms(device_ms);

    match &sr.outcome {
        StreamOutcome::Completed(Ok(samples)) => {
            let best = samples.iter().map(|r| r.energy_milli).min();
            // Solutions at or below the session threshold: the count the
            // coordinator scores the attempt on. Without a target none of them
            // qualify yet, so report the raw sample count instead.
            let valid = match pending.and_then(|p| p.max_energy_milli) {
                Some(max) => samples.iter().filter(|r| r.energy_milli <= max).count(),
                None => samples.len(),
            };
            tracing::info!(
                "[quip-miner-{backend}] attempt {job}: energy {}, valid {valid}/{} \
                 | {wall} wall, {device} device",
                best.map_or_else(|| "n/a".to_owned(), |e| energy_units(e).to_string()),
                samples.len(),
            );
        }
        StreamOutcome::Completed(Err(reason)) => {
            tracing::warn!(
                "[quip-miner-{backend}] attempt {job}: rejected {} | {wall} wall",
                reason.as_str_name(),
            );
        }
        // Expected on every reseed, so this is not a degraded condition.
        StreamOutcome::Cancelled => {
            tracing::debug!("[quip-miner-{backend}] attempt {job}: cancelled after {wall}");
        }
    }
}

/// Outbound half of a session, and the SOLE owner of the outbound sender.
///
/// Reading the inbound stream and writing to the outbound one must not live in
/// the same task. A `tx.send().await` blocks once the outbound channel fills,
/// which happens whenever the coordinator is slow to read — and if that await
/// sits in the same loop as `inbound.message()`, the miner stops reading jobs
/// precisely because it is busy returning results. The coordinator,
/// symmetrically blocked writing jobs, then stops reading results, and both
/// peers park forever. This was reproducible: a 480-credit grant (~89 MB of
/// inline h/J in flight) deadlocked the session within three dispatches.
///
/// So the read loop hands outbound traffic here and never awaits the wire
/// itself. Results (large) arrive on `res_rx` straight from the sampler;
/// control replies (small, rare) arrive on `ctrl_rx` from the read loop.
///
/// Returns once both inputs are finished, or as soon as the outbound channel
/// closes.
async fn outbound_writer(
    tx: mpsc::Sender<MinerMsg>,
    mut res_rx: mpsc::Receiver<StreamResult>,
    mut ctrl_rx: mpsc::Receiver<MinerMsg>,
    pending: PendingParams,
    jobs_done: Arc<AtomicU64>,
    backend: &'static str,
) {
    // Progress logging (mirrors v0.2 mine_work_item's every-N-attempts line).
    let session_start = std::time::Instant::now();
    let mut best_energy_milli: i64 = i64::MAX;
    let mut done: u64 = 0;
    // Disables the control branch once the read loop is gone. A closed
    // `ctrl_rx` completes `recv()` with `None` immediately and forever, so
    // leaving the branch enabled turns this `select!` into a hot spin for the
    // whole shutdown drain — the stretch between `drop(ctrl_tx)` and the sampler
    // releasing `res_tx`, which lasts as long as the last in-flight job. Results
    // still drain: the branch that stays enabled is the one that waits properly.
    let mut ctrl_open = true;
    loop {
        tokio::select! {
            biased;
            // Drain completed results first so a busy sampler never backs up.
            Some(sr) = res_rx.recv() => {
                let entry = {
                    let mut p = match pending.lock() {
                        Ok(p) => p,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    p.remove(&sr.job_id)
                };
                let (reads, sweeps) = entry
                    .as_ref()
                    .map_or((0, 0), |e| (e.num_reads, e.num_sweeps));
                // A Cancelled job neither advances progress nor updates
                // best energy; finalize_result just refunds its credit.
                let completed = matches!(sr.outcome, StreamOutcome::Completed(_));
                if let StreamOutcome::Completed(Ok(samples)) = &sr.outcome {
                    if let Some(e) = samples.iter().map(|r| r.energy_milli).min() {
                        best_energy_milli = best_energy_milli.min(e);
                    }
                }
                log_attempt(backend, &sr, entry.as_ref());
                for reply in finalize_result(sr, reads, sweeps, &mut done) {
                    if tx.send(reply).await.is_err() {
                        return;
                    }
                }
                jobs_done.store(done, Ordering::Relaxed);
                if completed && done > 0 && done.is_multiple_of(PROGRESS_LOG_INTERVAL) {
                    log_progress(
                        backend,
                        done,
                        session_start.elapsed(),
                        best_energy_milli,
                        entry.as_ref(),
                    );
                }
            }
            ctrl = ctrl_rx.recv(), if ctrl_open => {
                if let Some(msg) = ctrl {
                    if tx.send(msg).await.is_err() {
                        return;
                    }
                } else {
                    // Read loop is gone. Finish if the sampler has already
                    // drained too, otherwise keep draining results with this
                    // branch switched off.
                    ctrl_open = false;
                    if res_rx.is_closed() {
                        return;
                    }
                }
            }
            else => return,
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "single bidi session select-loop; splitting would obscure the control flow"
)]
async fn run_session<S: Sampler>(
    uri: &str,
    miner_id: &str,
    id: &BackendIdentity,
    sampler: Arc<S>,
    sweeps_per_beta: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Resolve token before any network I/O so a missing QUIP_SESSION_TOKEN
    // always maps to exit 77 (never InternalFatal from a connect failure).
    let hello = build_hello(
        miner_id,
        id.backend,
        id.algorithm,
        &[JobKind::IsingSample],
        BackendCaps {
            max_nodes: id.max_nodes,
            max_edges: id.max_edges,
        },
    )?;

    let path = uri.strip_prefix("unix://").unwrap_or(uri).to_string();
    let channel = Endpoint::try_from("http://[::]:50051")? // dummy authority for UDS
        // The keepalive the read loop's comment below assumes. It has to be
        // configured to exist: tonic's defaults leave HTTP/2 PINGs off, so a
        // coordinator that vanished without closing the socket left this session
        // parked in `inbound.message()` indefinitely, with no application-level
        // timeout to catch it by design.
        //
        // Scope, stated so the claim is not overread a second time: PINGs are
        // answered by the peer's HTTP/2 layer, not its application. This detects
        // a peer that is *gone*. It does nothing for a peer that is alive but
        // has stopped reading — that is a flow-control stall, and it needs
        // different tools.
        .http2_keep_alive_interval(KEEPALIVE_INTERVAL)
        .keep_alive_timeout(KEEPALIVE_TIMEOUT)
        .keep_alive_while_idle(true)
        .connect_with_connector(tower::service_fn(move |_: Uri| {
            let p = path.clone();
            async move {
                let s = tokio::net::UnixStream::connect(p).await?;
                Ok::<_, std::io::Error>(hyper_util::rt::TokioIo::new(s))
            }
        }))
        .await?;
    let mut client = MinerServiceClient::new(channel);

    let (tx, rx) = mpsc::channel::<MinerMsg>(16);
    tx.send(miner(miner_msg::Msg::Hello(hello))).await?;

    let mut inbound = client.session(ReceiverStream::new(rx)).await?.into_inner();

    // Streaming sampler on a blocking thread: it pulls StreamJobs and emits
    // StreamResults in completion order, keeping `stream_width` models in flight.
    let width = sampler.stream_width().max(1);
    // Prefetch a full extra batch beyond the active set: the backend keeps each
    // stream lane's NEXT slot pre-loaded so a completed slot rotates into a
    // ready one with no idle gap while it is downloaded and refilled (the
    // 3-slot pipeline in quip-miner-cuda). Without this look-ahead the lane
    // stalls every cycle waiting for the next job to arrive over the wire.
    let prefetch = width.saturating_mul(2);
    let cap = prefetch.max(8);
    let (job_tx, job_rx) = mpsc::channel::<StreamJob>(cap);
    let (res_tx, res_rx) = mpsc::channel::<StreamResult>(cap);
    // Control-plane cancellation watermark: bumped here on `Cancel`, read by the
    // sampler thread to skip/abort jobs from generations the coordinator
    // abandoned on reseed.
    let cancel = CancelGuard::default();
    let sampler_thread = {
        let s = Arc::clone(&sampler);
        let cancel = cancel.clone();
        // Named, because this is the thread anyone diagnosing a stalled miner
        // needs to find first. An unnamed thread inherits the process name, and
        // in a live stall this one sat among three other threads all reporting
        // `quip-cuda-sa` in `top -H`, identifiable only by elimination. Held to
        // 15 bytes, which is what Linux stores in `/proc/<pid>/task/*/comm`.
        std::thread::Builder::new()
            .name("quip-sampler".to_owned())
            .spawn(move || s.sample_stream(job_rx, res_tx, cancel))?
    };

    let mut grace_ms: u64 = 5000;
    let mut num_sweeps = DEFAULT_NUM_SWEEPS;
    let mut topology: Option<TopologyCache> = None;
    let mut target: Option<SessionTarget> = None;
    // job_id → (num_reads, num_sweeps) resolved at prepare, for the result meta.
    // Shared: the read loop inserts at prepare, the writer task removes at
    // finalize. The lock is never held across an await.
    let pending: PendingParams = Arc::new(StdMutex::new(HashMap::new()));
    // Published by the writer, read here for `Status.jobs_done`.
    let jobs_done = Arc::new(AtomicU64::new(0));

    let (ctrl_tx, ctrl_rx) = mpsc::channel::<MinerMsg>(CTRL_CHANNEL_DEPTH);
    let writer = tokio::spawn(outbound_writer(
        tx.clone(),
        res_rx,
        ctrl_rx,
        Arc::clone(&pending),
        Arc::clone(&jobs_done),
        id.backend,
    ));

    loop {
        // Read loop: owns the inbound stream and nothing else. Every outbound
        // message goes through `ctrl_tx` so this loop cannot be blocked by a
        // slow or backed-up peer.
        //
        // No wall-clock idle timeout: a miner grinding a hard nonce for
        // minutes-to-hours legitimately receives nothing from the
        // coordinator meanwhile — it is busy, not dead. Liveness of a
        // truly-gone peer surfaces here as a closed stream (`Ok(None)`) or
        // a transport error (`Err`); dead-peer detection over the network
        // belongs to HTTP/2 keepalive, not an application quiet-period.
        let msg = inbound.message().await;
        {
            let cm: CoordMsg = match msg {
                Ok(Some(cm)) => cm,
                Ok(None) => break,
                Err(status) => return Err(status.into()),
            };
            match cm.msg {
                Some(coord_msg::Msg::Welcome(w)) => {
                    if w.protocol_version != 1 {
                        return Err(SessionError::BadWelcome(w.protocol_version).into());
                    }
                }
                Some(coord_msg::Msg::Configure(c)) => {
                    // Hand the verbatim config subsection to the backend to
                    // parse against its own schema (overrides CLI, warns on
                    // unknown fields / overrides) before mining starts.
                    sampler.apply_config(&c.backend_toml);
                    num_sweeps = num_sweeps_from_toml(&c.backend_toml);
                    let config = SessionConfig::from_configure(miner_id.into(), &c);
                    ctrl_tx.send(miner(miner_msg::Msg::Ready(Ready {}))).await?;
                    // Request enough credits to keep the prefetch buffer
                    // full (active + next per lane), so the backend always
                    // has a NEXT slot to rotate into.
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "stream width / prefetch are small device-local counts well under u32::MAX"
                    )]
                    let depth = config.queue_depth.max(prefetch as u32);
                    // INVARIANT: the granted pool must not exceed the job
                    // channel's capacity (`cap`), or the read loop can block
                    // on `job_tx.send` with jobs still arriving — the same
                    // deadlock one layer down. `cap == prefetch` and
                    // `depth <= prefetch` unless the coordinator's
                    // `queue_depth` is larger, so clamp to `cap`.
                    let depth = depth.min(u32::try_from(cap).unwrap_or(u32::MAX));
                    ctrl_tx
                        .send(miner(miner_msg::Msg::JobRequest(JobRequest {
                            credits: depth,
                        })))
                        .await?;
                }
                Some(coord_msg::Msg::Topology(t)) => {
                    topology = Some(TopologyCache::from_proto(&t));
                }
                Some(coord_msg::Msg::SetTarget(s)) => {
                    target = Some(SessionTarget::from_proto(&s));
                }
                Some(coord_msg::Msg::Job(job)) => {
                    match prepare_job(
                        job,
                        &*sampler,
                        id,
                        num_sweeps,
                        sweeps_per_beta,
                        topology.as_ref(),
                        target.as_ref(),
                    ) {
                        Prepared::Reject(msg) => {
                            // Rejecting at prepare time is terminal for the
                            // job, so ask for a replacement credit — same as
                            // a completion — to keep the coordinator's
                            // consume-on-dispatch pool from leaking a slot.
                            ctrl_tx.send(msg).await?;
                            ctrl_tx
                                .send(miner(miner_msg::Msg::JobRequest(JobRequest { credits: 1 })))
                                .await?;
                        }
                        Prepared::Sample {
                            job,
                            num_reads,
                            num_sweeps: ns,
                        } => {
                            tracing::debug!(
                                "{} received job {}: {} nodes, {} edges | reads={num_reads} sweeps={ns}",
                                id.backend,
                                short_job_id(&job.job_id),
                                job.graph.num_nodes(),
                                job.graph.edges.len(),
                            );
                            {
                                let mut p = match pending.lock() {
                                    Ok(p) => p,
                                    Err(poisoned) => poisoned.into_inner(),
                                };
                                let _ = p.insert(
                                    job.job_id.clone(),
                                    PendingJob {
                                        num_reads,
                                        num_sweeps: ns,
                                        started: std::time::Instant::now(),
                                        // `drive` mode sets no real threshold
                                        // and sends `i64::MAX`. Reporting that
                                        // as a requirement is noise.
                                        max_energy_milli: target
                                            .as_ref()
                                            .map(|t| t.max_energy_milli)
                                            .filter(|&m| m != i64::MAX),
                                        min_solutions: target
                                            .as_ref()
                                            .map_or(0, |t| t.min_solutions),
                                    },
                                );
                            }
                            if job_tx.send(job).await.is_err() {
                                break;
                            }
                        }
                    }
                }
                Some(coord_msg::Msg::Cancel(c)) => {
                    // Abandon every generation at/below the watermark; the
                    // sampler skips buffered stale jobs at dequeue and (in
                    // backends that override sample_stream) aborts the
                    // in-flight one at its next checkpoint.
                    cancel.cancel_through(c.max_generation);
                    ctrl_tx
                        .send(status_msg(
                            miner_id,
                            jobs_done.load(Ordering::Relaxed),
                            sampler.utilization(),
                        ))
                        .await?;
                }
                Some(coord_msg::Msg::Ping(_)) => {
                    ctrl_tx
                        .send(status_msg(
                            miner_id,
                            jobs_done.load(Ordering::Relaxed),
                            sampler.utilization(),
                        ))
                        .await?;
                }
                Some(coord_msg::Msg::Shutdown(s)) => {
                    grace_ms = if s.grace_ms == 0 {
                        5000
                    } else {
                        u64::from(s.grace_ms)
                    };
                    break;
                }
                None => {}
            }
        }
    }

    // Stop feeding, then let the writer flush in-flight results within the
    // grace window. Closing both of its inputs is what ends it: `job_tx` stops
    // the sampler, which drops `res_tx`; `ctrl_tx` closes the control side.
    // The writer drains whatever the sampler already produced before returning,
    // so the drain that used to live here now happens there.
    drop(job_tx);
    drop(ctrl_tx);
    let grace = Duration::from_millis(grace_ms);
    if tokio::time::timeout(grace, writer).await.is_err() {
        tracing::warn!("outbound writer did not finish within the shutdown grace window");
    }
    // Bounded wait for the sampler worker, then surface panics: a join `Err` is
    // a panic payload, not a clean drain.
    //
    // `JoinHandle::join` is unbounded and uninterruptible, and the worker is not
    // guaranteed to return. One parked in `blocking_send` on a result channel
    // nobody is draining never does, so joining unconditionally made the process
    // ignore SIGTERM and sit until something external sent SIGKILL — observed as
    // multi-minute container restarts on a miner that had already stopped
    // producing. Poll `is_finished` against the same grace window instead and
    // abandon the thread if it overruns: the process is exiting, so a detached
    // worker costs nothing, while a hung join costs the entire shutdown.
    let sampler_deadline = tokio::time::Instant::now() + grace;
    while !sampler_thread.is_finished() && tokio::time::Instant::now() < sampler_deadline {
        tokio::time::sleep(SAMPLER_JOIN_POLL).await;
    }
    if !sampler_thread.is_finished() {
        tracing::error!(
            "sampler thread did not finish within the shutdown grace window; abandoning it"
        );
    } else if let Err(panic) = sampler_thread.join() {
        let payload = panic_payload_message(&*panic);
        tracing::error!(panic = %payload, "sampler thread panicked");
        return Err(format!("sampler thread panicked: {payload}").into());
    }

    drop(tx);
    let drain = async {
        while inbound.message().await?.is_some() {}
        Ok::<(), tonic::Status>(())
    };
    let _ = tokio::time::timeout(grace, drain).await;
    Ok(())
}

/// Format a `JoinHandle` panic payload for logging / error messages.
fn panic_payload_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

fn map_err_to_exit(err: Box<dyn std::error::Error>, backend: &str) -> StdExitCode {
    // Prefer the canonical `SessionError -> ExitCode` mapping (quip-protocol)
    // over a hand-rolled match, so real miners exit the same code as the mock
    // reference (e.g. BadWelcome -> ConfigInvalid/64, not InternalFatal/70).
    let err = match err.downcast::<SessionError>() {
        Ok(se) => return StdExitCode::from(ExitCode::from(*se) as u8),
        Err(err) => err,
    };
    // Type-erased fallback: the error crossed a boundary that lost the
    // concrete `SessionError` (e.g. tonic::Status::into()). Recover only the
    // two documented exit codes we can identify from the message.
    let msg = err.to_string();
    if msg.contains("QUIP_SESSION_TOKEN") || msg.contains("session token") {
        return StdExitCode::from(ExitCode::TokenRejected as u8);
    }
    if msg.contains("unexpected protocol version") {
        return StdExitCode::from(ExitCode::ConfigInvalid as u8);
    }
    tracing::error!("quip-miner-{backend} fatal: {err}");
    StdExitCode::from(ExitCode::InternalFatal as u8)
}

/// Miner entry point. Dispatches `--capabilities`/`--check`/session mode.
///
/// `open` opens the device and builds the [`Sampler`]; it runs for `--check`
/// (result discarded) and for session mode. `--capabilities` never calls it.
pub fn run<S: Sampler>(
    id: BackendIdentity,
    common: &CommonArgs,
    open: impl FnOnce() -> Result<S, OpenError>,
) -> StdExitCode {
    // Install the subscriber before anything else can log. `--capabilities`
    // writes JSON to stdout and must stay parseable, but the subscriber writes
    // to stderr, so installing first is safe for it too.
    if let Err(e) = crate::logging::init(&common.log_level) {
        #[expect(
            clippy::print_stderr,
            reason = "the subscriber failed to install, so tracing would discard this"
        )]
        {
            eprintln!("quip-miner-{}: {e}", id.backend);
        }
        return StdExitCode::from(ExitCode::ConfigInvalid as u8);
    }

    if common.capabilities {
        print_capabilities(&id);
        return StdExitCode::SUCCESS;
    }
    if common.check {
        return match open() {
            Ok(_) => StdExitCode::SUCCESS,
            Err(e) => {
                tracing::error!("{} check failed: {}", id.backend, e.0);
                StdExitCode::from(ExitCode::EnvIncompatible as u8)
            }
        };
    }

    let Some(uri) = common.quip_coordinator.clone() else {
        tracing::error!("error: --quip-coordinator required for session mode");
        return StdExitCode::from(ExitCode::ConfigInvalid as u8);
    };
    let miner_id = common
        .miner_id
        .clone()
        .unwrap_or_else(|| format!("{}-0", id.backend));

    let sampler = match open() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("failed to open {} device: {}", id.backend, e.0);
            return StdExitCode::from(ExitCode::EnvIncompatible as u8);
        }
    };

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!("failed to start tokio runtime: {e}");
            return StdExitCode::from(ExitCode::InternalFatal as u8);
        }
    };

    match rt.block_on(run_session(
        &uri,
        &miner_id,
        &id,
        Arc::new(sampler),
        common.sweeps_per_beta,
    )) {
        Ok(()) => StdExitCode::SUCCESS,
        Err(e) => map_err_to_exit(e, id.backend),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The writer waits, rather than spins, once the read loop is gone while the
    /// sampler still holds `res_tx`.
    ///
    /// That window is every shutdown: `run_session` drops `ctrl_tx` first and
    /// the sampler keeps `res_tx` until its last in-flight job finishes. A
    /// closed `ctrl_rx` completes `recv()` with `None` immediately and forever,
    /// so a `select!` that leaves the branch enabled burns a core for the whole
    /// drain.
    ///
    /// The paused clock is what makes that observable rather than a matter of
    /// opinion: tokio advances a paused clock only while every task is idle, so
    /// a spinning writer stops the sleep below from ever returning and this test
    /// hangs instead of passing.
    #[tokio::test(start_paused = true)]
    async fn writer_parks_when_the_read_loop_closes_before_the_sampler() {
        let (tx, mut out_rx) = mpsc::channel::<MinerMsg>(16);
        let (res_tx, res_rx) = mpsc::channel::<StreamResult>(4);
        let (ctrl_tx, ctrl_rx) = mpsc::channel::<MinerMsg>(4);
        let writer = tokio::spawn(outbound_writer(
            tx,
            res_rx,
            ctrl_rx,
            Arc::new(StdMutex::new(HashMap::new())),
            Arc::new(AtomicU64::new(0)),
            "test",
        ));

        // Read loop exits, as it does on Shutdown; the sampler is still working.
        drop(ctrl_tx);
        tokio::time::sleep(Duration::from_secs(30)).await;
        assert!(
            !writer.is_finished(),
            "writer must keep draining results until the sampler releases res_tx"
        );

        // And it is still serving the result side while the control side is shut.
        res_tx
            .send(StreamResult {
                job_id: vec![1, 2, 3],
                outcome: StreamOutcome::Cancelled,
                device_access_time_us: 0,
            })
            .await
            .unwrap();
        let msg = out_rx.recv().await.expect("credit refund reached the wire");
        assert!(
            matches!(msg.msg, Some(miner_msg::Msg::JobRequest(_))),
            "a Cancelled result refunds its credit and sends nothing else"
        );

        // Sampler finishes: both inputs are done, so the writer returns.
        drop(res_tx);
        tokio::time::timeout(Duration::from_secs(5), writer)
            .await
            .expect("writer returned once both inputs closed")
            .expect("writer did not panic");
    }
}
