//! tonic `MinerService` server: token verify, handshake, job dispatch, results.

use crate::chain::{ChainClient, Proof, SubmitAction};
use crate::config::LaunchEntry;
use crate::router::{MinerCaps, Router};
use crate::topology::Topology;
use crate::validate::{beats_current, validate_result, ResolvedTopo};
use quip_proto::v1::miner_service_server::{MinerService, MinerServiceServer};
use quip_proto::v1::{coord_msg, miner_msg, Configure, CoordMsg, MinerMsg, Shutdown, Welcome};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UnixListener;
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

/// Depth of the read loop's queue of credit grants to the dispatcher task.
/// Grants are single `u32`s, so this only absorbs a burst of `JobRequest`s
/// arriving while the dispatcher is mid-send.
const GRANT_CHANNEL_DEPTH: usize = 1024;

/// Grant `credits` and send every job the miner is now entitled to.
///
/// Jobs are collected under the state lock and sent after releasing it: the
/// lock guards coordinator-wide state, so holding it across a `tx.send().await`
/// blocks every *other* miner's session for as long as this one's outbound
/// channel stays full.
///
/// Returns `false` when the outbound channel is closed (session is over).
async fn dispatch_granted(
    state: &Arc<Mutex<CoordinatorState>>,
    miner_id: &str,
    credits: u32,
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
) -> bool {
    let jobs = {
        let mut st = state.lock().await;
        st.router.grant_credits(miner_id, credits);
        let mut jobs = Vec::new();
        while let Some(job) = st.router.next_job(miner_id) {
            st.dispatch_inflight(miner_id, job.clone());
            jobs.push(job);
        }
        tracing::debug!(
            miner = %miner_id,
            added = credits,
            dispatched = jobs.len(),
            staged = st.router.staged_len(miner_id),
            "dispatch drain"
        );
        st.metrics
            .record_contexts_dispatched(miner_id, jobs.len() as u64);
        jobs
    };
    for job in jobs {
        if tx.send(Ok(coord(coord_msg::Msg::Job(job)))).await.is_err() {
            return false;
        }
    }
    true
}

/// Record a stale drop without holding the coordinator mutex across an await.
async fn st_metrics_stale(state: &Arc<Mutex<CoordinatorState>>, miner_id: &str) {
    let metrics = Arc::clone(&state.lock().await.metrics);
    metrics.record_stale_drop(miner_id);
}

/// Unguessable per-spawn session token (32 random bytes, hex-encoded).
///
/// # Panics
///
/// Panics if the OS random number generator fails.
#[must_use]
pub fn gen_session_token() -> String {
    let mut buf = [0u8; 32];
    #[expect(
        clippy::expect_used,
        reason = "OS RNG failure is unrecoverable at process start"
    )]
    {
        getrandom::getrandom(&mut buf).expect("os rng");
    }
    let mut s = String::with_capacity(buf.len() * 2);
    for b in &buf {
        let _ = write!(s, "{b:02x}");
    }
    s
}

pub(crate) fn coord(msg: coord_msg::Msg) -> CoordMsg {
    CoordMsg { msg: Some(msg) }
}

/// Info-log one miner liveness transition. Stale-round logs carry the
/// coordinator's current qblock for operators.
fn log_liveness(miner_id: &str, qblock: Option<u64>, event: &crate::liveness::LivenessEvent) {
    use crate::liveness::LivenessEvent;
    match event {
        LivenessEvent::EnteredPaused { reason } => tracing::info!(
            miner = %miner_id,
            reason = reason.as_deref().unwrap_or("unspecified"),
            "miner paused"
        ),
        LivenessEvent::Resumed => tracing::info!(miner = %miner_id, "miner resumed mining"),
        LivenessEvent::StaleRound { reported, current } => tracing::info!(
            miner = %miner_id,
            reported_generation = reported,
            current_generation = current,
            qblock = qblock.unwrap_or(0),
            "miner reports mining a stale round"
        ),
    }
}

/// Render the leading bytes of a job id the way the miner does.
///
/// Eight bytes plus `..` is enough to correlate a stash decision with the
/// miner's per-attempt line.
fn short_job_id(job_id: &[u8]) -> String {
    let mut s = String::with_capacity(18);
    for b in job_id.iter().take(8) {
        let _ = write!(s, "{b:02x}");
    }
    if job_id.len() > 8 {
        s.push_str("..");
    }
    s
}

/// How many top candidates the win-time stash retains per generation.
const WIN_STASH_K: usize = 8;

/// Shared coordinator runtime state for one or more miner sessions.
pub struct CoordinatorState {
    /// Per-miner expected session tokens (`miner_id` → hex token).
    pub expected_tokens: HashMap<String, String>,
    /// Per-miner `Configure` messages to send after `Welcome`.
    pub configure: HashMap<String, Configure>,
    /// Active graph topology advertised to miners, if any.
    pub topology: Option<Topology>,
    /// Position-indexed scoring form of `topology`, resolved once via
    /// [`CoordinatorState::set_topology`]. A run constant every result borrows
    /// by `Arc`, so the graph is never rebuilt per result.
    pub resolved_topo: Arc<ResolvedTopo>,
    /// Difficulty target advertised to miners via `SetTarget`.
    pub target: Option<quip_proto::v1::SetTarget>,
    /// Work router: credits, staging queues, and miner caps.
    pub router: Router,
    /// `job_id` → `Job`, for validation context.
    pub inflight: HashMap<Vec<u8>, quip_proto::v1::Job>,
    /// `job_id` → owning `miner_id`, so a crashed miner's in-flight jobs can be
    /// isolated and re-routed. Maintained on the production path via
    /// [`CoordinatorState::dispatch_inflight`]/[`CoordinatorState::complete_inflight`];
    /// the drive harness leaves it empty (it never crash-requeues).
    pub inflight_owner: HashMap<Vec<u8>, String>,
    /// Live per-miner outbound channels, so the supervisor can push an in-band
    /// `Shutdown`/cancel to a running session. Registered on handshake success,
    /// removed when the session ends.
    pub outbound: HashMap<String, mpsc::Sender<Result<CoordMsg, Status>>>,
    /// Live per-miner dispatcher wakeups. A miner grants its credits as soon as
    /// it starts, which is usually before the first chain snapshot arrives, so
    /// the grant drains an empty queue and the dispatcher parks. Staging alone
    /// does not wake it, and the miner sends no further `JobRequest` until a
    /// job completes. The feeder sends a zero-credit grant here after it stages
    /// work, which re-runs the drain without changing the credit balance.
    pub wakeups: HashMap<String, mpsc::Sender<u32>>,
    /// `job_id` (nonce) → the 32-byte salt it was derived from, so the winning
    /// `Proof` carries the salt live submit requires. Recorded when the feeder
    /// stages a job, consumed on its `Result`, cleared on reseed.
    pub salts: HashMap<Vec<u8>, [u8; 32]>,
    /// Best accepted energy so far this generation, if any.
    pub current_best_milli: Option<i64>,
    /// Count of results that passed through validation.
    pub results_validated: u64,
    /// Highest abandoned generation reported by any miner via `Status`.
    pub last_abandoned_generation: u64,
    /// Sink for mining-attempt records + summaries (dashboard). `None` disables
    /// recording.
    pub attempt_tx: Option<std::sync::mpsc::Sender<crate::attempt::WriterMsg>>,
    /// Current chain quantum-block id, refreshed by the feeder on reseed; keys
    /// the per-qblock attempt logs.
    pub qblock_id: Option<u64>,
    /// Win-time stash of the most-viable sub-threshold candidates for the
    /// current generation, armed with the decay projection on reseed.
    pub stash: crate::stash::WinStash,
    /// Block-interval/lag tracker for current-block estimation.
    pub timing: crate::timing::TimingTracker,
    /// Feeder's current round; set on reseed. Compared against a miner's
    /// self-reported generation to detect stale-round mining.
    pub generation: u64,
    /// Per-miner last-known self-reported liveness (mining/paused + round),
    /// updated from ping-reply `Status` messages.
    pub miner_liveness: HashMap<String, crate::liveness::MinerLiveness>,
    /// Live controller counters. Shared with the dashboard router, so the HTTP
    /// handlers never take this mutex.
    pub metrics: Arc<crate::metrics::CoordinatorMetrics>,
}

impl CoordinatorState {
    /// Empty coordinator state with default stash size and timing tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            expected_tokens: HashMap::new(),
            configure: HashMap::new(),
            topology: None,
            resolved_topo: Arc::new(ResolvedTopo::default()),
            target: None,
            router: Router::new(),
            inflight: HashMap::new(),
            inflight_owner: HashMap::new(),
            outbound: HashMap::new(),
            wakeups: HashMap::new(),
            salts: HashMap::new(),
            current_best_milli: None,
            results_validated: 0,
            last_abandoned_generation: 0,
            attempt_tx: None,
            qblock_id: None,
            stash: crate::stash::WinStash::new(WIN_STASH_K),
            timing: crate::timing::TimingTracker::with_defaults(),
            generation: 0,
            miner_liveness: HashMap::new(),
            metrics: Arc::new(crate::metrics::CoordinatorMetrics::new(&[])),
        }
    }

    /// Set the session topology and resolve its position-indexed scoring form
    /// once. Both drive and production go through here so `validate_result`
    /// never rebuilds the graph per result.
    pub fn set_topology(&mut self, topology: Option<Topology>) {
        self.resolved_topo = Arc::new(
            topology
                .as_ref()
                .map(|t| ResolvedTopo::new(&t.nodes, &t.edge_pairs()))
                .unwrap_or_default(),
        );
        self.topology = topology;
    }

    /// Remember the salt a staged job was derived from (feeder path).
    pub fn record_salt(&mut self, job_id: &[u8], salt: [u8; 32]) {
        let _ = self.salts.insert(job_id.to_vec(), salt);
    }

    /// Consume the salt for a completed job, for its Proof.
    pub fn take_salt(&mut self, job_id: &[u8]) -> Option<[u8; 32]> {
        self.salts.remove(job_id)
    }

    /// Drop all remembered salts on reseed — the prior generation is cancelled.
    pub fn clear_salts(&mut self) {
        self.salts.clear();
    }

    /// Drop in-flight `PoW` jobs whose generation is at most `max_generation`.
    /// Mempool jobs (`generation == 0`) stay. Returns how many jobs were dropped.
    ///
    /// A late `Result` for a dropped id then sees `complete_inflight` return
    /// `None`, so the coordinator neither scores nor submits it.
    pub fn cancel_inflight(&mut self, max_generation: u64) -> usize {
        let ids: Vec<Vec<u8>> = self
            .inflight
            .iter()
            .filter(|(_, job)| job.generation != 0 && job.generation <= max_generation)
            .map(|(id, _)| id.clone())
            .collect();
        let n = ids.len();
        for id in ids {
            let _ = self.complete_inflight(&id);
        }
        n
    }

    /// Record a dispatched job as in-flight, attributing it to `miner_id`.
    pub fn dispatch_inflight(&mut self, miner_id: &str, job: quip_proto::v1::Job) {
        let _ = self
            .inflight_owner
            .insert(job.job_id.clone(), miner_id.to_string());
        let _ = self.inflight.insert(job.job_id.clone(), job);
    }

    /// Clear an in-flight job on a terminal event (Result/Reject); returns the
    /// job for validation context.
    pub fn complete_inflight(&mut self, job_id: &[u8]) -> Option<quip_proto::v1::Job> {
        let _ = self.inflight_owner.remove(job_id);
        self.inflight.remove(job_id)
    }

    /// Reclaim every job a miner owned — its in-flight jobs plus its staged
    /// queue — so they can be re-routed after a crash. Also drops its outbound
    /// channel.
    pub fn reclaim_miner(&mut self, miner_id: &str) -> Vec<quip_proto::v1::Job> {
        let owned: Vec<Vec<u8>> = self
            .inflight_owner
            .iter()
            .filter(|(_, m)| m.as_str() == miner_id)
            .map(|(id, _)| id.clone())
            .collect();
        let mut jobs = Vec::with_capacity(owned.len());
        for id in owned {
            let _ = self.inflight_owner.remove(&id);
            if let Some(job) = self.inflight.remove(&id) {
                jobs.push(job);
            }
        }
        jobs.extend(self.router.reclaim(miner_id));
        let _ = self.outbound.remove(miner_id);
        jobs
    }

    /// Register a live session's outbound channel for supervisor-initiated sends.
    pub fn register_outbound(
        &mut self,
        miner_id: &str,
        tx: mpsc::Sender<Result<CoordMsg, Status>>,
    ) {
        let _ = self.outbound.insert(miner_id.to_string(), tx);
    }

    /// Register a live session's dispatcher wakeup channel.
    pub fn register_wakeup(&mut self, miner_id: &str, grants: mpsc::Sender<u32>) {
        let _ = self.wakeups.insert(miner_id.to_string(), grants);
    }

    /// Wake a miner's dispatcher so it drains newly staged work against credits
    /// the miner already granted.
    ///
    /// The send is non-blocking on purpose. A full channel already holds an
    /// unread wakeup, and one wakeup drains the whole queue, so dropping this
    /// one loses nothing. Blocking here would stall the feeder for every other
    /// miner.
    pub fn wake_dispatcher(&self, miner_id: &str) {
        if let Some(grants) = self.wakeups.get(miner_id) {
            let _ = grants.try_send(0);
        }
    }

    /// Drop a session's outbound channel when it ends.
    pub fn deregister_outbound(&mut self, miner_id: &str) {
        let _ = self.outbound.remove(miner_id);
        let _ = self.wakeups.remove(miner_id);
    }
}

impl Default for CoordinatorState {
    fn default() -> Self {
        Self::new()
    }
}

/// gRPC service holding shared state + chain client.
pub struct CoordinatorService<C: ChainClient + 'static> {
    /// Shared session state.
    pub state: Arc<Mutex<CoordinatorState>>,
    /// Chain client for proof submit.
    pub chain: Arc<C>,
    /// Notifies waiters when a proof is submitted successfully.
    pub submit_notify: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

#[tonic::async_trait]
impl<C: ChainClient + 'static> MinerService for CoordinatorService<C> {
    type SessionStream = ReceiverStream<Result<CoordMsg, Status>>;

    async fn session(
        &self,
        request: Request<Streaming<MinerMsg>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let mut inbound = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<CoordMsg, Status>>(64);
        let state = Arc::clone(&self.state);
        let chain = Arc::clone(&self.chain);
        let submit_notify = Arc::clone(&self.submit_notify);

        drop(tokio::spawn(async move {
            run_session(&mut inbound, &tx, state, chain, submit_notify).await;
        }));

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "session loop is one cohesive handshake + message dispatch"
)]
async fn run_session<C: ChainClient>(
    inbound: &mut Streaming<MinerMsg>,
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
    state: Arc<Mutex<CoordinatorState>>,
    chain: Arc<C>,
    submit_notify: Arc<Mutex<Option<oneshot::Sender<()>>>>,
) {
    // 1. Hello
    let Ok(Some(MinerMsg {
        msg: Some(miner_msg::Msg::Hello(hello)),
    })) = inbound.message().await
    else {
        return;
    };

    let miner_id = hello.miner_id.clone();
    let (token_ok, protocol_ok, configure) = {
        let st = state.lock().await;
        let expected = st
            .expected_tokens
            .get(&miner_id)
            .cloned()
            .unwrap_or_default();
        let token_ok = !expected.is_empty() && hello.session_token == expected;
        let protocol_ok = hello.protocol_version == 1;
        let configure = st.configure.get(&miner_id).cloned().unwrap_or(Configure {
            queue_depth: 3,
            idle_timeout_s: 300,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: String::new(),
        });
        (token_ok, protocol_ok, configure)
    };

    if !token_ok || !protocol_ok {
        // Drop the stream; well-behaved miners exit 77 on handshake failure.
        // Say why: a rejected miner is otherwise invisible from both sides —
        // the coordinator drops the stream and the miner only sees EOF.
        tracing::warn!(
            miner = %miner_id,
            token_ok,
            protocol_ok,
            offered_protocol = hello.protocol_version,
            "miner handshake rejected; dropping session"
        );
        return;
    }

    {
        let mut st = state.lock().await;
        st.router.register_miner(
            miner_id.clone(),
            MinerCaps {
                backend: hello.backend.clone(),
                algorithm: hello.algorithm.clone(),
                supported_kinds: hello.supported_kinds.clone(),
                max_nodes: hello.max_nodes,
                max_edges: hello.max_edges,
            },
        );
    }
    tracing::info!(
        miner = %miner_id,
        backend = %hello.backend,
        algorithm = %hello.algorithm,
        max_nodes = hello.max_nodes,
        max_edges = hello.max_edges,
        "miner registered"
    );

    // 2. Welcome + Configure (+ Topology if cached)
    if tx
        .send(Ok(coord(coord_msg::Msg::Welcome(Welcome {
            protocol_version: 1,
        }))))
        .await
        .is_err()
    {
        return;
    }
    let configure_queue_depth = configure.queue_depth;
    if tx
        .send(Ok(coord(coord_msg::Msg::Configure(configure))))
        .await
        .is_err()
    {
        return;
    }
    {
        let st = state.lock().await;
        if let Some(topo) = st.topology.as_ref() {
            let _ = tx
                .send(Ok(coord(coord_msg::Msg::Topology(topo.to_proto()))))
                .await;
        }
        // Push the current difficulty so a miner joining mid-generation tracks
        // it immediately, rather than waiting for the next reseed broadcast.
        if let Some(target) = st.target {
            let _ = tx.send(Ok(coord(coord_msg::Msg::SetTarget(target)))).await;
        }
    }

    // Register this session's outbound channel so the supervisor can push an
    // in-band Shutdown/cancel while it runs; dropped when the loop exits.
    state.lock().await.register_outbound(&miner_id, tx.clone());

    let seed_credits = configure_queue_depth.max(1);
    // Declared before the dispatcher so the wakeup sender can be registered
    // alongside it; see `wake_dispatcher`.
    // Dispatcher task: the only writer of `Job`s for this session.
    //
    // Job sends must not happen on the read path. `tx.send().await` blocks once
    // the outbound channel fills, and this loop is the only reader of the
    // miner's `Result`s — so blocking here stops us consuming the results that
    // would free the miner's credits, while the miner (symmetrically blocked
    // writing those results) stops reading jobs. Both peers park permanently.
    let (grants, mut grant_rx) = mpsc::channel::<u32>(GRANT_CHANNEL_DEPTH);
    state
        .lock()
        .await
        .register_wakeup(&miner_id, grants.clone());
    let _dispatcher = {
        let state = Arc::clone(&state);
        let tx = tx.clone();
        let miner_id = miner_id.clone();
        tokio::spawn(async move {
            while let Some(credits) = grant_rx.recv().await {
                if !dispatch_granted(&state, &miner_id, credits, &tx).await {
                    return;
                }
            }
        })
    };

    // 3. Message loop
    'session: loop {
        let Ok(Some(msg)) = inbound.message().await else {
            break;
        };
        match msg.msg {
            // Seed credits from Configure so the first job can dispatch before
            // the miner sends JobRequest (quip-mock-miner only requests after
            // completing a job). Routed through the dispatcher like every other
            // grant, so it never sends on the read path.
            Some(miner_msg::Msg::Ready(_)) => {
                tracing::debug!(miner = %miner_id, credits = seed_credits, "miner ready; seeding credits");
                if grants.send(seed_credits).await.is_err() {
                    break 'session;
                }
            }
            Some(miner_msg::Msg::JobRequest(req)) => {
                tracing::debug!(miner = %miner_id, credits = req.credits, "miner requested work");
                if grants.send(req.credits).await.is_err() {
                    break 'session;
                }
            }
            Some(miner_msg::Msg::Result(result)) => {
                let (job, salt, topo, best, gates, generation) = {
                    let mut st = state.lock().await;
                    st.metrics.record_result_received(&miner_id);
                    let job = st.complete_inflight(&result.job_id);
                    if job.is_some() {
                        st.router.record_completion(&miner_id);
                    } else {
                        // The job id is unknown or already completed: either the
                        // miner sent the same result twice, or the job was
                        // cancelled when the round turned over.
                        st.metrics.record_duplicate_result_drop(&miner_id);
                    }
                    let salt = st.take_salt(&result.job_id);
                    let topo = Arc::clone(&st.resolved_topo);
                    let best = st.current_best_milli;
                    let gates = crate::validate::gates_from_target(st.target.as_ref());
                    (job, salt, topo, best, gates, st.generation)
                };
                if let Some(job) = job {
                    if job.generation < generation {
                        st_metrics_stale(&state, &miner_id).await;
                    }
                    if let Some(ising) = job.ising.as_ref() {
                        let validated = validate_result(ising, &result.solutions, &gates, &topo);
                        let mut submitted = false;
                        // Set when an accepted proof fails to submit for a
                        // transient reason: keep it for the win-time retry loop
                        // instead of dropping a genuine winner.
                        let mut retain_for_retry = false;
                        if validated.accepted && beats_current(validated.best_energy_milli, best) {
                            let proof = Proof {
                                job_id: result.job_id.clone(),
                                best_energy_milli: validated.best_energy_milli,
                                diversity_milli: validated.diversity_milli,
                                n_valid: validated.n_valid,
                                // Diverse gate-passing subset, capped at the
                                // pallet's MAX_PROOF_SOLUTIONS (not all raw rows,
                                // which would fail bounded-vec decode).
                                solutions: validated.selected_solutions.clone(),
                                is_pow: job.provenance.as_ref().is_some_and(|p| p.is_pow),
                                order_id: job
                                    .provenance
                                    .as_ref()
                                    .map(|p| p.order_id.clone())
                                    .unwrap_or_default(),
                                generation: job.generation,
                                // Salt is chosen by the feeder when the PoW job
                                // is derived and remembered by job_id; the live
                                // RealChainClient submit requires 32 bytes.
                                salt: salt.map_or_else(Vec::new, |s| s.to_vec()),
                                device_access_time_us: result
                                    .meta
                                    .as_ref()
                                    .map_or(0, |m| m.device_access_time_us),
                            };
                            let job_hex = crate::chain::extrinsic::hex_encode(&result.job_id);
                            match chain.submit_proof(&proof).await {
                                Ok(SubmitAction::Success) => {
                                    let mut st = state.lock().await;
                                    st.current_best_milli = Some(validated.best_energy_milli);
                                    st.metrics.record_proof_submitted(&miner_id);
                                    if let Some(n) = submit_notify.lock().await.take() {
                                        let _ = n.send(());
                                    }
                                    submitted = true;
                                }
                                Ok(SubmitAction::Retry) => {
                                    tracing::warn!(job = %job_hex, "session submit rejected (retryable); retaining accepted candidate for win-time retry");
                                    state
                                        .lock()
                                        .await
                                        .metrics
                                        .record_submission_error(&miner_id);
                                    retain_for_retry = true;
                                }
                                Ok(SubmitAction::StopRoundStale) => {
                                    tracing::info!(job = %job_hex, "session submit stale for round; dropping candidate");
                                    state.lock().await.metrics.record_stale_drop(&miner_id);
                                }
                                Ok(SubmitAction::StopFatal) => {
                                    tracing::error!(job = %job_hex, "session submit fatally rejected by pallet; dropping candidate");
                                    state
                                        .lock()
                                        .await
                                        .metrics
                                        .record_submission_error(&miner_id);
                                }
                                Err(e) => {
                                    tracing::error!(job = %job_hex, error = %e, "session submit failed (transient); retaining accepted candidate for win-time retry");
                                    state
                                        .lock()
                                        .await
                                        .metrics
                                        .record_submission_error(&miner_id);
                                    retain_for_retry = true;
                                }
                            }
                        }
                        // Record the attempt; stash sub-threshold candidates so
                        // the easing difficulty can still win them; refresh the
                        // per-qblock summary when the stash or a submit changed.
                        {
                            let mut st = state.lock().await;
                            st.results_validated += 1;
                            let qblock_id = st.qblock_id;
                            let device_us =
                                result.meta.as_ref().map_or(0, |m| m.device_access_time_us);

                            // A solution below the current threshold is submitted
                            // immediately (above); one not yet viable is stashed
                            // if the projection says the decay will clear it. An
                            // accepted candidate whose submit failed transiently
                            // (`retain_for_retry`) is also stashed so the win-time
                            // loop resubmits it instead of losing a winner.
                            let mut stash_changed = false;
                            if !validated.accepted || retain_for_retry {
                                let is_pow = job.provenance.as_ref().is_some_and(|p| p.is_pow);
                                let order_id = job
                                    .provenance
                                    .as_ref()
                                    .map(|p| p.order_id.clone())
                                    .unwrap_or_default();
                                stash_changed = st.stash.insert(crate::stash::Candidate {
                                    job_id: result.job_id.clone(),
                                    salt,
                                    generation: job.generation,
                                    // Raw best (gate-agnostic) so the decay
                                    // projection can decide when the easing gate
                                    // admits it. The rows resubmitted with it are
                                    // `stash_solutions`: the prefix-safe subset
                                    // (≤ MAX_PROOF_SOLUTIONS) that keeps clearing
                                    // the chain's diversity gate however tight the
                                    // ceiling is when the proof lands.
                                    best_energy_milli: validated.raw_best_energy_milli,
                                    diversity_milli: validated.diversity_milli,
                                    n_valid: validated.n_valid,
                                    solutions: validated.stash_solutions.clone(),
                                    is_pow,
                                    order_id,
                                    device_access_time_us: device_us,
                                    submitted: false,
                                });
                                let decision = if stash_changed {
                                    "stashed"
                                } else {
                                    "discarded"
                                };
                                let stash_txt = match st.stash.summary().retained_band_milli() {
                                    None => "empty".to_owned(),
                                    Some((worst, best)) => format!(
                                        "{} -> {}",
                                        crate::logging::energy_units(worst),
                                        crate::logging::energy_units(best)
                                    ),
                                };
                                let target_txt = crate::logging::display_energy(
                                    st.target.as_ref().map(|t| t.max_energy_milli),
                                );
                                tracing::info!(
                                    "[quip-miner-{miner_id}] attempt {}: {decision} (stash: {stash_txt}, target <= {target_txt})",
                                    short_job_id(&result.job_id),
                                );
                            }

                            let attempt = crate::attempt::AttemptRecord::new(
                                qblock_id,
                                &miner_id,
                                &result.job_id,
                                &job,
                                &validated,
                                submitted,
                                device_us,
                            );
                            let summary = (stash_changed || submitted).then(|| {
                                crate::attempt::summary_body(
                                    qblock_id,
                                    st.current_best_milli,
                                    st.results_validated,
                                    st.stash.summary(),
                                )
                            });
                            if let Some(tx) = st.attempt_tx.as_ref() {
                                let _ = tx.send(crate::attempt::WriterMsg::Attempt(attempt));
                                if let Some(body) = summary {
                                    let _ = tx.send(crate::attempt::WriterMsg::Summary {
                                        qblock_id,
                                        body,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            Some(miner_msg::Msg::Reject(rej)) => {
                let mut st = state.lock().await;
                // The rejecting miner grants its own replacement credit (see the
                // miner's reject path), so the coordinator only re-routes the
                // job to a capable miner; an unknown job_id needs nothing.
                if let Some(job) = st.complete_inflight(&rej.job_id) {
                    st.router.record_completion(&miner_id);
                    st.router.on_reject(&miner_id, job, rej.reason);
                }
            }
            Some(miner_msg::Msg::Status(s)) => {
                let mut st = state.lock().await;
                if s.abandoned_generation > 0 {
                    st.last_abandoned_generation = s.abandoned_generation;
                }
                let prev = st
                    .miner_liveness
                    .get(&miner_id)
                    .cloned()
                    .unwrap_or_default();
                let (next, events) =
                    crate::liveness::evaluate_status(&prev, &s.sampler_stats, st.generation);
                let _ = st.miner_liveness.insert(miner_id.clone(), next);
                let qblock = st.qblock_id;
                drop(st);
                for ev in &events {
                    log_liveness(&miner_id, qblock, ev);
                }
            }
            // A miner that gives up says so here. Dropping this silently is how
            // a backend failure reads as an idle miner: the process stays
            // alive, the supervisor sees no exit, and the queue never drains.
            Some(miner_msg::Msg::Fatal(f)) => {
                tracing::error!(
                    miner = %miner_id,
                    exit_code = f.exit_code,
                    restart_required = f.restart_required,
                    reason = %f.reason,
                    "miner reported a fatal error"
                );
            }
            // The handshake already consumed one Hello. A second one means the
            // miner restarted its session without reconnecting.
            Some(miner_msg::Msg::Hello(_)) => {
                tracing::warn!(miner = %miner_id, "miner sent Hello mid-session; ignoring");
            }
            // An empty `msg` is a message this build cannot name: either a
            // field number the miner uses and this coordinator does not, or the
            // reverse. Version skew between the two arrives here and nowhere
            // else, so it must not be silent.
            None => {
                tracing::warn!(
                    miner = %miner_id,
                    "unrecognized message from miner; check the miner and coordinator protocol versions"
                );
            }
        }
    }

    state.lock().await.deregister_outbound(&miner_id);
}

/// Push a cancel for generations `<= max_generation` to a live session channel.
///
/// # Errors
///
/// Returns a send error if the session channel is closed.
pub async fn send_cancel(
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
    max_generation: u64,
) -> Result<(), mpsc::error::SendError<Result<CoordMsg, Status>>> {
    tx.send(Ok(coord(coord_msg::Msg::Cancel(quip_proto::v1::Cancel {
        max_generation,
    }))))
    .await
}

/// Result of a one-shot handshake harness run.
#[derive(Debug)]
pub struct SessionHarnessReport {
    /// Miner caps registered on a successful handshake.
    pub caps: Option<MinerCaps>,
    /// Miner process exit code (`-1` if unknown).
    pub miner_exit_code: i32,
    /// Whether the handshake completed and caps were registered.
    pub handshake_ok: bool,
}

/// Spawn `miner_bin` against a coordinator on `sock_path`, expecting `token`.
///
/// Used by integration tests. `sock_path` is a filesystem path (no `unix://`).
///
/// # Errors
///
/// Returns [`SessionHarnessError`] when the handshake fails or caps are missing.
pub async fn serve_one_session(
    miner_bin: &str,
    sock_path: &str,
    miner_id: &str,
    token: &str,
) -> Result<MinerCaps, SessionHarnessError> {
    let report = serve_one_session_expecting(miner_bin, sock_path, miner_id, token, token).await;
    if report.handshake_ok {
        report.caps.ok_or(SessionHarnessError {
            miner_exit_code: report.miner_exit_code,
            reason: "handshake ok but no caps".into(),
        })
    } else {
        Err(SessionHarnessError {
            miner_exit_code: report.miner_exit_code,
            reason: "handshake failed".into(),
        })
    }
}

/// Failure from a one-shot handshake harness run.
#[derive(Debug)]
pub struct SessionHarnessError {
    /// Miner process exit code (`-1` if unknown).
    pub miner_exit_code: i32,
    /// Human-readable failure reason.
    pub reason: String,
}

impl std::fmt::Display for SessionHarnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "session harness error (exit {}): {}",
            self.miner_exit_code, self.reason
        )
    }
}

impl std::error::Error for SessionHarnessError {}

/// Like [`serve_one_session`], but the coordinator expects `coord_token` while
/// the miner is given `miner_token` (for mismatch tests).
///
/// # Panics
///
/// Panics if the Unix socket cannot be bound or the miner process cannot be
/// spawned.
pub async fn serve_one_session_expecting(
    miner_bin: &str,
    sock_path: &str,
    miner_id: &str,
    miner_token: &str,
    coord_token: &str,
) -> SessionHarnessReport {
    let _ = std::fs::remove_file(sock_path);
    if let Some(parent) = Path::new(sock_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    #[expect(
        clippy::expect_used,
        reason = "test harness: bind failure is a setup bug"
    )]
    let uds = UnixListener::bind(sock_path).expect("bind unix socket");
    let incoming = UnixListenerStream::new(uds);

    let mut st = CoordinatorState::new();
    let _ = st
        .expected_tokens
        .insert(miner_id.into(), coord_token.into());
    let _ = st.configure.insert(
        miner_id.into(),
        Configure {
            queue_depth: 3,
            idle_timeout_s: 5,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: String::new(),
        },
    );
    let state = Arc::new(Mutex::new(st));
    // Chain is unused for handshake-only; use a no-op fake with empty snapshot.
    let chain = Arc::new(crate::chain::FakeChain::new(
        crate::chain::MiningSnapshot {
            last_proof_block_hash: [0u8; 32],
            topology_hash: vec![0u8; 32],
            nodes: vec![],
            edges: vec![],
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![0],
            allowed_spin_milli: vec![-1000, 1000],
            min_solutions: 0,
            max_energy_milli: 0,
            min_diversity_milli: 0,
            block_number: 0,
        },
        None,
    ));

    let svc = CoordinatorService {
        state: Arc::clone(&state),
        chain,
        submit_notify: Arc::new(Mutex::new(None)),
    };
    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(MinerServiceServer::new(svc))
            .serve_with_incoming(incoming)
            .await
    });

    let socket_uri = format!("unix://{sock_path}");
    #[expect(
        clippy::expect_used,
        reason = "test harness: spawn failure is a setup bug"
    )]
    let mut child = Command::new(miner_bin)
        .arg("--quip-coordinator")
        .arg(&socket_uri)
        .arg("--miner-id")
        .arg(miner_id)
        .env("QUIP_SESSION_TOKEN", miner_token)
        .spawn()
        .expect("spawn miner");

    // Give the miner a moment to handshake, then shut it down cleanly so the
    // idle path doesn't dominate: send nothing further; for a successful
    // handshake the miner waits on idle_timeout (5s). For a failed handshake
    // the stream is already dropped and the miner exits promptly.
    let exit_code = match tokio::time::timeout(Duration::from_secs(15), child.wait()).await {
        Ok(Ok(status)) => status.code().unwrap_or(-1),
        Ok(Err(_)) => -1,
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            -1
        }
    };

    // After Ready, send Shutdown so a successful session exits 0 quickly
    // rather than waiting the full idle timeout — only applies if still alive.
    // (If already exited, we're done.)
    let _ = exit_code;

    server.abort();
    let _ = std::fs::remove_file(sock_path);

    let st = state.lock().await;
    let caps = st.router.caps(miner_id).cloned();
    let handshake_ok = caps.is_some();
    SessionHarnessReport {
        caps,
        miner_exit_code: exit_code,
        handshake_ok,
    }
}

/// Inputs for the e2e `PoW` drive harness.
pub struct DrivePowParams<'a, C: ChainClient + 'static> {
    /// Path to the miner binary to spawn.
    pub miner_bin: &'a str,
    /// Unix socket filesystem path (no `unix://` prefix).
    pub sock_path: &'a str,
    /// Miner id string the binary will announce.
    pub miner_id: &'a str,
    /// Session token shared by coordinator and miner.
    pub token: &'a str,
    /// Launch config providing the miner's `Configure` message.
    pub entry: &'a LaunchEntry,
    /// Topology staged into coordinator state before connect.
    pub topology: Topology,
    /// First `PoW` job staged for the miner.
    pub job: quip_proto::v1::Job,
    /// Chain client used for proof submit.
    pub chain: Arc<C>,
    /// After first submit, optionally cancel this generation and stage a second job.
    pub cancel_then_job: Option<(u64, quip_proto::v1::Job)>,
}

/// Full end-to-end drive: handshake, feed one `PoW` job, validate `Result`, submit.
///
/// # Panics
///
/// Panics if the Unix socket cannot be bound or the miner process cannot be
/// spawned.
pub async fn drive_pow_round<C: ChainClient + 'static>(p: DrivePowParams<'_, C>) -> DriveReport {
    let sock_path = p.sock_path;
    let miner_id = p.miner_id;
    let _ = std::fs::remove_file(sock_path);
    if let Some(parent) = Path::new(sock_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    #[expect(
        clippy::expect_used,
        reason = "test harness: bind failure is a setup bug"
    )]
    let uds = UnixListener::bind(sock_path).expect("bind");
    let incoming = UnixListenerStream::new(uds);

    let mut st = CoordinatorState::new();
    let _ = st.expected_tokens.insert(miner_id.into(), p.token.into());
    let _ = st
        .configure
        .insert(miner_id.into(), p.entry.configure.clone());
    st.set_topology(Some(p.topology));
    // Stage the PoW job before the miner connects so JobRequest can pull it.
    st.router.register_miner(
        miner_id,
        MinerCaps {
            backend: "pending".into(),
            algorithm: "pending".into(),
            supported_kinds: vec![quip_proto::v1::JobKind::IsingSample as i32],
            max_nodes: 0,
            max_edges: 0,
        },
    );
    // Route will re-register on Hello and overwrite caps; stage via direct insert
    // after Hello by using a channel. Instead: put job in a pre-stage list.
    let pre_jobs = Arc::new(Mutex::new(vec![p.job]));
    let cancel_then = Arc::new(Mutex::new(p.cancel_then_job));

    let state = Arc::new(Mutex::new(st));
    let (submit_tx, submit_rx) = oneshot::channel();
    let submit_notify = Arc::new(Mutex::new(Some(submit_tx)));

    // Wrap service with a custom session that injects pre-staged jobs on Ready.
    let svc = DriveService {
        inner_state: Arc::clone(&state),
        chain: Arc::clone(&p.chain),
        submit_notify: Arc::clone(&submit_notify),
        pre_jobs: Arc::clone(&pre_jobs),
        cancel_then: Arc::clone(&cancel_then),
        miner_id: miner_id.to_string(),
    };

    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(MinerServiceServer::new(svc))
            .serve_with_incoming(incoming)
            .await
    });

    let socket_uri = format!("unix://{sock_path}");
    #[expect(
        clippy::expect_used,
        reason = "test harness: spawn failure is a setup bug"
    )]
    let mut child = Command::new(p.miner_bin)
        .arg("--quip-coordinator")
        .arg(&socket_uri)
        .arg("--miner-id")
        .arg(miner_id)
        .env("QUIP_SESSION_TOKEN", p.token)
        .spawn()
        .expect("spawn miner");

    // Wait for first successful submit (or timeout).
    let submitted = tokio::time::timeout(Duration::from_secs(20), submit_rx)
        .await
        .is_ok();

    // Allow cancel/second-job path a moment, then shut down.
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Miner may still be running; kill after grace (shutdown is in-band but we
    // don't hold the session tx here — kill is fine for the harness).
    let _ = child.kill().await;
    let exit = child.wait().await.ok().and_then(|s| s.code()).unwrap_or(-1);
    server.abort();
    let _ = std::fs::remove_file(sock_path);

    let st = state.lock().await;
    DriveReport {
        handshake_ok: st.router.caps(miner_id).is_some(),
        results_validated: st.results_validated,
        submitted,
        abandoned_generation: st.last_abandoned_generation,
        miner_exit_code: exit,
    }
}

/// Outcome of a [`drive_pow_round`] harness run.
#[derive(Debug)]
pub struct DriveReport {
    /// Whether the miner completed handshake and registered caps.
    pub handshake_ok: bool,
    /// Number of results that passed through validation.
    pub results_validated: u64,
    /// Whether at least one proof submit succeeded.
    pub submitted: bool,
    /// Highest abandoned generation reported via `Status`.
    pub abandoned_generation: u64,
    /// Miner process exit code (`-1` if unknown).
    pub miner_exit_code: i32,
}

/// Session service used by the e2e harness: injects pre-staged jobs on `Ready`.
struct DriveService<C: ChainClient + 'static> {
    /// Shared coordinator state.
    inner_state: Arc<Mutex<CoordinatorState>>,
    /// Chain client for proof submit.
    chain: Arc<C>,
    /// Notifies the harness when a proof is submitted.
    submit_notify: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    /// Jobs staged into the router after handshake.
    pre_jobs: Arc<Mutex<Vec<quip_proto::v1::Job>>>,
    /// Optional cancel generation + follow-up job after first submit.
    cancel_then: Arc<Mutex<Option<(u64, quip_proto::v1::Job)>>>,
    /// Miner id this harness session is bound to.
    miner_id: String,
}

#[tonic::async_trait]
impl<C: ChainClient + 'static> MinerService for DriveService<C> {
    type SessionStream = ReceiverStream<Result<CoordMsg, Status>>;

    #[expect(
        clippy::too_many_lines,
        reason = "drive harness session is one cohesive handshake + inject loop"
    )]
    async fn session(
        &self,
        request: Request<Streaming<MinerMsg>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let mut inbound = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<CoordMsg, Status>>(64);
        let state = Arc::clone(&self.inner_state);
        let chain = Arc::clone(&self.chain);
        let submit_notify = Arc::clone(&self.submit_notify);
        let pre_jobs = Arc::clone(&self.pre_jobs);
        let cancel_then = Arc::clone(&self.cancel_then);
        let miner_id = self.miner_id.clone();

        drop(tokio::spawn(async move {
            // Reuse handshake logic by running a tailored loop.
            let Ok(Some(MinerMsg {
                msg: Some(miner_msg::Msg::Hello(hello)),
            })) = inbound.message().await
            else {
                return;
            };
            {
                let st = state.lock().await;
                let expected = st
                    .expected_tokens
                    .get(&miner_id)
                    .cloned()
                    .unwrap_or_default();
                if hello.session_token != expected || hello.protocol_version != 1 {
                    return;
                }
            }
            {
                let mut st = state.lock().await;
                st.router.register_miner(
                    miner_id.clone(),
                    MinerCaps {
                        backend: hello.backend,
                        algorithm: hello.algorithm,
                        supported_kinds: hello.supported_kinds,
                        max_nodes: hello.max_nodes,
                        max_edges: hello.max_edges,
                    },
                );
                // Stage pre-jobs now that the miner is registered.
                let jobs = pre_jobs.lock().await.drain(..).collect::<Vec<_>>();
                for j in jobs {
                    let _ = st.router.route(j);
                }
            }

            let configure = {
                let st = state.lock().await;
                st.configure.get(&miner_id).cloned().unwrap_or(Configure {
                    queue_depth: 3,
                    idle_timeout_s: 300,
                    heartbeat_s: 15,
                    reconnect_window_s: 60,
                    backend_toml: String::new(),
                })
            };

            let seed_credits = configure.queue_depth.max(1);
            let _ = tx
                .send(Ok(coord(coord_msg::Msg::Welcome(Welcome {
                    protocol_version: 1,
                }))))
                .await;
            let _ = tx
                .send(Ok(coord(coord_msg::Msg::Configure(configure))))
                .await;
            {
                let st = state.lock().await;
                if let Some(topo) = st.topology.as_ref() {
                    let _ = tx
                        .send(Ok(coord(coord_msg::Msg::Topology(topo.to_proto()))))
                        .await;
                }
            }

            let mut cancel_sent = false;

            loop {
                let Ok(Some(msg)) = inbound.message().await else {
                    break;
                };
                match msg.msg {
                    Some(miner_msg::Msg::Ready(_)) => {
                        let mut st = state.lock().await;
                        st.router.grant_credits(&miner_id, seed_credits);
                        while let Some(job) = st.router.next_job(&miner_id) {
                            let _ = st.inflight.insert(job.job_id.clone(), job.clone());
                            if tx.send(Ok(coord(coord_msg::Msg::Job(job)))).await.is_err() {
                                return;
                            }
                        }
                    }
                    Some(miner_msg::Msg::JobRequest(req)) => {
                        let mut st = state.lock().await;
                        st.router.grant_credits(&miner_id, req.credits);
                        while let Some(job) = st.router.next_job(&miner_id) {
                            let _ = st.inflight.insert(job.job_id.clone(), job.clone());
                            if tx.send(Ok(coord(coord_msg::Msg::Job(job)))).await.is_err() {
                                return;
                            }
                        }
                    }
                    Some(miner_msg::Msg::Result(result)) => {
                        let (job, topo, best, gates) = {
                            let mut st = state.lock().await;
                            let job = st.inflight.remove(&result.job_id);
                            let topo = Arc::clone(&st.resolved_topo);
                            let gates = crate::validate::gates_from_target(st.target.as_ref());
                            (job, topo, st.current_best_milli, gates)
                        };
                        if let Some(job) = job {
                            if let Some(ising) = job.ising.as_ref() {
                                let validated =
                                    validate_result(ising, &result.solutions, &gates, &topo);
                                {
                                    let mut st = state.lock().await;
                                    st.results_validated += 1;
                                }
                                if validated.accepted
                                    && beats_current(validated.best_energy_milli, best)
                                {
                                    let proof = Proof {
                                        job_id: result.job_id.clone(),
                                        best_energy_milli: validated.best_energy_milli,
                                        diversity_milli: validated.diversity_milli,
                                        n_valid: validated.n_valid,
                                        // Capped diverse subset (pallet bound).
                                        solutions: validated.selected_solutions.clone(),
                                        is_pow: job.provenance.as_ref().is_some_and(|p| p.is_pow),
                                        order_id: job
                                            .provenance
                                            .as_ref()
                                            .map(|p| p.order_id.clone())
                                            .unwrap_or_default(),
                                        generation: job.generation,
                                        salt: vec![],
                                        device_access_time_us: result
                                            .meta
                                            .as_ref()
                                            .map_or(0, |m| m.device_access_time_us),
                                    };
                                    let submit_result = chain.submit_proof(&proof).await;
                                    if !matches!(submit_result, Ok(SubmitAction::Success)) {
                                        match &submit_result {
                                            Ok(_) => tracing::warn!(
                                                job = %crate::chain::extrinsic::hex_encode(&result.job_id),
                                                "submit not successful; best not advanced"
                                            ),
                                            Err(e) => tracing::error!(
                                                job = %crate::chain::extrinsic::hex_encode(&result.job_id),
                                                error = %e,
                                                "submit failed; best not advanced"
                                            ),
                                        }
                                    }
                                    if let Ok(SubmitAction::Success) = submit_result {
                                        {
                                            let mut st = state.lock().await;
                                            st.current_best_milli =
                                                Some(validated.best_energy_milli);
                                        }
                                        if let Some(n) = submit_notify.lock().await.take() {
                                            let _ = n.send(());
                                        }
                                        // Optional cancel + re-stage path.
                                        if !cancel_sent {
                                            if let Some((max_gen, next_job)) =
                                                cancel_then.lock().await.take()
                                            {
                                                cancel_sent = true;
                                                {
                                                    let mut st = state.lock().await;
                                                    let _ = st.router.cancel(max_gen);
                                                }
                                                let _ = tx
                                                    .send(Ok(coord(coord_msg::Msg::Cancel(
                                                        quip_proto::v1::Cancel {
                                                            max_generation: max_gen,
                                                        },
                                                    ))))
                                                    .await;
                                                // Stage next generation job.
                                                {
                                                    let mut st = state.lock().await;
                                                    let _ = st.router.route(next_job);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Some(miner_msg::Msg::Reject(rej)) => {
                        let mut st = state.lock().await;
                        // The rejecting miner grants its own replacement credit,
                        // so the coordinator only re-routes the job to a capable
                        // miner; an unknown job_id needs nothing.
                        if let Some(job) = st.inflight.remove(&rej.job_id) {
                            st.router.on_reject(&miner_id, job, rej.reason);
                        }
                    }
                    Some(miner_msg::Msg::Status(s)) => {
                        let mut st = state.lock().await;
                        if s.abandoned_generation > 0 {
                            st.last_abandoned_generation = s.abandoned_generation;
                        }
                    }
                    _ => {}
                }
            }
        }));

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

/// Issue a `Shutdown` on an outbound channel (for supervisor).
#[must_use]
pub fn shutdown_msg(grace_ms: u32) -> CoordMsg {
    coord(coord_msg::Msg::Shutdown(Shutdown { grace_ms }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::MinerCaps;
    use quip_proto::v1::{IsingProblem, Job, JobKind, Provenance};

    fn job(id: &[u8]) -> Job {
        Job {
            job_id: id.to_vec(),
            kind: JobKind::IsingSample as i32,
            generation: 1,
            deadline_ms: 0,
            ising: Some(IsingProblem {
                graph: None,
                h_milli_le32: vec![0; 8], // 2 nodes
                j_milli_le32: vec![0; 4], // 1 edge
                num_reads: 0,
                num_sweeps: 0,
                anneal_time_us: 0,
            }),
            provenance: Some(Provenance {
                is_pow: true,
                order_id: vec![],
            }),
        }
    }

    fn caps() -> MinerCaps {
        MinerCaps {
            backend: "mock".into(),
            algorithm: "sa".into(),
            supported_kinds: vec![JobKind::IsingSample as i32],
            max_nodes: 0, // unlimited
            max_edges: 0,
        }
    }

    #[test]
    fn reclaim_miner_returns_inflight_plus_staged_and_isolates_by_owner() {
        let mut st = CoordinatorState::new();
        st.router.register_miner("cpu-0", caps());
        st.router.register_miner("cpu-1", caps());

        // cpu-0 owns two in-flight jobs; cpu-1 owns one.
        st.dispatch_inflight("cpu-0", job(b"a"));
        st.dispatch_inflight("cpu-0", job(b"b"));
        st.dispatch_inflight("cpu-1", job(b"c"));
        // One job staged on cpu-0 (route picks the first capable miner, cpu-0).
        assert_eq!(st.router.route(job(b"d")).as_deref(), Some("cpu-0"));

        let mut reclaimed = st.reclaim_miner("cpu-0");
        reclaimed.sort_by(|x, y| x.job_id.cmp(&y.job_id));
        let ids: Vec<&[u8]> = reclaimed.iter().map(|j| j.job_id.as_slice()).collect();
        assert_eq!(ids, vec![&b"a"[..], &b"b"[..], &b"d"[..]]);

        // cpu-0's in-flight is cleared from both maps; cpu-1's is untouched.
        assert!(!st.inflight.contains_key(&b"a"[..]));
        assert!(!st.inflight_owner.contains_key(&b"a"[..]));
        assert!(st.inflight.contains_key(&b"c"[..]));
        assert_eq!(st.router.staged_len("cpu-0"), 0);
    }

    fn job_with_generation(id: &[u8], generation: u64) -> Job {
        let mut j = job(id);
        j.generation = generation;
        j.provenance = Some(Provenance {
            is_pow: generation != 0,
            order_id: vec![],
        });
        j
    }

    #[test]
    fn cancel_inflight_drops_pow_keeps_mempool_and_newer() {
        let mut st = CoordinatorState::new();
        st.dispatch_inflight("cpu-0", job_with_generation(b"mempool", 0));
        st.dispatch_inflight("cpu-0", job_with_generation(b"old", 3));
        st.dispatch_inflight("cpu-0", job_with_generation(b"live", 5));
        assert_eq!(st.cancel_inflight(4), 1);
        assert!(st.inflight.contains_key(&b"mempool"[..]));
        assert!(!st.inflight.contains_key(&b"old"[..]));
        assert!(st.inflight.contains_key(&b"live"[..]));
        assert!(st.complete_inflight(b"old").is_none());
    }

    #[test]
    fn outbound_register_and_deregister() {
        let mut st = CoordinatorState::new();
        let (tx, _rx) = mpsc::channel::<Result<CoordMsg, Status>>(1);
        st.register_outbound("cpu-0", tx);
        assert!(st.outbound.contains_key("cpu-0"));
        st.deregister_outbound("cpu-0");
        assert!(!st.outbound.contains_key("cpu-0"));
    }

    /// A miner that grants its credits before any job is staged must still
    /// receive work once the feeder stages it. The grant drains an empty queue,
    /// so only the feeder's wakeup can start the dispatch. Without that wakeup
    /// the miner waits for a job and the coordinator waits for a request, and
    /// neither side ever moves.
    #[tokio::test]
    async fn credits_granted_before_staging_still_dispatch() {
        let state = Arc::new(Mutex::new(CoordinatorState::new()));
        let (tx, mut rx) = mpsc::channel::<Result<CoordMsg, Status>>(8);
        let (grants, mut grant_rx) = mpsc::channel::<u32>(GRANT_CHANNEL_DEPTH);
        {
            let mut st = state.lock().await;
            st.router.register_miner("cpu-0", caps());
            st.register_wakeup("cpu-0", grants);
        }

        // The miner grants credits first. Nothing is staged, so nothing is sent.
        assert!(dispatch_granted(&state, "cpu-0", 32, &tx).await);
        assert!(
            rx.try_recv().is_err(),
            "no job exists yet, so none can dispatch"
        );

        // The feeder stages work and wakes the dispatcher.
        {
            let mut st = state.lock().await;
            assert!(st.router.stage_on("cpu-0", job(b"late")));
            st.wake_dispatcher("cpu-0");
        }

        // The wakeup carries no credits: the balance from the first grant stands.
        assert_eq!(grant_rx.try_recv().expect("wakeup was queued"), 0);
        assert!(dispatch_granted(&state, "cpu-0", 0, &tx).await);
        let sent = rx
            .try_recv()
            .expect("the staged job dispatches on the wakeup");
        let Ok(CoordMsg {
            msg: Some(coord_msg::Msg::Job(j)),
        }) = sent
        else {
            panic!("expected a Job message");
        };
        assert_eq!(j.job_id, b"late".to_vec());
    }

    /// C7 wiring check: `contexts_dispatched` moves at `dispatch_granted`'s real
    /// call site, not just through the metrics API directly.
    #[tokio::test]
    async fn dispatch_granted_counts_one_context_per_job_dispatched() {
        let state = Arc::new(Mutex::new(CoordinatorState::new()));
        let (tx, mut rx) = mpsc::channel::<Result<CoordMsg, Status>>(8);
        {
            let mut st = state.lock().await;
            st.router.register_miner("cpu-0", caps());
        }

        assert_eq!(state.lock().await.metrics.global().contexts_dispatched, 0);

        // Granting credits with nothing staged dispatches nothing: this catches
        // a bump placed outside the `while let` drain.
        assert!(dispatch_granted(&state, "cpu-0", 32, &tx).await);
        assert!(rx.try_recv().is_err(), "no job exists yet");
        assert_eq!(state.lock().await.metrics.global().contexts_dispatched, 0);

        // Stage one job and dispatch it.
        {
            let mut st = state.lock().await;
            assert!(st.router.stage_on("cpu-0", job(b"one")));
        }
        assert!(dispatch_granted(&state, "cpu-0", 0, &tx).await);
        assert!(rx.try_recv().is_ok(), "the staged job dispatches");
        assert_eq!(state.lock().await.metrics.global().contexts_dispatched, 1);
    }

    #[tokio::test]
    async fn waking_an_unregistered_miner_is_harmless() {
        let st = CoordinatorState::new();
        st.wake_dispatcher("nobody");
    }

    #[tokio::test]
    async fn ending_a_session_drops_its_wakeup() {
        let mut st = CoordinatorState::new();
        let (grants, _rx) = mpsc::channel::<u32>(1);
        st.register_wakeup("cpu-0", grants);
        assert!(st.wakeups.contains_key("cpu-0"));
        st.deregister_outbound("cpu-0");
        assert!(
            !st.wakeups.contains_key("cpu-0"),
            "a dead session must not be woken"
        );
    }
}
