//! Runtime lifecycle: `run_runtime` binds the server, supervises a real
//! `quip-mock-miner`, and shuts down cleanly on signal. Job production (feeding
//! work, solve→submit) is exercised by `tests/e2e.rs`; this covers the process
//! wiring + graceful-shutdown seam.

#![expect(
    clippy::expect_used,
    reason = "helper builds mock miner outside #[test]"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "fixture drain rates fit u32"
)]
#![expect(
    clippy::items_after_statements,
    reason = "test-local constants next to usage"
)]

use quip_coordinator::chain::PendingProof;
use quip_coordinator::chain::{FakeChain, MiningSnapshot};
use quip_coordinator::config::LaunchEntry;
use quip_coordinator::router::MinerCaps;
use quip_coordinator::runtime::{feeder_loop, run_runtime, FeederParams, RuntimeParams};
use quip_coordinator::session::CoordinatorState;
use quip_coordinator::supervisor::BackoffPolicy;
use quip_proto::v1::{Configure, JobKind};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, watch, Mutex};

/// Build + resolve the sibling `quip-mock-miner` binary (not the same package,
/// so no `CARGO_BIN_EXE_*`).
fn mock_miner() -> String {
    let status = std::process::Command::new(env!("CARGO"))
        .args(["build", "-p", "quip-mock-miner"])
        .status()
        .expect("build quip-mock-miner");
    assert!(status.success(), "failed to build quip-mock-miner");
    let name = if cfg!(windows) {
        "quip-mock-miner.exe"
    } else {
        "quip-mock-miner"
    };
    let mut p = std::env::current_exe().expect("test exe path");
    let _ = p.pop();
    let _ = p.pop();
    p.push(name);
    p.to_string_lossy().into_owned()
}

/// A no-work snapshot: the runtime under test stages no jobs, so the chain is
/// only held by the service and never queried.
fn trivial_snapshot() -> MiningSnapshot {
    MiningSnapshot {
        head_hash: [0u8; 32],
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
    }
}

fn cpu_entry(binary: String) -> LaunchEntry {
    LaunchEntry {
        miner_id: "cpu-0".into(),
        binary,
        backend: "cpu".into(),
        device: None,
        configure: Configure {
            // Long idle so the miner stays connected until we shut it down.
            queue_depth: 3,
            idle_timeout_s: 60,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: String::new(),
        },
    }
}

#[tokio::test]
async fn runtime_serves_supervises_and_shuts_down_clean() {
    let miner = mock_miner();
    let sock = format!("/tmp/quip-rt-{}.sock", std::process::id());
    let chain = Arc::new(FakeChain::new(trivial_snapshot(), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    let params = RuntimeParams {
        max_submit_attempts: 5,
        sock_path: sock,
        grace_ms: 500,
        backoff: BackoffPolicy::default(),
        // Lifecycle test only: buffer_depth 0 disables feeding.
        miner_identity: [0u8; 32],
        miner_account: [0u8; 32],
        buffer_depth: 0,
        poll_interval_ms: 200,
        dashboard: None,
        log_level: quip_coordinator::logging::LogLevel::Info,
        funding: quip_coordinator::funding::FundingParams::default(),
        descriptor: quip_coordinator::config::DescriptorParams::default(),
        descriptor_filed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        miner_registered: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        solver_registered: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        identity: quip_coordinator::metrics::Identity::default(),
    };
    let (trigger_tx, trigger_rx) = oneshot::channel::<()>();

    let state_for_run = Arc::clone(&state);
    let run = tokio::spawn(async move {
        run_runtime(
            vec![cpu_entry(miner)],
            chain,
            state_for_run,
            params,
            async move {
                let _ = trigger_rx.await;
            },
        )
        .await
    });

    // The supervisor spawns the mock-miner, which handshakes and registers its
    // outbound channel — that is the observable "miner is live" signal.
    let mut handshook = false;
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if state.lock().await.outbound.contains_key("cpu-0") {
            handshook = true;
            break;
        }
    }
    assert!(
        handshook,
        "mock-miner never handshook / registered an outbound"
    );

    // Graceful shutdown: run_runtime fans out Shutdown, drains supervisors, and
    // returns Ok promptly (well within the grace + kill margin).
    trigger_tx.send(()).expect("send shutdown trigger");
    let res = tokio::time::timeout(Duration::from_secs(6), run)
        .await
        .expect("run_runtime did not return after shutdown")
        .expect("run_runtime task panicked");
    res.expect("run_runtime returned an error");
}

/// A non-trivial snapshot (real nodes/edges) so derived `PoW` jobs are staged.
fn ising_snapshot() -> MiningSnapshot {
    let nodes = vec![0, 1, 2, 3];
    let edges = vec![(0, 1), (1, 2), (2, 3), (0, 3)];
    let h = vec![-1000, 0, 1000];
    let j = vec![-1000, 1000];
    let spin = vec![-1000, 1000];
    let topology_hash =
        quip_coordinator::topology::topology_hash_sets(&nodes, &edges, &h, &j, &spin).to_vec();
    MiningSnapshot {
        head_hash: [0u8; 32],
        last_proof_block_hash: [7u8; 32],
        topology_hash,
        nodes,
        edges,
        allowed_h_milli: h,
        allowed_j_milli: j,
        allowed_spin_milli: spin,
        min_solutions: 1,
        max_energy_milli: i64::MAX / 2,
        min_diversity_milli: 0,
        block_number: 42,
    }
}

fn ising_caps() -> MinerCaps {
    MinerCaps {
        backend: "mock".into(),
        algorithm: "sa".into(),
        supported_kinds: vec![JobKind::IsingSample as i32],
        max_nodes: 0,
        max_edges: 0,
    }
}

#[tokio::test]
async fn feeder_tops_up_to_buffer_depth_records_salts_and_sets_target() {
    let chain = Arc::new(FakeChain::new(ising_snapshot(), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    // Pre-register a miner, as if it had handshaked.
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        FeederParams {
            max_submit_attempts: 5,
            miner_identity: [0u8; 32],
            miner_account: [0u8; 32],
            buffer_depth: 4,
            poll_interval: Duration::from_millis(50),
            funding: quip_coordinator::funding::FundingParams::default(),
            descriptor: quip_coordinator::config::DescriptorParams::default(),
            descriptor_filed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            miner_registered: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            solver_registered: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            metrics: Arc::new(quip_coordinator::metrics::CoordinatorMetrics::new(&[])),
        },
        stop_rx,
    ));

    let mut filled = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        if state.lock().await.router.staged_len("cpu-0") == 4 {
            filled = true;
            break;
        }
    }
    assert!(filled, "feeder never filled the buffer to depth 4");

    {
        let st = state.lock().await;
        assert_eq!(st.router.staged_len("cpu-0"), 4);
        assert_eq!(st.salts.len(), 4, "one salt recorded per staged job");
        // Reseed set topology + difficulty target from the snapshot.
        assert!(st.topology.is_some());
        assert!(st.target.is_some());
    }

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// The signing account and the `PoW` identity are two different 32-byte values
/// derived from one key, and each has exactly one correct destination. Funding
/// the identity leaves the signer broke, so every submit fails with "Inability
/// to pay some fees"; deriving nonces from the account makes every proof fail
/// `InvalidNonce`. Neither failure names the mix-up, so pin both routes here.
#[tokio::test]
async fn feeder_funds_the_account_and_derives_jobs_from_the_identity() {
    use quip_protocol::derive::derive_nonce;

    const IDENTITY: [u8; 32] = [0x11; 32];
    const ACCOUNT: [u8; 32] = [0x22; 32];
    const HEAD: [u8; 32] = [0x33; 32];

    let chain = Arc::new(FakeChain::new(snapshot_with_head(HEAD), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        FeederParams {
            max_submit_attempts: 5,
            miner_identity: IDENTITY,
            miner_account: ACCOUNT,
            buffer_depth: 2,
            poll_interval: Duration::from_millis(30),
            funding: quip_coordinator::funding::FundingParams::default(),
            descriptor: quip_coordinator::config::DescriptorParams::default(),
            descriptor_filed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            miner_registered: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            solver_registered: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            metrics: Arc::new(quip_coordinator::metrics::CoordinatorMetrics::new(&[])),
        },
        stop_rx,
    ));

    let mut filled = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(30)).await;
        if state.lock().await.router.staged_len("cpu-0") >= 2 {
            filled = true;
            break;
        }
    }
    assert!(filled, "feeder never staged any jobs");

    let checked = chain.take_balance_accounts();
    assert!(!checked.is_empty(), "funding never read a balance");
    assert!(
        checked.iter().all(|a| *a == ACCOUNT),
        "funding must read the signing account, not the PoW identity: {checked:?}"
    );

    let st = state.lock().await;
    assert!(!st.salts.is_empty(), "no salts recorded for staged jobs");
    for (job_id, salt) in &st.salts {
        assert_eq!(
            job_id.as_slice(),
            derive_nonce(HEAD, IDENTITY, *salt).as_slice(),
            "job nonce must derive from the PoW identity"
        );
        assert_ne!(
            job_id.as_slice(),
            derive_nonce(HEAD, ACCOUNT, *salt).as_slice(),
            "job nonce must not derive from the signing account"
        );
    }
    drop(st);

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// Registration sits between funding and requirements. A coordinator that
/// cannot register must not mine: the pallet rejects every proof from an
/// unregistered account, and the account still pays for each attempt.
#[tokio::test]
async fn feeder_registers_once_and_holds_off_mining_until_it_succeeds() {
    let chain = Arc::new(FakeChain::new(ising_snapshot(), None));
    chain.set_registration_result(Err(quip_coordinator::chain::ChainError::Unavailable(
        "rpc down".into(),
    )));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(2, 30),
        stop_rx,
    ));

    // Registration keeps failing, so the walk never reaches staging.
    for _ in 0..10 {
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(
            state.lock().await.router.staged_len("cpu-0"),
            0,
            "an unregistered miner must not be fed work"
        );
    }

    chain.set_registration_result(Ok(quip_coordinator::chain::RegistrationOutcome::Registered));
    let mut filled = false;
    for _ in 0..60 {
        tokio::time::sleep(Duration::from_millis(30)).await;
        if state.lock().await.router.staged_len("cpu-0") >= 2 {
            filled = true;
            break;
        }
    }
    assert!(filled, "mining never started after registration succeeded");

    let submits = chain.registration_submits();
    tokio::time::sleep(Duration::from_millis(120)).await;
    assert_eq!(
        chain.registration_submits(),
        submits,
        "later rounds must not re-submit register_miner"
    );

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

#[tokio::test]
async fn feeder_grows_window_for_drainer_and_holds_floor_for_idle() {
    let chain = Arc::new(FakeChain::new(ising_snapshot(), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    {
        let mut st = state.lock().await;
        st.router.register_miner("cpu-fast", ising_caps());
        st.router.register_miner("cpu-idle", ising_caps());
    }

    // Floor of 2; the fast miner drains a fixed ~8/interval so its adaptive
    // window converges to ~2x that (headroom), well above the floor. The idle
    // miner never consumes, so it stays pinned at the floor.
    const FLOOR: usize = 2;
    const DRAIN_RATE: usize = 8;

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        FeederParams {
            max_submit_attempts: 5,
            miner_identity: [0u8; 32],
            miner_account: [0u8; 32],
            buffer_depth: FLOOR,
            poll_interval: Duration::from_millis(30),
            funding: quip_coordinator::funding::FundingParams::default(),
            descriptor: quip_coordinator::config::DescriptorParams::default(),
            descriptor_filed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            miner_registered: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            solver_registered: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            metrics: Arc::new(quip_coordinator::metrics::CoordinatorMetrics::new(&[])),
        },
        stop_rx,
    ));

    // Simulate a fixed-throughput miner: each interval, grant credits and pull
    // up to DRAIN_RATE staged jobs (min with what's available, so an unramped
    // buffer can't force runaway growth). Watch the window climb past the floor.
    let mut grew = false;
    for _ in 0..80 {
        tokio::time::sleep(Duration::from_millis(30)).await;
        {
            let mut st = state.lock().await;
            st.router.grant_credits("cpu-fast", DRAIN_RATE as u32);
            for _ in 0..DRAIN_RATE {
                if st.router.next_job("cpu-fast").is_none() {
                    break;
                }
            }
        }
        if state.lock().await.router.staged_len("cpu-fast") >= DRAIN_RATE {
            grew = true;
            break;
        }
    }
    assert!(
        grew,
        "adaptive window never grew to the drain rate under sustained consumption"
    );

    // The idle miner consumed nothing, so its window is still the floor.
    assert_eq!(
        state.lock().await.router.staged_len("cpu-idle"),
        FLOOR,
        "idle miner should stay pinned at the buffer_depth floor"
    );

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

#[tokio::test]
async fn feeder_broadcasts_set_target_once_per_difficulty() {
    use quip_proto::v1::{coord_msg, CoordMsg};

    let chain = Arc::new(FakeChain::new(ising_snapshot(), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    let (tx, mut rx) = mpsc::channel::<Result<CoordMsg, tonic::Status>>(16);
    state.lock().await.register_outbound("cpu-0", tx);

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        FeederParams {
            max_submit_attempts: 5,
            miner_identity: [0u8; 32],
            miner_account: [0u8; 32],
            buffer_depth: 4,
            poll_interval: Duration::from_millis(50),
            funding: quip_coordinator::funding::FundingParams::default(),
            descriptor: quip_coordinator::config::DescriptorParams::default(),
            descriptor_filed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            miner_registered: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            solver_registered: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            metrics: Arc::new(quip_coordinator::metrics::CoordinatorMetrics::new(&[])),
        },
        stop_rx,
    ));

    // The first reseed pushes the topology (first availability) to the live
    // miner, then the current difficulty. Cancel is skipped on the first reseed
    // (generation 0 has nothing to cancel).
    let topo_msg = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("no Topology within 2s")
        .expect("outbound channel closed")
        .expect("status error");
    assert!(
        matches!(&topo_msg.msg, Some(coord_msg::Msg::Topology(_))),
        "first outbound message should be Topology (reseed push), got {:?}",
        topo_msg.msg
    );
    let target_msg = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("no SetTarget within 2s")
        .expect("outbound channel closed")
        .expect("status error");
    let snap = ising_snapshot();
    assert!(
        matches!(&target_msg.msg, Some(coord_msg::Msg::SetTarget(_))),
        "second outbound message should be SetTarget"
    );
    if let Some(coord_msg::Msg::SetTarget(t)) = target_msg.msg {
        assert_eq!(t.max_energy_milli, snap.max_energy_milli);
        assert_eq!(t.min_solutions, snap.min_solutions);
        assert_eq!(t.min_diversity_milli, snap.min_diversity_milli);
    }

    // Unchanged head/difficulty across further polls must not re-broadcast
    // (no reseed → no Topology/SetTarget re-push).
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        matches!(rx.try_recv(), Err(mpsc::error::TryRecvError::Empty)),
        "unchanged difficulty/topology must not re-broadcast"
    );

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// An open order on the `ising_snapshot` graph with no energy gate, so any
/// sample the mock miner returns clears it.
fn mempool_order() -> quip_coordinator::chain::JobOrder {
    quip_coordinator::chain::JobOrder {
        order_id: 7u64.to_le_bytes().to_vec(),
        nodes: vec![0, 1, 2, 3],
        edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
        h_milli: vec![0; 4],
        j_milli: vec![-1000; 4],
        min_energy_milli: None,
        min_diversity_milli: None,
        min_solutions: Some(1),
        deadline_ms: 0,
    }
}

async fn wait_for(mut check: impl FnMut() -> bool) -> bool {
    for _ in 0..100 {
        if check() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    false
}

/// The feeder registers the account as a solver once, then stages an open
/// order exactly once as a generation-0 job, however many blocks it sees.
#[tokio::test]
async fn feeder_registers_as_solver_and_stages_each_open_order_once() {
    let chain = Arc::new(FakeChain::new(ising_snapshot(), Some(mempool_order())));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(2, 30),
        stop_rx,
    ));

    let mut staged = false;
    for _ in 0..100 {
        tokio::time::sleep(Duration::from_millis(30)).await;
        if !state.lock().await.mempool_orders.is_empty() {
            staged = true;
            break;
        }
    }
    assert!(staged, "the open order was never staged");

    // Later blocks re-read the order map; the order must not be staged again.
    let mut next = ising_snapshot();
    next.block_number = 43;
    chain.set_snapshot(Some(next));
    tokio::time::sleep(Duration::from_millis(200)).await;

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");

    assert_eq!(chain.solver_registration_submits(), 1);
    let jobs = state.lock().await.router.reclaim("cpu-0");
    let orders: Vec<_> = jobs.iter().filter(|j| j.generation == 0).collect();
    assert_eq!(orders.len(), 1, "the order must be staged exactly once");
    assert_eq!(
        orders.first().map(|j| j.job_id.clone()),
        Some(mempool_order().order_id)
    );
}

/// `PoW` does not wait on solver registration, and no order is staged while it
/// keeps failing. A failure is retried once per round, not once per poll.
#[tokio::test]
async fn feeder_mines_pow_but_stages_no_order_while_solver_registration_fails() {
    let chain = Arc::new(FakeChain::new(ising_snapshot(), Some(mempool_order())));
    chain.set_solver_registration_result(Err(quip_coordinator::chain::ChainError::Unavailable(
        "rpc down".into(),
    )));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(2, 30),
        stop_rx,
    ));

    let mut mining = false;
    for _ in 0..100 {
        tokio::time::sleep(Duration::from_millis(30)).await;
        if state.lock().await.router.staged_len("cpu-0") >= 2 {
            mining = true;
            break;
        }
    }
    assert!(mining, "PoW must mine while solver registration fails");
    tokio::time::sleep(Duration::from_millis(200)).await;

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");

    assert!(state.lock().await.mempool_orders.is_empty());
    assert_eq!(
        chain.solver_registration_submits(),
        1,
        "one attempt per round, not one per poll"
    );
}

/// Through the real session path: a mock miner samples a staged order, and
/// the coordinator answers it with `submit_solution`, never `submit_proof`.
#[tokio::test]
async fn runtime_answers_a_mempool_order_with_submit_solution() {
    let miner = mock_miner();
    let sock = format!("/tmp/quip-rt-mempool-{}.sock", std::process::id());
    let chain = Arc::new(FakeChain::new(ising_snapshot(), Some(mempool_order())));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    let params = RuntimeParams {
        max_submit_attempts: 5,
        sock_path: sock,
        grace_ms: 500,
        backoff: BackoffPolicy::default(),
        miner_identity: [0u8; 32],
        miner_account: [0u8; 32],
        buffer_depth: 1,
        poll_interval_ms: 50,
        dashboard: None,
        log_level: quip_coordinator::logging::LogLevel::Info,
        funding: quip_coordinator::funding::FundingParams::default(),
        descriptor: quip_coordinator::config::DescriptorParams::default(),
        descriptor_filed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        miner_registered: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        solver_registered: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        identity: quip_coordinator::metrics::Identity::default(),
    };
    let (trigger_tx, trigger_rx) = oneshot::channel::<()>();
    let run = tokio::spawn(run_runtime(
        vec![cpu_entry(miner)],
        Arc::clone(&chain),
        state,
        params,
        async move {
            let _ = trigger_rx.await;
        },
    ));

    let answered = wait_for(|| !chain.solutions.lock().expect("solutions lock").is_empty()).await;

    trigger_tx.send(()).expect("send shutdown trigger");
    tokio::time::timeout(Duration::from_secs(6), run)
        .await
        .expect("run_runtime did not return after shutdown")
        .expect("run_runtime task panicked")
        .expect("run_runtime returned an error");

    assert!(
        answered,
        "the order was never answered with submit_solution"
    );
    let solutions = chain.take_solutions();
    let order_id = mempool_order().order_id;
    let first = solutions.first().expect("one captured solution");
    assert_eq!(first.order_id, order_id);
    assert!(!first.is_pow);
    assert!(!first.solutions.is_empty());
    assert!(
        chain
            .take_submitted()
            .iter()
            .all(|p| p.order_id != order_id),
        "a mempool result must never reach submit_proof"
    );
}

fn feeder_params(buffer_depth: usize, poll_ms: u64) -> FeederParams {
    FeederParams {
        max_submit_attempts: 5,
        miner_identity: [0u8; 32],
        miner_account: [0u8; 32],
        buffer_depth,
        poll_interval: Duration::from_millis(poll_ms),
        funding: quip_coordinator::funding::FundingParams::default(),
        descriptor: quip_coordinator::config::DescriptorParams::default(),
        descriptor_filed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        miner_registered: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        solver_registered: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        metrics: Arc::new(quip_coordinator::metrics::CoordinatorMetrics::new(&[])),
    }
}

fn snapshot_with_head(head: [u8; 32]) -> MiningSnapshot {
    let mut snap = ising_snapshot();
    snap.last_proof_block_hash = head;
    snap
}

fn new_generation_staged(st: &CoordinatorState, generation: u64) -> usize {
    if st.generation != generation {
        return 0;
    }
    st.router.staged_len("cpu-0")
}

/// A new qblock is not a target refresh. The feeder must stop the dead
/// generation, push the new round's `Topology` and `SetTarget`, and only then
/// stage jobs of the new generation. Same topology hash and same difficulty
/// gates still require that push: the miner has to hear the new round.
#[tokio::test]
async fn feeder_sends_requirements_before_staging_the_new_generation() {
    use quip_proto::v1::{coord_msg, CoordMsg};

    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    let (tx, mut rx) = mpsc::channel::<Result<CoordMsg, tonic::Status>>(32);
    {
        let mut st = state.lock().await;
        st.router.register_miner("cpu-0", ising_caps());
        st.register_outbound("cpu-0", tx);
    }

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(2, 40),
        stop_rx,
    ));

    // First round: Topology then SetTarget. Staging is not checked between
    // those two recv calls: the feeder continues after each send, so a
    // staged-length read here races the rest of the poll.
    let first = recv_coord(&mut rx).await;
    assert!(
        matches!(&first.msg, Some(coord_msg::Msg::Topology(_))),
        "first message must be Topology, got {:?}",
        first.msg
    );
    let second = recv_coord(&mut rx).await;
    assert!(
        matches!(&second.msg, Some(coord_msg::Msg::SetTarget(_))),
        "second message must be SetTarget, got {:?}",
        second.msg
    );

    let mut filled = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        if state.lock().await.router.staged_len("cpu-0") >= 2 {
            filled = true;
            break;
        }
    }
    assert!(filled, "first round never staged jobs");

    // Drain leftover first-round broadcasts so the next recv is the reseed.
    while rx.try_recv().is_ok() {}

    // Same topology, same difficulty, new head: still a new round.
    chain.set_snapshot(Some(snapshot_with_head([2u8; 32])));

    let mut saw_cancel = false;
    let mut saw_topology = false;
    let mut saw_target = false;
    for _ in 0..8 {
        let msg = recv_coord(&mut rx).await;
        match &msg.msg {
            Some(coord_msg::Msg::Cancel(c)) => {
                assert!(
                    !saw_topology && !saw_target,
                    "Cancel must precede requirements"
                );
                assert_eq!(c.max_generation, 1);
                saw_cancel = true;
            }
            Some(coord_msg::Msg::Topology(_)) => {
                assert!(saw_cancel, "Topology must follow Cancel on a later reseed");
                saw_topology = true;
            }
            Some(coord_msg::Msg::SetTarget(_)) => {
                assert!(
                    saw_cancel && saw_topology,
                    "SetTarget must follow Cancel and Topology"
                );
                saw_target = true;
            }
            other => panic!("unexpected outbound during reseed: {other:?}"),
        }
        if saw_cancel && saw_topology && saw_target {
            break;
        }
    }
    assert!(
        saw_cancel && saw_topology && saw_target,
        "reseed must send Cancel, Topology, and SetTarget"
    );

    let mut restaged = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        if new_generation_staged(&*state.lock().await, 2) > 0 {
            restaged = true;
            break;
        }
    }
    assert!(
        restaged,
        "new generation was never staged after requirements"
    );

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// A pending proof that clears `snap`. `ising_snapshot` sets the ceiling to
/// `i64::MAX / 2` with one required row and no diversity floor, so any
/// well-formed row clears once the nonce matches the root.
fn clearing_pending_proof(snap: &MiningSnapshot, account: [u8; 32]) -> PendingProof {
    use quantum_validation::packed::pack_solution;
    use quantum_validation::{derive_nonce, AllowedValueSpec};
    use quip_coordinator::chain::extrinsic::account_identity_bytes;
    use quip_coordinator::chain::scale_types::QuantumProof;

    let salt = [3u8; 32];
    let nonce = derive_nonce(
        &snap.last_proof_block_hash,
        &account_identity_bytes(&account),
        &salt,
    );
    let spins = vec![1000i32; snap.nodes.len()];
    let packed = pack_solution(
        &spins,
        &AllowedValueSpec::Set(snap.allowed_spin_milli.as_slice()),
    )
    .expect("pack");
    PendingProof {
        extrinsic_hash: account,
        account,
        proof: QuantumProof {
            topology_hash: sp_core::H256::from_slice(&snap.topology_hash),
            nonce,
            salt,
            solutions: vec![packed],
            device_access_time_us: 0,
        },
    }
}

/// Drive a feeder to its first staged round and drain the round's broadcasts.
async fn first_round_staged(
    state: &Arc<Mutex<CoordinatorState>>,
    rx: &mut mpsc::Receiver<Result<quip_proto::v1::CoordMsg, tonic::Status>>,
) {
    let mut filled = false;
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        if new_generation_staged(&*state.lock().await, 1) > 0 {
            filled = true;
            break;
        }
    }
    assert!(filled, "first round never staged jobs");
    while rx.try_recv().is_ok() {}
}

/// Expect the reseed triple for `cancelled` and return once all three arrived.
#[expect(
    clippy::panic,
    reason = "test helper rejects missing or unexpected reseed messages"
)]
async fn expect_reseed(
    rx: &mut mpsc::Receiver<Result<quip_proto::v1::CoordMsg, tonic::Status>>,
    cancelled: u64,
) {
    use quip_proto::v1::coord_msg;
    let (mut saw_cancel, mut saw_topology, mut saw_target) = (false, false, false);
    for _ in 0..8 {
        let msg = recv_coord(rx).await;
        match &msg.msg {
            Some(coord_msg::Msg::Cancel(c)) => {
                assert_eq!(c.max_generation, cancelled);
                saw_cancel = true;
            }
            Some(coord_msg::Msg::Topology(_)) => saw_topology = true,
            Some(coord_msg::Msg::SetTarget(_)) => saw_target = true,
            other => panic!("unexpected outbound during reseed: {other:?}"),
        }
        if saw_cancel && saw_topology && saw_target {
            return;
        }
    }
    panic!("reseed did not send Cancel({cancelled}), Topology, and SetTarget");
}

/// A proof in the pool that clears the round stops the miners: the current
/// generation is cancelled and nothing is staged until the block that
/// includes the proof arrives. That block's hash is the next root.
#[tokio::test]
async fn feeder_stops_on_a_clearing_pending_proof_and_mines_the_new_root() {
    use quip_coordinator::chain::extrinsic::hex_encode;
    use quip_proto::v1::coord_msg;

    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    let (tx, mut rx) = mpsc::channel::<Result<quip_proto::v1::CoordMsg, tonic::Status>>(64);
    {
        let mut st = state.lock().await;
        st.router.register_miner("cpu-0", ising_caps());
        st.register_outbound("cpu-0", tx);
    }
    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(2, 40),
        stop_rx,
    ));
    first_round_staged(&state, &mut rx).await;

    chain.set_pending_proofs(vec![clearing_pending_proof(
        &snapshot_with_head([1u8; 32]),
        [5u8; 32],
    )]);

    let msg = recv_coord(&mut rx).await;
    match &msg.msg {
        Some(coord_msg::Msg::Cancel(c)) => assert_eq!(c.max_generation, 1),
        other => panic!("expected Cancel(1) on a pending win, got {other:?}"),
    }
    {
        let st = state.lock().await;
        assert_eq!(st.generation, 2);
        assert_eq!(st.router.staged_len("cpu-0"), 0, "nothing stays staged");
    }
    // Waiting: no requirements, no staging, while the root is unchanged.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        rx.try_recv().is_err(),
        "no broadcast while awaiting the qblock"
    );
    assert_eq!(state.lock().await.router.staged_len("cpu-0"), 0);

    // The block that includes the proof arrives: its hash is the new root.
    chain.set_pending_proofs(Vec::new());
    chain.set_snapshot(Some(snapshot_with_head([2u8; 32])));
    expect_reseed(&mut rx, 2).await;
    assert_eq!(
        state.lock().await.last_proof_block_hash,
        hex_encode(&[2u8; 32])
    );

    let mut restaged = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        if new_generation_staged(&*state.lock().await, 3) > 0 {
            restaged = true;
            break;
        }
    }
    assert!(restaged, "new root was never staged");

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// The pending proof leaves the pool and no qblock follows: the round
/// resumes on the same root with a fresh generation.
#[tokio::test]
async fn feeder_resumes_when_the_pending_proof_leaves_the_pool_without_a_qblock() {
    use quip_proto::v1::coord_msg;

    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    let (tx, mut rx) = mpsc::channel::<Result<quip_proto::v1::CoordMsg, tonic::Status>>(64);
    {
        let mut st = state.lock().await;
        st.router.register_miner("cpu-0", ising_caps());
        st.register_outbound("cpu-0", tx);
    }
    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(2, 40),
        stop_rx,
    ));
    first_round_staged(&state, &mut rx).await;

    chain.set_pending_proofs(vec![clearing_pending_proof(
        &snapshot_with_head([1u8; 32]),
        [5u8; 32],
    )]);
    let msg = recv_coord(&mut rx).await;
    assert!(
        matches!(&msg.msg, Some(coord_msg::Msg::Cancel(c)) if c.max_generation == 1),
        "expected Cancel(1), got {:?}",
        msg.msg
    );

    chain.set_pending_proofs(Vec::new());
    expect_reseed(&mut rx, 2).await;

    let mut restaged = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        if new_generation_staged(&*state.lock().await, 3) > 0 {
            restaged = true;
            break;
        }
    }
    assert!(restaged, "the round never resumed on the same root");
    assert_eq!(
        state.lock().await.last_proof_block_hash,
        quip_coordinator::chain::extrinsic::hex_encode(&[1u8; 32]),
        "resume keeps the root"
    );

    let mut resubmitted = clearing_pending_proof(&snapshot_with_head([1u8; 32]), [5u8; 32]);
    resubmitted.extrinsic_hash = [9u8; 32];
    chain.set_pending_proofs(vec![resubmitted]);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        rx.try_recv().is_err(),
        "a resubmitted proof must not stop the resumed round"
    );

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// The proof stays pending but two blocks pass with no qblock: resume.
#[tokio::test]
async fn feeder_resumes_after_two_blocks_when_the_pending_proof_never_lands() {
    use quip_proto::v1::coord_msg;

    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    let (tx, mut rx) = mpsc::channel::<Result<quip_proto::v1::CoordMsg, tonic::Status>>(64);
    {
        let mut st = state.lock().await;
        st.router.register_miner("cpu-0", ising_caps());
        st.register_outbound("cpu-0", tx);
    }
    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(2, 40),
        stop_rx,
    ));
    first_round_staged(&state, &mut rx).await;

    chain.set_pending_proofs(vec![clearing_pending_proof(
        &snapshot_with_head([1u8; 32]),
        [5u8; 32],
    )]);
    let msg = recv_coord(&mut rx).await;
    assert!(
        matches!(&msg.msg, Some(coord_msg::Msg::Cancel(c)) if c.max_generation == 1),
        "expected Cancel(1), got {:?}",
        msg.msg
    );

    // Same root, two blocks later, proof still pending.
    let mut later = snapshot_with_head([1u8; 32]);
    later.block_number += 2;
    chain.set_snapshot(Some(later));
    expect_reseed(&mut rx, 2).await;

    let mut restaged = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        if new_generation_staged(&*state.lock().await, 3) > 0 {
            restaged = true;
            break;
        }
    }
    assert!(restaged, "the round never resumed on the same root");
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        rx.try_recv().is_err(),
        "the timed-out proof must not stop the resumed round"
    );

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// A pending proof derived from another root is not this round's win.
#[tokio::test]
async fn feeder_ignores_a_pending_proof_for_another_round() {
    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    let (tx, mut rx) = mpsc::channel::<Result<quip_proto::v1::CoordMsg, tonic::Status>>(64);
    {
        let mut st = state.lock().await;
        st.router.register_miner("cpu-0", ising_caps());
        st.register_outbound("cpu-0", tx);
    }
    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(2, 40),
        stop_rx,
    ));
    first_round_staged(&state, &mut rx).await;

    chain.set_pending_proofs(vec![clearing_pending_proof(
        &snapshot_with_head([9u8; 32]),
        [5u8; 32],
    )]);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        rx.try_recv().is_err(),
        "no Cancel for another round's proof"
    );
    let st = state.lock().await;
    assert_eq!(st.generation, 1);
    assert!(st.router.staged_len("cpu-0") > 0);
    drop(st);

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// Mid-run funding failure must not exit. It holds off staging the new
/// generation and retries. Startup still exits 64; this path must not.
#[tokio::test]
async fn feeder_holds_off_new_round_when_account_is_underfunded() {
    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(2, 40),
        stop_rx,
    ));

    let mut filled = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        if state.lock().await.router.staged_len("cpu-0") >= 2 {
            filled = true;
            break;
        }
    }
    assert!(filled, "first round never staged jobs");
    let first_round_gen = state.lock().await.generation;

    // Next round cannot pay fees. No faucet is configured on the default
    // FundingParams, so ensure_funded fails immediately.
    chain.set_balance(Ok(0));
    chain.set_snapshot(Some(snapshot_with_head([2u8; 32])));

    tokio::time::sleep(Duration::from_millis(250)).await;
    {
        let st = state.lock().await;
        assert_eq!(
            new_generation_staged(&st, first_round_gen.saturating_add(1)),
            0,
            "must not stage the new generation while the account is underfunded"
        );
        assert_eq!(
            st.router.staged_len("cpu-0"),
            0,
            "prior-generation staged jobs must have been cancelled"
        );
    }

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// A job already dispatched under the dead generation is dropped from
/// in-flight on reseed. A late `Result` cannot be scored or submitted, and the
/// job is not re-queued into the new round.
#[tokio::test]
async fn feeder_drops_dead_generation_inflight_and_does_not_submit_it() {
    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(2, 40),
        stop_rx,
    ));

    let mut job_id = None;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        let mut st = state.lock().await;
        if st.router.staged_len("cpu-0") == 0 {
            continue;
        }
        st.router.grant_credits("cpu-0", 1);
        if let Some(job) = st.router.next_job("cpu-0") {
            job_id = Some(job.job_id.clone());
            st.dispatch_inflight("cpu-0", job);
            break;
        }
    }
    let job_id = job_id.expect("never dispatched a first-round job");
    assert!(state.lock().await.inflight.contains_key(&job_id));

    chain.set_snapshot(Some(snapshot_with_head([2u8; 32])));
    let mut dropped = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        if !state.lock().await.inflight.contains_key(&job_id) {
            dropped = true;
            break;
        }
    }
    assert!(dropped, "dead-generation in-flight job was not dropped");
    assert_eq!(
        chain.submitted_count(),
        0,
        "cancelled job must not be submitted"
    );
    {
        let mut st = state.lock().await;
        assert!(
            st.complete_inflight(&job_id).is_none(),
            "complete_inflight on a cancelled id must be a no-op"
        );
    }

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// The readiness walk re-reads sync state and balance on every round, not
/// only at process start.
#[tokio::test]
async fn feeder_re_runs_sync_and_funding_on_every_round() {
    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(1, 40),
        stop_rx,
    ));

    let mut first = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        if state.lock().await.generation >= 1 && state.lock().await.router.staged_len("cpu-0") >= 1
        {
            first = true;
            break;
        }
    }
    assert!(first, "first round never became ready");
    let sync_after_first = chain.sync_calls();
    let balance_after_first = chain.balance_calls();
    assert!(
        sync_after_first >= 1,
        "first round must read sync status, got {sync_after_first}"
    );
    assert!(
        balance_after_first >= 1,
        "first round must read balance, got {balance_after_first}"
    );

    chain.set_snapshot(Some(snapshot_with_head([2u8; 32])));
    let mut second = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        if state.lock().await.generation >= 2 {
            second = true;
            break;
        }
    }
    assert!(second, "second round never started");
    tokio::time::sleep(Duration::from_millis(80)).await;
    assert!(
        chain.sync_calls() > sync_after_first,
        "second round must re-read sync status ({} then {})",
        sync_after_first,
        chain.sync_calls()
    );
    assert!(
        chain.balance_calls() > balance_after_first,
        "second round must re-read balance ({} then {})",
        balance_after_first,
        chain.balance_calls()
    );

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

async fn recv_coord(
    rx: &mut mpsc::Receiver<Result<quip_proto::v1::CoordMsg, tonic::Status>>,
) -> quip_proto::v1::CoordMsg {
    tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("no outbound")
        .expect("closed")
        .expect("status")
}

async fn wait_generation(state: &Arc<Mutex<CoordinatorState>>, generation: u64) -> bool {
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        if state.lock().await.generation >= generation
            && state.lock().await.router.staged_len("cpu-0") >= 1
        {
            return true;
        }
    }
    false
}

/// Play one Result for `generation` the way the session path does: dispatch a
/// staged job to cpu-0 and complete it as this round's work. False when no job
/// of that generation is staged in time.
async fn return_one_result(state: &Arc<Mutex<CoordinatorState>>, generation: u64) -> bool {
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        let mut st = state.lock().await;
        if st.generation != generation || st.router.staged_len("cpu-0") == 0 {
            continue;
        }
        st.router.grant_credits("cpu-0", 1);
        let Some(job) = st.router.next_job("cpu-0") else {
            continue;
        };
        let job_id = job.job_id.clone();
        let job_generation = job.generation;
        st.dispatch_inflight("cpu-0", job);
        let _ = st.complete_inflight(&job_id);
        st.note_result(job_generation);
        return true;
    }
    false
}

/// Wait until the fake chain has seen `n` participate calls.
async fn wait_participations(chain: &FakeChain, n: usize) -> bool {
    for _ in 0..40 {
        if chain.participation_calls() >= n {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(40)).await;
    }
    false
}

/// Staging work is not participating. The declaration waits for a miner to
/// return a Result for the round: a QPU that sits a round out by withholding
/// credits (or any miner that is down) must not be recorded as a participant.
#[tokio::test]
async fn feeder_declares_participation_only_after_a_result_for_the_round() {
    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    chain.set_qblock_id(Some(10));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(1, 40),
        stop_rx,
    ));

    assert!(wait_generation(&state, 1).await, "first round never staged");
    // Several polls with work staged and nothing returned.
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        chain.participation_calls(),
        0,
        "staged work alone must not declare participation"
    );

    assert!(return_one_result(&state, 1).await, "no job to complete");
    assert!(
        wait_participations(&chain, 1).await,
        "a Result for the round must declare participation"
    );
    assert_eq!(chain.take_participations(), vec![11]);

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// A pending win must not erase participation owed for the stopped round.
#[tokio::test]
async fn feeder_declares_participation_for_a_round_stopped_by_a_pending_win() {
    use quip_proto::v1::coord_msg;

    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    chain.set_qblock_id(Some(10));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    let (tx, mut rx) = mpsc::channel::<Result<quip_proto::v1::CoordMsg, tonic::Status>>(64);
    {
        let mut st = state.lock().await;
        st.router.register_miner("cpu-0", ising_caps());
        st.register_outbound("cpu-0", tx);
    }
    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(2, 40),
        stop_rx,
    ));
    first_round_staged(&state, &mut rx).await;

    // No await between the returned Result and setting the proof: on this
    // current-thread runtime, the feeder's next poll sees both together.
    assert!(return_one_result(&state, 1).await, "no job to complete");
    chain.set_pending_proofs(vec![clearing_pending_proof(
        &snapshot_with_head([1u8; 32]),
        [5u8; 32],
    )]);

    let msg = recv_coord(&mut rx).await;
    match &msg.msg {
        Some(coord_msg::Msg::Cancel(c)) => assert_eq!(c.max_generation, 1),
        other => panic!("expected Cancel(1) on a pending win, got {other:?}"),
    }
    assert!(
        wait_participations(&chain, 1).await,
        "a round mined before the stop must be declared"
    );
    assert_eq!(chain.take_participations(), vec![11]);

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// Two reseeds of the same minted qblock send one participate call.
#[tokio::test]
async fn feeder_declares_once_for_the_same_qblock() {
    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    chain.set_qblock_id(Some(10));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(1, 40),
        stop_rx,
    ));

    assert!(
        return_one_result(&state, 1).await,
        "first round never mined"
    );
    assert!(
        wait_participations(&chain, 1).await,
        "first round not declared"
    );
    chain.set_snapshot(Some(snapshot_with_head([2u8; 32])));
    assert!(
        return_one_result(&state, 2).await,
        "second round never mined"
    );
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        chain.participation_calls(),
        1,
        "same candidate must not be declared twice"
    );
    assert_eq!(chain.take_participations(), vec![11]);

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// A new minted qblock sends a second participate call.
#[tokio::test]
async fn feeder_declares_again_on_a_new_qblock() {
    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    chain.set_qblock_id(Some(10));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(1, 40),
        stop_rx,
    ));

    assert!(
        return_one_result(&state, 1).await,
        "first round never mined"
    );
    assert!(
        wait_participations(&chain, 1).await,
        "first round not declared"
    );
    chain.set_qblock_id(Some(11));
    chain.set_snapshot(Some(snapshot_with_head([2u8; 32])));
    assert!(
        return_one_result(&state, 2).await,
        "second round never mined"
    );
    assert!(
        wait_participations(&chain, 2).await,
        "second round not declared"
    );
    assert_eq!(chain.take_participations(), vec![11, 12]);

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// Pallet participation errors must not hold off mining.
#[tokio::test]
async fn feeder_keeps_mining_when_participation_pallet_errors() {
    use quip_coordinator::chain::ParticipationOutcome;

    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    chain.set_qblock_id(Some(4));
    chain.set_participation_result(Ok(ParticipationOutcome::AlreadyDeclared));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(1, 40),
        stop_rx,
    ));

    assert!(
        return_one_result(&state, 1).await,
        "first round never mined"
    );
    assert!(
        wait_participations(&chain, 1).await,
        "DuplicateParticipation must not block mining"
    );

    chain.set_qblock_id(Some(5));
    chain.set_participation_result(Ok(ParticipationOutcome::StaleQBlock));
    chain.set_snapshot(Some(snapshot_with_head([2u8; 32])));
    assert!(
        return_one_result(&state, 2).await,
        "second round never mined"
    );
    assert!(
        wait_participations(&chain, 2).await,
        "InvalidQBlockId must not block mining"
    );

    chain.set_qblock_id(Some(6));
    chain.set_participation_result(Ok(ParticipationOutcome::DescriptorMissing));
    chain.set_snapshot(Some(snapshot_with_head([3u8; 32])));
    assert!(
        return_one_result(&state, 3).await,
        "third round never mined"
    );
    assert!(
        wait_participations(&chain, 3).await,
        "DescriptorRequired must not block mining"
    );
    assert_eq!(chain.participation_calls(), 3);

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

fn named_feeder_params(buffer_depth: usize, poll_ms: u64) -> FeederParams {
    let mut params = feeder_params(buffer_depth, poll_ms);
    params.descriptor = quip_coordinator::config::DescriptorParams {
        node_name: Some("Tesla".into()),
        public_host: Some("96.233.112.201".into()),
        rpc_endpoints: vec!["ws://127.0.0.1:9944".into()],
        miners: vec![
            quip_coordinator::chain::MinerSpecScale {
                kind: quip_coordinator::chain::MinerKind::Cpu,
                label: Some(b"cpu-0".to_vec()),
                backend: Some(b"cpu".to_vec()),
                device_id: None,
            },
            quip_coordinator::chain::MinerSpecScale {
                kind: quip_coordinator::chain::MinerKind::Metal,
                label: Some(b"metal-0".to_vec()),
                backend: Some(b"metal".to_vec()),
                device_id: None,
            },
        ],
        ..quip_coordinator::config::DescriptorParams::default()
    };
    params
}

/// Two rounds in one process file the descriptor once.
#[tokio::test]
async fn feeder_files_descriptor_once_across_two_rounds() {
    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    chain.set_qblock_id(Some(10));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        named_feeder_params(1, 40),
        stop_rx,
    ));

    assert!(wait_generation(&state, 1).await, "first round never staged");
    chain.set_snapshot(Some(snapshot_with_head([2u8; 32])));
    assert!(
        wait_generation(&state, 2).await,
        "second round never staged"
    );
    assert_eq!(chain.descriptor_calls(), 1);
    let filed = chain.take_descriptors();
    let desc = filed.first().expect("one descriptor");
    assert_eq!(desc.node_name, b"Tesla");
    assert_eq!(
        desc.miners.iter().map(|m| m.kind).collect::<Vec<_>>(),
        vec![
            quip_coordinator::chain::MinerKind::Cpu,
            quip_coordinator::chain::MinerKind::Metal
        ]
    );

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// Missing `[miner].node_name` files nothing and still starts mining.
#[tokio::test]
async fn feeder_reaches_mining_when_node_name_is_missing() {
    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    chain.set_qblock_id(Some(10));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(1, 40),
        stop_rx,
    ));

    assert!(
        wait_generation(&state, 1).await,
        "missing node_name must not block mining"
    );
    assert_eq!(chain.descriptor_calls(), 0);
    assert!(
        return_one_result(&state, 1).await,
        "first round never mined"
    );
    assert!(
        wait_participations(&chain, 1).await,
        "a skipped descriptor must not stop participation"
    );

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// Transient descriptor or participation errors must not hold off mining.
#[tokio::test]
async fn feeder_keeps_mining_when_descriptor_or_participation_is_transient() {
    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    chain.set_qblock_id(Some(10));
    chain.set_descriptor_result(Err(quip_coordinator::chain::ChainError::Unavailable(
        "rpc down".into(),
    )));
    chain.set_participation_result(Err(quip_coordinator::chain::ChainError::Unavailable(
        "rpc down".into(),
    )));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        named_feeder_params(1, 40),
        stop_rx,
    ));

    assert!(
        wait_generation(&state, 1).await,
        "transient errors must not block mining"
    );
    assert_eq!(chain.descriptor_calls(), 3);
    // Participation retries on later polls while the error persists, and
    // mining carries on around it.
    assert!(
        return_one_result(&state, 1).await,
        "first round never mined"
    );
    assert!(
        wait_participations(&chain, 2).await,
        "a transient participation error must be retried"
    );
    assert!(
        state.lock().await.router.staged_len("cpu-0") >= 1,
        "participation retries must not stop staging"
    );

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// Pallet rejection of the descriptor must not hold off mining.
#[tokio::test]
async fn feeder_keeps_mining_when_descriptor_is_rejected() {
    let chain = Arc::new(FakeChain::new(snapshot_with_head([1u8; 32]), None));
    chain.set_qblock_id(Some(10));
    chain.set_descriptor_result(Ok(quip_coordinator::chain::DescriptorOutcome::Rejected));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    state
        .lock()
        .await
        .router
        .register_miner("cpu-0", ising_caps());

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        named_feeder_params(1, 40),
        stop_rx,
    ));

    assert!(
        wait_generation(&state, 1).await,
        "descriptor rejection must not block mining"
    );
    assert_eq!(chain.descriptor_calls(), 1);
    assert!(
        return_one_result(&state, 1).await,
        "first round never mined"
    );
    assert!(
        wait_participations(&chain, 1).await,
        "a rejected descriptor must not stop participation"
    );

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}

/// Difficulty gates that demand three mutually distant solutions, with the
/// energy ceiling the caller wants live on the chain.
fn gated_snapshot(max_energy_milli: i64) -> MiningSnapshot {
    let mut snap = snapshot_with_head([9u8; 32]);
    snap.min_solutions = 3;
    snap.min_diversity_milli = 300;
    snap.max_energy_milli = max_energy_milli;
    snap
}

/// Wire spin bytes: `+1 -> 0x01`, `-1 -> 0xFF`.
fn spin_bytes(spins: &[i8]) -> Vec<u8> {
    spins
        .iter()
        .map(|&s| if s > 0 { 0x01 } else { 0xFF })
        .collect()
}

/// Three mutually distant rows, each pair at symmetric Hamming distance 2 of 4
/// spins, so any two score 500 milli and all three score 500 milli.
fn distant_rows() -> Vec<quip_proto::v1::Solution> {
    [
        (vec![1, 1, 1, 1], -3000),
        (vec![1, 1, -1, -1], -2000),
        (vec![1, -1, 1, -1], -1000),
    ]
    .into_iter()
    .map(|(spins, energy_milli)| quip_proto::v1::Solution {
        spins_bytes: spin_bytes(&spins),
        energy_milli,
    })
    .collect()
}

/// A stashed candidate whose viability projection has already arrived, so the
/// only thing standing between it and a submission is the live difficulty.
fn due_candidate() -> quip_coordinator::stash::Candidate {
    quip_coordinator::stash::Candidate {
        job_id: vec![0xaa],
        salt: Some([1u8; 32]),
        generation: 1,
        best_energy_milli: -3000,
        diversity_milli: 500,
        n_valid: 3,
        solutions: distant_rows(),
        is_pow: true,
        order_id: vec![],
        device_access_time_us: 0,
        submitted: false,
    }
}

/// The decay projection says a stashed candidate is due, but the chain gates it
/// would meet at the inclusion block are the ones live *now*, not the ones it
/// was stashed under. While the ceiling admits only two of its three rows the
/// chain would reject it for want of solutions, so the feeder has to hold it;
/// once the ceiling eases to admit all three, it submits.
#[tokio::test]
async fn feeder_holds_a_stashed_candidate_until_the_live_difficulty_admits_it() {
    let chain = Arc::new(FakeChain::new(gated_snapshot(-1500), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));

    let (stop_tx, stop_rx) = watch::channel(false);
    let feeder = tokio::spawn(feeder_loop(
        Arc::clone(&chain),
        Arc::clone(&state),
        feeder_params(2, 40),
        stop_rx,
    ));

    // Wait for the first reseed to arm the round, then plant the candidate. The
    // head never changes below, so no later reseed clears the stash.
    let mut armed = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        if state.lock().await.target.is_some() {
            armed = true;
            break;
        }
    }
    assert!(armed, "feeder never reseeded the round");
    {
        let mut st = state.lock().await;
        // A schedule that clears at step 0 makes the candidate due immediately,
        // so the projection is never what withholds it.
        let generation = st.generation;
        st.stash.reset(generation, vec![i64::MAX], 0, 1);
        assert!(
            st.stash.insert(due_candidate()),
            "candidate must be stashed"
        );
    }

    // The ceiling admits two of the three rows: below `min_solutions`.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        chain.submitted.lock().expect("submitted lock").is_empty(),
        "a candidate the live difficulty rejects must not be submitted"
    );
    assert!(
        !state.lock().await.stash.is_empty(),
        "the withheld candidate must stay stashed for a later window"
    );

    // Ease the ceiling past every row. Same head, so this is a difficulty
    // refresh rather than a new round.
    chain.set_snapshot(Some(gated_snapshot(0)));

    let mut submitted = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(40)).await;
        if !chain.submitted.lock().expect("submitted lock").is_empty() {
            submitted = true;
            break;
        }
    }
    assert!(
        submitted,
        "the eased ceiling must let the candidate through"
    );
    {
        let sent = chain.submitted.lock().expect("submitted lock");
        assert_eq!(sent.len(), 1);
        let proof = sent.first().expect("one recorded proof");
        assert_eq!(proof.job_id, vec![0xaa]);
        assert_eq!(proof.solutions.len(), 3);
    }

    let _ = stop_tx.send(true);
    tokio::time::timeout(Duration::from_secs(2), feeder)
        .await
        .expect("feeder did not stop")
        .expect("feeder task panicked");
}
