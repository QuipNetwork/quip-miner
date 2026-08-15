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

    assert!(wait_generation(&state, 1).await, "first round never staged");
    chain.set_snapshot(Some(snapshot_with_head([2u8; 32])));
    assert!(
        wait_generation(&state, 2).await,
        "second round never staged"
    );
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

    assert!(wait_generation(&state, 1).await, "first round never staged");
    chain.set_qblock_id(Some(11));
    chain.set_snapshot(Some(snapshot_with_head([2u8; 32])));
    assert!(
        wait_generation(&state, 2).await,
        "second round never staged"
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
        wait_generation(&state, 1).await,
        "DuplicateParticipation must not block mining"
    );

    chain.set_qblock_id(Some(5));
    chain.set_participation_result(Ok(ParticipationOutcome::StaleQBlock));
    chain.set_snapshot(Some(snapshot_with_head([2u8; 32])));
    assert!(
        wait_generation(&state, 2).await,
        "InvalidQBlockId must not block mining"
    );

    chain.set_qblock_id(Some(6));
    chain.set_participation_result(Ok(ParticipationOutcome::DescriptorMissing));
    chain.set_snapshot(Some(snapshot_with_head([3u8; 32])));
    assert!(
        wait_generation(&state, 3).await,
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
    assert_eq!(chain.participation_calls(), 1);

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
    assert_eq!(chain.participation_calls(), 3);

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
    assert_eq!(chain.participation_calls(), 1);

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
