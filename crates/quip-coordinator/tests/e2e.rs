//! End-to-end: coordinator drives `quip-mock-miner` with `FakeChain`.

#![expect(
    clippy::expect_used,
    reason = "helper builds mock miner outside #[test]"
)]
#![expect(
    clippy::indexing_slicing,
    reason = "config launch table is fixture-sized"
)]

use quip_coordinator::chain::{FakeChain, MiningSnapshot};
use quip_coordinator::config::{parse_config, LaunchEntry};
use quip_coordinator::runtime::{run_runtime, RuntimeParams};
use quip_coordinator::session::CoordinatorState;
use quip_coordinator::supervisor::BackoffPolicy;
use quip_proto::v1::Configure;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{oneshot, Mutex};

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

fn loose_snapshot() -> MiningSnapshot {
    // Gates loose enough that mock-miner's all-+1 solution is accepted.
    let nodes = vec![0, 1, 2, 3];
    let edges = vec![(0, 1), (1, 2), (2, 3), (0, 3)];
    let h = [-1000, 0, 1000];
    let j = [-1000, 1000];
    let spin = [-1000, 1000];
    let hash =
        quip_coordinator::topology::topology_hash_sets(&nodes, &edges, &h, &j, &spin).to_vec();
    MiningSnapshot {
        head_hash: [0u8; 32],
        last_proof_block_hash: [7u8; 32],
        topology_hash: hash,
        nodes,
        edges,
        allowed_h_milli: h.to_vec(),
        allowed_j_milli: j.to_vec(),
        allowed_spin_milli: spin.to_vec(),
        min_solutions: 1,
        max_energy_milli: i64::MAX / 2, // energy ceiling (strict <)
        min_diversity_milli: 0,
        block_number: 42,
        spec_version: 117,
    }
}

#[tokio::test]
async fn e2e_mock_miner_pow_submit_via_fake_chain() {
    let miner = mock_miner();
    let chain = Arc::new(FakeChain::new(loose_snapshot(), None));
    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    let params = RuntimeParams {
        max_submit_attempts: 5,
        sock_path: format!("/tmp/quip-e2e-{}.sock", std::process::id()),
        grace_ms: 500,
        backoff: BackoffPolicy::default(),
        miner_identity: [0x11; 32],
        miner_account: [0x22; 32],
        buffer_depth: 2,
        poll_interval_ms: 100,
        dashboard: None,
        log_level: quip_coordinator::logging::LogLevel::Info,
        funding: quip_coordinator::funding::FundingParams::default(),
        descriptor: quip_coordinator::config::DescriptorParams::default(),
        descriptor_filed: Arc::new(AtomicBool::new(false)),
        miner_registered: Arc::new(AtomicBool::new(false)),
        solver_registered: Arc::new(AtomicBool::new(false)),
        identity: quip_coordinator::metrics::Identity::default(),
    };
    let entry = LaunchEntry {
        miner_id: "cpu-0".into(),
        binary: miner,
        backend: "cpu".into(),
        device: None,
        configure: Configure {
            queue_depth: 3,
            idle_timeout_s: 30,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: String::new(),
        },
    };

    let (trigger_tx, trigger_rx) = oneshot::channel::<()>();
    let chain_for_run = Arc::clone(&chain);
    let run = tokio::spawn(async move {
        run_runtime(vec![entry], chain_for_run, state, params, async move {
            let _ = trigger_rx.await;
        })
        .await
    });

    // The supervisor spawns the mock miner. The feeder stages derived PoW
    // jobs. The mock miner answers all +1, which clears the loose gates, and
    // the session validates the result and submits it to FakeChain.
    let mut submitted = false;
    for _ in 0..100 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if chain.submitted_count() >= 1 {
            submitted = true;
            break;
        }
    }
    assert!(
        submitted,
        "FakeChain never captured a submit from the mock miner"
    );

    trigger_tx.send(()).expect("send shutdown trigger");
    tokio::time::timeout(Duration::from_secs(6), run)
        .await
        .expect("run_runtime did not return after shutdown")
        .expect("run_runtime task panicked")
        .expect("run_runtime returned an error");
}

#[test]
fn config_maps_mock_miner_launch() {
    let toml = r#"
[miner]
validators = ["ws://127.0.0.1:9944"]
signer_key = "//Alice"
public_host = "203.0.113.10"
public_port = 20050

[cpu]
binary = "quip-mock-miner"
queue_depth = 3
"#;
    let c = parse_config(toml).unwrap();
    assert_eq!(c.launch[0].miner_id, "cpu-0");
    assert_eq!(c.launch[0].binary, "quip-mock-miner");
}
