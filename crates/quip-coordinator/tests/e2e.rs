//! End-to-end: coordinator drives `quip-mock-miner` with `FakeChain`.

#![expect(
    clippy::expect_used,
    reason = "helper builds mock miner outside #[test]"
)]
#![expect(
    clippy::unwrap_used,
    reason = "now_ms / config helpers outside #[test]"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "test wall-clock millis fit u64"
)]
#![expect(
    clippy::indexing_slicing,
    reason = "config launch table is fixture-sized"
)]

use quip_coordinator::chain::{FakeChain, MiningSnapshot};
use quip_coordinator::config::{parse_config, LaunchEntry};
use quip_coordinator::producer::derive_pow_job;
use quip_coordinator::session::{drive_pow_round, DrivePowParams};
use quip_coordinator::topology::Topology;
use quip_proto::v1::Configure;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

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

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
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
    }
}

#[tokio::test]
async fn e2e_mock_miner_pow_submit_via_fake_chain() {
    let snap = loose_snapshot();
    let chain = Arc::new(FakeChain::new(snap.clone(), None));

    let topology = Topology::from_nodes_edges(
        snap.nodes.clone(),
        snap.edges.clone(),
        &snap.allowed_h_milli,
        &snap.allowed_j_milli,
        &snap.allowed_spin_milli,
    );
    // Ensure topology hash matches snapshot.
    assert_eq!(topology.hash, snap.topology_hash);

    let deadline = now_ms() + 3_600_000;
    let job = derive_pow_job(&snap, [1u8; 32], [2u8; 32], 1, deadline).unwrap();

    // Second-generation job for cancel path.
    let mut snap2 = snap.clone();
    snap2.last_proof_block_hash = [8u8; 32];
    let job2 = derive_pow_job(&snap2, [1u8; 32], [3u8; 32], 2, deadline).unwrap();

    let miner = mock_miner();
    let entry = LaunchEntry {
        miner_id: "cpu-0".into(),
        binary: miner.clone(),
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

    let sock = format!("/tmp/quip-e2e-{}.sock", std::process::id());
    let report = drive_pow_round(DrivePowParams {
        miner_bin: &miner,
        sock_path: &sock,
        miner_id: "cpu-0",
        token: "e2e-token",
        entry: &entry,
        topology,
        job,
        chain: Arc::clone(&chain),
        cancel_then_job: Some((1, job2)),
    })
    .await;

    assert!(report.handshake_ok, "handshake failed");
    assert!(
        report.results_validated >= 1,
        "expected ≥1 validated result, got {}",
        report.results_validated
    );
    assert!(
        chain.submitted_count() >= 1,
        "FakeChain should capture ≥1 submit"
    );
    // Cancel path: mock-miner reports Status on Cancel (abandoned_generation
    // may stay 0 because the mock has no buffered work). At least the first
    // submit path is the hard gate.
    let _ = report.abandoned_generation;
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
