//! End-to-end tests for the `download` subcommand: qblock → redraw →
//! self-verify → `hardest_models` dataset → reload via the drive harness.
//!
//! The offline test (mock `FakeChain`) always runs. The live devnet test is
//! gated on `QUIP_DEVNET=<ws-url>`; a no-op (passes) when unset. Run against a
//! live node with:
//!
//! ```text
//! QUIP_DEVNET=ws://localhost:9944 \
//!   cargo test -p quip-coordinator --test download_e2e -- --nocapture --ignored
//! ```

#![expect(clippy::expect_used, reason = "integration test helpers")]
#![expect(clippy::print_stdout, reason = "devnet test diagnostic output")]
#![expect(clippy::print_stderr, reason = "devnet test diagnostic output")]

use quip_coordinator::chain::qblock::{QBlockRecord, TopologyInputs};
use quip_coordinator::chain::scale_types::MinerKind;
use quip_coordinator::chain::snapshot::MiningSnapshot;
use quip_coordinator::chain::{ChainClient, FakeChain, RealChainClient};
use quip_coordinator::download::record::hex_plain;
use quip_coordinator::download::{
    run_download, DownloadParams, Selection, DEFAULT_ENERGY_FLOOR_MILLI,
};
use quip_coordinator::drive::{drain_all, ListSource};
use quip_coordinator::topology::topology_hash_sets;

fn empty_snapshot() -> MiningSnapshot {
    MiningSnapshot {
        head_hash: [0u8; 32],
        last_proof_block_hash: [0u8; 32],
        topology_hash: vec![],
        nodes: vec![],
        edges: vec![],
        allowed_h_milli: vec![],
        allowed_j_milli: vec![],
        allowed_spin_milli: vec![],
        min_solutions: 0,
        max_energy_milli: 0,
        min_diversity_milli: 0,
        block_number: 0,
        spec_version: 117,
    }
}

fn qblock_at(id: u64, hash: [u8; 32]) -> QBlockRecord {
    let b = u8::try_from(id).expect("test id fits u8");
    QBlockRecord {
        qblock_id: id,
        miner: [1; 32],
        salt: [b; 32],
        energy_milli: -(100 * i64::try_from(id).expect("test id fits i64")),
        reward: 0,
        submitted_at: u32::try_from(id).expect("test id fits u32"),
        last_proof_block_hash: [0; 32],
        topology_hash: hash,
        device_access_time_us: 0,
        max_energy_milli: 0,
        min_solutions: 1,
        min_diversity_milli: 0,
        nonce: [b; 32],
    }
}

// ----------------------------------------------------------------------------
// Offline mock-chain e2e (always runs).
// ----------------------------------------------------------------------------

#[tokio::test]
async fn download_writes_dataset_that_reloads_via_list_source() {
    let topo = TopologyInputs {
        nodes: vec![0, 1, 2, 3],
        edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
        allowed_h_milli: vec![-1000, 0, 1000],
        allowed_j_milli: vec![-1000, 1000],
        allowed_spin_milli: vec![-1000, 1000],
    };
    let hash = topology_hash_sets(
        &topo.nodes,
        &topo.edges,
        &topo.allowed_h_milli,
        &topo.allowed_j_milli,
        &topo.allowed_spin_milli,
    );

    let fake = FakeChain::new(empty_snapshot(), None);
    fake.set_registered_topology(hash, topo.clone(), 0);
    for id in 1..=3u64 {
        fake.set_qblock(id, qblock_at(id, hash));
    }

    let dir = std::env::temp_dir().join(format!("hm-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let summary = run_download(
        &fake,
        &DownloadParams {
            selection: Selection::Range { from: 1, to: 3 },
            out_dir: dir.clone(),
            topology_filter: None,
            cap: 10_000,
            energy_floor_milli: DEFAULT_ENERGY_FLOOR_MILLI,
        },
    )
    .await
    .expect("run_download");
    assert_eq!(summary.verified, 3);

    let snap = MiningSnapshot {
        head_hash: [0u8; 32],
        last_proof_block_hash: [0u8; 32],
        topology_hash: hash.to_vec(),
        nodes: topo.nodes,
        edges: topo.edges,
        allowed_h_milli: topo.allowed_h_milli,
        allowed_j_milli: topo.allowed_j_milli,
        allowed_spin_milli: topo.allowed_spin_milli,
        min_solutions: 1,
        max_energy_milli: i64::MAX,
        min_diversity_milli: 0,
        block_number: 0,
        spec_version: 117,
    };
    let jsonl = dir.join(hex_plain(&hash)).join("instances.jsonl");
    let mut src = ListSource::load(&jsonl, Some(&snap), 0).expect("reload via --source list");
    let jobs = drain_all(&mut src);
    assert_eq!(jobs.len(), summary.verified);
    assert!(jobs.iter().all(|j| j.provenance.as_ref().unwrap().is_pow));
    let _ = std::fs::remove_dir_all(&dir);
}

// ----------------------------------------------------------------------------
// Live devnet e2e (#[ignore], QUIP_DEVNET gated).
// ----------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires a live devnet; set QUIP_DEVNET=ws://host:port"]
async fn devnet_download_qblocks_end_to_end() {
    let Ok(url) = std::env::var("QUIP_DEVNET") else {
        eprintln!("QUIP_DEVNET unset; skipping live download e2e");
        return;
    };
    let client = RealChainClient::new(vec![url.clone()], String::new(), MinerKind::Cpu);
    let latest = client
        .fetch_latest_qblock_id()
        .await
        .expect("latest_qblock_id");
    let Some(latest) = latest else {
        // A fresh devnet may have no winning qblocks yet (mining is
        // frontier-hard). The decode/redraw path can't be exercised without
        // one — pass with a note, exactly as the read-path milestones do when
        // state is absent.
        eprintln!("no qblocks on chain yet; skipping download decode assertions");
        return;
    };

    let dir = std::env::temp_dir().join(format!("hm-devnet-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let summary = run_download(
        &client,
        &DownloadParams {
            selection: Selection::Range {
                from: 1,
                to: latest.min(50),
            }, // bound the pull
            out_dir: dir.clone(),
            topology_filter: None,
            cap: 10_000,
            energy_floor_milli: DEFAULT_ENERGY_FLOOR_MILLI,
        },
    )
    .await
    .expect("run_download");
    // Every fetched qblock that decoded must also self-verify (redraw is
    // deterministic from consensus inputs) — no unverified skips on real data.
    assert_eq!(
        summary.skipped_unverified, 0,
        "a live qblock failed redraw self-verify: {summary:?}"
    );
    assert!(
        summary.verified >= 1,
        "expected at least one winning qblock"
    );

    // Reload the default-topology bucket through the drive harness path.
    let snap = client
        .fetch_mining_snapshot(None, [0u8; 32], None)
        .await
        .expect("snapshot rpc")
        .expect("snapshot present");
    let hash_hex = if snap.topology_hash.len() == 32 {
        let mut h = [0u8; 32];
        h.copy_from_slice(&snap.topology_hash);
        hex_plain(&h)
    } else {
        String::new()
    };
    let jsonl = dir.join(&hash_hex).join("instances.jsonl");
    if jsonl.exists() {
        let mut src = ListSource::load(&jsonl, Some(&snap), 0).expect("reload via --source list");
        let jobs = drain_all(&mut src);
        assert!(!jobs.is_empty(), "default bucket reloaded empty");
        println!(
            "devnet download: {} verified, default bucket reloaded {} jobs",
            summary.verified,
            jobs.len()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
