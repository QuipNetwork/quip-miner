//! Live devnet integration test for the v0.3 coordinator chain client.
//!
//! Gated on `QUIP_DEVNET=<ws-url>`; a no-op (passes) when unset so the normal
//! `cargo test` run stays offline. Run against a live node with:
//!
//! ```text
//! QUIP_DEVNET=ws://localhost:9944 \
//!   cargo test -p quip-coordinator --test devnet_submit -- --nocapture --ignored
//! ```
//!
//! Milestone 1 (read path): `RealChainClient::fetch_mining_snapshot` decodes a
//! real `MiningSnapshot` (runtime API `state_call` + SCALE decode) and reports
//! the topology size and default difficulty.
//!
//! Milestone 2 (submit path): register `//Alice` as a miner, derive the `PoW`
//! Ising from the snapshot, solve it locally (greedy spin-glass descent),
//! re-validate with `quantum_validation` exactly as the pallet does, then drive
//! the coordinator's `RealChainClient::submit_proof` and assert the pallet
//! accepts it (persistent `Miners.proofs_submitted` increment + `ProofAccepted`
//! event), or report the exact pallet error.

#![expect(clippy::expect_used, reason = "integration test helpers")]
#![expect(clippy::unwrap_used, reason = "integration test helpers")]
#![expect(clippy::panic, reason = "tests panic on hard failures")]
#![expect(clippy::print_stdout, reason = "devnet test diagnostic output")]
#![expect(clippy::print_stderr, reason = "devnet test diagnostic output")]
#![expect(
    clippy::indexing_slicing,
    reason = "event-scan and solver use fixture-bounded indices"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "RPC / solution counts fit target widths in fixtures"
)]
#![expect(
    clippy::too_many_lines,
    reason = "end-to-end devnet scenario is intentionally linear"
)]

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use parity_scale_codec::{Decode, Encode};
use serde_json::{json, Value};

use quantum_validation::{
    calculate_diversity, derive_nonce, energy_of_solution, generate_ising_model, select_diverse,
    MilliValue,
};
use quip_coordinator::chain::extrinsic::{
    build_hybrid_signed_extrinsic, hex_decode, hex_encode, miner_identity_bytes,
    miners_storage_key, SignedExtensionContext,
};
use quip_coordinator::chain::scale_types::{
    encode_register_miner_call, IsingParams, JobMode, MiningSnapshotScale, ResultDelivery,
    RewardResolution,
};
use quip_coordinator::chain::submit::SubmitAction;
use quip_coordinator::chain::{ChainClient, JobOrder, RealChainClient};
use quip_coordinator::drive::parse_topology_spec;
use quip_coordinator::presets::preset_spec;
use quip_coordinator::validate::MAX_PROOF_SOLUTIONS;
use quip_proto::v1::Solution;
use quip_protocol::wire::encode_spins;
use quip_transaction_crypto::{account_id_from_public, HybridPair};
use sp_core::Pair;

const QUANTUM_POW_PALLET: u8 = 10;

/// `QuantumPow` `Error` variants in declaration order (module error index).
const POW_ERRORS: &[&str] = &[
    "MinerAlreadyRegistered",
    "MinerNotRegistered",
    "TopologyAlreadyRegistered",
    "TopologyNotRegistered",
    "InvalidCurve",
    "GraphTooSmall",
    "InvalidTopology",
    "ProofLimitReached",
    "InvalidNonce",
    "NoSolutionsSubmitted",
    "InvalidSpinValues",
    "SolutionLengthMismatch",
    "InsufficientEnergy",
    "InsufficientDiversity",
    "InsufficientSolutions",
    "ArithmeticOverflow",
    "EmptyAllowedValues",
    "EncodingTooWide",
    "PackedSolutionLengthMismatch",
    "InvalidEncodedSpin",
    "PackedSolutionTooLarge",
    "TopologyNotMineable",
    "TopologyIsDefault",
    "MineableTopologyConflict",
];

// ----------------------------------------------------------------------------
// RPC
// ----------------------------------------------------------------------------

async fn rpc_try(url: &str, method: &str, params: Vec<Value>) -> Result<Value, String> {
    use jsonrpsee::core::client::ClientT;
    use jsonrpsee::ws_client::WsClientBuilder;
    let client = WsClientBuilder::default()
        .build(url)
        .await
        .map_err(|e| format!("ws build: {e}"))?;
    let mut p = jsonrpsee::core::params::ArrayParams::new();
    for v in params {
        p.insert(v).map_err(|e| e.to_string())?;
    }
    client
        .request::<Value, _>(method, p)
        .await
        .map_err(|e| e.to_string())
}

async fn rpc(url: &str, method: &str, params: Vec<Value>) -> Value {
    rpc_try(url, method, params)
        .await
        .unwrap_or_else(|e| panic!("rpc {method}: {e}"))
}

async fn get_storage(url: &str, key: &[u8], at: Option<&str>) -> Option<Vec<u8>> {
    let mut params = vec![json!(hex_encode(key))];
    if let Some(h) = at {
        params.push(json!(h));
    }
    let v = rpc(url, "state_getStorage", params).await;
    v.as_str().map(|s| hex_decode(s).expect("hex storage"))
}

// ----------------------------------------------------------------------------
// Storage keys
// ----------------------------------------------------------------------------

fn twox128(data: &[u8]) -> [u8; 16] {
    use std::hash::Hasher;
    use twox_hash::XxHash64;
    let mut h0 = XxHash64::with_seed(0);
    h0.write(data);
    let mut h1 = XxHash64::with_seed(1);
    h1.write(data);
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&h0.finish().to_le_bytes());
    out[8..].copy_from_slice(&h1.finish().to_le_bytes());
    out
}

fn blake2_128(data: &[u8]) -> [u8; 16] {
    use blake2::digest::{Update, VariableOutput};
    use blake2::Blake2bVar;
    let mut hasher = Blake2bVar::new(16).expect("16-byte blake2b");
    hasher.update(data);
    let mut out = [0u8; 16];
    hasher.finalize_variable(&mut out).expect("finalize");
    out
}

/// `Blake2_128Concat` storage-map key for `pallet.item(account)`.
fn map_key(pallet: &[u8], item: &[u8], acct: &[u8; 32]) -> Vec<u8> {
    let mut k = Vec::new();
    k.extend_from_slice(&twox128(pallet));
    k.extend_from_slice(&twox128(item));
    k.extend_from_slice(&blake2_128(acct));
    k.extend_from_slice(acct);
    k
}

fn events_key() -> Vec<u8> {
    let mut k = Vec::new();
    k.extend_from_slice(&twox128(b"System"));
    k.extend_from_slice(&twox128(b"Events"));
    k
}

/// SCALE-lite mirror of `frame_system::AccountInfo` (only the leading nonce).
#[derive(Decode)]
struct AccountNonce {
    nonce: u32,
    #[expect(dead_code, reason = "decoded for layout parity only")]
    consumers: u32,
    #[expect(dead_code, reason = "decoded for layout parity only")]
    providers: u32,
    #[expect(dead_code, reason = "decoded for layout parity only")]
    sufficients: u32,
}

/// SCALE mirror of `pallet_quantum_pow::MinerInfo<u128, u32>`.
#[derive(Decode)]
struct MinerInfoLite {
    #[expect(dead_code, reason = "decoded for layout parity only")]
    registered_at: u32,
    #[expect(dead_code, reason = "decoded for layout parity only")]
    deposit: u128,
    proofs_submitted: u32,
    #[expect(dead_code, reason = "decoded for layout parity only")]
    proofs_won: u32,
    #[expect(dead_code, reason = "decoded for layout parity only")]
    rewards_earned: u128,
}

async fn account_nonce(url: &str, acct: &[u8; 32]) -> u32 {
    let key = map_key(b"System", b"Account", acct);
    match get_storage(url, &key, None).await {
        Some(bytes) => AccountNonce::decode(&mut &bytes[..]).map_or(0, |a| a.nonce),
        None => 0,
    }
}

async fn miner_proofs_submitted(url: &str, acct: &[u8; 32]) -> Option<u32> {
    // The production key builder, so a live chain checks it too.
    let key = miners_storage_key(acct);
    let bytes = get_storage(url, &key, None).await?;
    MinerInfoLite::decode(&mut &bytes[..])
        .map(|m| m.proofs_submitted)
        .ok()
}

// ----------------------------------------------------------------------------
// Signed-extension context (genesis / spec / tx version)
// ----------------------------------------------------------------------------

async fn fetch_ext_ctx(url: &str, nonce: u32) -> SignedExtensionContext {
    let genesis_hex = rpc(url, "chain_getBlockHash", vec![json!(0)])
        .await
        .as_str()
        .expect("genesis hash")
        .to_string();
    let gb = hex_decode(&genesis_hex).expect("genesis hex");
    let mut genesis_hash = [0u8; 32];
    genesis_hash.copy_from_slice(&gb);
    let rv = rpc(url, "state_getRuntimeVersion", vec![]).await;
    SignedExtensionContext {
        account_nonce: nonce,
        genesis_hash,
        spec_version: rv["specVersion"].as_u64().unwrap() as u32,
        transaction_version: rv["transactionVersion"].as_u64().unwrap() as u32,
        tip: 0,
    }
}

// ----------------------------------------------------------------------------
// Snapshot (full scale form, for the exact allowed-value specs)
// ----------------------------------------------------------------------------

async fn fetch_scale_snapshot(url: &str) -> MiningSnapshotScale {
    let head = rpc(url, "chain_getBlockHash", vec![])
        .await
        .as_str()
        .expect("head")
        .to_string();
    // Runtime API arg: Option<H256>::None.
    let result = rpc(
        url,
        "state_call",
        vec![
            json!("QuantumPowApi_mining_snapshot"),
            json!(hex_encode(&[0u8])),
            json!(head),
        ],
    )
    .await;
    let bytes = hex_decode(result.as_str().expect("snapshot hex")).expect("hex");
    let decoded: Option<MiningSnapshotScale> =
        Decode::decode(&mut &bytes[..]).expect("decode MiningSnapshotScale");
    decoded.expect("snapshot is Some")
}

// ----------------------------------------------------------------------------
// Extrinsic submit + confirmation
// ----------------------------------------------------------------------------

async fn head_number(url: &str) -> u64 {
    let header = rpc(url, "chain_getHeader", vec![]).await;
    let n = header["number"].as_str().expect("number");
    u64::from_str_radix(n.trim_start_matches("0x"), 16).unwrap()
}

/// Submit a raw extrinsic and wait until `ready(state)` holds (or timeout).
/// Returns the observed head block number when the condition first held.
async fn submit_and_confirm<F, Fut>(
    url: &str,
    ext_hex: &str,
    label: &str,
    ready: F,
) -> Result<u64, String>
where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let tx = rpc_try(url, "author_submitExtrinsic", vec![json!(ext_hex)]).await;
    match &tx {
        Ok(h) => println!("  [{label}] injected tx {}", h.as_str().unwrap_or("?")),
        Err(e) => return Err(format!("author_submitExtrinsic rejected: {e}")),
    }
    for _ in 0..40 {
        let url_owned = url.to_string();
        if ready(url_owned).await {
            return Ok(head_number(url).await);
        }
        tokio::time::sleep(Duration::from_millis(750)).await;
    }
    Err(format!("[{label}] condition not met within timeout"))
}

/// Scan raw `System.Events` across recent blocks for a `QuantumPow` module error
/// or a `QuantumPow` event naming `acct`. Returns a human-readable finding.
async fn scan_recent_events(url: &str, acct: &[u8; 32], depth: u64) -> Option<String> {
    let head = head_number(url).await;
    let mut seen = HashSet::new();
    for n in (head.saturating_sub(depth)..=head).rev() {
        let hash = rpc(url, "chain_getBlockHash", vec![json!(n)]).await;
        let Some(hash) = hash.as_str() else { continue };
        if !seen.insert(hash.to_string()) {
            continue;
        }
        let Some(bytes) = get_storage(url, &events_key(), Some(hash)).await else {
            continue;
        };
        if let Some(f) = scan_events_blob(&bytes, acct) {
            return Some(format!("block #{n}: {f}"));
        }
    }
    None
}

fn scan_events_blob(bytes: &[u8], acct: &[u8; 32]) -> Option<String> {
    // System(0).ExtrinsicFailed(1) with DispatchError::Module(3){index,error[4]}.
    for i in 0..bytes.len().saturating_sub(6) {
        if bytes[i] == 0x00 && bytes[i + 1] == 0x01 && bytes[i + 2] == 0x03 {
            let module = bytes[i + 3];
            let err = bytes[i + 4];
            let name = if module == QUANTUM_POW_PALLET {
                POW_ERRORS.get(err as usize).copied().unwrap_or("Unknown")
            } else {
                "(non-QuantumPow pallet)"
            };
            return Some(format!(
                "ExtrinsicFailed Module{{ index: {module}, error: {err} }} => {name}"
            ));
        }
    }
    // QuantumPow(10) event naming our account (MinerRegistered / ProofAccepted:
    // both encode the miner AccountId immediately after pallet+variant byte).
    for i in 0..bytes.len().saturating_sub(2 + 32) {
        if bytes[i] == QUANTUM_POW_PALLET && bytes[i + 2..i + 2 + 32] == acct[..] {
            return Some(format!(
                "QuantumPow event variant {} for miner (matches account)",
                bytes[i + 1]
            ));
        }
    }
    None
}

// ----------------------------------------------------------------------------
// Local spin-glass solver + pallet-mirror validation
// ----------------------------------------------------------------------------

/// Greedy single-flip descent to a local minimum from a random start.
fn solve_one(
    rng: &mut impl rand::Rng,
    n: usize,
    h: &[MilliValue],
    adj: &[Vec<(usize, i32)>],
) -> Vec<i8> {
    let mut s: Vec<i8> = (0..n)
        .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
        .collect();
    let mut lf: Vec<i64> = (0..n)
        .map(|p| {
            i64::from(h[p])
                + adj[p]
                    .iter()
                    .map(|&(q, j)| i64::from(j) * i64::from(s[q]))
                    .sum::<i64>()
        })
        .collect();
    loop {
        let mut improved = false;
        for p in 0..n {
            // Flip lowers energy iff s[p] * local_field[p] > 0.
            if i64::from(s[p]) * lf[p] > 0 {
                let ds = i64::from(-2 * s[p]);
                s[p] = -s[p];
                for &(q, j) in &adj[p] {
                    lf[q] += i64::from(j) * ds;
                }
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    s
}

fn build_adjacency(
    nodes: &[u32],
    edges: &[(u32, u32)],
    j: &[MilliValue],
) -> Vec<Vec<(usize, i32)>> {
    let pos: HashMap<u32, usize> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let mut adj = vec![Vec::new(); nodes.len()];
    for (e, &(u, v)) in edges.iter().enumerate() {
        let (pu, pv) = (pos[&u], pos[&v]);
        adj[pu].push((pv, j[e]));
        adj[pv].push((pu, j[e]));
    }
    adj
}

// ----------------------------------------------------------------------------
// Test
// ----------------------------------------------------------------------------

/// Install the coordinator's log subscriber so a failed submit reports the
/// module error it hit. Without this, `submit_proof` logs the dispatch error at
/// `warn` and the test sees only the classified action, which cannot say why.
/// Ignores a second call, because both tests in this file may run together.
fn init_test_logging() {
    let level = std::env::var("QUIP_DEVNET_LOG")
        .ok()
        .and_then(|s| match s.as_str() {
            "trace" => Some(quip_coordinator::logging::LogLevel::Trace),
            "debug" => Some(quip_coordinator::logging::LogLevel::Debug),
            _ => None,
        })
        .unwrap_or(quip_coordinator::logging::LogLevel::Info);
    let _ = quip_coordinator::logging::init(Some(level));
}

#[tokio::test]
#[ignore = "requires a live devnet; set QUIP_DEVNET=ws://host:port"]
async fn devnet_submit_proof_end_to_end() {
    let Ok(url) = std::env::var("QUIP_DEVNET") else {
        eprintln!("QUIP_DEVNET unset; skipping live devnet test");
        return;
    };
    init_test_logging();

    let alice = HybridPair::from_string("//Alice", None).expect("//Alice");
    let alice_acct = account_id_from_public(&alice.public());
    let alice_bytes: [u8; 32] = *AsRef::<[u8; 32]>::as_ref(&alice_acct);

    let client = RealChainClient::new(
        vec![url.clone()],
        "//Alice".to_string(),
        quip_coordinator::chain::MinerKind::Cpu,
    );

    // ---------------- Milestone 1: read path ----------------
    let snap = client
        .fetch_mining_snapshot(None, alice_bytes, None)
        .await
        .expect("fetch_mining_snapshot RPC")
        .expect("snapshot present");
    println!(
        "M1 snapshot: nodes={} edges={} min_solutions={} max_energy_milli={} min_diversity_milli={} last_proof_block_hash={}",
        snap.nodes.len(),
        snap.edges.len(),
        snap.min_solutions,
        snap.max_energy_milli,
        snap.min_diversity_milli,
        hex_encode(&snap.last_proof_block_hash),
    );
    // The devnet is seeded from the `advantage2-system1` preset, so the
    // snapshot must report that graph exactly. Deriving the counts from the
    // preset rather than hard-coding them keeps this assertion correct if the
    // fixture ever changes, and catches a devnet seeded with the wrong graph.
    let seeded = parse_topology_spec(preset_spec("advantage2-system1").expect("preset resolves"))
        .expect("preset parses");
    assert_eq!(
        snap.nodes.len(),
        seeded.topology.nodes.len(),
        "node count must match the advantage2-system1 preset"
    );
    assert_eq!(
        snap.edges.len(),
        seeded.topology.edges.0.len(),
        "edge count must match the advantage2-system1 preset"
    );
    // Difficulty parameters are chain-state-dependent (they ratchet/decay and are
    // reconfigured), so assert decode sanity rather than pinning volatile values.
    assert!(snap.min_solutions >= 1, "min_solutions should be >= 1");
    assert!(
        snap.max_energy_milli < 0,
        "max_energy gate should be negative, got {}",
        snap.max_energy_milli
    );
    println!("M1 PASS: real MiningSnapshot decoded from live chain");

    // ---------------- Milestone 2: submit path ----------------
    // Full-precision snapshot for the exact allowed-value specs (needed to
    // regenerate the same Ising the pallet will).
    let scale = fetch_scale_snapshot(&url).await;
    let nodes = &scale.nodes;
    let edges = &scale.edges;
    let last_proof = snap.last_proof_block_hash;

    // (1) Register //Alice as a miner (idempotent).
    if miner_proofs_submitted(&url, &alice_bytes).await.is_none() {
        let nonce = account_nonce(&url, &alice_bytes).await;
        let ctx = fetch_ext_ctx(&url, nonce).await;
        let call = encode_register_miner_call();
        let ext = build_hybrid_signed_extrinsic(&alice, &call, &ctx);
        let block = submit_and_confirm(&url, &hex_encode(&ext), "register_miner", |u| async move {
            miner_proofs_submitted(&u, &alice_bytes).await.is_some()
        })
        .await
        .expect("register_miner not confirmed");
        println!("  register_miner: Miners[//Alice] present by block #{block}");
    } else {
        println!("  register_miner: //Alice already registered, skipping");
    }
    let proofs_before = miner_proofs_submitted(&url, &alice_bytes)
        .await
        .expect("miner registered");

    // (2) Derive the PoW nonce + Ising, exactly as the pallet does.
    let salt: [u8; 32] = rand::random();
    let miner_id = miner_identity_bytes(&alice);
    let nonce = derive_nonce(&last_proof, &miner_id, &salt);
    let (h, j) = generate_ising_model(
        nonce,
        nodes,
        edges,
        &scale.allowed_h_values.as_slice(),
        &scale.allowed_j_values.as_slice(),
    )
    .expect("generate_ising_model");
    println!(
        "  Ising derived: |h|={} |j|={} nonce=0x{:x}",
        h.len(),
        j.len(),
        nonce
    );

    // (3) Best-effort local solve. The live gate is frontier-hard (real-miner
    // territory), so this toy solver may find few or no qualifying rows — that
    // is fine: this test verifies F9's inclusion confirmation, not mining.
    let adj = build_adjacency(nodes, edges, &j);
    let ceiling = snap.max_energy_milli;
    let mut rng = rand::thread_rng();
    let mut valid: Vec<(Vec<i8>, i64)> = Vec::new();
    for _ in 0..80 {
        let s = solve_one(&mut rng, nodes.len(), &h, &adj);
        let e = energy_of_solution(&s, &h, edges, &j, nodes).expect("energy");
        if e < ceiling && !valid.iter().any(|(o, _)| *o == s) {
            valid.push((s, e));
        }
    }
    // The runtime bounds a proof at `QuantumPowMaxSolutions` rows. More than
    // that fails to decode, so the node answers the submission with a codec
    // error instead of a dispatch result and the test learns nothing.
    valid.truncate(MAX_PROOF_SOLUTIONS);

    // (4) Best-effort diversity over whatever we found (0 if < 2 rows).
    let valid_slices: Vec<&[i8]> = valid.iter().map(|(s, _)| s.as_slice()).collect();
    let diversity = if valid_slices.len() >= 2 {
        let target = valid_slices.len().min((snap.min_solutions.max(1)) as usize);
        select_diverse(&valid_slices, target)
            .ok()
            .and_then(|idx| {
                let sel: Vec<&[i8]> = idx.iter().map(|&i| valid_slices[i]).collect();
                calculate_diversity(&sel).ok()
            })
            .unwrap_or(0)
    } else {
        0
    };
    let best_energy = valid.iter().map(|(_, e)| *e).min().unwrap_or(i64::MAX);
    println!(
        "  local solve: {} rows below {ceiling} (best_energy_milli={best_energy} diversity_milli={diversity})",
        valid.len()
    );

    // (5) Drive the coordinator's real submit path.
    let proof = quip_coordinator::chain::submit::Proof {
        job_id: vec![],
        best_energy_milli: best_energy,
        diversity_milli: diversity,
        n_valid: valid.len() as u32,
        solutions: valid
            .iter()
            .map(|(s, e)| Solution {
                spins_bytes: encode_spins(s),
                energy_milli: *e,
            })
            .collect(),
        is_pow: true,
        order_id: vec![],
        generation: 0,
        salt: salt.to_vec(),
        device_access_time_us: 0,
    };
    let action = client.submit_proof(&proof).await;
    println!("  RealChainClient::submit_proof -> {action:?}");

    // (6) F9 contract: submit_proof returns Ok(Success) IFF the proof actually
    // landed and dispatched OK on-chain (proofs_submitted incremented). The old
    // fire-and-forget path returned Success on mere pool acceptance, so a proof
    // the pallet rejects would falsely report Success while the counter never
    // moved. F9 (submit_and_watch + ExtrinsicSuccess/Failed) must never do that.
    let mut incremented = false;
    for _ in 0..40 {
        if let Some(now) = miner_proofs_submitted(&url, &alice_bytes).await {
            if now > proofs_before {
                incremented = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(750)).await;
    }
    let reported_success = matches!(action.as_ref().map(|r| r.action), Ok(SubmitAction::Success));
    let event = scan_recent_events(&url, &alice_bytes, 8).await;
    println!(
        "  F9 check: reported_success={reported_success} on_chain_incremented={incremented} event={}",
        event.as_deref().unwrap_or("(none)")
    );
    assert_eq!(
        reported_success, incremented,
        "F9 VIOLATED: submit_proof returned {action:?} but on-chain increment={incremented}; \
         Ok(Success) must mean the proof landed and dispatched successfully"
    );
    if reported_success {
        println!("M2 PASS: submit_proof confirmed a real on-chain ExtrinsicSuccess");
    } else {
        println!(
            "M2 PASS: submit_proof reported the on-chain outcome with no false Success -> {action:?}"
        );
    }
}

// ----------------------------------------------------------------------------
// Milestone 3: mempool JobProposed decode (quip-0j6)
// ----------------------------------------------------------------------------

/// `QuantumComputeMempool` pallet + `propose_job` call indices (runtime
/// `construct_runtime`: `pallet_index` 9, `call_index` 3).
const MEMPOOL_PALLET: u8 = 9;
const PROPOSE_JOB_CALL: u8 = 3;

/// Canonical default plain-Ising job spec name (`DEFAULT_ISING_SPEC_NAME`),
/// seeded at genesis by the runtime. Its `spec_id` is deterministic, so M3 can
/// propose against it on a fresh devnet without registering a spec.
const DEFAULT_ISING_SPEC_NAME: &[u8] = b"plain-ising-v1";

/// `propose_job` reward floor (`QuantumMinReward = UNIT`).
const MIN_REWARD: u128 = 1_000_000_000_000;

fn blake2_256(data: &[u8]) -> [u8; 32] {
    use blake2::digest::{Update, VariableOutput};
    use blake2::Blake2bVar;
    let mut hasher = Blake2bVar::new(32).expect("32-byte blake2b");
    hasher.update(data);
    let mut out = [0u8; 32];
    hasher.finalize_variable(&mut out).expect("finalize");
    out
}

/// Derive the canonical default plain-Ising `spec_id`, mirroring the pallet's
/// `job_spec_id`: `BlakeTwo256::hash_of(&(name, Formulation::Ising, None, None))`.
/// The SCALE preimage is `Vec<u8>(name) ++ 0x00 (Ising) ++ 0x00 ++ 0x00`.
fn default_ising_spec_id() -> [u8; 32] {
    let mut preimage = DEFAULT_ISING_SPEC_NAME.to_vec().encode();
    preimage.extend_from_slice(&[0u8, 0u8, 0u8]);
    blake2_256(&preimage)
}

/// `Blake2_128Concat` storage key for `QuantumComputeMempool.JobSpecs(spec_id)`.
fn job_specs_storage_key(spec_id: &[u8; 32]) -> Vec<u8> {
    let mut k = Vec::new();
    k.extend_from_slice(&twox128(b"QuantumComputeMempool"));
    k.extend_from_slice(&twox128(b"JobSpecs"));
    k.extend_from_slice(&blake2_128(spec_id));
    k.extend_from_slice(spec_id);
    k
}

/// Storage key for the plain `QuantumComputeMempool.NextOrderId` value.
fn next_order_id_key() -> Vec<u8> {
    let mut k = Vec::new();
    k.extend_from_slice(&twox128(b"QuantumComputeMempool"));
    k.extend_from_slice(&twox128(b"NextOrderId"));
    k
}

async fn read_next_order_id(url: &str) -> u64 {
    match get_storage(url, &next_order_id_key(), None).await {
        Some(b) => u64::decode(&mut &b[..]).unwrap_or(0),
        None => 0,
    }
}

/// SCALE-encode a `propose_job(spec_id, ising_params, reward, mode, resolution,
/// deadline_blocks, block_wait, delivery)` call with an open, on-chain-only order.
fn encode_propose_job_call(spec_id: &[u8; 32], ising: &IsingParams, reward: u128) -> Vec<u8> {
    let mut out = vec![MEMPOOL_PALLET, PROPOSE_JOB_CALL];
    out.extend_from_slice(spec_id);
    out.extend(ising.encode());
    out.extend(reward.encode());
    out.extend(JobMode::Open.encode());
    out.extend(RewardResolution::SingleBest.encode());
    out.extend(100u32.encode()); // deadline_blocks (<= MaxDeadlineBlocks = 1000)
    out.extend(10u32.encode()); // block_wait (<= MaxBlockWait = 100)
    out.extend(ResultDelivery::OnChainOnly.encode());
    out
}

/// End-to-end: propose a fresh job order against the genesis default Ising spec,
/// then confirm the coordinator's `fetch_mempool_orders` discovers its
/// `order_id` from the head-block `JobProposed` event, storage-reads
/// `JobOrders(oid)`, and builds a `JobOrder`. Self-seeding — needs only the
/// genesis default spec, so it runs on a fresh devnet with an empty mempool.
#[tokio::test]
#[ignore = "requires a live devnet; set QUIP_DEVNET=ws://host:port"]
async fn devnet_mempool_job_proposed_end_to_end() {
    let Ok(url) = std::env::var("QUIP_DEVNET") else {
        eprintln!("QUIP_DEVNET unset; skipping live devnet mempool test");
        return;
    };
    init_test_logging();

    let alice = HybridPair::from_string("//Alice", None).expect("//Alice");
    let alice_acct = account_id_from_public(&alice.public());
    let alice_bytes: [u8; 32] = *AsRef::<[u8; 32]>::as_ref(&alice_acct);
    let client = RealChainClient::new(
        vec![url.clone()],
        "//Alice".to_string(),
        quip_coordinator::chain::MinerKind::Cpu,
    );

    // The canonical default plain-Ising spec is seeded at genesis; propose
    // against it so no root-gated spec registration is needed.
    let spec_id = default_ising_spec_id();
    assert!(
        get_storage(&url, &job_specs_storage_key(&spec_id), None)
            .await
            .is_some(),
        "genesis default Ising spec {} not found on chain; the devnet must seed \
         default_ising_spec_builder",
        hex_encode(&spec_id)
    );

    // A minimal structurally-consistent Ising: 2 nodes, 1 edge. propose_job
    // validates topology with no allowed-value constraints, so any graph with
    // matching h/j counts and in-range edges is accepted.
    let ising = IsingParams {
        nodes: vec![0, 1],
        edges: vec![(0, 1)],
        h_values: vec![0, 0],
        j_values: vec![-1000],
        min_energy_milli: None,
        min_diversity_milli: None,
        min_solutions: None,
    };

    // Build + sign propose_job as //Alice; the assigned order_id is the current
    // NextOrderId (no competing proposer takes it on an otherwise-idle chain).
    let expected_oid = read_next_order_id(&url).await;
    let nonce = account_nonce(&url, &alice_bytes).await;
    let ctx = fetch_ext_ctx(&url, nonce).await;
    let call = encode_propose_job_call(&spec_id, &ising, MIN_REWARD);
    let ext = build_hybrid_signed_extrinsic(&alice, &call, &ctx);
    let tx = rpc_try(
        &url,
        "author_submitExtrinsic",
        vec![json!(hex_encode(&ext))],
    )
    .await;
    let tx = tx.expect("author_submitExtrinsic accepted propose_job");
    println!(
        "  propose_job injected (spec_id={}, expected order_id={expected_oid}), tx {}",
        hex_encode(&spec_id),
        tx.as_str().unwrap_or("?")
    );

    // Poll the coordinator's real read path: fetch_mempool_orders discovers
    // order ids from JobProposed events at head, then storage-reads each. The
    // event is only visible while the inclusion block is head (~6s), so poll
    // fast enough to land inside that window.
    let want_id_le = expected_oid.to_le_bytes().to_vec();
    let mut found: Option<JobOrder> = None;
    for _ in 0..60 {
        match client.fetch_mempool_orders(alice_bytes).await {
            Ok(orders) => {
                if let Some(o) = orders.into_iter().find(|o| o.order_id == want_id_le) {
                    found = Some(o);
                    break;
                }
            }
            Err(e) => eprintln!("  fetch_mempool_orders err (retrying): {e}"),
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // Secondary confirmation the order actually landed (clearer failure than a
    // bare timeout if the extrinsic was rejected in-block).
    let after = read_next_order_id(&url).await;
    assert!(
        after > expected_oid,
        "propose_job did not create an order: NextOrderId stayed at {expected_oid}"
    );

    let order = found.unwrap_or_else(|| {
        panic!("fetch_mempool_orders never surfaced order #{expected_oid} at head within timeout")
    });

    // The built Job must match the proposed Ising end-to-end.
    assert_eq!(order.order_id, want_id_le, "order_id mismatch");
    assert_eq!(order.nodes, ising.nodes, "nodes mismatch");
    assert_eq!(order.edges, ising.edges, "edges mismatch");
    assert_eq!(order.h_milli, ising.h_values, "h_values mismatch");
    assert_eq!(order.j_milli, ising.j_values, "j_values mismatch");
    println!(
        "M3 PASS: propose_job(order_id={expected_oid}) -> JobProposed discovered at head; \
         JobOrders storage decoded + Job built (nodes={} edges={})",
        order.nodes.len(),
        order.edges.len()
    );
}
