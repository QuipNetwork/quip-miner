//! Startup driver for the round state machine.
//!
//! Process start has no miners to stop and does not stage work. This module
//! drives [`crate::round::RoundState`] through validator-synced, account-funded,
//! miner-registered, requirements-downloaded, descriptor-filed, and
//! participation-declared. The feeder drives the same machine, including
//! stop-mining and start-mining, on every later round.
//!
//! A funding failure at startup is fatal. A missing snapshot is not: the
//! caller warns and the feeder retries after miners connect.

use crate::chain::extrinsic::hex_encode;
use crate::chain::scale_types::{
    MAX_NODE_ID_BYTES, MAX_NODE_NAME_BYTES, MAX_PUBLIC_HOST_BYTES, MAX_RPC_ENDPOINTS,
    MAX_RPC_ENDPOINT_BYTES,
};
use crate::chain::sync::{wait_until_synced, SyncOutcome, SyncSource};
use crate::chain::{
    ChainClient, DescriptorOutcome, MiningSnapshot, NodeDescriptorV2Input, ParticipationOutcome,
    RegistrationOutcome,
};
use crate::config::DescriptorParams;
use crate::funding::{
    ensure_funded, BalanceSource, Faucet, FundingError, FundingParams, HttpFaucet,
};
use crate::round::{RoundEvent, RoundState};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Bounded retries for a transient descriptor or participation submit.
pub(crate) const SUBMIT_ATTEMPTS: u32 = 3;

/// Why the coordinator is not ready to mine.
#[derive(Debug)]
pub enum ReadinessError {
    /// The miner account cannot pay submit fees.
    Funding(FundingError),
    /// The next qblock's requirements could not be loaded.
    Snapshot(String),
}

impl std::fmt::Display for ReadinessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Funding(e) => write!(f, "{e}"),
            Self::Snapshot(e) => write!(f, "cannot load next qblock requirements: {e}"),
        }
    }
}

impl std::error::Error for ReadinessError {}

/// Steps that must happen once per process, not once per round.
///
/// Both start clear and are set when their step is known done, so the feeder's
/// re-walk of the same states is cheap and does not re-submit.
pub struct ProcessLatches<'a> {
    /// Set after the node descriptor is filed, rejected, or given up on.
    pub descriptor_filed: &'a AtomicBool,
    /// Set once `QuantumPow.Miners` is known to hold the signing account.
    pub miner_registered: &'a AtomicBool,
}

/// Build a faucet client from a URL. A URL that will not even construct a
/// client is reported and then treated as absent.
#[must_use]
pub fn build_faucet(url: Option<&str>) -> Option<HttpFaucet> {
    let url = url?;
    match HttpFaucet::new(url) {
        Ok(f) => Some(f),
        Err(e) => {
            tracing::error!(url = %url, error = %e, "cannot build faucet client");
            None
        }
    }
}

/// Drive the round machine through validator-synced, account-funded,
/// miner-registered, requirements-downloaded, descriptor-filed, and
/// participation-declared. Used at process start. Registration, descriptor and
/// participation never fail the walk. The feeder walks the same states, plus
/// stop-mining and start-mining, on every later round.
///
/// `account` is the signing `AccountId32`, which is what pays fees, holds the
/// balance, and keys the on-chain maps. The `PoW` miner identity derived from
/// the same key is a different value and is not used here.
///
/// `sleep` is injected so tests do not wait on the real clock. A validator
/// that never answers sync is not a failure here: that matches startup, which
/// warns and continues.
///
/// # Errors
/// Returns [`ReadinessError::Funding`] when the account cannot be brought up
/// to the floor, or [`ReadinessError::Snapshot`] when the next round's
/// requirements cannot be read.
pub async fn prepare_round<C, F, S, Fut>(
    chain: &C,
    faucet: Option<&F>,
    account: [u8; 32],
    funding: &FundingParams,
    sleep: S,
    descriptor: &DescriptorParams,
    latches: &ProcessLatches<'_>,
) -> Result<MiningSnapshot, ReadinessError>
where
    C: ChainClient + SyncSource + BalanceSource,
    F: Faucet + ?Sized,
    S: FnMut(Duration) -> Fut + Send + Clone,
    Fut: std::future::Future<Output = ()> + Send,
{
    let mut state = RoundState::ValidatorSynced;
    state.log_entry(0);

    match wait_until_synced(chain, sleep.clone()).await {
        SyncOutcome::Synced => {}
        SyncOutcome::Unknown(reason) => {
            tracing::warn!(
                reason = %reason,
                "cannot confirm the validator has caught up; continuing, but funding and \
                 mining may fail until it does"
            );
        }
    }
    state = match state.transition(RoundEvent::Succeeded) {
        Some(next) => {
            next.log_entry(0);
            next
        }
        None => state,
    };

    let _ = ensure_funded(chain, faucet, account, funding, sleep)
        .await
        .map_err(ReadinessError::Funding)?;
    state = match state.transition(RoundEvent::Succeeded) {
        Some(next) => {
            next.log_entry(0);
            next
        }
        None => state,
    };

    // Not fatal at startup, for the same reason an unreachable validator is
    // not: the node manager starts the coordinator and its validator together,
    // so exiting here would crash-loop while the node boots. The latch stays
    // clear, and the feeder holds off mining and retries every round until it
    // succeeds.
    let _ = register_round_miner(chain, latches.miner_registered).await;
    state = match state.transition(RoundEvent::Succeeded) {
        Some(next) => {
            next.log_entry(0);
            next
        }
        None => state,
    };

    let snap = chain
        .fetch_mining_snapshot(None, account, None)
        .await
        .map_err(|e| ReadinessError::Snapshot(e.to_string()))?
        .ok_or_else(|| ReadinessError::Snapshot("chain has no mining snapshot".into()))?;
    state = match state.transition(RoundEvent::Succeeded) {
        Some(next) => {
            next.log_entry(0);
            next
        }
        None => state,
    };

    file_round_descriptor(chain, latches.descriptor_filed, descriptor, account).await;
    state = match state.transition(RoundEvent::Succeeded) {
        Some(next) => {
            next.log_entry(0);
            next
        }
        None => state,
    };

    let mut last_declared = None;
    for _ in 0..SUBMIT_ATTEMPTS {
        declare_round_participation(chain, &mut last_declared, account).await;
        if last_declared.is_some() {
            break;
        }
    }
    let _ = state.transition(RoundEvent::Succeeded);
    Ok(snap)
}

/// Register the signing account with `QuantumPow.register_miner`.
///
/// Returns whether the account is known to be registered. The chain client
/// reads `QuantumPow.Miners` first, so an account that registered in an earlier
/// process submits nothing; `miner_registered` then latches that answer for the
/// rest of this process so later rounds skip the read too.
///
/// Unlike the descriptor step, a failure does not latch. Without registration
/// every `submit_proof` is rejected with `MinerNotRegistered`, so the caller
/// holds off mining and this runs again on the next round.
pub(crate) async fn register_round_miner<C: ChainClient>(
    chain: &C,
    miner_registered: &AtomicBool,
) -> bool {
    if miner_registered.load(Ordering::Relaxed) {
        tracing::trace!("miner already registered this process");
        return true;
    }
    for attempt in 1..=SUBMIT_ATTEMPTS {
        match chain.ensure_miner_registered().await {
            Ok(RegistrationOutcome::Registered) => {
                tracing::info!("registered this account as a miner on chain");
                miner_registered.store(true, Ordering::Relaxed);
                return true;
            }
            Ok(RegistrationOutcome::AlreadyRegistered) => {
                tracing::debug!("this account is already a registered miner");
                miner_registered.store(true, Ordering::Relaxed);
                return true;
            }
            Err(e) => {
                if attempt == SUBMIT_ATTEMPTS {
                    tracing::warn!(
                        error = %e,
                        attempts = SUBMIT_ATTEMPTS,
                        "cannot register this account as a miner; proofs would be rejected \
                         with MinerNotRegistered, so mining waits until this succeeds"
                    );
                }
            }
        }
    }
    false
}

/// Default `node_id`: 64-char lowercase hex of the miner account, with no
/// `0x` prefix. Fits `MaxNodeIdBytes`.
#[must_use]
pub fn node_id_from_account(account: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in account {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Build a V2 descriptor, or `None` when a required value is missing.
///
/// Warns once and names the missing `[miner]` key. Does not fail the walk.
pub(crate) fn build_descriptor_payload(
    params: &DescriptorParams,
    account: [u8; 32],
) -> Option<NodeDescriptorV2Input> {
    let Some(name) = params
        .node_name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        tracing::warn!(
            key = "node_name",
            section = "miner",
            "missing [miner].node_name; not filing a node descriptor"
        );
        return None;
    };
    if name.len() > MAX_NODE_NAME_BYTES {
        tracing::warn!(
            key = "node_name",
            section = "miner",
            max = MAX_NODE_NAME_BYTES,
            "[miner].node_name is longer than the pallet bound; not filing a node descriptor"
        );
        return None;
    }
    let node_id = params
        .node_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map_or_else(|| node_id_from_account(&account), str::to_string);
    if node_id.is_empty() || node_id.len() > MAX_NODE_ID_BYTES {
        tracing::warn!(
            key = "node_id",
            section = "miner",
            "[miner].node_id is empty or longer than the pallet bound; not filing a node descriptor"
        );
        return None;
    }
    if params.miners.is_empty() {
        tracing::warn!("launch plan has no miners; not filing a node descriptor");
        return None;
    }

    let public_host = match params
        .public_host
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(host) if host.len() > MAX_PUBLIC_HOST_BYTES => {
            tracing::warn!(
                key = "public_host",
                section = "miner",
                "[miner].public_host is longer than the pallet bound; not filing a node descriptor"
            );
            return None;
        }
        Some(host) => Some(host.as_bytes().to_vec()),
        None => None,
    };

    let mut rpc_endpoints = Vec::new();
    for endpoint in params.rpc_endpoints.iter().take(MAX_RPC_ENDPOINTS) {
        let trimmed = endpoint.trim();
        if trimmed.is_empty() || trimmed.len() > MAX_RPC_ENDPOINT_BYTES {
            continue;
        }
        rpc_endpoints.push(trimmed.as_bytes().to_vec());
    }

    Some(NodeDescriptorV2Input {
        node_id: node_id.into_bytes(),
        node_name: name.as_bytes().to_vec(),
        public_host,
        public_port: params.public_port.filter(|&p| p > 0),
        rpc_endpoints,
        auto_mine: params.auto_mine,
        log_level: params.log_level,
        miners: params.miners.clone(),
        system_info: None,
        runtime: None,
    })
}

/// File a node descriptor on the first walk after process start.
///
/// Later calls are a no-op. A missing required value, a pallet rejection, or
/// three transient failures still mark the step done so mining continues.
pub(crate) async fn file_round_descriptor<C: ChainClient>(
    chain: &C,
    descriptor_filed: &AtomicBool,
    params: &DescriptorParams,
    account: [u8; 32],
) {
    if descriptor_filed.load(Ordering::Relaxed) {
        tracing::trace!("node descriptor already filed this process");
        return;
    }
    let Some(payload) = build_descriptor_payload(params, account) else {
        descriptor_filed.store(true, Ordering::Relaxed);
        return;
    };
    for attempt in 1..=SUBMIT_ATTEMPTS {
        match chain.file_descriptor(&payload).await {
            Ok(DescriptorOutcome::Filed) => {
                tracing::info!("filed node descriptor");
                descriptor_filed.store(true, Ordering::Relaxed);
                return;
            }
            Ok(DescriptorOutcome::Rejected) => {
                tracing::warn!("pallet rejected the node descriptor; mining continues without one");
                descriptor_filed.store(true, Ordering::Relaxed);
                return;
            }
            Err(e) => {
                if attempt == SUBMIT_ATTEMPTS {
                    tracing::warn!(
                        error = %e,
                        attempts = SUBMIT_ATTEMPTS,
                        "descriptor submit failed; mining continues without one"
                    );
                    descriptor_filed.store(true, Ordering::Relaxed);
                    return;
                }
            }
        }
    }
}

/// Candidate qblock id the pallet accepts: one past the last minted qblock.
#[must_use]
pub(crate) fn candidate_qblock_id(latest: Option<u64>) -> u64 {
    latest.unwrap_or(0).saturating_add(1)
}

/// Submit `MinerRegistry.participate` for the current candidate qblock.
///
/// Mining does not wait on the result. A second call for the same qblock does
/// not submit again. Pallet errors never become a readiness failure.
pub(crate) async fn declare_round_participation<C: ChainClient>(
    chain: &C,
    last_declared: &mut Option<u64>,
    account: [u8; 32],
) {
    let latest = match chain.fetch_latest_qblock_id().await {
        Ok(id) => id,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "cannot read qblock id; will retry participation next round"
            );
            return;
        }
    };
    let qblock_id = candidate_qblock_id(latest);
    if *last_declared == Some(qblock_id) {
        return;
    }
    match chain.declare_participation(qblock_id).await {
        Ok(ParticipationOutcome::Declared | ParticipationOutcome::AlreadyDeclared) => {
            *last_declared = Some(qblock_id);
        }
        Ok(ParticipationOutcome::StaleQBlock) => {
            tracing::debug!(
                qblock_id,
                "candidate qblock moved; will declare the next one"
            );
            *last_declared = Some(qblock_id);
        }
        Ok(ParticipationOutcome::DescriptorMissing) => {
            tracing::warn!(
                account = %hex_encode(&account),
                qblock_id,
                "MinerRegistry has no descriptor for this account; participation cannot be recorded until one is set"
            );
            *last_declared = Some(qblock_id);
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                qblock_id,
                "participation submit failed; will retry next round"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        candidate_qblock_id, declare_round_participation, file_round_descriptor, prepare_round,
        register_round_miner, HttpFaucet, ProcessLatches, SUBMIT_ATTEMPTS,
    };
    use crate::chain::{
        ChainError, DescriptorOutcome, FakeChain, MinerKind, MinerSpecScale, MiningSnapshot,
        ParticipationOutcome, RegistrationOutcome,
    };
    use crate::config::DescriptorParams;
    use crate::funding::FundingParams;
    use std::io::{self, Write};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tracing_subscriber::fmt::MakeWriter;

    fn snap() -> MiningSnapshot {
        MiningSnapshot {
            head_hash: [0u8; 32],
            last_proof_block_hash: [1u8; 32],
            topology_hash: vec![0u8; 32],
            nodes: vec![0, 1],
            edges: vec![(0, 1)],
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![0],
            allowed_spin_milli: vec![-1000, 1000],
            min_solutions: 1,
            max_energy_milli: 0,
            min_diversity_milli: 0,
            block_number: 1,
        }
    }

    async fn no_sleep(_d: Duration) {}

    fn account() -> [u8; 32] {
        [0xab; 32]
    }

    #[test]
    fn candidate_is_one_past_latest() {
        assert_eq!(candidate_qblock_id(None), 1);
        assert_eq!(candidate_qblock_id(Some(0)), 1);
        assert_eq!(candidate_qblock_id(Some(4581)), 4582);
    }

    #[test]
    fn default_node_id_is_account_hex_without_prefix() {
        let id = super::node_id_from_account(&account());
        assert_eq!(id.len(), 64);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(!id.starts_with("0x"));
        assert_eq!(id, "ab".repeat(32));
    }

    #[tokio::test]
    async fn same_qblock_declares_once() {
        let chain = FakeChain::new(snap(), None);
        chain.set_qblock_id(Some(10));
        let mut last = None;
        declare_round_participation(&chain, &mut last, account()).await;
        declare_round_participation(&chain, &mut last, account()).await;
        assert_eq!(chain.participation_calls(), 1);
        assert_eq!(chain.take_participations(), vec![11]);
        assert_eq!(last, Some(11));
    }

    #[tokio::test]
    async fn different_qblocks_declare_twice() {
        let chain = FakeChain::new(snap(), None);
        let mut last = None;
        chain.set_qblock_id(Some(10));
        declare_round_participation(&chain, &mut last, account()).await;
        chain.set_qblock_id(Some(11));
        declare_round_participation(&chain, &mut last, account()).await;
        assert_eq!(chain.participation_calls(), 2);
        assert_eq!(chain.take_participations(), vec![11, 12]);
    }

    fn named_descriptor() -> DescriptorParams {
        DescriptorParams {
            node_name: Some("Tesla".into()),
            public_host: Some("96.233.112.201".into()),
            rpc_endpoints: vec!["ws://127.0.0.1:9944".into()],
            miners: vec![
                MinerSpecScale {
                    kind: MinerKind::Cpu,
                    label: Some(b"cpu-0".to_vec()),
                    backend: Some(b"cpu".to_vec()),
                    device_id: None,
                },
                MinerSpecScale {
                    kind: MinerKind::Metal,
                    label: Some(b"metal-0".to_vec()),
                    backend: Some(b"metal".to_vec()),
                    device_id: None,
                },
            ],
            ..DescriptorParams::default()
        }
    }

    #[tokio::test]
    async fn prepare_round_declares_the_candidate() {
        let chain = FakeChain::new(snap(), None);
        chain.set_qblock_id(Some(20));
        let _snap = prepare_round(
            &chain,
            None::<&HttpFaucet>,
            account(),
            &FundingParams::default(),
            no_sleep,
            &DescriptorParams::default(),
            &ProcessLatches {
                descriptor_filed: &AtomicBool::new(false),
                miner_registered: &AtomicBool::new(false),
            },
        )
        .await
        .expect("prepare_round");
        assert_eq!(chain.participation_calls(), 1);
        assert_eq!(chain.take_participations(), vec![21]);
        assert_eq!(chain.descriptor_calls(), 0);
    }

    #[tokio::test]
    async fn prepare_round_files_a_v2_descriptor_once() {
        let chain = FakeChain::new(snap(), None);
        chain.set_qblock_id(Some(20));
        let filed = AtomicBool::new(false);
        let params = named_descriptor();
        let _snap = prepare_round(
            &chain,
            None::<&HttpFaucet>,
            account(),
            &FundingParams::default(),
            no_sleep,
            &params,
            &ProcessLatches {
                descriptor_filed: &filed,
                miner_registered: &AtomicBool::new(false),
            },
        )
        .await
        .expect("prepare_round");
        assert!(filed.load(Ordering::Relaxed));
        assert_eq!(chain.descriptor_calls(), 1);
        let filed_payload = chain.take_descriptors();
        let desc = filed_payload.first().expect("one descriptor");
        assert_eq!(
            desc.node_id,
            super::node_id_from_account(&account()).into_bytes()
        );
        assert_eq!(desc.node_name, b"Tesla");
        assert_eq!(
            desc.public_host.as_deref(),
            Some(b"96.233.112.201".as_slice())
        );
        assert_eq!(desc.public_port, None);
        assert!(desc.auto_mine);
        assert_eq!(desc.miners.len(), 2);
        assert_eq!(desc.miners.get(1).map(|m| m.kind), Some(MinerKind::Metal));
        assert!(desc.system_info.is_none());
        assert!(desc.runtime.is_none());
        file_round_descriptor(&chain, &filed, &params, account()).await;
        assert_eq!(chain.descriptor_calls(), 0);
    }

    #[tokio::test]
    async fn an_unregistered_account_registers_once_and_then_latches() {
        let chain = FakeChain::new(snap(), None);
        let registered = AtomicBool::new(false);
        assert!(register_round_miner(&chain, &registered).await);
        assert_eq!(chain.registration_submits(), 1);
        assert!(registered.load(Ordering::Relaxed));
        // The latch keeps later rounds off the chain entirely.
        assert!(register_round_miner(&chain, &registered).await);
        assert_eq!(chain.registration_calls(), 1);
        assert_eq!(chain.registration_submits(), 1);
    }

    #[tokio::test]
    async fn an_already_registered_account_submits_nothing() {
        let chain = FakeChain::new(snap(), None);
        chain.set_registered(true);
        let registered = AtomicBool::new(false);
        assert!(register_round_miner(&chain, &registered).await);
        assert_eq!(chain.registration_calls(), 1);
        assert_eq!(
            chain.registration_submits(),
            0,
            "a second register_miner would fail with MinerAlreadyRegistered and burn fees"
        );
        assert!(registered.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn a_failed_registration_retries_and_does_not_latch() {
        let chain = FakeChain::new(snap(), None);
        chain.set_registration_result(Err(ChainError::Unavailable("rpc down".into())));
        let registered = AtomicBool::new(false);
        assert!(!register_round_miner(&chain, &registered).await);
        assert_eq!(chain.registration_calls(), SUBMIT_ATTEMPTS as usize);
        assert!(
            !registered.load(Ordering::Relaxed),
            "a failed registration must not latch; mining cannot proceed without it"
        );
        // The next round tries again rather than assuming the miner is ready.
        chain.set_registration_result(Ok(RegistrationOutcome::Registered));
        assert!(register_round_miner(&chain, &registered).await);
        assert!(registered.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn prepare_round_registers_the_miner_before_reading_requirements() {
        let chain = FakeChain::new(snap(), None);
        let registered = AtomicBool::new(false);
        let _snap = prepare_round(
            &chain,
            None::<&HttpFaucet>,
            account(),
            &FundingParams::default(),
            no_sleep,
            &DescriptorParams::default(),
            &ProcessLatches {
                descriptor_filed: &AtomicBool::new(false),
                miner_registered: &registered,
            },
        )
        .await
        .expect("prepare_round");
        assert_eq!(chain.registration_submits(), 1);
        assert!(registered.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn prepare_round_funds_the_account_it_was_given() {
        let chain = FakeChain::new(snap(), None);
        let _snap = prepare_round(
            &chain,
            None::<&HttpFaucet>,
            account(),
            &FundingParams::default(),
            no_sleep,
            &DescriptorParams::default(),
            &ProcessLatches {
                descriptor_filed: &AtomicBool::new(false),
                miner_registered: &AtomicBool::new(false),
            },
        )
        .await
        .expect("prepare_round");
        assert_eq!(chain.take_balance_accounts(), vec![account()]);
    }

    #[tokio::test]
    async fn a_registration_failure_does_not_fail_the_startup_walk() {
        let chain = FakeChain::new(snap(), None);
        chain.set_registration_result(Err(ChainError::Unavailable("rpc down".into())));
        let registered = AtomicBool::new(false);
        let _snap = prepare_round(
            &chain,
            None::<&HttpFaucet>,
            account(),
            &FundingParams::default(),
            no_sleep,
            &DescriptorParams::default(),
            &ProcessLatches {
                descriptor_filed: &AtomicBool::new(false),
                miner_registered: &registered,
            },
        )
        .await
        .expect("startup continues so the feeder can retry");
        assert!(!registered.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn missing_node_name_files_nothing_and_names_the_key() {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_writer(Capture(Arc::clone(&buf)))
            .with_ansi(false)
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);
        let chain = FakeChain::new(snap(), None);
        let filed = AtomicBool::new(false);
        file_round_descriptor(&chain, &filed, &DescriptorParams::default(), account()).await;
        assert_eq!(chain.descriptor_calls(), 0);
        assert!(filed.load(Ordering::Relaxed));
        let text = drain(&buf);
        assert!(
            text.contains("[miner].node_name"),
            "must name the missing key, got {text:?}"
        );
    }

    #[tokio::test]
    async fn descriptor_transient_error_retries_then_marks_done() {
        let chain = FakeChain::new(snap(), None);
        chain.set_descriptor_result(Err(ChainError::Unavailable("rpc down".into())));
        let filed = AtomicBool::new(false);
        file_round_descriptor(&chain, &filed, &named_descriptor(), account()).await;
        assert_eq!(chain.descriptor_calls(), 3);
        assert!(filed.load(Ordering::Relaxed));
        file_round_descriptor(&chain, &filed, &named_descriptor(), account()).await;
        assert_eq!(chain.descriptor_calls(), 3);
    }

    #[tokio::test]
    async fn descriptor_pallet_rejection_does_not_retry() {
        let chain = FakeChain::new(snap(), None);
        chain.set_descriptor_result(Ok(DescriptorOutcome::Rejected));
        let filed = AtomicBool::new(false);
        file_round_descriptor(&chain, &filed, &named_descriptor(), account()).await;
        assert_eq!(chain.descriptor_calls(), 1);
        assert!(filed.load(Ordering::Relaxed));
    }

    #[derive(Clone)]
    struct Capture(Arc<Mutex<Vec<u8>>>);

    impl Write for Capture {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().expect("capture lock").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for Capture {
        type Writer = Self;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    fn drain(buf: &Arc<Mutex<Vec<u8>>>) -> String {
        let mut g = buf.lock().expect("capture lock");
        let text = String::from_utf8_lossy(&g).into_owned();
        g.clear();
        text
    }

    #[tokio::test]
    async fn only_descriptor_required_warns_among_pallet_errors() {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_writer(Capture(Arc::clone(&buf)))
            .with_ansi(false)
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);
        let chain = FakeChain::new(snap(), None);
        chain.set_qblock_id(Some(4));

        chain.set_participation_result(Ok(ParticipationOutcome::AlreadyDeclared));
        let mut last = None;
        declare_round_participation(&chain, &mut last, account()).await;
        let dup = drain(&buf);
        assert!(
            !dup.contains("WARN") && !dup.to_ascii_lowercase().contains("warn"),
            "DuplicateParticipation must not warn, got {dup:?}"
        );

        chain.set_qblock_id(Some(5));
        chain.set_participation_result(Ok(ParticipationOutcome::StaleQBlock));
        declare_round_participation(&chain, &mut last, account()).await;
        let stale = drain(&buf);
        assert!(
            stale.is_empty(),
            "InvalidQBlockId must not warn, got {stale:?}"
        );

        chain.set_qblock_id(Some(6));
        chain.set_participation_result(Ok(ParticipationOutcome::DescriptorMissing));
        declare_round_participation(&chain, &mut last, account()).await;
        let missing = drain(&buf);
        assert!(
            missing.contains("no descriptor"),
            "DescriptorRequired must warn, got {missing:?}"
        );
        assert!(
            missing.contains("0xabababab"),
            "DescriptorRequired must name the account, got {missing:?}"
        );
    }

    #[tokio::test]
    async fn transient_error_does_not_record_the_qblock() {
        let chain = FakeChain::new(snap(), None);
        chain.set_qblock_id(Some(7));
        chain.set_participation_result(Err(ChainError::Unavailable("rpc down".into())));
        let mut last = None;
        declare_round_participation(&chain, &mut last, account()).await;
        assert_eq!(last, None);
        chain.set_participation_result(Ok(ParticipationOutcome::Declared));
        declare_round_participation(&chain, &mut last, account()).await;
        assert_eq!(chain.participation_calls(), 2);
        assert_eq!(last, Some(8));
    }
}
