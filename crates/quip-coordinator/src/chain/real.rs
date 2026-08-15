//! Production chain client over Substrate JSON-RPC.
//!
//! Live end-to-end verification requires a running node. Offline unit tests
//! cover SCALE encode/decode, hybrid extrinsic assembly, and proof packing.
//! Methods return transport errors when no validator is reachable.

use super::extrinsic::{
    build_hybrid_signed_extrinsic, default_topology_storage_key, difficulties_storage_key,
    extrinsic_hash, hex_decode, hex_encode, job_orders_storage_key, last_proof_block_storage_key,
    load_hybrid_pair, miner_identity_bytes, miners_storage_key, node_descriptors_storage_key,
    participants_by_qblock_storage_key, qblocks_storage_key, signer_account_bytes,
    topology_curve_c_storage_key, SignedExtensionContext,
};
use super::proof_encode::{build_quantum_proof, ProofBuildContext};
use super::scale_types::{
    encode_participate_call, encode_register_miner_call, encode_set_descriptor_call,
    encode_submit_proof_call, require_set_values, CurveCScale, DifficultyConfig, JobOrderScale,
    MinerInfoScale, MinerKind, MiningSnapshotScale, NodeDescriptorV2Input, OrderStatus,
};
use super::submit::{
    classify_descriptor, classify_participation, classify_receipt, classify_registration,
    DescriptorOutcome, ParticipationOutcome, Proof, RegistrationOutcome, SubmitAction,
};
use super::transport::RpcTransport;
use super::transport_jsonrpsee::JsonrpseeTransport;
use super::watch::{parse_tx_status, TxStatus};
use super::{ChainClient, ChainError, DecayParams, JobOrder, MinerInfo, MiningSnapshot};
use crate::decay::{
    DEFAULT_BASE_MAX_ENERGY_MILLI, DEFAULT_C_EASY_MILLI, DEFAULT_C_HARD_MILLI,
    DEFAULT_C_KNEE_MILLI, EPOCH_LENGTH_BLOCKS,
};
use async_trait::async_trait;
use futures::StreamExt as _;
use parity_scale_codec::Decode;
use quantum_validation::AllowedValueSpec;
use quip_transaction_crypto::{account_id_from_public, HybridPair};
use serde_json::Value;
use sp_core::crypto::Ss58Codec;
use sp_core::Pair as _;
use std::sync::{Arc, Mutex};

/// Keys requested per `state_getKeysPaged` call when walking the order map.
///
/// The mempool holds tens of orders in practice. A page of 200 fetches them in
/// one round trip while staying far below any node response limit.
const ORDER_PAGE_SIZE: u32 = 200;

/// Production chain client (RPC + hybrid-signed submit).
pub struct RealChainClient {
    /// Validator WebSocket / HTTP RPC URLs (primary first).
    pub validators: Vec<String>,
    /// Hybrid keystore path, any substrate secret URI, or 32-byte hex seed.
    pub signer_key: String,
    /// Cached hybrid pair (loaded lazily from `signer_key`).
    pair: Mutex<Option<HybridPair>>,
    /// Last fetched mining snapshot — used by `submit_proof` to pack spins
    /// and derive the nonce. Updated by `fetch_mining_snapshot`.
    last_snapshot: Mutex<Option<MiningSnapshot>>,
    /// Last `allowed_spin` spec (full `AllowedValueSpec`, not just Set values).
    last_spin_spec: Mutex<Option<AllowedValueSpec<Vec<i32>>>>,
    /// How this client reaches the validator. Boxed so a WebAssembly build can
    /// supply a browser transport in place of the native one.
    transport: Arc<dyn RpcTransport>,
    /// Whether the last RPC round-trip reached the validator. `None` before the
    /// first call. Every RPC builds a fresh client (see `rpc_http` / `rpc_ws`),
    /// so a per-call "connected" line would run to thousands an hour; this
    /// tracks reachability so only the *transitions* are reported.
    reachable: Mutex<Option<bool>>,
    /// Kind declared on `participate`. Derived from the miners this process starts.
    participate_kind: MinerKind,
}

impl RealChainClient {
    /// Construct a client over the given validators and signer material.
    #[must_use]
    pub fn new(validators: Vec<String>, signer_key: String, participate_kind: MinerKind) -> Self {
        Self {
            validators,
            signer_key,
            pair: Mutex::new(None),
            last_snapshot: Mutex::new(None),
            last_spin_spec: Mutex::new(None),
            transport: Arc::new(JsonrpseeTransport),
            reachable: Mutex::new(None),
            participate_kind,
        }
    }

    /// Construct a client over a caller-supplied transport.
    #[must_use]
    pub fn with_transport(
        validators: Vec<String>,
        signer_key: String,
        participate_kind: MinerKind,
        transport: Arc<dyn RpcTransport>,
    ) -> Self {
        let mut c = Self::new(validators, signer_key, participate_kind);
        c.transport = transport;
        c
    }

    /// Record the outcome of one RPC round-trip and log reachability changes.
    ///
    /// A validator that goes away (or comes back) is the single most common
    /// reason a coordinator stops mining, and v0.2.1 narrated it on every
    /// reconnect. Reporting only the edges keeps that signal without the
    /// per-call volume.
    fn note_reachability(&self, url: &str, outcome: Result<(), &ChainError>) {
        let now = outcome.is_ok();
        // Poisoned lock: reachability is advisory, never fail an RPC over it.
        let Ok(mut guard) = self.reachable.lock() else {
            return;
        };
        if *guard == Some(now) {
            return;
        }
        match outcome {
            Ok(()) => tracing::info!(url = %url, "validator RPC reachable"),
            Err(e) => {
                tracing::warn!(url = %url, error = %e, "validator RPC unreachable");
            }
        }
        *guard = Some(now);
    }

    fn pair(&self) -> Result<HybridPair, ChainError> {
        let mut guard = self
            .pair
            .lock()
            .map_err(|e| ChainError::Unavailable(e.to_string()))?;
        if guard.is_none() {
            let p = load_hybrid_pair(&self.signer_key)
                .map_err(|e| ChainError::Unavailable(format!("load signer: {e}")))?;
            *guard = Some(p);
        }
        // HybridPair is not Clone in all builds — re-load from key each time
        // if needed. Prefer re-load for simplicity.
        load_hybrid_pair(&self.signer_key)
            .map_err(|e| ChainError::Unavailable(format!("load signer: {e}")))
    }

    fn primary_url(&self) -> Result<&str, ChainError> {
        self.validators
            .first()
            .map(String::as_str)
            .ok_or_else(|| ChainError::Unavailable("no validators configured".into()))
    }

    /// Issue `method` against the configured validators in order, returning the
    /// first success.
    ///
    /// `validators` is documented as an ordered failover list, and the default
    /// pair leads with a container-network name that does not resolve on a host
    /// install. Trying only the first entry makes that default unusable off
    /// Docker, so every entry gets a turn before the call is called failed.
    async fn rpc_call(&self, method: &str, params: Value) -> Result<Value, ChainError> {
        if self.validators.is_empty() {
            return Err(ChainError::Unavailable("no validators configured".into()));
        }
        let mut last: Option<ChainError> = None;
        for url in &self.validators {
            let out = self.transport.request(url, method, params.clone()).await;
            tracing::trace!(url = %url, method = %method, ok = out.is_ok(), "rpc call");
            match out {
                Ok(v) => {
                    // Lock only after the await: `reachable` is a std Mutex and
                    // must never be held across a suspension point.
                    self.note_reachability(url, Ok(()));
                    return Ok(v);
                }
                Err(e) => {
                    tracing::debug!(url = %url, method = %method, error = %e, "validator failed; trying next");
                    last = Some(e);
                }
            }
        }
        // Every endpoint failed: report against the primary, which is the one an
        // operator will look at first.
        let err =
            last.unwrap_or_else(|| ChainError::Unavailable("no validators configured".into()));
        if let Some(primary) = self.validators.first() {
            self.note_reachability(primary, Err(&err));
        }
        Err(err)
    }

    /// Raw `state_getRuntimeVersion` response, for the startup compatibility
    /// check in [`super::preflight`].
    ///
    /// # Errors
    /// Returns a transport error when the validator cannot be reached.
    pub(crate) async fn runtime_version_raw(&self) -> Result<Value, ChainError> {
        self.rpc_call("state_getRuntimeVersion", Value::Array(vec![]))
            .await
    }

    /// Raw `system_health` response, for the startup sync gate in
    /// [`super::sync`].
    ///
    /// # Errors
    /// Returns a transport error when the validator cannot be reached.
    pub(crate) async fn system_health_raw(&self) -> Result<Value, ChainError> {
        self.rpc_call("system_health", Value::Array(vec![])).await
    }

    /// Raw `system_syncState` response: the block heights the sync gate reports.
    ///
    /// # Errors
    /// Returns a transport error when the validator cannot be reached, or when
    /// it does not serve this method.
    pub(crate) async fn sync_state_raw(&self) -> Result<Value, ChainError> {
        self.rpc_call("system_syncState", Value::Array(vec![]))
            .await
    }

    /// Read + SCALE-decode a storage value at `at_hex`. `Ok(None)` when the key
    /// is unset (null result).
    async fn read_storage<T: Decode>(
        &self,
        key: &[u8],
        at_hex: &str,
    ) -> Result<Option<T>, ChainError> {
        let raw = self
            .rpc_call(
                "state_getStorage",
                Value::Array(vec![
                    Value::String(hex_encode(key)),
                    Value::String(at_hex.to_string()),
                ]),
            )
            .await?;
        let Some(hex) = raw.as_str() else {
            return Ok(None);
        };
        let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
        T::decode(&mut &bytes[..])
            .map(Some)
            .map_err(|e| ChainError::Decode(e.to_string()))
    }

    /// Read `QuantumPow.DefaultTopology` at the current head.
    ///
    /// `Ok(None)` means no topology is registered, which is the state a fresh
    /// chain starts in and the only state the seed path accepts.
    ///
    /// # Errors
    /// Returns a transport error when the validator cannot be reached, or a
    /// decode error when the stored value is not a 32-byte hash.
    pub(crate) async fn default_topology(&self) -> Result<Option<[u8; 32]>, ChainError> {
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let at = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?;
        self.read_storage::<[u8; 32]>(&default_topology_storage_key(), at)
            .await
    }

    /// Nonce, genesis hash, and runtime versions for a signed extrinsic.
    async fn signed_extension_context(
        &self,
        pair: &HybridPair,
    ) -> Result<SignedExtensionContext, ChainError> {
        let account_ss58 = account_id_from_public(&pair.public()).to_ss58check();
        let nonce_val = self
            .rpc_call(
                "system_accountNextIndex",
                Value::Array(vec![Value::String(account_ss58)]),
            )
            .await?;
        let account_nonce = u32::try_from(
            nonce_val
                .as_u64()
                .ok_or_else(|| ChainError::Decode("system_accountNextIndex not a u64".into()))?,
        )
        .map_err(|_| ChainError::Decode("account nonce exceeds u32".into()))?;

        let genesis = self
            .rpc_call(
                "chain_getBlockHash",
                Value::Array(vec![Value::Number(0.into())]),
            )
            .await?;
        let genesis_hex = genesis
            .as_str()
            .ok_or_else(|| ChainError::Decode("genesis hash not a string".into()))?;
        let genesis_bytes = hex_decode(genesis_hex).map_err(ChainError::Decode)?;
        let mut genesis_hash = [0u8; 32];
        if genesis_bytes.len() != 32 {
            return Err(ChainError::Decode("genesis hash length".into()));
        }
        genesis_hash.copy_from_slice(&genesis_bytes);

        let rv = self
            .rpc_call("state_getRuntimeVersion", Value::Array(vec![]))
            .await?;
        let spec_version = u32::try_from(
            rv.get("specVersion")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    ChainError::Decode("runtime specVersion missing/not a u64".into())
                })?,
        )
        .map_err(|_| ChainError::Decode("specVersion exceeds u32".into()))?;
        let transaction_version = u32::try_from(
            rv.get("transactionVersion")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    ChainError::Decode("runtime transactionVersion missing/not a u64".into())
                })?,
        )
        .map_err(|_| ChainError::Decode("transactionVersion exceeds u32".into()))?;

        Ok(SignedExtensionContext {
            account_nonce,
            genesis_hash,
            spec_version,
            transaction_version,
            tip: 0,
        })
    }

    /// Hybrid-sign `call`, submit it, and confirm the result from chain state.
    ///
    /// The status subscription reports pool and inclusion progress only. It
    /// cannot say whether the dispatch inside the block succeeded, because that
    /// lives in the block's events and decoding those needs runtime metadata.
    /// So each call names the storage entry its own success writes. Inclusion
    /// is confirmed from the block body. Both reads are metadata-free.
    pub(crate) async fn submit_signed_call(
        &self,
        call: &[u8],
        confirmation: Confirmation,
    ) -> Result<SignedCallOutcome, ChainError> {
        let pair = self.pair()?;
        let signed_ctx = self.signed_extension_context(&pair).await?;
        let ext = build_hybrid_signed_extrinsic(&pair, call, &signed_ctx);
        let want_hash = extrinsic_hash(&ext);

        let url = self.primary_url()?.to_string();
        let mut stream = match self
            .transport
            .subscribe(
                &url,
                "author_submitAndWatchExtrinsic",
                Value::Array(vec![Value::String(hex_encode(&ext))]),
                "author_unwatchExtrinsic",
            )
            .await
        {
            Ok(s) => s,
            // A node that answers the submit with a rejection is delivering a
            // verdict about this extrinsic, not reporting a transport failure.
            // Resubmitting the same bytes cannot change it. Returning an error
            // here would let the caller read a permanent rejection as transient
            // and retry the same proof forever, which is how a lost win hides.
            Err(ChainError::Submit(message)) => return Ok(SignedCallOutcome::Invalid { message }),
            Err(e) => return Err(e),
        };

        while let Some(item) = stream.next().await {
            let value = item?;
            match parse_tx_status(&value) {
                TxStatus::InBlock(block) | TxStatus::Finalized(block) => {
                    return self
                        .confirm_in_block(&block, &want_hash, confirmation)
                        .await;
                }
                TxStatus::Invalid(message) => return Ok(SignedCallOutcome::Invalid { message }),
                TxStatus::Dropped(message) => return Ok(SignedCallOutcome::Dropped { message }),
                TxStatus::Other(s) => {
                    tracing::debug!(status = %s, "unmodelled transaction status");
                }
                TxStatus::Ready | TxStatus::Broadcast | TxStatus::Future => {}
            }
        }
        Err(ChainError::Unavailable(
            "transaction status stream ended before inclusion".into(),
        ))
    }

    /// Read the block that claimed to include our extrinsic and decide.
    async fn confirm_in_block(
        &self,
        block_hex: &str,
        want_hash: &[u8; 32],
        confirmation: Confirmation,
    ) -> Result<SignedCallOutcome, ChainError> {
        let included = self.block_contains(block_hex, want_hash).await?;
        let confirmed = self.confirmation_present(block_hex, &confirmation).await?;
        match classify_state_outcome(included, confirmed) {
            StateOutcome::Won => Ok(SignedCallOutcome::Success {
                block: block_hex.to_string(),
            }),
            StateOutcome::IncludedButNotWon => {
                let (error, lost_race) = self.explain_failure(block_hex, &confirmation).await;
                Ok(SignedCallOutcome::DispatchFailed { error, lost_race })
            }
            StateOutcome::NotIncluded => Ok(SignedCallOutcome::Dropped {
                message: format!("extrinsic absent from block {block_hex}"),
            }),
        }
    }

    /// Is our extrinsic in this block?
    ///
    /// `chain_getBlock` is in the safe RPC set. Each extrinsic comes back as a
    /// hex blob, so inclusion is a hash comparison and needs no metadata.
    async fn block_contains(
        &self,
        block_hex: &str,
        want_hash: &[u8; 32],
    ) -> Result<bool, ChainError> {
        let body = self
            .rpc_call(
                "chain_getBlock",
                Value::Array(vec![Value::String(block_hex.to_string())]),
            )
            .await?;
        let Some(exts) = body
            .get("block")
            .and_then(|b| b.get("extrinsics"))
            .and_then(Value::as_array)
        else {
            return Err(ChainError::Decode(
                "chain_getBlock has no block.extrinsics array".into(),
            ));
        };
        for e in exts {
            let Some(hex) = e.as_str() else { continue };
            let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
            if extrinsic_hash(&bytes) == *want_hash {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Is the storage entry this call should have written present at `block_hex`?
    async fn confirmation_present(
        &self,
        block_hex: &str,
        confirmation: &Confirmation,
    ) -> Result<bool, ChainError> {
        match *confirmation {
            Confirmation::ProofWin { account } => self.proof_win_at(block_hex, &account).await,
            Confirmation::Descriptor { account } => {
                self.storage_value_present(&node_descriptors_storage_key(&account), block_hex)
                    .await
            }
            Confirmation::MinerRegistered { account } => {
                self.storage_value_present(&miners_storage_key(&account), block_hex)
                    .await
            }
            Confirmation::Participation { qblock_id, account } => {
                self.storage_value_present(
                    &participants_by_qblock_storage_key(qblock_id, &account),
                    block_hex,
                )
                .await
            }
            Confirmation::DefaultTopology => {
                self.storage_value_present(&default_topology_storage_key(), block_hex)
                    .await
            }
            Confirmation::Difficulty { topology_hash } => {
                self.storage_value_present(&difficulties_storage_key(&topology_hash), block_hex)
                    .await
            }
        }
    }

    /// Which account won the qblock recorded at `block_hex`, if any?
    ///
    /// `QuantumPow::QBlocks` is keyed by block number and its value begins with
    /// the winning account, so this answers the question exactly. `None` means
    /// no proof was accepted in this block at all.
    async fn qblock_winner_at(&self, block_hex: &str) -> Result<Option<[u8; 32]>, ChainError> {
        let header = self
            .rpc_call(
                "chain_getHeader",
                Value::Array(vec![Value::String(block_hex.to_string())]),
            )
            .await?;
        let block_number = u32::try_from(parse_block_number(&header)?)
            .map_err(|_| ChainError::Decode("block number exceeds u32".into()))?;

        let key = qblocks_storage_key(block_number);
        let raw = self
            .rpc_call(
                "state_getStorage",
                Value::Array(vec![
                    Value::String(hex_encode(&key)),
                    Value::String(block_hex.to_string()),
                ]),
            )
            .await?;
        let Some(hex) = raw.as_str() else {
            return Ok(None);
        };
        let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
        // `miner` is the first field of `QBlock`, so it occupies the leading 32
        // bytes. Reading only those survives the migrations that appended
        // fields to the end of the struct.
        let Some(miner) = bytes.get(..32) else {
            return Err(ChainError::Decode(format!(
                "QBlocks value is {} bytes, too short to hold an account",
                bytes.len()
            )));
        };
        let mut account = [0u8; 32];
        account.copy_from_slice(miner);
        Ok(Some(account))
    }

    /// Did this account win the qblock recorded at `block_hex`?
    ///
    /// Comparing `LastProofBlock` instead would report a false success whenever
    /// another miner won the same block.
    async fn proof_win_at(
        &self,
        block_hex: &str,
        our_account: &[u8; 32],
    ) -> Result<bool, ChainError> {
        Ok(self.qblock_winner_at(block_hex).await? == Some(*our_account))
    }

    /// Is `account` already in `QuantumPow.Miners` at the current head?
    async fn miner_is_registered(&self, account: &[u8; 32]) -> Result<bool, ChainError> {
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let at = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?;
        self.storage_value_present(&miners_storage_key(account), at)
            .await
    }

    /// Is any value stored at `key` in the block named by `at_hex`?
    async fn storage_value_present(&self, key: &[u8], at_hex: &str) -> Result<bool, ChainError> {
        let raw = self
            .rpc_call(
                "state_getStorage",
                Value::Array(vec![
                    Value::String(hex_encode(key)),
                    Value::String(at_hex.to_string()),
                ]),
            )
            .await?;
        Ok(raw.as_str().is_some())
    }

    /// Best-effort reason for a dispatch that was included but did not take
    /// effect.
    ///
    /// The pallet's own error variant lives in this block's `System.Events`,
    /// and decoding those needs runtime metadata this client does not carry.
    /// Replaying the extrinsic through `system_dryRun` cannot stand in for it:
    /// at the head the nonce is already spent, and at the parent block the
    /// replay omits whatever was ordered ahead of us inside the same block,
    /// which is the very thing that beats a proof. So this reads the storage
    /// the call competes for and reports what is actually there.
    ///
    /// The caller must treat this as log text only.
    async fn explain_failure(
        &self,
        block_hex: &str,
        confirmation: &Confirmation,
    ) -> (String, bool) {
        // Only a proof competes with other miners, so only a proof can lose a
        // race. Every other call fails on its own merits.
        if let Confirmation::ProofWin { account } = confirmation {
            return match self.qblock_winner_at(block_hex).await {
                Ok(winner) => explain_proof_failure(account, winner),
                Err(e) => (
                    format!("dispatch failed and the qblock winner could not be read: {e}"),
                    false,
                ),
            };
        }
        let text = match confirmation {
            // Handled above.
            Confirmation::ProofWin { .. } => unreachable!(),
            Confirmation::MinerRegistered { .. } => {
                "dispatch failed and QuantumPow.Miners still has no entry for this account; \
                 the deposit may be unaffordable"
                    .to_string()
            }
            Confirmation::Descriptor { .. } => {
                "dispatch failed and MinerRegistry.NodeDescriptors still has no entry for this \
                 account"
                    .to_string()
            }
            Confirmation::Participation { qblock_id, .. } => format!(
                "dispatch failed and this account is absent from the participant set for qblock \
                 {qblock_id}"
            ),
            Confirmation::DefaultTopology => {
                "dispatch failed and QuantumPow.DefaultTopology is still unset".to_string()
            }
            Confirmation::Difficulty { .. } => {
                "dispatch failed and QuantumPow.Difficulties has no entry for this topology"
                    .to_string()
            }
        };
        (text, false)
    }
}

/// Turn the qblock winner recorded at a block into a reason and a race verdict.
///
/// Split from the RPC read so the verdict can be tested without a chain. The
/// returned flag is true only when the qblock went to somebody else, which is
/// the one failure that says nothing is wrong with our proof.
fn explain_proof_failure(our_account: &[u8; 32], winner: Option<[u8; 32]>) -> (String, bool) {
    match winner {
        Some(w) if w != *our_account => (
            format!(
                "lost the qblock to miner {}; this proof was valid but arrived behind theirs",
                hex_encode(&w)
            ),
            true,
        ),
        Some(_) => (
            "we hold the qblock but the win was not observed at this block; treat as a read \
             race, not a rejection"
                .to_string(),
            false,
        ),
        None => (
            "no qblock was recorded at this block, so the pallet rejected the proof itself; \
             check diversity, energy, and registration"
                .to_string(),
            false,
        ),
    }
}

/// What chain state proves a submitted call succeeded.
///
/// The status subscription reports pool and inclusion progress only. It cannot
/// say whether the dispatch inside the block succeeded, because that lives in
/// the block's events and decoding those needs runtime metadata. So each call
/// names the storage entry its own success writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Confirmation {
    /// `QuantumPow.QBlocks[block_number].miner` equals this account.
    ProofWin { account: [u8; 32] },
    /// `MinerRegistry.NodeDescriptors[account]` is present.
    Descriptor { account: [u8; 32] },
    /// `QuantumPow.Miners[account]` is present.
    MinerRegistered { account: [u8; 32] },
    /// `MinerRegistry.ParticipantsByQBlock[qblock_id][account]` is present.
    Participation { qblock_id: u64, account: [u8; 32] },
    /// `QuantumPow.DefaultTopology` is present. Used by seed-chain.
    DefaultTopology,
    /// `QuantumPow.Difficulties[topology_hash]` is present. Used by seed-chain.
    Difficulty { topology_hash: [u8; 32] },
}

/// On-chain result of a hybrid-signed extrinsic, before pallet-specific classify.
pub(crate) enum SignedCallOutcome {
    /// Included and the dispatch succeeded.
    Success { block: String },
    /// Included but the dispatch failed.
    DispatchFailed {
        error: String,
        /// The call was sound and another miner simply took the qblock first.
        /// Distinguished so a lost race is not reported as a rejection.
        lost_race: bool,
    },
    /// The transaction pool rejected the extrinsic.
    Invalid { message: String },
    /// Dropped or errored before inclusion.
    Dropped { message: String },
}

#[async_trait]
impl crate::funding::BalanceSource for RealChainClient {
    async fn free_balance(&self, account: [u8; 32]) -> Result<u128, String> {
        let key = super::account::system_account_storage_key(&account);
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await
            .map_err(|e| e.to_string())?;
        let at = head
            .as_str()
            .ok_or_else(|| "chain_getBlockHash did not return a string".to_string())?;
        let raw = self
            .rpc_call(
                "state_getStorage",
                Value::Array(vec![
                    Value::String(hex_encode(&key)),
                    Value::String(at.to_string()),
                ]),
            )
            .await
            .map_err(|e| e.to_string())?;
        // A null result means the account has never been touched on chain,
        // which is a zero balance rather than a read failure.
        let Some(hex) = raw.as_str() else {
            return Ok(0);
        };
        let bytes = hex_decode(hex)?;
        super::account::free_from_account_bytes(&bytes)
    }
}

#[async_trait]
impl ChainClient for RealChainClient {
    async fn fetch_mining_snapshot(
        &self,
        at: Option<[u8; 32]>,
        _miner_account: [u8; 32],
        topology_hash: Option<[u8; 32]>,
    ) -> Result<Option<MiningSnapshot>, ChainError> {
        let block_hash = if let Some(h) = at {
            h
        } else {
            let head = self
                .rpc_call("chain_getBlockHash", Value::Array(vec![]))
                .await?;
            let hex = head
                .as_str()
                .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?;
            let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
            if bytes.len() != 32 {
                return Err(ChainError::Decode(format!(
                    "block hash len {}",
                    bytes.len()
                )));
            }
            let mut h = [0u8; 32];
            h.copy_from_slice(&bytes);
            h
        };

        // Parameter: Option<H256> SCALE-encoded.
        let param = match topology_hash {
            None => vec![0u8],
            Some(th) => {
                let mut p = vec![1u8];
                p.extend_from_slice(&th);
                p
            }
        };

        let result = self
            .rpc_call(
                "state_call",
                Value::Array(vec![
                    Value::String("QuantumPowApi_mining_snapshot".into()),
                    Value::String(hex_encode(&param)),
                    Value::String(hex_encode(&block_hash)),
                ]),
            )
            .await?;

        let Some(hex) = result.as_str() else {
            // Null result → no topology / empty response.
            return Ok(None);
        };
        let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
        let decoded: Option<MiningSnapshotScale> =
            Decode::decode(&mut &bytes[..]).map_err(|e| ChainError::Decode(e.to_string()))?;
        let Some(scale) = decoded else {
            return Ok(None);
        };

        // Block number from header (snapshot has no block_number field).
        let header = self
            .rpc_call(
                "chain_getHeader",
                Value::Array(vec![Value::String(hex_encode(&block_hash))]),
            )
            .await?;
        let block_number = parse_block_number(&header)?;

        let snap = MiningSnapshot {
            head_hash: block_hash,
            last_proof_block_hash: scale.last_proof_block_hash.0,
            topology_hash: scale.topology_hash.0.to_vec(),
            nodes: scale.nodes,
            edges: scale.edges,
            allowed_h_milli: require_set_values(&scale.allowed_h_values)
                .map_err(ChainError::Decode)?,
            allowed_j_milli: require_set_values(&scale.allowed_j_values)
                .map_err(ChainError::Decode)?,
            allowed_spin_milli: require_set_values(&scale.allowed_spin_values)
                .map_err(ChainError::Decode)?,
            min_solutions: scale.difficulty.min_solutions,
            max_energy_milli: scale.difficulty.max_energy_milli,
            min_diversity_milli: scale.difficulty.min_diversity_milli,
            block_number,
        };

        if let Ok(mut g) = self.last_snapshot.lock() {
            *g = Some(snap.clone());
        }
        if let Ok(mut g) = self.last_spin_spec.lock() {
            *g = Some(scale.allowed_spin_values);
        }
        Ok(Some(snap))
    }

    async fn fetch_latest_qblock_id(&self) -> Result<Option<u64>, ChainError> {
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let block_hash = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))
            .and_then(|hex| hex_decode(hex).map_err(ChainError::Decode))?;

        // `QuantumPowApi_latest_qblock_id()` takes no args → empty SCALE params.
        let result = self
            .rpc_call(
                "state_call",
                Value::Array(vec![
                    Value::String("QuantumPowApi_latest_qblock_id".into()),
                    Value::String(hex_encode(&[])),
                    Value::String(hex_encode(&block_hash)),
                ]),
            )
            .await?;

        let Some(hex) = result.as_str() else {
            return Ok(None);
        };
        let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
        Decode::decode(&mut &bytes[..]).map_err(|e| ChainError::Decode(e.to_string()))
    }

    async fn fetch_miner_info(&self, account: [u8; 32]) -> Result<Option<MinerInfo>, ChainError> {
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let at = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?;
        let scale: Option<MinerInfoScale> =
            self.read_storage(&miners_storage_key(&account), at).await?;
        Ok(scale.map(|s| MinerInfo {
            registered_at: u64::from(s.registered_at),
            deposit: s.deposit,
            proofs_submitted: u64::from(s.proofs_submitted),
            proofs_won: u64::from(s.proofs_won),
            rewards_earned: s.rewards_earned,
        }))
    }

    async fn fetch_decay_params(
        &self,
        topology_hash: [u8; 32],
    ) -> Result<Option<DecayParams>, ChainError> {
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let at = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?
            .to_string();

        // Base (un-decayed) difficulty; unset topology → chain genesis default.
        let base: Option<DifficultyConfig> = self
            .read_storage(&difficulties_storage_key(&topology_hash), &at)
            .await?;
        // LastProofBlock is a u32 BlockNumber (ValueQuery → 0 when unset).
        let last_proof_block: u32 = self
            .read_storage(&last_proof_block_storage_key(), &at)
            .await?
            .unwrap_or(0);
        // Per-topology curve override; unset → the runtime CurveC* constants.
        let curve: Option<CurveCScale> = self
            .read_storage(&topology_curve_c_storage_key(&topology_hash), &at)
            .await?;
        let (c_easy_milli, c_knee_milli, c_hard_milli) = curve.map_or(
            (
                DEFAULT_C_EASY_MILLI,
                DEFAULT_C_KNEE_MILLI,
                DEFAULT_C_HARD_MILLI,
            ),
            |c| (c.easy_milli, c.knee_milli, c.hard_milli),
        );

        Ok(Some(DecayParams {
            base_max_energy_milli: base
                .map_or(DEFAULT_BASE_MAX_ENERGY_MILLI, |d| d.max_energy_milli),
            last_proof_block: u64::from(last_proof_block),
            epoch_length: EPOCH_LENGTH_BLOCKS,
            c_easy_milli,
            c_knee_milli,
            c_hard_milli,
        }))
    }

    async fn fetch_mempool_orders(
        &self,
        _miner_account: [u8; 32],
    ) -> Result<Vec<JobOrder>, ChainError> {
        // Discover order ids by walking the JobOrders map at head, then
        // storage-read each order. Without a live node this returns transport
        // errors; with a node, empty open-order sets yield Ok(vec![]).
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let head_hex = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?;

        let head_bytes = hex_decode(head_hex).map_err(ChainError::Decode)?;
        if head_bytes.len() != 32 {
            return Err(ChainError::Decode(format!(
                "head hash is {} bytes, expected 32",
                head_bytes.len()
            )));
        }

        // Walk the JobOrders map rather than decoding JobProposed events. The
        // map holds every open order, not only those proposed in the head
        // block, and it needs no runtime metadata.
        let prefix = hex_encode(&super::orders::job_orders_prefix());
        let mut order_ids: Vec<u64> = Vec::new();
        let mut start_key: Option<String> = None;
        loop {
            let mut params = vec![
                Value::String(prefix.clone()),
                Value::Number(ORDER_PAGE_SIZE.into()),
            ];
            // state_getKeysPaged takes the previous page's last key as the
            // resume point. The block hash is always the final argument, so a
            // missing start key must still be sent as null.
            params.push(match &start_key {
                Some(k) => Value::String(k.clone()),
                None => Value::Null,
            });
            params.push(Value::String(head_hex.to_string()));

            let page = self
                .rpc_call("state_getKeysPaged", Value::Array(params))
                .await?;
            let Some(keys) = page.as_array() else {
                break;
            };
            if keys.is_empty() {
                break;
            }
            for k in keys {
                let Some(hex) = k.as_str() else { continue };
                let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
                if let Some(oid) = super::orders::order_id_from_key(&bytes) {
                    order_ids.push(oid);
                }
            }
            let full_page = keys.len() == ORDER_PAGE_SIZE as usize;
            start_key = keys.last().and_then(|k| k.as_str()).map(String::from);
            if !full_page {
                break;
            }
        }

        let mut orders = Vec::new();
        for oid in order_ids {
            let key = job_orders_storage_key(oid);
            let raw = self
                .rpc_call(
                    "state_getStorage",
                    Value::Array(vec![
                        Value::String(hex_encode(&key)),
                        Value::String(head_hex.to_string()),
                    ]),
                )
                .await?;
            let Some(hex) = raw.as_str() else {
                continue;
            };
            let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
            let scale = JobOrderScale::decode(&mut &bytes[..])
                .map_err(|e| ChainError::Decode(format!("JobOrder {oid}: {e}")))?;
            if scale.status != OrderStatus::Opened {
                continue;
            }
            orders.push(JobOrder {
                order_id: oid.to_le_bytes().to_vec(),
                nodes: scale.ising_params.nodes,
                edges: scale.ising_params.edges,
                h_milli: scale.ising_params.h_values,
                j_milli: scale.ising_params.j_values,
                min_energy_milli: scale.ising_params.min_energy_milli,
                min_diversity_milli: scale.ising_params.min_diversity_milli,
                min_solutions: scale.ising_params.min_solutions,
                // Convert deadline_blocks → a soft ms deadline using a 6s
                // block time estimate. Callers that need exact expiry should
                // re-query chain head.
                deadline_ms: u64::from(scale.timing.deadline_blocks).saturating_mul(6_000),
            });
        }
        Ok(orders)
    }

    async fn submit_proof(&self, proof: &Proof) -> Result<SubmitAction, ChainError> {
        let pair = self.pair()?;
        let snap = self
            .last_snapshot
            .lock()
            .map_err(|e| ChainError::Unavailable(e.to_string()))?
            .clone()
            .ok_or_else(|| {
                ChainError::Unavailable(
                    "submit_proof requires a prior fetch_mining_snapshot".into(),
                )
            })?;
        let spin_spec = self
            .last_spin_spec
            .lock()
            .map_err(|e| ChainError::Unavailable(e.to_string()))?
            .clone()
            .unwrap_or_else(|| AllowedValueSpec::Set(snap.allowed_spin_milli.clone()));

        let salt = extract_salt(proof).ok_or_else(|| {
            ChainError::Submit(
                "proof.salt must be 32 bytes for live submit_proof (session does not yet thread salt through Job)".into(),
            )
        })?;

        let mut topo = [0u8; 32];
        if snap.topology_hash.len() == 32 {
            topo.copy_from_slice(&snap.topology_hash);
        }

        let ctx = ProofBuildContext {
            topology_hash: topo,
            last_proof_block_hash: snap.last_proof_block_hash,
            miner_identity: miner_identity_bytes(&pair),
            salt,
            num_nodes: snap.nodes.len(),
            allowed_spin: spin_spec,
        };
        let quantum = build_quantum_proof(proof, &ctx)
            .map_err(|e| ChainError::Submit(format!("encode proof: {e}")))?;
        let call = encode_submit_proof_call(&quantum);
        let account = signer_account_bytes(&pair);

        match self
            .submit_signed_call(&call, Confirmation::ProofWin { account })
            .await?
        {
            SignedCallOutcome::Success { block } => {
                tracing::info!(block = %block, "proof included and dispatched successfully");
                Ok(SubmitAction::Success)
            }
            SignedCallOutcome::DispatchFailed {
                error,
                lost_race: true,
            } => {
                tracing::info!(reason = %error, "proof included but another miner took the qblock");
                Ok(SubmitAction::StopRoundStale)
            }
            SignedCallOutcome::DispatchFailed { error, .. } => {
                tracing::warn!(error = %error, "proof included but ExtrinsicFailed");
                Ok(classify_receipt(Some(&error)))
            }
            SignedCallOutcome::Invalid { message } => {
                tracing::warn!(%message, "proof rejected as invalid before inclusion");
                Ok(classify_receipt(Some(&message)))
            }
            SignedCallOutcome::Dropped { message } => {
                tracing::warn!(%message, "proof dropped by node before inclusion");
                Ok(SubmitAction::Retry)
            }
        }
    }

    async fn ensure_miner_registered(&self) -> Result<RegistrationOutcome, ChainError> {
        let account = signer_account_bytes(&self.pair()?);
        if self.miner_is_registered(&account).await? {
            return Ok(RegistrationOutcome::AlreadyRegistered);
        }
        let call = encode_register_miner_call();
        match self
            .submit_signed_call(&call, Confirmation::MinerRegistered { account })
            .await?
        {
            SignedCallOutcome::Success { .. } => Ok(RegistrationOutcome::Registered),
            SignedCallOutcome::DispatchFailed { error, .. }
            | SignedCallOutcome::Invalid { message: error } => {
                classify_registration(Some(&error)).ok_or(ChainError::Submit(error))
            }
            SignedCallOutcome::Dropped { message } => Err(ChainError::Submit(message)),
        }
    }

    async fn file_descriptor(
        &self,
        descriptor: &NodeDescriptorV2Input,
    ) -> Result<DescriptorOutcome, ChainError> {
        let call = encode_set_descriptor_call(descriptor);
        let account = signer_account_bytes(&self.pair()?);
        match self
            .submit_signed_call(&call, Confirmation::Descriptor { account })
            .await?
        {
            SignedCallOutcome::Success { .. } => Ok(DescriptorOutcome::Filed),
            SignedCallOutcome::DispatchFailed { error, .. }
            | SignedCallOutcome::Invalid { message: error } => {
                classify_descriptor(Some(&error)).ok_or(ChainError::Submit(error))
            }
            SignedCallOutcome::Dropped { message } => Err(ChainError::Submit(message)),
        }
    }

    async fn declare_participation(
        &self,
        qblock_id: u64,
    ) -> Result<ParticipationOutcome, ChainError> {
        let call = encode_participate_call(qblock_id, self.participate_kind, None);
        let account = signer_account_bytes(&self.pair()?);
        match self
            .submit_signed_call(&call, Confirmation::Participation { qblock_id, account })
            .await?
        {
            SignedCallOutcome::Success { .. } => Ok(ParticipationOutcome::Declared),
            SignedCallOutcome::DispatchFailed { error, .. }
            | SignedCallOutcome::Invalid { message: error } => {
                classify_participation(Some(&error)).ok_or(ChainError::Submit(error))
            }
            SignedCallOutcome::Dropped { message } => Err(ChainError::Submit(message)),
        }
    }
}

fn extract_salt(proof: &Proof) -> Option<[u8; 32]> {
    if proof.salt.len() != 32 {
        return None;
    }
    let mut s = [0u8; 32];
    s.copy_from_slice(&proof.salt);
    Some(s)
}

/// What chain state says about a submitted extrinsic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StateOutcome {
    /// Our extrinsic is in the block and the expected storage write is present.
    Won,
    /// Our extrinsic is in the block but the expected storage write is absent.
    /// The dispatch failed, and state does not say why.
    IncludedButNotWon,
    /// Our extrinsic is not in the block.
    NotIncluded,
}

/// Fold the two state reads into an outcome.
///
/// The second argument is true when the storage entry this call should have
/// written is present. That is not the same as winning a quantum block. A
/// descriptor or a participation write never touches `QBlocks`. Confirmation
/// only counts when our own extrinsic is in the block. Another account's
/// extrinsic can write the same entry, and claiming that would mark a failed
/// call as successful.
const fn classify_state_outcome(included: bool, won_here: bool) -> StateOutcome {
    match (included, won_here) {
        (true, true) => StateOutcome::Won,
        (true, false) => StateOutcome::IncludedButNotWon,
        (false, _) => StateOutcome::NotIncluded,
    }
}

fn parse_block_number(header: &Value) -> Result<u64, ChainError> {
    let number = header
        .get("number")
        .ok_or_else(|| ChainError::Decode("header missing number".into()))?;
    if let Some(n) = number.as_u64() {
        return Ok(n);
    }
    if let Some(s) = number.as_str() {
        let s = s.strip_prefix("0x").unwrap_or(s);
        return u64::from_str_radix(s, 16).map_err(|e| ChainError::Decode(e.to_string()));
    }
    Err(ChainError::Decode("header.number unparseable".into()))
}

#[cfg(test)]
mod tests {
    use super::super::transport_jsonrpsee::{
        rpc_request, RPC_CONNECT_TIMEOUT, RPC_REQUEST_TIMEOUT,
    };

    /// Losing the qblock to another miner is the one included-but-failed case
    /// that must not be reported as a rejection: the proof itself was sound.
    #[test]
    fn only_a_rival_winner_counts_as_a_lost_race() {
        let ours = [1u8; 32];
        let rival = [2u8; 32];

        let (text, lost_race) = explain_proof_failure(&ours, Some(rival));
        assert!(lost_race, "a rival winner is a lost race");
        assert!(
            text.contains(&hex_encode(&rival)),
            "the reason must name the winner so the loss can be verified on chain, got: {text}"
        );

        // No qblock at all means the pallet threw the proof out. That is a real
        // rejection and must stay distinguishable from a lost race.
        let (_, lost_race) = explain_proof_failure(&ours, None);
        assert!(!lost_race, "an empty qblock slot is a rejection");

        // Seeing our own account means the read raced the write, not a loss.
        let (_, lost_race) = explain_proof_failure(&ours, Some(ours));
        assert!(!lost_race, "winning it ourselves is not a lost race");
    }
    use super::{
        classify_state_outcome, explain_proof_failure, hex_encode, RealChainClient, StateOutcome,
    };
    use crate::chain::scale_types::MinerKind;
    use crate::chain::ChainError;
    use serde_json::Value as JsonValue;
    use std::time::{Duration, Instant};
    use tokio::net::TcpListener;

    /// Accept connections and never write a response byte. Models a wedged peer
    /// that passes TCP but blocks the client forever without timeouts.
    async fn spawn_blackhole_listener() -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind blackhole listener");
        let addr = listener.local_addr().expect("blackhole local addr");
        drop(tokio::spawn(async move {
            loop {
                let Ok((sock, _)) = listener.accept().await else {
                    break;
                };
                // Hold the accepted socket open and never write. The peer must
                // time out on its own; that is the behaviour under test.
                drop(tokio::spawn(async move {
                    let _sock = sock;
                    std::future::pending::<()>().await;
                }));
            }
        }));
        format!("{addr}")
    }

    /// Upper bound for a single-endpoint blackhole: connect budget plus slack
    /// for scheduler jitter on a loaded CI host. Must stay well under the
    /// test's outer timeout so a hang fails the test, not the harness.
    fn single_endpoint_budget() -> Duration {
        RPC_CONNECT_TIMEOUT
            .saturating_mul(2)
            .saturating_add(Duration::from_secs(5))
    }

    /// Upper bound when the first endpoint blackholes and the second refuses:
    /// one connect timeout plus a refused connect, with CI slack.
    fn failover_budget() -> Duration {
        RPC_CONNECT_TIMEOUT
            .saturating_mul(3)
            .saturating_add(RPC_REQUEST_TIMEOUT)
            .saturating_add(Duration::from_secs(5))
    }

    #[tokio::test]
    async fn ws_blackhole_endpoint_errors_within_connect_budget() {
        let addr = spawn_blackhole_listener().await;
        let url = format!("ws://{addr}");
        let started = Instant::now();
        let outcome = tokio::time::timeout(
            single_endpoint_budget(),
            rpc_request(&url, "system_health", JsonValue::Array(vec![])),
        )
        .await;
        assert!(
            outcome.is_ok(),
            "ws blackhole call hung past {:?}",
            single_endpoint_budget()
        );
        let err = outcome
            .expect("outer timeout")
            .expect_err("blackhole must not succeed");
        assert!(
            matches!(err, ChainError::Unavailable(_)),
            "timeout must classify as Unavailable for failover, got {err}"
        );
        assert!(
            started.elapsed() < single_endpoint_budget(),
            "elapsed {:?} exceeds budget",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn http_blackhole_endpoint_errors_within_request_budget() {
        let addr = spawn_blackhole_listener().await;
        let url = format!("http://{addr}");
        // HTTP has no separate connect timeout; request_timeout covers the hang.
        let budget = RPC_REQUEST_TIMEOUT.saturating_add(Duration::from_secs(10));
        let started = Instant::now();
        let outcome = tokio::time::timeout(
            budget,
            rpc_request(&url, "system_health", JsonValue::Array(vec![])),
        )
        .await;
        assert!(outcome.is_ok(), "http blackhole call hung past {budget:?}");
        let err = outcome
            .expect("outer timeout")
            .expect_err("blackhole must not succeed");
        assert!(
            matches!(err, ChainError::Unavailable(_)),
            "timeout must classify as Unavailable for failover, got {err}"
        );
        assert!(
            started.elapsed() < budget,
            "elapsed {:?} exceeds budget",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn rpc_call_fails_over_past_blackhole_first_validator() {
        let addr = spawn_blackhole_listener().await;
        // Primary accepts and stays silent. Secondary refuses immediately so
        // the call ends after one connect timeout plus a fast connection error.
        let client = RealChainClient::new(
            vec![format!("ws://{addr}"), "ws://127.0.0.1:1".to_string()],
            "//Alice".to_string(),
            MinerKind::Cpu,
        );
        let started = Instant::now();
        let outcome = tokio::time::timeout(failover_budget(), client.runtime_version_raw()).await;
        assert!(
            outcome.is_ok(),
            "failover hung past {:?}",
            failover_budget()
        );
        let err = outcome
            .expect("outer timeout")
            .expect_err("no working validator in the list");
        assert!(
            matches!(err, ChainError::Unavailable(_)),
            "all-endpoint failure must be Unavailable, got {err}"
        );
        // Must have left the blackhole: elapsed is under one connect budget
        // times a small factor, not an unbounded hang.
        assert!(
            started.elapsed() < failover_budget(),
            "elapsed {:?} exceeds failover budget",
            started.elapsed()
        );
        // And it must have waited long enough that the connect timeout fired
        // (not instant success / skip). A refused-only pair would be near 0.
        assert!(
            started.elapsed() >= RPC_CONNECT_TIMEOUT,
            "expected at least one connect timeout ({RPC_CONNECT_TIMEOUT:?}), got {:?}",
            started.elapsed()
        );
    }

    /// The submit path must not treat inclusion as success. A block that
    /// includes the extrinsic but does not advance `LastProofBlock` is a failed
    /// dispatch, and reporting it as success would mark a losing proof as won.
    #[test]
    fn inclusion_without_a_win_is_not_success() {
        assert_eq!(
            classify_state_outcome(true, false),
            StateOutcome::IncludedButNotWon
        );
        assert_eq!(classify_state_outcome(true, true), StateOutcome::Won);
        assert_eq!(
            classify_state_outcome(false, false),
            StateOutcome::NotIncluded
        );
    }

    /// A win recorded while our extrinsic never made the block belongs to
    /// another miner and must not be claimed.
    #[test]
    fn a_win_without_our_extrinsic_is_not_ours() {
        assert_eq!(
            classify_state_outcome(false, true),
            StateOutcome::NotIncluded
        );
    }
}
