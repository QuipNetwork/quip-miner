//! Production chain client over Substrate JSON-RPC.
//!
//! Live end-to-end verification requires a running node. Offline unit tests
//! cover SCALE encode/decode, hybrid extrinsic assembly, and proof packing.
//! Methods return transport errors when no validator is reachable.

use super::extrinsic::{
    build_hybrid_signed_extrinsic, default_topology_storage_key, difficulties_storage_key,
    extrinsic_hash, hex_decode, hex_encode, job_orders_storage_key,
    last_proof_block_hash_storage_key, last_proof_block_storage_key, load_hybrid_pair,
    miner_identity_bytes, miners_storage_key, node_descriptors_storage_key,
    order_solutions_storage_key, participants_by_qblock_storage_key, qblocks_storage_key,
    signer_account_bytes, solvers_storage_key, topology_curve_c_storage_key, RuntimeTxContext,
    SignedExtensionContext,
};
use super::pool::{decode_pending_proof, PendingProof};
use super::proof_encode::{build_quantum_proof, ProofBuildContext};
use super::scale_types::{
    encode_deregister_solver_call, encode_participate_call, encode_register_miner_call,
    encode_register_solver_call, encode_set_descriptor_call, encode_submit_proof_call,
    encode_submit_solution_call, require_set_values, CurveCScale, DifficultyConfig, JobOrderScale,
    JobSolutionScale, MinerInfoScale, MinerKind, NodeDescriptorV2Input, OrderStatus, OrderTiming,
    SolverInfoScale, SolverType, TopologyMetaScale,
};
use super::submit::{
    classify_descriptor, classify_participation, classify_receipt, classify_registration,
    DescriptorOutcome, ParticipationOutcome, Proof, RegistrationOutcome, SubmitAction,
    SubmitReceipt,
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

/// A registered topology and the hash it was fetched under.
struct CachedTopology {
    hash: [u8; 32],
    meta: TopologyMetaScale,
}

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
    /// The registered topology the last snapshot was built from, with its
    /// hash. A topology never changes under its hash, so a poll on the same
    /// hash reuses it and only the difficulty and root are read again. One
    /// slot: a poll on another hash downloads that topology and replaces it.
    topology: Mutex<Option<CachedTopology>>,
    /// Genesis hash and runtime versions that pool signatures commit to.
    /// Reused while it matches the last snapshot's `spec_version`.
    tx_context: Mutex<Option<RuntimeTxContext>>,
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
    /// Registered mempool type, scoped to the account that was read.
    registered_solver: Mutex<Option<([u8; 32], SolverType)>>,
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
            topology: Mutex::new(None),
            tx_context: Mutex::new(None),
            transport: Arc::new(JsonrpseeTransport),
            reachable: Mutex::new(None),
            participate_kind,
            registered_solver: Mutex::new(None),
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

    /// Read the account's mempool solver type, reusing a successful registration read.
    async fn registered_solver_type(
        &self,
        account: [u8; 32],
        at_hex: &str,
    ) -> Result<Option<SolverType>, ChainError> {
        let cached = *self
            .registered_solver
            .lock()
            .map_err(|e| ChainError::Unavailable(e.to_string()))?;
        let cached_type = cached
            .filter(|(cached_account, _)| *cached_account == account)
            .map(|(_, solver_type)| solver_type);
        if cached_type.is_some() {
            return Ok(cached_type);
        }
        let stored = self
            .read_storage::<SolverInfoScale>(&solvers_storage_key(&account), at_hex)
            .await?;
        let solver_type = stored.map(|info| info.solver_type);
        *self
            .registered_solver
            .lock()
            .map_err(|e| ChainError::Unavailable(e.to_string()))? =
            solver_type.map(|kind| (account, kind));
        Ok(solver_type)
    }

    /// `state_call` a `QuantumPowApi` method that takes one `H256` and returns
    /// an `Option`. `Ok(None)` when the runtime answers `None` or null.
    async fn quantum_pow_call_by_hash<T: Decode>(
        &self,
        method: &str,
        hash: &[u8; 32],
        at_hex: &str,
    ) -> Result<Option<T>, ChainError> {
        let result = self
            .rpc_call(
                "state_call",
                Value::Array(vec![
                    Value::String(method.to_string()),
                    Value::String(hex_encode(hash)),
                    Value::String(at_hex.to_string()),
                ]),
            )
            .await?;
        let Some(hex) = result.as_str() else {
            return Ok(None);
        };
        let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
        Option::<T>::decode(&mut &bytes[..]).map_err(|e| ChainError::Decode(e.to_string()))
    }

    /// The topology the snapshot mines on: the caller's choice, else the
    /// chain's `DefaultTopology` at `at_hex`. `Ok(None)` when neither names one.
    async fn snapshot_topology_hash(
        &self,
        chosen: Option<[u8; 32]>,
        at_hex: &str,
    ) -> Result<Option<[u8; 32]>, ChainError> {
        match chosen {
            Some(hash) => Ok(Some(hash)),
            None => {
                self.read_storage::<[u8; 32]>(&default_topology_storage_key(), at_hex)
                    .await
            }
        }
    }

    /// The registered topology under `hash`: the cached one when the last
    /// snapshot used the same hash, else `QuantumPowApi_topology_meta`, whose
    /// answer replaces the cache. `Ok(None)` when the hash is not registered.
    async fn topology_meta(
        &self,
        hash: [u8; 32],
        at_hex: &str,
    ) -> Result<Option<TopologyMetaScale>, ChainError> {
        let cached = self.topology.lock().ok().and_then(|g| {
            g.as_ref()
                .filter(|c| c.hash == hash)
                .map(|c| c.meta.clone())
        });
        if cached.is_some() {
            return Ok(cached);
        }
        let meta = self
            .quantum_pow_call_by_hash::<TopologyMetaScale>(
                "QuantumPowApi_topology_meta",
                &hash,
                at_hex,
            )
            .await?;
        if let Some(meta) = &meta {
            tracing::info!(
                topology = %hex_encode(&hash),
                nodes = meta.nodes.len(),
                edges = meta.edges.len(),
                "topology downloaded"
            );
            if let Ok(mut g) = self.topology.lock() {
                *g = Some(CachedTopology {
                    hash,
                    meta: meta.clone(),
                });
            }
        }
        Ok(meta)
    }

    /// Fetch the chain's default topology at one best-block hash.
    ///
    /// Returns `None` when `QuantumPow.DefaultTopology` is unset. This reads
    /// topology metadata only and never loads signing material.
    ///
    /// # Errors
    /// Returns transport or decode errors. A stored default whose metadata
    /// is absent returns a decode error.
    pub async fn fetch_default_topology(
        &self,
    ) -> Result<Option<([u8; 32], TopologyMetaScale)>, ChainError> {
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let at = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?;
        let Some(hash) = self.snapshot_topology_hash(None, at).await? else {
            return Ok(None);
        };
        let metadata = self.topology_meta(hash, at).await?.ok_or_else(|| {
            ChainError::Decode(format!(
                "default topology {} is not registered",
                hex_encode(&hash)
            ))
        })?;
        Ok(Some((hash, metadata)))
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

    /// Read the genesis hash and the runtime versions from the validator.
    async fn fetch_runtime_tx_context(&self) -> Result<RuntimeTxContext, ChainError> {
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
        let spec_version = parse_spec_version(&rv)?;
        let transaction_version = u32::try_from(
            rv.get("transactionVersion")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    ChainError::Decode("runtime transactionVersion missing/not a u64".into())
                })?,
        )
        .map_err(|_| ChainError::Decode("transactionVersion exceeds u32".into()))?;

        Ok(RuntimeTxContext {
            genesis_hash,
            spec_version,
            transaction_version,
        })
    }

    /// The runtime context, reused while it matches the last snapshot's
    /// `spec_version`. A runtime upgrade moves `spec_version` and forces a read.
    async fn runtime_tx_context(&self) -> Result<RuntimeTxContext, ChainError> {
        let current_spec = self
            .last_snapshot
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|s| s.spec_version));
        let cached = self.tx_context.lock().ok().and_then(|g| *g);
        if let (Some(ctx), Some(spec)) = (cached, current_spec) {
            if ctx.spec_version == spec {
                return Ok(ctx);
            }
        }
        let ctx = self.fetch_runtime_tx_context().await?;
        if let Ok(mut g) = self.tx_context.lock() {
            *g = Some(ctx);
        }
        Ok(ctx)
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

        let runtime = self.fetch_runtime_tx_context().await?;
        Ok(SignedExtensionContext {
            account_nonce,
            genesis_hash: runtime.genesis_hash,
            spec_version: runtime.spec_version,
            transaction_version: runtime.transaction_version,
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
    ) -> Result<(SignedCallOutcome, [u8; 32]), ChainError> {
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
            Err(ChainError::Submit(message)) => {
                return Ok((SignedCallOutcome::Invalid { message }, want_hash))
            }
            Err(e) => return Err(e),
        };

        while let Some(item) = stream.next().await {
            let value = item?;
            match parse_tx_status(&value) {
                TxStatus::InBlock(block) | TxStatus::Finalized(block) => {
                    let outcome = self
                        .confirm_in_block(&block, &want_hash, confirmation)
                        .await?;
                    return Ok((outcome, want_hash));
                }
                TxStatus::Invalid(message) => {
                    return Ok((SignedCallOutcome::Invalid { message }, want_hash))
                }
                TxStatus::Dropped(message) => {
                    return Ok((SignedCallOutcome::Dropped { message }, want_hash))
                }
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
            StateOutcome::Won => {
                let header = self
                    .rpc_call(
                        "chain_getHeader",
                        Value::Array(vec![Value::String(block_hex.to_string())]),
                    )
                    .await?;
                Ok(SignedCallOutcome::Success {
                    block: block_hex.to_string(),
                    number: parse_block_number(&header)?,
                })
            }
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
            Confirmation::SolverRegistered { account } => {
                let stored = self
                    .read_storage::<SolverInfoScale>(&solvers_storage_key(&account), block_hex)
                    .await?;
                let registered = stored
                    .filter(|info| info.solver_type == SolverType::from(self.participate_kind))
                    .map(|info| (account, info.solver_type));
                *self
                    .registered_solver
                    .lock()
                    .map_err(|e| ChainError::Unavailable(e.to_string()))? = registered;
                Ok(registered.is_some())
            }
            Confirmation::SolverDeregistered { account } => Ok(!self
                .storage_value_present(&solvers_storage_key(&account), block_hex)
                .await?),
            Confirmation::SolutionAccepted { order_id, account } => {
                let header = self
                    .rpc_call(
                        "chain_getHeader",
                        Value::Array(vec![Value::String(block_hex.to_string())]),
                    )
                    .await?;
                let number = parse_block_number(&header)?;
                let stored = self
                    .read_storage::<JobSolutionScale>(
                        &order_solutions_storage_key(order_id, &account),
                        block_hex,
                    )
                    .await?;
                Ok(stored.is_some_and(|s| u64::from(s.submitted_at) == number))
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
            Confirmation::SolverRegistered { .. } => {
                "dispatch failed and QuantumComputeMempool.Solvers has no entry with the requested \
                 solver type for this account"
                    .to_string()
            }
            Confirmation::SolverDeregistered { .. } => {
                "dispatch failed and QuantumComputeMempool.Solvers still holds this account"
                    .to_string()
            }
            Confirmation::SolutionAccepted { order_id, .. } => format!(
                "dispatch failed and OrderSolutions for order {order_id} holds no submission from \
                 this block; the order may have closed, or the solutions missed its gates"
            ),
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
    /// `QuantumComputeMempool.Solvers[account]` holds the configured solver type.
    SolverRegistered { account: [u8; 32] },
    /// `QuantumComputeMempool.Solvers[account]` is absent.
    SolverDeregistered { account: [u8; 32] },
    /// `QuantumComputeMempool.OrderSolutions[order_id][account]` was written by
    /// this block. Presence alone is not enough: an earlier submission for the
    /// same order would already be there.
    SolutionAccepted { order_id: u64, account: [u8; 32] },
}

/// On-chain result of a hybrid-signed extrinsic, before pallet-specific classify.
pub(crate) enum SignedCallOutcome {
    /// Included and the dispatch succeeded.
    Success {
        /// `0x`-prefixed hash of the including block.
        block: String,
        /// Height of that block.
        number: u64,
    },
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
        let at_hex = hex_encode(&block_hash);

        // Every read is pinned at `block_hash`, so the five round trips that
        // only depend on it run together. The topology itself is not among
        // them: it is fetched once per hash and cached (see `topology_meta`).
        // `LastProofBlock` at the head names the winning block one block
        // before `LastProofBlockHash` catches up. Both are ValueQuery: 0 and
        // the zero hash when unset.
        let last_proof_block_key = last_proof_block_storage_key();
        let stored_root_key = last_proof_block_hash_storage_key();
        let (header, topology_hash, last_proof_block, stored_root, runtime_version) = futures::try_join!(
            self.rpc_call(
                "chain_getHeader",
                Value::Array(vec![Value::String(at_hex.clone())]),
            ),
            self.snapshot_topology_hash(topology_hash, &at_hex),
            self.read_storage::<u32>(&last_proof_block_key, &at_hex),
            self.read_storage::<[u8; 32]>(&stored_root_key, &at_hex),
            self.rpc_call(
                "state_getRuntimeVersion",
                Value::Array(vec![Value::String(at_hex.clone())]),
            ),
        )?;
        let Some(topology_hash) = topology_hash else {
            return Ok(None);
        };
        let block_number = parse_block_number(&header)?;
        let spec_version = parse_spec_version(&runtime_version)?;

        // Decayed difficulty for this block. `None` means the hash is not
        // registered, which is the same "nothing to mine" as no topology.
        let difficulty = self
            .quantum_pow_call_by_hash::<DifficultyConfig>(
                "QuantumPowApi_difficulty_for",
                &topology_hash,
                &at_hex,
            )
            .await?;
        let Some(difficulty) = difficulty else {
            return Ok(None);
        };
        let Some(meta) = self.topology_meta(topology_hash, &at_hex).await? else {
            return Ok(None);
        };

        let snap = MiningSnapshot {
            head_hash: block_hash,
            last_proof_block_hash: super::snapshot::round_root(
                block_hash,
                block_number,
                u64::from(last_proof_block.unwrap_or(0)),
                stored_root.unwrap_or([0u8; 32]),
            ),
            topology_hash: topology_hash.to_vec(),
            nodes: meta.nodes,
            edges: meta.edges,
            allowed_h_milli: require_set_values(&meta.allowed_h_values)
                .map_err(ChainError::Decode)?,
            allowed_j_milli: require_set_values(&meta.allowed_j_values)
                .map_err(ChainError::Decode)?,
            allowed_spin_milli: require_set_values(&meta.allowed_spin_values)
                .map_err(ChainError::Decode)?,
            min_solutions: difficulty.min_solutions,
            max_energy_milli: difficulty.max_energy_milli,
            min_diversity_milli: difficulty.min_diversity_milli,
            block_number,
            spec_version,
        };

        if let Ok(mut g) = self.last_snapshot.lock() {
            *g = Some(snap.clone());
        }
        if let Ok(mut g) = self.last_spin_spec.lock() {
            *g = Some(meta.allowed_spin_values);
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

    #[expect(
        clippy::too_many_lines,
        reason = "one order walk: registration, paged keys, status, expiry, and eligibility"
    )]
    async fn fetch_mempool_orders(
        &self,
        miner_account: [u8; 32],
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

        let header = self
            .rpc_call(
                "chain_getHeader",
                Value::Array(vec![Value::String(head_hex.to_string())]),
            )
            .await?;
        let head_number = parse_block_number(&header)?;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
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
            // The pallet expires orders lazily, so an `Opened` status can
            // outlive the order. Skip one already past its expiry.
            let Some(deadline_ms) = order_deadline_ms(
                scale.created_at,
                scale.first_solution_at,
                &scale.timing,
                head_number,
                now_ms,
            ) else {
                continue;
            };
            let order = JobOrder {
                order_id: oid.to_le_bytes().to_vec(),
                mode: scale.mode,
                nodes: scale.ising_params.nodes,
                edges: scale.ising_params.edges,
                h_milli: scale.ising_params.h_values,
                j_milli: scale.ising_params.j_values,
                min_energy_milli: scale.ising_params.min_energy_milli,
                min_diversity_milli: scale.ising_params.min_diversity_milli,
                min_solutions: scale.ising_params.min_solutions,
                deadline_ms,
            };
            let eligible = match &order.mode {
                super::scale_types::JobMode::Open => true,
                super::scale_types::JobMode::Bid { .. } => self
                    .registered_solver_type(miner_account, head_hex)
                    .await?
                    .is_some_and(|solver_type| order.is_eligible_for(&miner_account, solver_type)),
            };
            if !eligible {
                tracing::debug!(
                    order_id = oid,
                    "mempool order excludes this solver; skipping it"
                );
                continue;
            }
            orders.push(order);
        }
        Ok(orders)
    }

    async fn fetch_pending_proofs(&self) -> Result<Vec<PendingProof>, ChainError> {
        let raw = self
            .rpc_call("author_pendingExtrinsics", Value::Array(vec![]))
            .await?;
        let Some(entries) = raw.as_array() else {
            return Err(ChainError::Decode(
                "author_pendingExtrinsics did not return an array".into(),
            ));
        };
        let ctx = self.runtime_tx_context().await?;
        let mut proofs = Vec::new();
        for entry in entries {
            let Some(hex) = entry.as_str() else { continue };
            let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
            if let Some(p) = decode_pending_proof(&bytes, &ctx) {
                proofs.push(p);
            }
        }
        Ok(proofs)
    }

    async fn submit_proof(&self, proof: &Proof) -> Result<SubmitReceipt, ChainError> {
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

        let (outcome, extrinsic_hash) = self
            .submit_signed_call(&call, Confirmation::ProofWin { account })
            .await?;
        match outcome {
            SignedCallOutcome::Success { block, number } => {
                tracing::info!(block = %block, number, "proof included and dispatched successfully");
                let block_hash = hex_decode(&block)
                    .ok()
                    .and_then(|b| <[u8; 32]>::try_from(b.as_slice()).ok());
                Ok(SubmitReceipt {
                    action: SubmitAction::Success,
                    extrinsic_hash: Some(extrinsic_hash),
                    block_hash,
                    block_number: Some(number),
                })
            }
            SignedCallOutcome::DispatchFailed {
                error,
                lost_race: true,
            } => {
                tracing::info!(reason = %error, "proof included but another miner took the qblock");
                Ok(SubmitReceipt {
                    action: SubmitAction::StopRoundStale,
                    extrinsic_hash: Some(extrinsic_hash),
                    block_hash: None,
                    block_number: None,
                })
            }
            SignedCallOutcome::DispatchFailed { error, .. } => {
                tracing::warn!(error = %error, "proof included but ExtrinsicFailed");
                Ok(SubmitReceipt {
                    action: classify_receipt(Some(&error)),
                    extrinsic_hash: Some(extrinsic_hash),
                    block_hash: None,
                    block_number: None,
                })
            }
            SignedCallOutcome::Invalid { message } => {
                tracing::warn!(%message, "proof rejected as invalid before inclusion");
                Ok(SubmitReceipt {
                    action: classify_receipt(Some(&message)),
                    extrinsic_hash: Some(extrinsic_hash),
                    block_hash: None,
                    block_number: None,
                })
            }
            SignedCallOutcome::Dropped { message } => {
                tracing::warn!(%message, "proof dropped by node before inclusion");
                Ok(SubmitReceipt {
                    action: SubmitAction::Retry,
                    extrinsic_hash: Some(extrinsic_hash),
                    block_hash: None,
                    block_number: None,
                })
            }
        }
    }

    async fn ensure_miner_registered(&self) -> Result<RegistrationOutcome, ChainError> {
        let account = signer_account_bytes(&self.pair()?);
        if self.miner_is_registered(&account).await? {
            return Ok(RegistrationOutcome::AlreadyRegistered);
        }
        let call = encode_register_miner_call();
        let (outcome, _) = self
            .submit_signed_call(&call, Confirmation::MinerRegistered { account })
            .await?;
        match outcome {
            SignedCallOutcome::Success { .. } => Ok(RegistrationOutcome::Registered),
            SignedCallOutcome::DispatchFailed { error, .. }
            | SignedCallOutcome::Invalid { message: error } => {
                classify_registration(Some(&error)).ok_or(ChainError::Submit(error))
            }
            SignedCallOutcome::Dropped { message } => Err(ChainError::Submit(message)),
        }
    }

    async fn ensure_solver_registered(&self) -> Result<RegistrationOutcome, ChainError> {
        *self
            .registered_solver
            .lock()
            .map_err(|e| ChainError::Unavailable(e.to_string()))? = None;
        let account = signer_account_bytes(&self.pair()?);
        let want = SolverType::from(self.participate_kind);
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let at = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?;
        match self.registered_solver_type(account, at).await? {
            Some(registered) if registered == want => {
                return Ok(RegistrationOutcome::AlreadyRegistered);
            }
            Some(registered) => {
                tracing::info!(
                    registered = ?registered,
                    configured = ?want,
                    "mempool solver type changed; deregistering before re-registering"
                );
                *self
                    .registered_solver
                    .lock()
                    .map_err(|e| ChainError::Unavailable(e.to_string()))? = None;
                let (outcome, _) = self
                    .submit_signed_call(
                        &encode_deregister_solver_call(),
                        Confirmation::SolverDeregistered { account },
                    )
                    .await?;
                match outcome {
                    SignedCallOutcome::Success { .. } => {}
                    SignedCallOutcome::DispatchFailed { error, .. }
                    | SignedCallOutcome::Invalid { message: error }
                    | SignedCallOutcome::Dropped { message: error } => {
                        return Err(ChainError::Submit(format!("deregister_solver: {error}")));
                    }
                }
            }
            None => {}
        }
        let (outcome, _) = self
            .submit_signed_call(
                &encode_register_solver_call(want),
                Confirmation::SolverRegistered { account },
            )
            .await?;
        match outcome {
            SignedCallOutcome::Success { .. } => Ok(RegistrationOutcome::Registered),
            SignedCallOutcome::DispatchFailed { error, .. }
            | SignedCallOutcome::Invalid { message: error } => {
                classify_registration(Some(&error)).ok_or(ChainError::Submit(error))
            }
            SignedCallOutcome::Dropped { message } => Err(ChainError::Submit(message)),
        }
    }

    async fn submit_solution(&self, proof: &Proof) -> Result<SubmitReceipt, ChainError> {
        let order_id = <[u8; 8]>::try_from(proof.order_id.as_slice())
            .map(u64::from_le_bytes)
            .map_err(|_| {
                ChainError::Submit(format!(
                    "mempool order id is {} bytes, expected 8",
                    proof.order_id.len()
                ))
            })?;
        // Proof rows contain validated +1/-1 i8 spins.
        let rows: Vec<Vec<i8>> = proof.solutions.iter().map(|r| r.spins.clone()).collect();
        let account = signer_account_bytes(&self.pair()?);
        let call = encode_submit_solution_call(order_id, &rows);
        let (outcome, extrinsic_hash) = self
            .submit_signed_call(&call, Confirmation::SolutionAccepted { order_id, account })
            .await?;
        let action = match &outcome {
            SignedCallOutcome::Success { block, number } => {
                tracing::info!(order_id, block = %block, number, rows = rows.len(), "mempool solution accepted");
                let block_hash = hex_decode(block)
                    .ok()
                    .and_then(|b| <[u8; 32]>::try_from(b.as_slice()).ok());
                return Ok(SubmitReceipt {
                    action: SubmitAction::Success,
                    extrinsic_hash: Some(extrinsic_hash),
                    block_hash,
                    block_number: Some(*number),
                });
            }
            // The failure text explains state; it carries no pallet error name
            // to classify, and a rejected solution is not worth resending.
            SignedCallOutcome::DispatchFailed { error: reason, .. }
            | SignedCallOutcome::Invalid { message: reason } => {
                tracing::warn!(order_id, reason = %reason, "mempool solution rejected");
                SubmitAction::StopFatal
            }
            SignedCallOutcome::Dropped { message } => {
                tracing::warn!(order_id, reason = %message, "mempool solution dropped before inclusion");
                SubmitAction::Retry
            }
        };
        Ok(SubmitReceipt {
            action,
            extrinsic_hash: Some(extrinsic_hash),
            block_hash: None,
            block_number: None,
        })
    }

    async fn file_descriptor(
        &self,
        descriptor: &NodeDescriptorV2Input,
    ) -> Result<DescriptorOutcome, ChainError> {
        let call = encode_set_descriptor_call(descriptor);
        let account = signer_account_bytes(&self.pair()?);
        let (outcome, _) = self
            .submit_signed_call(&call, Confirmation::Descriptor { account })
            .await?;
        match outcome {
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
        let (outcome, _) = self
            .submit_signed_call(&call, Confirmation::Participation { qblock_id, account })
            .await?;
        match outcome {
            SignedCallOutcome::Success { .. } => Ok(ParticipationOutcome::Declared),
            SignedCallOutcome::DispatchFailed { error, .. }
            | SignedCallOutcome::Invalid { message: error } => {
                classify_participation(Some(&error)).ok_or(ChainError::Submit(error))
            }
            SignedCallOutcome::Dropped { message } => Err(ChainError::Submit(message)),
        }
    }
}

/// Block time assumed when turning a block count into wall-clock time.
const BLOCK_TIME_ESTIMATE_MS: u64 = 6_000;

/// Absolute deadline, in Unix milliseconds, for a mempool order seen at `head`.
///
/// Miners compare `deadline_ms` with their own clock, so it must be absolute.
/// Mirrors the pallet's `effective_expiry`: the hard deadline, cut short by
/// `block_wait` once a first solution lands. `None` once `head` has reached
/// that expiry, when `submit_solution` would find the order closed.
fn order_deadline_ms(
    created_at: u32,
    first_solution_at: Option<u32>,
    timing: &OrderTiming,
    head: u64,
    now_ms: u64,
) -> Option<u64> {
    let hard = created_at.saturating_add(timing.deadline_blocks);
    let expiry = first_solution_at.map_or(hard, |first| {
        hard.min(first.saturating_add(timing.block_wait))
    });
    let remaining = u64::from(expiry).checked_sub(head).filter(|&b| b > 0)?;
    Some(now_ms.saturating_add(remaining.saturating_mul(BLOCK_TIME_ESTIMATE_MS)))
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

/// `specVersion` from a `state_getRuntimeVersion` response.
pub(crate) fn parse_spec_version(rv: &Value) -> Result<u32, ChainError> {
    u32::try_from(
        rv.get("specVersion")
            .and_then(Value::as_u64)
            .ok_or_else(|| ChainError::Decode("runtime specVersion missing/not a u64".into()))?,
    )
    .map_err(|_| ChainError::Decode("specVersion exceeds u32".into()))
}

#[cfg(test)]
mod tests {
    use super::super::transport_jsonrpsee::{
        rpc_request, RPC_CONNECT_TIMEOUT, RPC_REQUEST_TIMEOUT,
    };

    /// Losing the qblock to another miner is the one included-but-failed case
    /// that must not be reported as a rejection: the proof itself was sound.
    #[test]
    fn an_order_deadline_is_absolute_and_honours_block_wait() {
        let timing = super::OrderTiming {
            deadline_blocks: 50,
            block_wait: 5,
        };
        assert_eq!(
            super::order_deadline_ms(100, None, &timing, 120, 1_000),
            Some(1_000 + 30 * 6_000)
        );
        assert_eq!(
            super::order_deadline_ms(100, Some(110), &timing, 112, 0),
            Some(3 * 6_000)
        );
        assert_eq!(
            super::order_deadline_ms(100, Some(110), &timing, 115, 0),
            None,
            "block_wait has elapsed since the first solution"
        );
        assert_eq!(
            super::order_deadline_ms(100, None, &timing, 150, 0),
            None,
            "the hard deadline has been reached"
        );
    }

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
        classify_state_outcome, explain_proof_failure, hex_encode, parse_spec_version,
        RealChainClient, StateOutcome,
    };

    #[test]
    fn parse_spec_version_reads_the_number_and_rejects_absence() {
        let rv = serde_json::json!({ "specName": "quip-runtime", "specVersion": 118 });
        assert_eq!(parse_spec_version(&rv).map_err(|e| e.to_string()), Ok(118));
        let missing = serde_json::json!({ "specName": "quip-runtime" });
        assert!(parse_spec_version(&missing).is_err());
    }
    use crate::chain::scale_types::MinerKind;
    use crate::chain::transport::RpcTransport;
    use crate::chain::{ChainError, MiningSnapshot};
    use quip_transaction_crypto::HybridPair;
    use serde_json::Value as JsonValue;
    use sp_core::Pair as _;
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

    /// A validator scripted for the snapshot path: a fixed head at block 16,
    /// one registered topology per hash, a `DefaultTopology` the test can
    /// move, and a `LastProofBlock` the test can set. `None` answers null for
    /// both `LastProofBlock` and `LastProofBlockHash`, which is genesis.
    struct SnapshotValidator {
        default_topology: std::sync::Mutex<[u8; 32]>,
        last_proof_block: std::sync::Mutex<Option<u32>>,
        topology_meta_calls: std::sync::Mutex<Vec<[u8; 32]>>,
    }

    const HEAD: [u8; 32] = [0xAA; 32];
    const STORED_ROOT: [u8; 32] = [0xBB; 32];

    fn topology_for(hash: [u8; 32]) -> crate::chain::scale_types::TopologyMetaScale {
        use quantum_validation::AllowedValueSpec;
        let n = u32::from(hash[0]);
        crate::chain::scale_types::TopologyMetaScale {
            nodes: vec![0, 1, n],
            edges: vec![(0, 1), (1, n)],
            allowed_h_values: AllowedValueSpec::Set(vec![-1000, 1000]),
            allowed_j_values: AllowedValueSpec::Set(vec![-1000, 1000]),
            allowed_spin_values: AllowedValueSpec::Set(vec![-1000, 1000]),
            registered_at: 3,
        }
    }

    #[async_trait::async_trait]
    impl RpcTransport for SnapshotValidator {
        async fn request(
            &self,
            _url: &str,
            method: &str,
            params: JsonValue,
        ) -> Result<JsonValue, ChainError> {
            use crate::chain::extrinsic::{
                default_topology_storage_key, hex_decode, last_proof_block_hash_storage_key,
                last_proof_block_storage_key,
            };
            use parity_scale_codec::Encode as _;
            let arg = |i: usize| {
                params
                    .get(i)
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default()
                    .to_string()
            };
            let hex = |bytes: &[u8]| JsonValue::String(hex_encode(bytes));
            Ok(match method {
                "chain_getBlockHash" => hex(&HEAD),
                "chain_getHeader" => {
                    assert_eq!(arg(0), hex_encode(&HEAD), "header read pinned at the head");
                    serde_json::json!({ "number": "0x10" })
                }
                "state_getRuntimeVersion" => {
                    assert_eq!(
                        arg(0),
                        hex_encode(&HEAD),
                        "runtime version read pinned at the head"
                    );
                    serde_json::json!({ "specName": "quip-runtime", "specVersion": 118 })
                }
                "state_getStorage" => {
                    assert_eq!(arg(1), hex_encode(&HEAD), "storage read pinned at the head");
                    let key = hex_decode(&arg(0)).expect("storage key hex");
                    if key == default_topology_storage_key() {
                        hex(&*self.default_topology.lock().unwrap())
                    } else if key == last_proof_block_storage_key() {
                        match *self.last_proof_block.lock().unwrap() {
                            Some(n) => hex(&n.encode()),
                            None => JsonValue::Null,
                        }
                    } else if key == last_proof_block_hash_storage_key() {
                        match *self.last_proof_block.lock().unwrap() {
                            Some(_) => hex(&STORED_ROOT),
                            None => JsonValue::Null,
                        }
                    } else {
                        panic!("unexpected storage key {}", arg(0))
                    }
                }
                "state_call" => {
                    assert_eq!(arg(2), hex_encode(&HEAD), "runtime call pinned at the head");
                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&hex_decode(&arg(1)).expect("hash hex"));
                    match arg(0).as_str() {
                        "QuantumPowApi_difficulty_for" => {
                            hex(&Some(crate::chain::scale_types::DifficultyConfig {
                                min_solutions: 1,
                                max_energy_milli: -5_000,
                                min_diversity_milli: 0,
                            })
                            .encode())
                        }
                        "QuantumPowApi_topology_meta" => {
                            self.topology_meta_calls.lock().unwrap().push(hash);
                            hex(&Some(topology_for(hash)).encode())
                        }
                        other => panic!("unexpected runtime call {other}"),
                    }
                }
                other => panic!("unexpected rpc method {other}"),
            })
        }

        async fn subscribe(
            &self,
            _url: &str,
            _sub: &str,
            _params: JsonValue,
            _unsub: &str,
        ) -> Result<
            crate::chain::transport::BoxStream<'static, Result<JsonValue, ChainError>>,
            ChainError,
        > {
            Err(ChainError::Unavailable(
                "no subscriptions in this test".into(),
            ))
        }
    }

    /// The topology is the one payload that never changes under its hash, so
    /// polling the same hash must not download it again. A new default
    /// topology is a new hash and is downloaded when it first appears. The
    /// root and the difficulty are read on every poll.
    #[tokio::test]
    async fn the_topology_is_downloaded_only_when_the_hash_changes() {
        use crate::chain::ChainClient as _;
        let first = [0x11; 32];
        let second = [0x22; 32];
        let validator = std::sync::Arc::new(SnapshotValidator {
            default_topology: std::sync::Mutex::new(first),
            last_proof_block: std::sync::Mutex::new(Some(10)),
            topology_meta_calls: std::sync::Mutex::new(Vec::new()),
        });
        let client = RealChainClient::with_transport(
            vec!["ws://scripted".to_string()],
            "//Alice".to_string(),
            MinerKind::Cpu,
            validator.clone(),
        );

        let snap = client
            .fetch_mining_snapshot(None, [0u8; 32], None)
            .await
            .expect("first snapshot")
            .expect("a topology is registered");
        assert_eq!(snap.head_hash, HEAD);
        assert_eq!(
            snap.last_proof_block_hash, STORED_ROOT,
            "the win was at block 10, so the stored hash is the root"
        );
        assert_eq!(snap.block_number, 16);
        assert_eq!(snap.spec_version, 118);
        assert_eq!(snap.topology_hash, first.to_vec());
        assert_eq!(snap.nodes, vec![0, 1, 0x11]);
        assert_eq!(snap.edges, vec![(0, 1), (1, 0x11)]);
        assert_eq!(snap.allowed_spin_milli, vec![-1000, 1000]);
        assert_eq!(snap.min_solutions, 1);
        assert_eq!(snap.max_energy_milli, -5_000);

        *validator.last_proof_block.lock().unwrap() = Some(16);
        let won = client
            .fetch_mining_snapshot(None, [0u8; 32], None)
            .await
            .expect("snapshot at the winning block")
            .expect("still registered");
        assert_eq!(
            won.last_proof_block_hash, HEAD,
            "LastProofBlock equals the head number, so the head is the root"
        );

        *validator.last_proof_block.lock().unwrap() = None;
        let genesis = client
            .fetch_mining_snapshot(None, [0u8; 32], None)
            .await
            .expect("snapshot before any win")
            .expect("still registered");
        assert_eq!(
            genesis.last_proof_block_hash, [0u8; 32],
            "unset storage reads as the ValueQuery defaults: block 0, zero hash"
        );

        let again = client
            .fetch_mining_snapshot(None, [0u8; 32], None)
            .await
            .expect("second snapshot")
            .expect("still registered");
        assert_eq!(again.nodes, snap.nodes);
        assert_eq!(
            validator.topology_meta_calls.lock().unwrap().as_slice(),
            &[first],
            "the second poll on the same hash reuses the cached topology"
        );

        *validator.default_topology.lock().unwrap() = second;
        let moved = client
            .fetch_mining_snapshot(None, [0u8; 32], None)
            .await
            .expect("third snapshot")
            .expect("the new default is registered");
        assert_eq!(moved.topology_hash, second.to_vec());
        assert_eq!(moved.nodes, vec![0, 1, 0x22]);
        assert_eq!(
            validator.topology_meta_calls.lock().unwrap().as_slice(),
            &[first, second],
            "a new hash is downloaded when it first appears"
        );

        // The caller's own choice bypasses `DefaultTopology` and hits the cache.
        let chosen = client
            .fetch_mining_snapshot(None, [0u8; 32], Some(second))
            .await
            .expect("chosen snapshot")
            .expect("registered");
        assert_eq!(chosen.nodes, moved.nodes);
        assert_eq!(validator.topology_meta_calls.lock().unwrap().len(), 2);
    }

    /// Answers the pool decoder's reads and counts the runtime-version calls.
    struct PoolValidator {
        pending: Vec<String>,
        runtime_version_calls: std::sync::Mutex<u32>,
        spec_version: std::sync::Mutex<u32>,
        transaction_version: std::sync::Mutex<u32>,
        fail_runtime_version: bool,
    }

    #[async_trait::async_trait]
    impl RpcTransport for PoolValidator {
        async fn request(
            &self,
            _url: &str,
            method: &str,
            _params: JsonValue,
        ) -> Result<JsonValue, ChainError> {
            match method {
                "chain_getBlockHash" => Ok(JsonValue::String(hex_encode(&[0x11; 32]))),
                "state_getRuntimeVersion" => {
                    *self.runtime_version_calls.lock().unwrap() += 1;
                    if self.fail_runtime_version {
                        return Err(ChainError::Unavailable("scripted outage".into()));
                    }
                    Ok(serde_json::json!({
                        "specVersion": *self.spec_version.lock().unwrap(),
                        "transactionVersion": *self.transaction_version.lock().unwrap(),
                    }))
                }
                "author_pendingExtrinsics" => Ok(JsonValue::Array(
                    self.pending
                        .iter()
                        .cloned()
                        .map(JsonValue::String)
                        .collect(),
                )),
                other => panic!("unexpected rpc method {other}"),
            }
        }

        async fn subscribe(
            &self,
            _url: &str,
            _sub: &str,
            _params: JsonValue,
            _unsub: &str,
        ) -> Result<
            crate::chain::transport::BoxStream<'static, Result<JsonValue, ChainError>>,
            ChainError,
        > {
            Err(ChainError::Unavailable(
                "no subscriptions in this test".into(),
            ))
        }
    }

    fn pool_snapshot(spec_version: u32) -> MiningSnapshot {
        MiningSnapshot {
            head_hash: [0u8; 32],
            last_proof_block_hash: [0u8; 32],
            topology_hash: vec![0u8; 32],
            nodes: vec![0, 1],
            edges: vec![(0, 1)],
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![-1000, 1000],
            allowed_spin_milli: vec![-1000, 1000],
            min_solutions: 1,
            max_energy_milli: 0,
            min_diversity_milli: 0,
            block_number: 1,
            spec_version,
        }
    }

    fn pool_entry(genesis_hash: [u8; 32], spec_version: u32, transaction_version: u32) -> String {
        use crate::chain::extrinsic::{build_hybrid_signed_extrinsic, SignedExtensionContext};
        use crate::chain::scale_types::{encode_submit_proof_call, QuantumProof};
        let pair = HybridPair::from_string("//Alice", None).expect("alice");
        let call = encode_submit_proof_call(&QuantumProof {
            topology_hash: sp_core::H256::repeat_byte(1),
            nonce: sp_core::U256::from(42u64),
            salt: [7u8; 32],
            solutions: vec![vec![0b01]],
            device_access_time_us: 5,
        });
        let ctx = SignedExtensionContext {
            account_nonce: 0,
            genesis_hash,
            spec_version,
            transaction_version,
            tip: 0,
        };
        hex_encode(&build_hybrid_signed_extrinsic(&pair, &call, &ctx))
    }

    #[tokio::test]
    async fn pending_proofs_are_verified_and_the_context_is_cached_per_spec_version() {
        use crate::chain::extrinsic::{extrinsic_hash, hex_decode};
        use crate::chain::ChainClient as _;
        let upgraded_entry = pool_entry([0x11; 32], 101, 2);
        let upgraded_hash = extrinsic_hash(&hex_decode(&upgraded_entry).expect("upgraded hex"));
        let validator = std::sync::Arc::new(PoolValidator {
            pending: vec![
                pool_entry([0x11; 32], 100, 1),
                pool_entry([0x22; 32], 100, 1),
                upgraded_entry,
            ],
            runtime_version_calls: std::sync::Mutex::new(0),
            spec_version: std::sync::Mutex::new(100),
            transaction_version: std::sync::Mutex::new(1),
            fail_runtime_version: false,
        });
        let client = RealChainClient::with_transport(
            vec!["ws://scripted".to_string()],
            "//Alice".to_string(),
            MinerKind::Cpu,
            validator.clone(),
        );
        *client.last_snapshot.lock().unwrap() = Some(pool_snapshot(100));

        let proofs = client.fetch_pending_proofs().await.expect("pool read");
        assert_eq!(
            proofs.len(),
            1,
            "entries signed for another genesis or runtime are dropped"
        );
        let _ = client.fetch_pending_proofs().await.expect("pool read");
        assert_eq!(
            *validator.runtime_version_calls.lock().unwrap(),
            1,
            "the second poll at the same spec_version reuses the context"
        );

        *validator.spec_version.lock().unwrap() = 101;
        *validator.transaction_version.lock().unwrap() = 2;
        *client.last_snapshot.lock().unwrap() = Some(pool_snapshot(101));
        let after_upgrade = client.fetch_pending_proofs().await.expect("pool read");
        assert_eq!(*validator.runtime_version_calls.lock().unwrap(), 2);
        assert_eq!(
            after_upgrade.len(),
            1,
            "only the entry signed for spec 101 / transaction version 2 is accepted"
        );
        assert_eq!(
            after_upgrade
                .first()
                .expect("upgraded proof")
                .extrinsic_hash,
            upgraded_hash,
            "entries signed for spec 100 / transaction version 1 are dropped"
        );
    }

    #[tokio::test]
    async fn a_runtime_version_outage_is_an_error_not_a_panic() {
        use crate::chain::ChainClient as _;
        let validator = std::sync::Arc::new(PoolValidator {
            pending: vec![pool_entry([0x11; 32], 100, 1)],
            runtime_version_calls: std::sync::Mutex::new(0),
            spec_version: std::sync::Mutex::new(100),
            transaction_version: std::sync::Mutex::new(1),
            fail_runtime_version: true,
        });
        let client = RealChainClient::with_transport(
            vec!["ws://scripted".to_string()],
            "//Alice".to_string(),
            MinerKind::Cpu,
            validator,
        );
        assert!(client.fetch_pending_proofs().await.is_err());
    }

    mod bid_orders {
        use super::super::Confirmation;
        use crate::chain::extrinsic::{
            hex_decode, hex_encode, job_orders_storage_key, signer_account_bytes,
            solvers_storage_key,
        };
        use crate::chain::scale_types::{
            IsingParams, JobMode, JobOrderScale, MinerKind, OrderStatus, OrderTiming,
            ResultDelivery, RewardResolution, SolverInfoScale, SolverType,
        };
        use crate::chain::{
            BoxStream, ChainClient, ChainError, RealChainClient, RegistrationOutcome, RpcTransport,
        };
        use parity_scale_codec::Encode as _;
        use quip_transaction_crypto::HybridPair;
        use serde_json::Value;
        use sp_core::{Pair as _, H256};
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};

        const HEAD: [u8; 32] = [0xab; 32];

        struct BidValidator {
            account: [u8; 32],
            registration: Mutex<Value>,
            solver_reads: AtomicUsize,
            orders: Vec<(u64, JobOrderScale)>,
        }

        fn alice_account() -> [u8; 32] {
            signer_account_bytes(
                &HybridPair::from_string("//Alice", None).expect("public dev account"),
            )
        }

        fn registration(account: [u8; 32], solver_type: SolverType) -> Value {
            Value::String(hex_encode(
                &SolverInfoScale {
                    account,
                    solver_type,
                    registered_at: 1,
                    solutions_submitted: 0,
                    rewards_earned: 0,
                }
                .encode(),
            ))
        }

        fn scale_order(mode: JobMode) -> JobOrderScale {
            JobOrderScale {
                spec_id: H256::repeat_byte(7),
                proposer: [3; 32],
                ising_params: IsingParams {
                    nodes: vec![0, 1],
                    edges: vec![(0, 1)],
                    h_values: vec![0, 0],
                    j_values: vec![-1000],
                    min_energy_milli: None,
                    min_diversity_milli: None,
                    min_solutions: Some(1),
                },
                reward: 100,
                mode,
                resolution: RewardResolution::SingleBest,
                timing: OrderTiming {
                    deadline_blocks: 50,
                    block_wait: 5,
                },
                delivery: ResultDelivery::OnChainOnly,
                status: OrderStatus::Opened,
                created_at: 1,
                first_solution_at: None,
                solution_count: 0,
            }
        }

        fn type_bid(tag: u8) -> JobOrderScale {
            scale_order(JobMode::Bid {
                miners: None,
                miner_types: Some(vec![tag]),
            })
        }

        fn client_for(
            account: [u8; 32],
            solver_type: Option<SolverType>,
            orders: Vec<JobOrderScale>,
            configured_kind: MinerKind,
        ) -> (Arc<BidValidator>, RealChainClient) {
            let validator = Arc::new(BidValidator {
                account,
                registration: Mutex::new(
                    solver_type.map_or(Value::Null, |kind| registration(account, kind)),
                ),
                solver_reads: AtomicUsize::new(0),
                orders: orders
                    .into_iter()
                    .enumerate()
                    .map(|(index, order)| (u64::try_from(index + 1).expect("small fixture"), order))
                    .collect(),
            });
            let client = RealChainClient::with_transport(
                vec!["ws://scripted".to_string()],
                "//Alice".to_string(),
                configured_kind,
                validator.clone(),
            );
            (validator, client)
        }

        async fn assert_order_count(client: &RealChainClient, account: [u8; 32], expected: usize) {
            let orders = client.fetch_mempool_orders(account).await.expect("orders");
            assert_eq!(orders.len(), expected);
        }

        fn assert_decode_error(error: &ChainError) {
            assert!(match error {
                ChainError::Decode(_) => true,
                ChainError::Unavailable(_) | ChainError::Submit(_) => false,
            });
        }

        #[async_trait::async_trait]
        impl RpcTransport for BidValidator {
            async fn request(
                &self,
                _url: &str,
                method: &str,
                params: Value,
            ) -> Result<Value, ChainError> {
                let arg = |index: usize| {
                    params
                        .get(index)
                        .and_then(Value::as_str)
                        .expect("string RPC argument")
                };
                match method {
                    "chain_getBlockHash" => Ok(Value::String(hex_encode(&HEAD))),
                    "chain_getHeader" => {
                        assert_eq!(arg(0), hex_encode(&HEAD));
                        Ok(serde_json::json!({ "number": "0x02" }))
                    }
                    "state_getKeysPaged" => {
                        assert_eq!(arg(0), hex_encode(&crate::chain::job_orders_prefix()));
                        assert_eq!(params.get(1).and_then(Value::as_u64), Some(200));
                        assert_eq!(params.get(2), Some(&Value::Null));
                        assert_eq!(arg(3), hex_encode(&HEAD));
                        Ok(Value::Array(
                            self.orders
                                .iter()
                                .map(|(id, _)| {
                                    Value::String(hex_encode(&job_orders_storage_key(*id)))
                                })
                                .collect(),
                        ))
                    }
                    "state_getStorage" => {
                        assert_eq!(arg(1), hex_encode(&HEAD));
                        let key = hex_decode(arg(0)).map_err(ChainError::Decode)?;
                        if key == solvers_storage_key(&self.account) {
                            let _ = self.solver_reads.fetch_add(1, Ordering::Relaxed);
                            let stored = self.registration.lock().expect("registration").clone();
                            if let Some(error) = stored.get("rpc_error").and_then(Value::as_str) {
                                return Err(ChainError::Unavailable(error.to_string()));
                            }
                            return Ok(stored);
                        }
                        for (id, order) in &self.orders {
                            if key == job_orders_storage_key(*id) {
                                return Ok(Value::String(hex_encode(&order.encode())));
                            }
                        }
                        Ok(Value::Null)
                    }
                    other => Err(ChainError::Unavailable(format!(
                        "unexpected RPC method {other}"
                    ))),
                }
            }

            async fn subscribe(
                &self,
                _url: &str,
                _sub: &str,
                _params: Value,
                _unsub: &str,
            ) -> Result<BoxStream<'static, Result<Value, ChainError>>, ChainError> {
                Err(ChainError::Unavailable(
                    "no subscriptions in this test".into(),
                ))
            }
        }

        #[tokio::test]
        async fn fetch_excludes_bid_and_preserves_allowed_modes() {
            let account = alice_account();
            let account_bid = JobMode::Bid {
                miners: Some(vec![account]),
                miner_types: Some(vec![1]),
            };
            let type_bid = JobMode::Bid {
                miners: Some(vec![[2; 32]]),
                miner_types: Some(vec![0]),
            };
            let (validator, client) = client_for(
                account,
                Some(SolverType::Cpu),
                vec![
                    scale_order(JobMode::Bid {
                        miners: Some(vec![[2; 32]]),
                        miner_types: Some(vec![1]),
                    }),
                    scale_order(account_bid.clone()),
                    scale_order(type_bid.clone()),
                    scale_order(JobMode::Open),
                ],
                MinerKind::QpuDwave,
            );
            let orders = client.fetch_mempool_orders(account).await.expect("orders");
            let ids: Vec<Vec<u8>> = orders.iter().map(|order| order.order_id.clone()).collect();
            assert_eq!(
                ids,
                vec![
                    2u64.to_le_bytes().to_vec(),
                    3u64.to_le_bytes().to_vec(),
                    4u64.to_le_bytes().to_vec(),
                ],
            );
            let modes: Vec<JobMode> = orders.iter().map(|order| order.mode.clone()).collect();
            assert_eq!(modes, vec![account_bid, type_bid, JobMode::Open]);
            let again = client
                .fetch_mempool_orders(account)
                .await
                .expect("second read");
            assert_eq!(again.len(), 3);
            assert_eq!(validator.solver_reads.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn startup_registration_seeds_the_gpu_type_for_metal() {
            let account = alice_account();
            let (validator, client) = client_for(
                account,
                Some(SolverType::Gpu),
                vec![type_bid(1)],
                MinerKind::Metal,
            );
            assert_eq!(
                client.ensure_solver_registered().await.expect("registered"),
                RegistrationOutcome::AlreadyRegistered,
            );
            for _ in 0..2 {
                assert_order_count(&client, account, 1).await;
            }
            assert_eq!(validator.solver_reads.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn registration_confirmation_refreshes_the_cached_type() {
            let account = alice_account();
            let (validator, client) = client_for(
                account,
                Some(SolverType::Cpu),
                vec![type_bid(1)],
                MinerKind::Metal,
            );
            assert_order_count(&client, account, 0).await;
            *validator.registration.lock().expect("registration") =
                registration(account, SolverType::Gpu);
            assert!(client
                .confirmation_present(
                    &hex_encode(&HEAD),
                    &Confirmation::SolverRegistered { account }
                )
                .await
                .expect("GPU confirmation"));
            assert_order_count(&client, account, 1).await;
            assert_eq!(validator.solver_reads.load(Ordering::Relaxed), 2);
        }

        #[tokio::test]
        async fn registration_confirmation_rejects_the_wrong_type() {
            let account = alice_account();
            let (validator, client) = client_for(
                account,
                Some(SolverType::Cpu),
                vec![type_bid(1)],
                MinerKind::Metal,
            );
            assert!(!client
                .confirmation_present(
                    &hex_encode(&HEAD),
                    &Confirmation::SolverRegistered { account }
                )
                .await
                .expect("confirmation"));
            assert_order_count(&client, account, 0).await;
            assert_eq!(validator.solver_reads.load(Ordering::Relaxed), 2);
        }

        #[tokio::test]
        async fn missing_registration_is_not_cached_as_a_permanent_absence() {
            let account = alice_account();
            let (validator, client) = client_for(account, None, vec![type_bid(0)], MinerKind::Cpu);
            assert_order_count(&client, account, 0).await;
            *validator.registration.lock().expect("registration") =
                registration(account, SolverType::Cpu);
            assert_order_count(&client, account, 1).await;
            assert_eq!(validator.solver_reads.load(Ordering::Relaxed), 2);
        }

        #[tokio::test]
        async fn malformed_registration_is_an_error_and_can_be_retried() {
            let account = alice_account();
            let (validator, client) = client_for(
                account,
                Some(SolverType::Cpu),
                vec![type_bid(0)],
                MinerKind::Cpu,
            );
            *validator.registration.lock().expect("registration") = Value::String("0xff".into());
            let error = client
                .fetch_mempool_orders(account)
                .await
                .expect_err("truncated registration");
            assert_decode_error(&error);
            *validator.registration.lock().expect("registration") =
                registration(account, SolverType::Cpu);
            assert_order_count(&client, account, 1).await;
            assert_eq!(validator.solver_reads.load(Ordering::Relaxed), 2);
        }

        #[tokio::test]
        async fn solver_registration_outage_is_retryable() {
            let account = alice_account();
            let (validator, client) = client_for(
                account,
                Some(SolverType::Cpu),
                vec![type_bid(0)],
                MinerKind::Cpu,
            );
            *validator.registration.lock().expect("registration") =
                serde_json::json!({ "rpc_error": "scripted outage" });
            let error = client
                .fetch_mempool_orders(account)
                .await
                .expect_err("RPC failure");
            assert!(match error {
                ChainError::Unavailable(_) => true,
                ChainError::Decode(_) | ChainError::Submit(_) => false,
            });
            *validator.registration.lock().expect("registration") =
                registration(account, SolverType::Cpu);
            assert_order_count(&client, account, 1).await;
            assert_eq!(validator.solver_reads.load(Ordering::Relaxed), 2);
        }

        #[tokio::test]
        async fn failed_registration_refresh_clears_the_previous_type() {
            let account = alice_account();
            let (validator, client) = client_for(
                account,
                Some(SolverType::Gpu),
                vec![type_bid(1)],
                MinerKind::Metal,
            );
            let _ = client
                .ensure_solver_registered()
                .await
                .expect("GPU registration");
            *validator.registration.lock().expect("registration") = Value::String("0xff".into());
            let error = client
                .ensure_solver_registered()
                .await
                .expect_err("bad refresh");
            assert_decode_error(&error);
            *validator.registration.lock().expect("registration") =
                registration(account, SolverType::Cpu);
            assert_order_count(&client, account, 0).await;
            assert_eq!(validator.solver_reads.load(Ordering::Relaxed), 3);
        }

        #[tokio::test]
        async fn a_cached_type_is_never_used_for_another_account() {
            let account = alice_account();
            let (validator, client) = client_for(
                account,
                Some(SolverType::Cpu),
                vec![type_bid(0)],
                MinerKind::Cpu,
            );
            assert_order_count(&client, account, 1).await;
            assert_order_count(&client, [2; 32], 0).await;
            assert_order_count(&client, account, 1).await;
            assert_eq!(validator.solver_reads.load(Ordering::Relaxed), 2);
        }

        #[tokio::test]
        async fn open_discovery_does_not_require_a_solver_type_read() {
            let account = alice_account();
            let (validator, client) = client_for(
                account,
                None,
                vec![scale_order(JobMode::Open)],
                MinerKind::Cpu,
            );
            assert_order_count(&client, account, 1).await;
            assert_eq!(validator.solver_reads.load(Ordering::Relaxed), 0);
        }
    }
}

#[cfg(test)]
mod default_topology_tests {
    use super::RealChainClient;
    use crate::chain::extrinsic::{default_topology_storage_key, hex_encode};
    use crate::chain::scale_types::TopologyMetaScale;
    use crate::chain::{BoxStream, ChainError, MinerKind, RpcTransport};
    use parity_scale_codec::Encode;
    use quantum_validation::AllowedValueSpec;
    use serde_json::Value;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    const HEAD: [u8; 32] = [0xaa; 32];
    const URL: &str = "ws://topology-fixture";

    struct ExpectedCall {
        method: &'static str,
        params: Value,
        response: Result<Value, ChainError>,
    }

    struct TopologyTransport {
        steps: Mutex<VecDeque<ExpectedCall>>,
    }

    #[async_trait::async_trait]
    impl RpcTransport for TopologyTransport {
        async fn request(
            &self,
            url: &str,
            method: &str,
            params: Value,
        ) -> Result<Value, ChainError> {
            let step = self
                .steps
                .lock()
                .map_err(|_| ChainError::Unavailable("RPC script lock poisoned".into()))?
                .pop_front()
                .ok_or_else(|| ChainError::Unavailable("unexpected extra RPC".into()))?;
            assert_eq!(url, URL);
            assert_eq!(method, step.method);
            assert_eq!(params, step.params);
            step.response
        }

        async fn subscribe(
            &self,
            _url: &str,
            _sub: &str,
            _params: Value,
            _unsub: &str,
        ) -> Result<BoxStream<'static, Result<Value, ChainError>>, ChainError> {
            Err(ChainError::Unavailable(
                "topology reads do not subscribe".into(),
            ))
        }
    }

    fn meta() -> TopologyMetaScale {
        TopologyMetaScale {
            nodes: vec![10, 30],
            edges: vec![(10, 30)],
            allowed_h_values: AllowedValueSpec::Set(vec![0]),
            allowed_j_values: AllowedValueSpec::Set(vec![-1000, 1000]),
            allowed_spin_values: AllowedValueSpec::Set(vec![-1000, 1000]),
            registered_at: 7,
        }
    }

    fn topology_steps(
        hash: Option<[u8; 32]>,
        metadata: Result<Value, ChainError>,
    ) -> Vec<ExpectedCall> {
        let mut steps = vec![
            ExpectedCall {
                method: "chain_getBlockHash",
                params: serde_json::json!([]),
                response: Ok(Value::String(hex_encode(&HEAD))),
            },
            ExpectedCall {
                method: "state_getStorage",
                params: serde_json::json!([
                    hex_encode(&default_topology_storage_key()),
                    hex_encode(&HEAD),
                ]),
                response: Ok(hash.map_or(Value::Null, |hash| Value::String(hex_encode(&hash)))),
            },
        ];
        if let Some(hash) = hash {
            steps.push(ExpectedCall {
                method: "state_call",
                params: serde_json::json!([
                    "QuantumPowApi_topology_meta",
                    hex_encode(&hash),
                    hex_encode(&HEAD),
                ]),
                response: metadata,
            });
        }
        steps
    }

    fn client(steps: Vec<ExpectedCall>) -> (RealChainClient, Arc<TopologyTransport>) {
        let transport = Arc::new(TopologyTransport {
            steps: Mutex::new(steps.into()),
        });
        let client = RealChainClient::with_transport(
            vec![URL.to_string()],
            String::new(),
            MinerKind::Cpu,
            transport.clone(),
        );
        (client, transport)
    }

    fn assert_drained(transport: &TopologyTransport) {
        assert!(transport
            .steps
            .lock()
            .expect("RPC script lock poisoned")
            .is_empty());
    }

    fn assert_topology_read_error(error: &ChainError, expected: &ChainError, message: &str) {
        assert_eq!(
            std::mem::discriminant(error),
            std::mem::discriminant(expected),
            "unexpected chain error variant: {error}"
        );
        assert!(error.to_string().contains(message), "{error}");
    }

    #[tokio::test]
    async fn default_topology_uses_one_head_and_needs_no_signer() {
        let hash = [0x11; 32];
        let metadata = meta();
        let response = Value::String(hex_encode(&Some(metadata.clone()).encode()));
        let (client, transport) = client(topology_steps(Some(hash), Ok(response)));
        let fetched = client
            .fetch_default_topology()
            .await
            .expect("fetch default topology")
            .expect("missing default topology");
        assert_eq!(fetched, (hash, metadata));
        assert!(client.pair.lock().expect("signer lock poisoned").is_none());
        assert_drained(&transport);
    }

    #[tokio::test]
    async fn unset_default_topology_skips_metadata() {
        let (client, transport) = client(topology_steps(None, Ok(Value::Null)));
        assert!(client
            .fetch_default_topology()
            .await
            .expect("fetch unset default topology")
            .is_none());
        assert_drained(&transport);
    }

    #[tokio::test]
    async fn stored_default_without_metadata_is_an_error() {
        let response = Value::String(hex_encode(&Option::<TopologyMetaScale>::None.encode()));
        let (client, transport) = client(topology_steps(Some([0x11; 32]), Ok(response)));
        let error = client
            .fetch_default_topology()
            .await
            .expect_err("expected missing metadata error");
        assert_topology_read_error(
            &error,
            &ChainError::Decode(String::new()),
            "is not registered",
        );
        assert_drained(&transport);
    }

    #[tokio::test]
    async fn topology_read_preserves_transport_and_decode_errors() {
        for (response, expected_variant, expected) in [
            (
                Err(ChainError::Unavailable("fixture unavailable".into())),
                ChainError::Unavailable(String::new()),
                "chain unavailable: fixture unavailable",
            ),
            (
                Ok(Value::String("0x03".into())),
                ChainError::Decode(String::new()),
                "decode error:",
            ),
        ] {
            let (client, transport) = client(topology_steps(Some([0x11; 32]), response));
            let error = client
                .fetch_default_topology()
                .await
                .expect_err("expected topology read failure");
            assert_topology_read_error(&error, &expected_variant, expected);
            assert_drained(&transport);
        }
    }

    #[tokio::test]
    async fn non_string_head_fails_before_storage_reads() {
        let (client, transport) = client(vec![ExpectedCall {
            method: "chain_getBlockHash",
            params: serde_json::json!([]),
            response: Ok(Value::Null),
        }]);
        let error = client
            .fetch_default_topology()
            .await
            .expect_err("expected invalid head error");
        assert_topology_read_error(
            &error,
            &ChainError::Decode(String::new()),
            "chain_getBlockHash not a string",
        );
        assert_drained(&transport);
    }
}
