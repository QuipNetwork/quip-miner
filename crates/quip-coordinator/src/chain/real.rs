//! Production chain client over Substrate JSON-RPC.
//!
//! Live end-to-end verification requires a running node. Offline unit tests
//! cover SCALE encode/decode, hybrid extrinsic assembly, and proof packing.
//! Methods return transport errors when no validator is reachable.

use super::extrinsic::{
    build_hybrid_signed_extrinsic, difficulties_storage_key, hex_decode, hex_encode,
    job_orders_storage_key, last_proof_block_storage_key, load_hybrid_pair, miner_identity_bytes,
    registered_topologies_prefix, topology_curve_c_storage_key, SignedExtensionContext,
};
use super::proof_encode::{build_quantum_proof, ProofBuildContext};
use super::scale_types::{
    encode_submit_proof_call, require_set_values, CurveCScale, DifficultyConfig, JobOrderScale,
    MiningSnapshotScale, OrderStatus, QBlockWithNonceScale, TopologyMetaScale,
};
use super::submit::{classify_receipt, Proof, SubmitAction};
use super::{
    ChainClient, ChainError, DecayParams, JobOrder, MiningSnapshot, QBlockRecord, TopologyInputs,
};
use crate::decay::{
    DEFAULT_BASE_MAX_ENERGY_MILLI, DEFAULT_C_EASY_MILLI, DEFAULT_C_HARD_MILLI,
    DEFAULT_C_KNEE_MILLI, EPOCH_LENGTH_BLOCKS,
};
use async_trait::async_trait;
use parity_scale_codec::{Decode, Encode};
use quantum_validation::AllowedValueSpec;
use quip_transaction_crypto::HybridPair;
use serde_json::Value;
use sp_core::crypto::Ss58Codec;
use sp_core::Pair as _;
use std::sync::Mutex;
use subxt::config::substrate::H256;
use subxt::ext::scale_value::{Composite, Primitive, ValueDef};
use subxt::transactions::TransactionStatus;
use tokio::sync::OnceCell;

/// subxt client for read-only, metadata-aware event decoding. Submit stays on
/// the hybrid-signed jsonrpsee path — subxt is used only to decode mempool
/// `JobProposed` events, which needs the runtime type registry.
type SubxtClient = subxt::OnlineClient<subxt::SubstrateConfig>;

/// Production chain client (RPC + hybrid-signed submit).
pub struct RealChainClient {
    /// Validator WebSocket / HTTP RPC URLs (primary first).
    pub validators: Vec<String>,
    /// Hybrid keystore path, `//DevUri`, or 32-byte hex seed.
    pub signer_key: String,
    /// Cached hybrid pair (loaded lazily from `signer_key`).
    pair: Mutex<Option<HybridPair>>,
    /// Last fetched mining snapshot — used by `submit_proof` to pack spins
    /// and derive the nonce. Updated by `fetch_mining_snapshot`.
    last_snapshot: Mutex<Option<MiningSnapshot>>,
    /// Last `allowed_spin` spec (full `AllowedValueSpec`, not just Set values).
    last_spin_spec: Mutex<Option<AllowedValueSpec<Vec<i32>>>>,
    /// Lazily-connected subxt client for mempool `JobProposed` event decoding.
    subxt: OnceCell<SubxtClient>,
}

impl RealChainClient {
    /// Construct a client over the given validators and signer material.
    #[must_use]
    pub fn new(validators: Vec<String>, signer_key: String) -> Self {
        Self {
            validators,
            signer_key,
            pair: Mutex::new(None),
            last_snapshot: Mutex::new(None),
            last_spin_spec: Mutex::new(None),
            subxt: OnceCell::new(),
        }
    }

    /// Lazily connect the read-only subxt client to the primary validator.
    async fn subxt_client(&self) -> Result<&SubxtClient, ChainError> {
        let url = self.primary_url()?.to_string();
        self.subxt
            .get_or_try_init(|| async move {
                SubxtClient::from_url(&url)
                    .await
                    .map_err(|e| ChainError::Unavailable(format!("subxt connect: {e}")))
            })
            .await
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

    async fn rpc_call(&self, method: &str, params: Value) -> Result<Value, ChainError> {
        let url = self.primary_url()?;
        rpc_request(url, method, params).await
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

    /// Run a `QuantumPowApi` runtime call at best head with SCALE-encoded
    /// `params`, returning the raw result bytes (`None` on null result).
    async fn quantum_pow_call_at_head(
        &self,
        method: &str,
        params: &[u8],
    ) -> Result<Option<Vec<u8>>, ChainError> {
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let block_hash = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))
            .and_then(|hex| hex_decode(hex).map_err(ChainError::Decode))?;
        let result = self
            .rpc_call(
                "state_call",
                Value::Array(vec![
                    Value::String(method.to_string()),
                    Value::String(hex_encode(params)),
                    Value::String(hex_encode(&block_hash)),
                ]),
            )
            .await?;
        match result.as_str() {
            None => Ok(None),
            Some(hex) => hex_decode(hex).map(Some).map_err(ChainError::Decode),
        }
    }

    /// SCALE-decode `topology_meta(hash)` at best head. `None` on a null result
    /// (hash never registered).
    async fn topology_meta_scale(
        &self,
        topology_hash: [u8; 32],
    ) -> Result<Option<TopologyMetaScale>, ChainError> {
        // Arg: SCALE(H256) == the 32 hash bytes verbatim.
        let Some(bytes) = self
            .quantum_pow_call_at_head("QuantumPowApi_topology_meta", &topology_hash)
            .await?
        else {
            return Ok(None);
        };
        let decoded: Option<TopologyMetaScale> =
            Decode::decode(&mut &bytes[..]).map_err(|e| ChainError::Decode(e.to_string()))?;
        Ok(decoded)
    }

    /// Enumerate the 32-byte topology hashes stored under `RegisteredTopologies`
    /// via paged `state_getKeysPaged`. Each `Blake2_128Concat` map key is
    /// `prefix ++ blake2_128(hash) ++ hash`, so the topology hash is the final
    /// 32 bytes of every returned key.
    async fn registered_topology_hashes(&self) -> Result<Vec<[u8; 32]>, ChainError> {
        const PAGE: u64 = 256;
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let at = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?
            .to_string();
        let prefix = registered_topologies_prefix();
        let prefix_hex = hex_encode(&prefix);
        let mut hashes = Vec::new();
        let mut start = prefix_hex.clone();
        loop {
            let page = self
                .rpc_call(
                    "state_getKeysPaged",
                    Value::Array(vec![
                        Value::String(prefix_hex.clone()),
                        Value::Number(PAGE.into()),
                        Value::String(start.clone()),
                        Value::String(at.clone()),
                    ]),
                )
                .await?;
            let Some(arr) = page.as_array() else {
                break;
            };
            if arr.is_empty() {
                break;
            }
            for k in arr {
                let key_hex = k.as_str().ok_or_else(|| {
                    ChainError::Decode("state_getKeysPaged item not a string".into())
                })?;
                let key = hex_decode(key_hex).map_err(ChainError::Decode)?;
                let len = key.len();
                if len < 32 {
                    return Err(ChainError::Decode(format!(
                        "RegisteredTopologies key too short ({len} bytes)"
                    )));
                }
                let mut h = [0u8; 32];
                #[expect(
                    clippy::indexing_slicing,
                    reason = "len >= 32 checked immediately above"
                )]
                h.copy_from_slice(&key[len - 32..]);
                hashes.push(h);
            }
            if (arr.len() as u64) < PAGE {
                break;
            }
            start = arr
                .last()
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| ChainError::Decode("empty page tail".into()))?;
        }
        Ok(hashes)
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
        // Discover recent order ids from system events at head, then storage-read
        // each JobOrders(order_id). Without a live node this returns transport
        // errors; with a node, empty open-order sets yield Ok(vec![]).
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let head_hex = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?;

        // Decode System.Events at head via subxt (metadata-aware) and collect
        // the order_id of every QuantumComputeMempool::JobProposed. This finds
        // orders proposed in the head block; still-open orders from earlier
        // blocks are re-surfaced as they are re-proposed or by the storage-status
        // filter below. Matches the Python reference (get_events_at → filter
        // module_id/event_id → attributes.order_id).
        let head_bytes = hex_decode(head_hex).map_err(ChainError::Decode)?;
        if head_bytes.len() != 32 {
            return Err(ChainError::Decode(format!(
                "head hash is {} bytes, expected 32",
                head_bytes.len()
            )));
        }
        let head_hash = H256::from_slice(&head_bytes);

        let client = self.subxt_client().await?;
        let at = client
            .at_block(head_hash)
            .await
            .map_err(|e| ChainError::Unavailable(format!("subxt at_block: {e}")))?;
        let events = at
            .events()
            .fetch()
            .await
            .map_err(|e| ChainError::Unavailable(format!("subxt events fetch: {e}")))?;

        let mut order_ids: Vec<u64> = Vec::new();
        for ev in events.iter() {
            let ev = ev.map_err(|e| ChainError::Decode(format!("event decode: {e}")))?;
            if ev.pallet_name() != "QuantumComputeMempool" || ev.event_name() != "JobProposed" {
                continue;
            }
            let fields = ev
                .decode_fields_unchecked_as::<Composite<()>>()
                .map_err(|e| ChainError::Decode(format!("JobProposed fields: {e}")))?;
            if let Some(oid) = order_id_from_fields(&fields) {
                order_ids.push(oid);
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

    async fn fetch_qblock_by_id(&self, qblock_id: u64) -> Result<Option<QBlockRecord>, ChainError> {
        // Arg: SCALE(u64).
        let Some(bytes) = self
            .quantum_pow_call_at_head("QuantumPowApi_qblock_by_id", &qblock_id.encode())
            .await?
        else {
            return Ok(None);
        };
        let decoded: Option<QBlockWithNonceScale> =
            Decode::decode(&mut &bytes[..]).map_err(|e| ChainError::Decode(e.to_string()))?;
        Ok(decoded.map(|qn| qn.into_record(qblock_id)))
    }

    async fn fetch_topology_meta(
        &self,
        topology_hash: [u8; 32],
    ) -> Result<Option<TopologyInputs>, ChainError> {
        self.topology_meta_scale(topology_hash)
            .await?
            .map(|m| m.into_inputs().map_err(ChainError::Decode))
            .transpose()
    }

    async fn fetch_registered_topologies(
        &self,
    ) -> Result<Vec<super::RegisteredTopology>, ChainError> {
        let hashes = self.registered_topology_hashes().await?;
        let mut out = Vec::with_capacity(hashes.len());
        for hash in hashes {
            // A hash listed in RegisteredTopologies must resolve; a null here is
            // a chain inconsistency worth surfacing rather than silently dropping.
            let Some(scale) = self.topology_meta_scale(hash).await? else {
                return Err(ChainError::Decode(format!(
                    "RegisteredTopologies listed {} but topology_meta returned null",
                    hex_encode(&hash)
                )));
            };
            out.push(scale.into_registered(hash).map_err(ChainError::Decode)?);
        }
        Ok(out)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "hybrid sign + RPC submit path is one linear procedure"
    )]
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

        // Chain state for signed extensions. `system_accountNextIndex` expects
        // an SS58-encoded address (the node rejects a hex account with a
        // "Base 58 requirement is violated" param error), so encode the
        // derived account with the default SS58 prefix.
        let account_ss58 =
            quip_transaction_crypto::account_id_from_public(&pair.public()).to_ss58check();
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

        let signed_ctx = SignedExtensionContext {
            account_nonce,
            genesis_hash,
            spec_version,
            transaction_version,
            tip: 0,
        };
        let ext = build_hybrid_signed_extrinsic(&pair, &call, &signed_ctx);

        // Submit via subxt's `author_submitAndWatchExtrinsic` and CONFIRM the
        // on-chain outcome, rather than treating pool acceptance as success.
        // We watch the status stream to in-best-block (not finality, to stay
        // responsive — a re-org after this is rare on the local validator and
        // the per-generation `current_best` reset bounds any wrong advance),
        // then check the extrinsic's own events: `ExtrinsicSuccess` -> Success,
        // `ExtrinsicFailed` -> classify the pallet error via `classify_receipt`.
        let client = self.subxt_client().await?;
        let tx_client = client
            .tx()
            .await
            .map_err(|e| ChainError::Unavailable(format!("subxt tx client: {e}")))?;
        let mut progress = tx_client
            .from_bytes(ext)
            .submit_and_watch()
            .await
            .map_err(|e| ChainError::Submit(format!("submit_and_watch: {e}")))?;
        loop {
            let status = progress
                .next()
                .await
                .ok_or_else(|| ChainError::Unavailable("tx status stream ended".into()))?
                .map_err(|e| ChainError::Unavailable(format!("tx progress: {e}")))?;
            match status {
                TransactionStatus::InBestBlock(in_block)
                | TransactionStatus::InFinalizedBlock(in_block) => {
                    return match in_block.wait_for_success().await {
                        Ok(_events) => {
                            tracing::info!(
                                block = %in_block.block_hash(),
                                "proof included and dispatched successfully"
                            );
                            Ok(SubmitAction::Success)
                        }
                        Err(e) => {
                            // Included, but the pallet dispatch failed
                            // (`ExtrinsicFailed`). Extract the `Pallet::Variant`
                            // name for a precise classification (e.g. InvalidNonce
                            // -> StopRoundStale/retry, not the default Fatal);
                            // fall back to the Display string otherwise.
                            let msg = match &e {
                                subxt::error::TransactionEventsError::ExtrinsicFailed(
                                    subxt::error::DispatchError::Module(m),
                                ) => m.details_string(),
                                other => other.to_string(),
                            };
                            tracing::warn!(error = %msg, "proof included but ExtrinsicFailed");
                            Ok(classify_receipt(Some(&msg)))
                        }
                    };
                }
                TransactionStatus::Invalid { message } => {
                    // Bad nonce / signature: stale for this round.
                    tracing::warn!(%message, "proof rejected as invalid before inclusion");
                    return Ok(classify_receipt(Some(&message)));
                }
                TransactionStatus::Dropped { message } | TransactionStatus::Error { message } => {
                    // Pool drop / transient node error: retry the candidate.
                    tracing::warn!(%message, "proof dropped by node before inclusion");
                    return Ok(SubmitAction::Retry);
                }
                // Validated / Broadcasted / NoLongerInBestBlock: keep watching.
                _ => {}
            }
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

/// Extract the `order_id` (u64) from a decoded `QuantumComputeMempool::
/// JobProposed` event's fields. Prefers the named `order_id` field (matching
/// the pallet's named event attributes); falls back to the first field of an
/// unnamed composite. `None` if absent or not an unsigned primitive. Extra
/// fields (proposer, reward, …) are ignored, so it survives event-shape growth.
fn order_id_from_fields(fields: &Composite<()>) -> Option<u64> {
    let value = match fields {
        Composite::Named(named) => named
            .iter()
            .find(|(name, _)| name == "order_id")
            .map(|(_, v)| v),
        Composite::Unnamed(vals) => vals.first(),
    }?;
    match &value.value {
        ValueDef::Primitive(Primitive::U128(n)) => u64::try_from(*n).ok(),
        _ => None,
    }
}

async fn rpc_request(url: &str, method: &str, params: Value) -> Result<Value, ChainError> {
    // Support both ws:// and http(s):// via jsonrpsee.
    if url.starts_with("ws://") || url.starts_with("wss://") {
        rpc_ws(url, method, params).await
    } else {
        rpc_http(url, method, params).await
    }
}

async fn rpc_http(url: &str, method: &str, params: Value) -> Result<Value, ChainError> {
    use jsonrpsee::core::client::ClientT;
    use jsonrpsee::http_client::HttpClientBuilder;

    let client = HttpClientBuilder::default()
        .build(url)
        .map_err(|e| ChainError::Unavailable(format!("http client: {e}")))?;
    let result: Value = client
        .request(method, rpc_params_from_value(params))
        .await
        .map_err(|e| ChainError::Unavailable(format!("rpc {method}: {e}")))?;
    Ok(result)
}

async fn rpc_ws(url: &str, method: &str, params: Value) -> Result<Value, ChainError> {
    use jsonrpsee::core::client::ClientT;
    use jsonrpsee::ws_client::WsClientBuilder;

    let client = WsClientBuilder::default()
        .build(url)
        .await
        .map_err(|e| ChainError::Unavailable(format!("ws client: {e}")))?;
    let result: Value = client
        .request(method, rpc_params_from_value(params))
        .await
        .map_err(|e| ChainError::Unavailable(format!("rpc {method}: {e}")))?;
    Ok(result)
}

fn rpc_params_from_value(params: Value) -> jsonrpsee::core::params::ArrayParams {
    match params {
        Value::Array(arr) => {
            let mut p = jsonrpsee::core::params::ArrayParams::new();
            for v in arr {
                let _ = p.insert(v);
            }
            p
        }
        other => {
            let mut p = jsonrpsee::core::params::ArrayParams::new();
            let _ = p.insert(other);
            p
        }
    }
}

#[cfg(test)]
mod tests {
    use super::order_id_from_fields;
    use subxt::ext::scale_value::{Composite, Value};

    #[test]
    fn order_id_from_named_ignores_other_fields() {
        // Real JobProposed carries more than order_id (proposer, reward, …);
        // the named lookup must pick order_id regardless of position.
        let fields = Composite::Named(vec![
            ("proposer".to_string(), Value::u128(999)),
            ("order_id".to_string(), Value::u128(42)),
            ("reward".to_string(), Value::u128(7)),
        ]);
        assert_eq!(order_id_from_fields(&fields), Some(42));
    }

    #[test]
    fn order_id_from_unnamed_takes_first() {
        let fields = Composite::Unnamed(vec![Value::u128(7), Value::u128(99)]);
        assert_eq!(order_id_from_fields(&fields), Some(7));
    }

    #[test]
    fn order_id_absent_wrong_type_or_empty_is_none() {
        // No order_id field.
        assert_eq!(
            order_id_from_fields(&Composite::Named(vec![("x".to_string(), Value::u128(1))])),
            None
        );
        // order_id present but not an unsigned primitive.
        let nested = Value::named_composite(vec![("inner".to_string(), Value::u128(1))]);
        assert_eq!(
            order_id_from_fields(&Composite::Named(vec![("order_id".to_string(), nested)])),
            None
        );
        // Empty unnamed composite.
        assert_eq!(order_id_from_fields(&Composite::Unnamed(vec![])), None);
    }
}
