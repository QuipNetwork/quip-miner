//! Hybrid-signed extrinsic assembly (mirrors Python
//! `substrate/scale_codec.py::_build_hybrid_signed_extrinsic`).

use parity_scale_codec::{Compact, Encode};
use quip_transaction_crypto::{account_id_from_public, HybridPair, HybridTxSignature};
use sp_core::hashing::blake2_256;
use sp_core::Pair as _;

/// Chain state required to build signed-extension extras / additional.
#[derive(Clone, Debug)]
pub struct SignedExtensionContext {
    /// Account nonce for `CheckNonce`.
    pub account_nonce: u32,
    /// Genesis block hash (also used for immortal era).
    pub genesis_hash: [u8; 32],
    /// Runtime `spec_version`.
    pub spec_version: u32,
    /// Runtime `transaction_version`.
    pub transaction_version: u32,
    /// Tip in plancks (`Compact<u128>` on the wire).
    pub tip: u128,
}

/// Build a signed v4 extrinsic for the given call bytes.
///
/// Layout:
/// ```text
/// compact_len(body) || body
/// body = 0x84 || MultiAddress::Id || AccountId32 || HybridTxSignature
///        || extra || call
/// ```
///
/// Signing payload = `call || extra || additional`, blake2_256-hashed when
/// longer than 256 bytes. The hybrid pair applies its own domain prefix
/// inside `HybridTxSignature::sign`.
#[must_use]
pub fn build_hybrid_signed_extrinsic(
    pair: &HybridPair,
    call_bytes: &[u8],
    ctx: &SignedExtensionContext,
) -> Vec<u8> {
    let account = account_id_from_public(&pair.public());
    let account_bytes: [u8; 32] = *AsRef::<[u8; 32]>::as_ref(&account);

    let extra = encode_extra(ctx.account_nonce, ctx.tip);
    let additional =
        encode_additional(ctx.spec_version, ctx.transaction_version, &ctx.genesis_hash);

    let payload = {
        let mut p = Vec::with_capacity(call_bytes.len() + extra.len() + additional.len());
        p.extend_from_slice(call_bytes);
        p.extend_from_slice(&extra);
        p.extend_from_slice(&additional);
        p
    };
    let payload_to_sign: Vec<u8> = if payload.len() > 256 {
        blake2_256(&payload).to_vec()
    } else {
        payload
    };

    let sig = HybridTxSignature::sign(pair, &payload_to_sign);
    let hybrid_sig_scale = sig.encode(); // public || signature

    let mut body = Vec::new();
    body.push(0x84); // v4 | signed
    body.push(0x00); // MultiAddress::Id
    body.extend_from_slice(&account_bytes);
    body.extend_from_slice(&hybrid_sig_scale);
    body.extend_from_slice(&extra);
    body.extend_from_slice(call_bytes);

    #[expect(
        clippy::cast_possible_truncation,
        reason = "signed extrinsic body length is far below u32::MAX"
    )]
    let mut full = Compact(body.len() as u32).encode();
    full.extend_from_slice(&body);
    full
}

/// Signed-extension extras in metadata order (immortal era, tip).
fn encode_extra(nonce: u32, tip: u128) -> Vec<u8> {
    let mut out = Vec::new();
    // AuthorizeCall, CheckNonZeroSender, CheckSpecVersion, CheckTxVersion,
    // CheckGenesis, CheckWeight, WeightReclaim → empty
    out.push(0x00); // CheckMortality: Era::Immortal
    out.extend(Compact(nonce).encode()); // CheckNonce
    out.extend(Compact(tip).encode()); // ChargeTransactionPayment
    out.push(0x00); // CheckMetadataHash: Mode::Disabled
    out
}

/// Signed-extension `additional_signed` in metadata order.
fn encode_additional(
    spec_version: u32,
    transaction_version: u32,
    genesis_hash: &[u8; 32],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&spec_version.to_le_bytes());
    out.extend_from_slice(&transaction_version.to_le_bytes());
    out.extend_from_slice(genesis_hash); // CheckGenesis
    out.extend_from_slice(genesis_hash); // CheckMortality (immortal → genesis)
    out.push(0x00); // CheckMetadataHash: Option::None
    out
}

/// Hex-encode bytes with a `0x` prefix (for RPC params).
#[must_use]
pub fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(2 + bytes.len() * 2);
    s.push_str("0x");
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Decode a `0x`-optional hex string.
///
/// # Errors
/// Returns an error when the hex string has odd length or a non-hex nibble.
pub fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    if !s.len().is_multiple_of(2) {
        return Err(format!("odd-length hex: {s}"));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

/// Load a hybrid pair from a Python-compatible keystore JSON path, a raw
/// 32-byte hex seed, or a `//DevUri` string.
///
/// # Errors
/// Returns an error when the path cannot be read/parsed, the seed is not 32
/// bytes, or a `//` URI fails to derive a pair.
pub fn load_hybrid_pair(signer_key: &str) -> Result<HybridPair, String> {
    let path = std::path::Path::new(signer_key);
    if path.exists() {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("keystore json: {e}"))?;
        let seed_hex = v
            .get("master_seed_hex")
            .and_then(|x| x.as_str())
            .ok_or_else(|| "keystore missing master_seed_hex".to_string())?;
        let seed_bytes = hex_decode(seed_hex)?;
        if seed_bytes.len() != 32 {
            return Err(format!(
                "master_seed must be 32 bytes, got {}",
                seed_bytes.len()
            ));
        }
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&seed_bytes);
        return Ok(HybridPair::from_seed(&seed));
    }
    if signer_key.starts_with("//") {
        return HybridPair::from_string(signer_key, None)
            .map_err(|e| format!("HybridPair::from_string: {e:?}"));
    }
    // Raw hex seed.
    let seed_bytes = hex_decode(signer_key)?;
    if seed_bytes.len() != 32 {
        return Err(format!(
            "signer seed must be 32-byte hex or keystore path, got {} bytes",
            seed_bytes.len()
        ));
    }
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&seed_bytes);
    Ok(HybridPair::from_seed(&seed))
}

/// Derive the 32-byte miner identity used in `derive_nonce`.
///
/// Matches the pallet: `blake2_256(account.encode())` where account is the
/// SCALE-encoded `AccountId32` (32 raw bytes, no length prefix beyond the
/// fixed array encoding).
#[must_use]
pub fn miner_identity_bytes(pair: &HybridPair) -> [u8; 32] {
    let account = account_id_from_public(&pair.public());
    // AccountId32 SCALE-encodes as the raw 32 bytes.
    blake2_256(account.encode().as_slice())
}

/// Substrate storage key for `QuantumComputeMempool.JobOrders(order_id)`.
#[must_use]
pub fn job_orders_storage_key(order_id: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(16 + 16 + 16 + 8);
    key.extend_from_slice(&twox128(b"QuantumComputeMempool"));
    key.extend_from_slice(&twox128(b"JobOrders"));
    // Blake2_128Concat(u64)
    let encoded_id = order_id.encode();
    key.extend_from_slice(&blake2_128(&encoded_id));
    key.extend_from_slice(&encoded_id);
    key
}

/// `QuantumPow` `Blake2_128Concat` storage-map key for a 32-byte topology hash
/// (`H256` encodes as its raw 32 bytes, no length prefix).
fn quantum_pow_map_key(item: &[u8], topology_hash: &[u8; 32]) -> Vec<u8> {
    let mut key = Vec::with_capacity(16 + 16 + 16 + 32);
    key.extend_from_slice(&twox128(b"QuantumPow"));
    key.extend_from_slice(&twox128(item));
    key.extend_from_slice(&blake2_128(topology_hash));
    key.extend_from_slice(topology_hash);
    key
}

/// `QuantumPow::Difficulties[topology_hash]` — base (un-decayed) `DifficultyConfig`.
#[must_use]
pub fn difficulties_storage_key(topology_hash: &[u8; 32]) -> Vec<u8> {
    quantum_pow_map_key(b"Difficulties", topology_hash)
}

/// `QuantumPow::TopologyCurveC[topology_hash]` — per-topology c-triple override.
#[must_use]
pub fn topology_curve_c_storage_key(topology_hash: &[u8; 32]) -> Vec<u8> {
    quantum_pow_map_key(b"TopologyCurveC", topology_hash)
}

/// `QuantumPow::RegisteredTopologies` storage-map prefix (twox128 pallet ++
/// twox128 item). Used with `state_getKeysPaged` to enumerate every registered
/// topology hash — the corrupted per-qblock `topology_hash` (migration-clobbered)
/// cannot be trusted, so the download rebuilds the mining-era timeline from this
/// authoritative set instead.
#[must_use]
pub fn registered_topologies_prefix() -> Vec<u8> {
    let mut key = Vec::with_capacity(32);
    key.extend_from_slice(&twox128(b"QuantumPow"));
    key.extend_from_slice(&twox128(b"RegisteredTopologies"));
    key
}

/// `QuantumPow::LastProofBlock` — plain `StorageValue` (block number of the last
/// winning proof).
#[must_use]
pub fn last_proof_block_storage_key() -> Vec<u8> {
    let mut key = Vec::with_capacity(32);
    key.extend_from_slice(&twox128(b"QuantumPow"));
    key.extend_from_slice(&twox128(b"LastProofBlock"));
    key
}

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
    #[expect(
        clippy::expect_used,
        reason = "Blake2bVar::new(16) is infallible for a fixed 16-byte digest"
    )]
    let mut hasher = Blake2bVar::new(16).expect("16-byte blake2b");
    hasher.update(data);
    let mut out = [0u8; 16];
    #[expect(
        clippy::expect_used,
        reason = "finalize into a 16-byte buffer matching the hasher size"
    )]
    {
        hasher.finalize_variable(&mut out).expect("finalize");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::scale_types::{encode_submit_proof_call, QuantumProof};
    use sp_core::{H256, U256};

    #[test]
    fn hybrid_extrinsic_has_signed_v4_prefix_and_compact_len() {
        let pair = HybridPair::from_string("//Alice", None).expect("alice");
        let proof = QuantumProof {
            topology_hash: H256::repeat_byte(1),
            nonce: U256::from(42u64),
            salt: [7u8; 32],
            solutions: vec![vec![0b01]],
            device_access_time_us: 0,
        };
        let call = encode_submit_proof_call(&proof);
        let ctx = SignedExtensionContext {
            account_nonce: 0,
            genesis_hash: [0x11; 32],
            spec_version: 100,
            transaction_version: 1,
            tip: 0,
        };
        let ext = build_hybrid_signed_extrinsic(&pair, &call, &ctx);
        // Compact length for a large body uses multi-byte encoding; body
        // still starts with 0x84 after the compact prefix.
        assert!(ext.len() > 100, "extrinsic too short: {}", ext.len());
        // Find signed version byte: after compact length.
        #[expect(
            clippy::indexing_slicing,
            reason = "extrinsic is asserted >100 bytes; compact prefix is 1–5 bytes"
        )]
        {
            let body_start = compact_len_bytes(ext[0]);
            assert_eq!(ext[body_start], 0x84);
            assert_eq!(ext[body_start + 1], 0x00); // MultiAddress::Id
        }
    }

    #[test]
    fn hybrid_sign_is_deterministic_for_same_payload() {
        let pair = HybridPair::from_string("//Alice", None).expect("alice");
        let msg = b"test-payload-for-hybrid-sign";
        let a = HybridTxSignature::sign(&pair, msg);
        let b = HybridTxSignature::sign(&pair, msg);
        assert_eq!(a.encode(), b.encode());
        // Public is 1344 bytes; signature 2484; SCALE is just concatenation.
        assert_eq!(a.encode().len(), 1344 + 2484);
    }

    #[test]
    fn job_orders_key_is_stable() {
        let k1 = job_orders_storage_key(7);
        let k2 = job_orders_storage_key(7);
        assert_eq!(k1, k2);
        assert_ne!(job_orders_storage_key(7), job_orders_storage_key(8));
        // prefix (32) + blake2_128 (16) + u64 (8) = 56
        assert_eq!(k1.len(), 56);
    }

    #[test]
    fn miner_identity_is_32_bytes() {
        let pair = HybridPair::from_string("//Alice", None).expect("alice");
        let id = miner_identity_bytes(&pair);
        assert_eq!(id.len(), 32);
        // Deterministic.
        assert_eq!(id, miner_identity_bytes(&pair));
    }

    fn compact_len_bytes(first: u8) -> usize {
        match first & 0b11 {
            0b00 => 1,
            0b01 => 2,
            0b10 => 4,
            _ => 1 + ((first >> 2) as usize + 4),
        }
    }
}
