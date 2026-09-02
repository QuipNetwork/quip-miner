//! H4 hybrid-signed extrinsic assembly for the Quip runtime transaction format.

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

/// Load a hybrid pair from signer material.
///
/// Accepted forms:
/// - a Python-compatible keystore JSON path
/// - a 32-byte hex seed, with or without the `0x` prefix
/// - a `//DevUri` string such as `//Alice`
/// - any substrate secret URI, including a bare BIP39 mnemonic
///
/// # Errors
/// Returns an error when the path cannot be read or parsed, the keystore seed
/// is not 32 bytes, or the secret URI fails to derive a pair.
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
    // A 32-byte hex seed, with or without the 0x prefix. Checked before the
    // secret-URI branch because a bare hex string is also a legal URI and
    // would derive a different key through the phrase path.
    let hex_body = signer_key.strip_prefix("0x").unwrap_or(signer_key);
    if hex_body.len() == 64 && hex_body.chars().all(|c| c.is_ascii_hexdigit()) {
        let seed_bytes = hex_decode(hex_body)?;
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&seed_bytes);
        return Ok(HybridPair::from_seed(&seed));
    }
    // Any substrate secret URI: a dev path (//Alice), a BIP39 mnemonic, or a
    // mnemonic with derivation and password (phrase//hard/soft///password).
    HybridPair::from_string(signer_key, None)
        .map_err(|e| format!("cannot derive a signer from {signer_key:?}: {e:?}"))
}

/// The 32-byte `AccountId32` this pair signs and pays with.
///
/// This is the address the chain debits for fees, reserves deposits from, and
/// keys every per-signer storage map by. It is **not** [`miner_identity_bytes`].
#[must_use]
pub fn signer_account_bytes(pair: &HybridPair) -> [u8; 32] {
    *AsRef::<[u8; 32]>::as_ref(&account_id_from_public(&pair.public()))
}

/// Derive the 32-byte miner identity used in `derive_nonce`.
///
/// Matches the pallet: `blake2_256(account.encode())` where account is the
/// SCALE-encoded `AccountId32` (32 raw bytes, no length prefix beyond the
/// fixed array encoding). This is a `PoW` input only — it is not an address,
/// holds no balance, and nothing on chain is keyed by it.
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

/// `QuantumPow` `Blake2_128Concat` storage-map key for any 32-byte map key.
/// Both `H256` and `AccountId32` encode as their raw 32 bytes, with no length
/// prefix.
fn quantum_pow_map_key(item: &[u8], map_key: &[u8; 32]) -> Vec<u8> {
    let mut key = Vec::with_capacity(16 + 16 + 16 + 32);
    key.extend_from_slice(&twox128(b"QuantumPow"));
    key.extend_from_slice(&twox128(item));
    key.extend_from_slice(&blake2_128(map_key));
    key.extend_from_slice(map_key);
    key
}

/// `QuantumPow::Miners[account]` — presence proves the account registered and
/// its deposit is reserved. `submit_proof` rejects any other account.
#[must_use]
pub fn miners_storage_key(account: &[u8; 32]) -> Vec<u8> {
    quantum_pow_map_key(b"Miners", account)
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

/// `QuantumPow::LastProofBlock` — plain `StorageValue` (block number of the last
/// winning proof).
#[must_use]
pub fn last_proof_block_storage_key() -> Vec<u8> {
    let mut key = Vec::with_capacity(32);
    key.extend_from_slice(&twox128(b"QuantumPow"));
    key.extend_from_slice(&twox128(b"LastProofBlock"));
    key
}

/// Substrate storage key for the `QuantumPow.DefaultTopology` storage value.
///
/// A `StorageValue` has no key hasher, so the key is the two twox128 name
/// hashes concatenated.
#[must_use]
pub fn default_topology_storage_key() -> Vec<u8> {
    let mut key = Vec::with_capacity(32);
    key.extend_from_slice(&twox128(b"QuantumPow"));
    key.extend_from_slice(&twox128(b"DefaultTopology"));
    key
}

/// `QuantumPow::QBlocks[block_number]` — the accepted proof for one block.
///
/// The map is `Blake2_128Concat` over `BlockNumberFor<T>`, which is `u32` on
/// this runtime. The value begins with the winning account.
#[must_use]
pub fn qblocks_storage_key(block_number: u32) -> Vec<u8> {
    let mut key = Vec::with_capacity(16 + 16 + 16 + 4);
    key.extend_from_slice(&twox128(b"QuantumPow"));
    key.extend_from_slice(&twox128(b"QBlocks"));
    let encoded = block_number.encode();
    key.extend_from_slice(&blake2_128(&encoded));
    key.extend_from_slice(&encoded);
    key
}

/// `MinerRegistry::NodeDescriptors[account]` — presence proves the descriptor.
#[must_use]
pub fn node_descriptors_storage_key(account: &[u8; 32]) -> Vec<u8> {
    let mut key = Vec::with_capacity(16 + 16 + 16 + 32);
    key.extend_from_slice(&twox128(b"MinerRegistry"));
    key.extend_from_slice(&twox128(b"NodeDescriptors"));
    key.extend_from_slice(&blake2_128(account));
    key.extend_from_slice(account);
    key
}

/// `MinerRegistry::ParticipantsByQBlock[qblock_id][account]`.
///
/// A double map key is `twox128(pallet) ++ twox128(item) ++ blake2_128(k1) ++
/// k1 ++ blake2_128(k2) ++ k2`. Presence proves participation.
#[must_use]
pub fn participants_by_qblock_storage_key(qblock_id: u64, account: &[u8; 32]) -> Vec<u8> {
    let encoded_id = qblock_id.encode();
    let mut key = Vec::with_capacity(16 + 16 + 16 + 8 + 16 + 32);
    key.extend_from_slice(&twox128(b"MinerRegistry"));
    key.extend_from_slice(&twox128(b"ParticipantsByQBlock"));
    key.extend_from_slice(&blake2_128(&encoded_id));
    key.extend_from_slice(&encoded_id);
    key.extend_from_slice(&blake2_128(account));
    key.extend_from_slice(account);
    key
}

/// Substrate `Twox128` of `data`: two `XxHash64` digests, seeds 0 and 1,
/// concatenated little-endian.
#[must_use]
pub fn twox128(data: &[u8]) -> [u8; 16] {
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

/// Substrate `Blake2_128` of `data`.
///
/// # Panics
/// Panics only if the blake2 crate rejects a 16-byte digest, which it does not.
#[must_use]
pub fn blake2_128(data: &[u8]) -> [u8; 16] {
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

/// Hash of a submitted extrinsic, as the node reports it in a block body.
///
/// Substrate hashes the full SCALE-encoded extrinsic, length prefix included,
/// with Blake2-256. Inclusion is confirmed by matching this against the
/// extrinsics in the block the status stream named.
#[must_use]
pub fn extrinsic_hash(ext: &[u8]) -> [u8; 32] {
    blake2_256(ext)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::scale_types::{encode_submit_proof_call, QuantumProof};
    use quip_transaction_crypto_core::{HYBRID_PUBLIC_LEN, HYBRID_SIGNATURE_LEN};
    use sp_core::{H256, U256};

    /// `hex_encode` already prepends `0x` (see its doc comment); a caller that
    /// wraps its output in another `format!("0x{}", ...)` doubles the prefix.
    /// This pins the contract so that regression is a one-line diff away from
    /// this test failing, not just a doc comment to reread.
    #[test]
    fn hex_encode_prepends_exactly_one_0x_prefix() {
        let out = hex_encode(&[0xab_u8; 32]);
        assert_eq!(out.len(), 66, "0x plus 64 hex digits for a 32-byte hash");
        assert!(!out.starts_with("0x0x"));
        assert_eq!(out, format!("0x{}", "ab".repeat(32)));
    }

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
        // The envelope SCALE-encodes as the pinned suite's public key followed
        // by its fixed-size signature buffer.
        assert_eq!(a.encode().len(), HYBRID_PUBLIC_LEN + HYBRID_SIGNATURE_LEN);
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

    #[test]
    fn the_signing_account_is_never_the_miner_identity() {
        let pair = HybridPair::from_string("//Alice", None).expect("alice");
        let account = signer_account_bytes(&pair);
        let identity = miner_identity_bytes(&pair);
        // The identity is a hash *of* the account, so the two addresses can
        // never coincide. Paying fees from one while funding the other is the
        // shape of the bug this asserts against.
        assert_ne!(account, identity);
        assert_eq!(identity, blake2_256(account.as_slice()));
        assert_eq!(account, signer_account_bytes(&pair));
    }

    /// Pins the identity to the pallet expression it mirrors, character for
    /// character: `pallet_quantum_pow::Pallet::account_to_bytes` is
    /// `blake2_256(&account.encode())`, and `submit_proof` feeds its result to
    /// `derive_nonce` before rejecting a mismatch with `InvalidNonce`. Drift
    /// here is silent — every proof simply stops verifying — so assert against
    /// the SCALE encoding rather than against the raw account bytes.
    #[test]
    fn the_miner_identity_mirrors_the_pallet_account_to_bytes() {
        let pair = HybridPair::from_string("//Alice", None).expect("alice");
        let account_id = account_id_from_public(&pair.public());

        // The pallet expression, spelled out.
        let pallet_account_to_bytes = blake2_256(&account_id.encode());
        assert_eq!(miner_identity_bytes(&pair), pallet_account_to_bytes);

        // The two agree only because AccountId32 SCALE-encodes as its 32 raw
        // bytes with no length prefix. Pin that, so a codec change surfaces
        // here instead of as InvalidNonce on a live chain.
        let raw: [u8; 32] = *AsRef::<[u8; 32]>::as_ref(&account_id);
        assert_eq!(account_id.encode(), raw.to_vec());
    }

    /// The coordinator derives job nonces with `quip_protocol::derive`, while
    /// the pallet validates them with `quantum_validation`. These are separate
    /// implementations of the same BLAKE3 composition, so a change to either
    /// one alone invalidates every proof. Cross-check them over the identity.
    #[test]
    fn both_derive_nonce_implementations_agree_over_the_identity() {
        let pair = HybridPair::from_string("//Alice", None).expect("alice");
        let identity = miner_identity_bytes(&pair);
        let head = [0x33u8; 32];
        let salt = [0x44u8; 32];

        let coordinator = quip_protocol::derive::derive_nonce(head, identity, salt);
        let pallet: U256 = quantum_validation::derive_nonce(&head, &identity, &salt);
        assert_eq!(pallet, U256::from_big_endian(&coordinator));

        // The account is not an accepted substitute for the identity anywhere
        // on this path.
        let from_account =
            quip_protocol::derive::derive_nonce(head, signer_account_bytes(&pair), salt);
        assert_ne!(coordinator, from_account);
    }

    #[test]
    fn the_miners_key_ends_with_the_account() {
        let account = [5u8; 32];
        let key = miners_storage_key(&account);
        assert_eq!(key.len(), 16 + 16 + 16 + 32);
        assert_eq!(key.get(key.len() - 32..), Some(account.as_slice()));
        #[expect(
            clippy::indexing_slicing,
            reason = "length is asserted to 80 bytes above"
        )]
        {
            assert_eq!(&key[..16], &twox128(b"QuantumPow")[..]);
            assert_eq!(&key[16..32], &twox128(b"Miners")[..]);
            assert_eq!(&key[32..48], &blake2_128(&account)[..]);
        }
        assert_ne!(
            miners_storage_key(&[1u8; 32]),
            miners_storage_key(&[2u8; 32])
        );
        // Same hasher shape, different item: the two must not collide.
        assert_ne!(
            miners_storage_key(&account),
            difficulties_storage_key(&account)
        );
    }

    #[test]
    fn the_extrinsic_hash_is_blake2_256_of_the_whole_blob() {
        let ext = vec![1u8, 2, 3, 4];
        assert_eq!(extrinsic_hash(&ext), blake2_256(&ext));
        assert_ne!(extrinsic_hash(&ext), extrinsic_hash(&[1u8, 2, 3, 5]));
    }

    fn compact_len_bytes(first: u8) -> usize {
        match first & 0b11 {
            0b00 => 1,
            0b01 => 2,
            0b10 => 4,
            _ => 1 + ((first >> 2) as usize + 4),
        }
    }

    #[test]
    fn dev_phrase_with_a_derivation_path_matches_the_bare_dev_uri() {
        // sp_core substitutes DEV_PHRASE for an empty phrase, so the full
        // phrase plus //Alice must derive the same pair as //Alice alone.
        // This exercises the mnemonic branch against a value we can check.
        let uri = format!("{}//Alice", sp_core::crypto::DEV_PHRASE);
        let from_phrase = load_hybrid_pair(&uri).expect("mnemonic URI derives");
        let from_dev_uri = load_hybrid_pair("//Alice").expect("dev URI derives");
        assert_eq!(
            from_phrase.public().encode(),
            from_dev_uri.public().encode()
        );
    }

    #[test]
    fn bare_dev_phrase_derives_a_pair() {
        let pair = load_hybrid_pair(sp_core::crypto::DEV_PHRASE).expect("bare phrase derives");
        // The root account differs from //Alice; only assert it is not that.
        let alice = load_hybrid_pair("//Alice").expect("dev URI derives");
        assert_ne!(pair.public().encode(), alice.public().encode());
    }

    #[test]
    fn garbage_signer_material_is_rejected_with_the_input_named() {
        // HybridPair is not Debug, so Result::expect_err does not compile.
        let err = load_hybrid_pair("not a key")
            .err()
            .expect("garbage is rejected");
        assert!(err.contains("not a key"), "error names the input: {err}");
    }

    #[test]
    fn default_topology_key_is_the_two_storage_hashes() {
        let key = default_topology_storage_key();
        assert_eq!(key.len(), 32);
        #[expect(
            clippy::indexing_slicing,
            reason = "length is asserted to 32 bytes above"
        )]
        {
            assert_eq!(&key[..16], &twox128(b"QuantumPow")[..]);
            assert_eq!(&key[16..], &twox128(b"DefaultTopology")[..]);
        }
    }

    #[test]
    fn the_qblocks_key_is_a_blake2_128_concat_map_key() {
        let key = qblocks_storage_key(1_121_300);
        assert_eq!(key.len(), 16 + 16 + 16 + 4);
        // Blake2_128Concat appends the SCALE-encoded key after its hash.
        assert_eq!(
            key.get(key.len() - 4..),
            Some(1_121_300u32.encode().as_slice())
        );
        assert_ne!(qblocks_storage_key(1), qblocks_storage_key(2));
    }

    #[test]
    fn the_node_descriptors_key_ends_with_the_account() {
        let account = [7u8; 32];
        let key = node_descriptors_storage_key(&account);
        assert_eq!(key.len(), 16 + 16 + 16 + 32);
        assert_eq!(key.get(key.len() - 32..), Some(account.as_slice()));
        assert_ne!(
            node_descriptors_storage_key(&[1u8; 32]),
            node_descriptors_storage_key(&[2u8; 32])
        );
    }

    #[test]
    fn the_participants_key_is_a_blake2_128_concat_double_map() {
        let account = [9u8; 32];
        let key = participants_by_qblock_storage_key(42, &account);
        assert_eq!(key.len(), 16 + 16 + 16 + 8 + 16 + 32);
        assert_eq!(key.get(key.len() - 32..), Some(account.as_slice()));
        let encoded = 42u64.encode();
        assert_eq!(key.get(32 + 16..32 + 16 + 8), Some(encoded.as_slice()));
        assert_ne!(
            participants_by_qblock_storage_key(1, &account),
            participants_by_qblock_storage_key(2, &account)
        );
    }
}
