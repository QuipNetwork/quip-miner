//! Pending `QuantumPow.submit_proof` extrinsics from the transaction pool.
//!
//! `author_pendingExtrinsics` returns every pool entry as opaque bytes. The
//! coordinator wants only signed `submit_proof` calls, so the decoder here
//! walks the signed v4 envelope the coordinator itself emits
//! (`extrinsic::build_hybrid_signed_extrinsic`) and returns nothing for
//! every other shape. That is not an error: the pool holds every kind of
//! call.

use super::extrinsic::{encode_additional, extrinsic_hash, signing_payload, RuntimeTxContext};
use super::scale_types::{QuantumProof, QUANTUM_POW_PALLET_INDEX, SUBMIT_PROOF_CALL_INDEX};
use parity_scale_codec::{Compact, Decode, Input};
use quip_transaction_crypto::{HybridPair, HybridTxSignature};
use sp_core::Pair as _;

/// A `QuantumPow.submit_proof` waiting in the pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingProof {
    /// `blake2_256` of the extrinsic bytes. Keys the per-round verdict cache.
    pub extrinsic_hash: [u8; 32],
    /// The signing `AccountId32`.
    pub account: [u8; 32],
    /// The proof as submitted.
    pub proof: QuantumProof,
}

/// Version byte of a signed v4 extrinsic, the only envelope the coordinator emits.
const SIGNED_V4: u8 = 0x84;
/// `MultiAddress::Id` discriminant.
const MULTI_ADDRESS_ID: u8 = 0x00;
/// `Era::Immortal` is one zero byte. A mortal era is two bytes.
const ERA_IMMORTAL: u8 = 0x00;
/// `CheckMetadataHash` mode: no metadata hash in `additional_signed`.
const METADATA_HASH_DISABLED: u8 = 0x00;
/// `CheckMetadataHash` mode: the runtime metadata hash in `additional_signed`.
const METADATA_HASH_ENABLED: u8 = 0x01;

/// Decode one pool entry and check its signature against `ctx`.
///
/// `None` for anything that is not a signed v4 `submit_proof` whose signature
/// holds: unsigned extrinsics, other calls, other envelope versions, a compact
/// length that differs from the bytes after it, or bytes left over after the
/// call. Also `None` for a mortal era. The coordinator signs immortal
/// extrinsics, so it does not look up era birth blocks.
///
/// A signer that used `CheckMetadataHash` mode Enabled committed to the
/// runtime metadata hash, which the coordinator does not compute. For that
/// mode only the account binding is checked: the embedded public key must
/// derive the claimed account. The pool-watch judge still replays the qblock
/// gates on the proof, and the proof nonce derives from that account.
#[must_use]
pub fn decode_pending_proof(bytes: &[u8], ctx: &RuntimeTxContext) -> Option<PendingProof> {
    let input = &mut &bytes[..];
    let declared = Compact::<u32>::decode(input).ok()?.0;
    if usize::try_from(declared).ok()? != input.len() {
        return None;
    }
    if input.read_byte().ok()? != SIGNED_V4 {
        return None;
    }
    if input.read_byte().ok()? != MULTI_ADDRESS_ID {
        return None;
    }
    let account = <[u8; 32]>::decode(input).ok()?;
    let signature = HybridTxSignature::decode(input).ok()?;
    let extra_start = offset(bytes, input)?;
    let metadata_hash_mode = read_signed_extensions(input)?;
    let call_start = offset(bytes, input)?;
    if input.read_byte().ok()? != QUANTUM_POW_PALLET_INDEX {
        return None;
    }
    if input.read_byte().ok()? != SUBMIT_PROOF_CALL_INDEX {
        return None;
    }
    let proof = QuantumProof::decode(input).ok()?;
    if !input.is_empty() {
        return None;
    }
    let extra = bytes.get(extra_start..call_start)?;
    let call = bytes.get(call_start..)?;
    if !signature_holds(&signature, &account, call, extra, metadata_hash_mode, ctx) {
        return None;
    }
    Some(PendingProof {
        extrinsic_hash: extrinsic_hash(bytes),
        account,
        proof,
    })
}

/// Position of `rest` inside `bytes`, where `rest` is a suffix of `bytes`.
fn offset(bytes: &[u8], rest: &[u8]) -> Option<usize> {
    bytes.len().checked_sub(rest.len())
}

/// Read the signed extensions in metadata order, mirroring
/// `extrinsic::encode_extra`: era, nonce, tip, metadata-hash mode.
///
/// Returns the metadata-hash mode. `None` for a mortal era or an unknown mode.
fn read_signed_extensions(input: &mut &[u8]) -> Option<u8> {
    if input.read_byte().ok()? != ERA_IMMORTAL {
        return None;
    }
    let _nonce = Compact::<u32>::decode(input).ok()?;
    let _tip = Compact::<u128>::decode(input).ok()?;
    match input.read_byte().ok()? {
        mode @ (METADATA_HASH_DISABLED | METADATA_HASH_ENABLED) => Some(mode),
        _ => None,
    }
}

/// Whether `signature` binds `account` and, unless the signer committed to
/// a metadata hash, signs `call || extra || additional` under `ctx`.
fn signature_holds(
    signature: &HybridTxSignature,
    account: &[u8; 32],
    call: &[u8],
    extra: &[u8],
    metadata_hash_mode: u8,
    ctx: &RuntimeTxContext,
) -> bool {
    let derived = signature.derived_account_id();
    if AsRef::<[u8; 32]>::as_ref(&derived) != account {
        return false;
    }
    if metadata_hash_mode == METADATA_HASH_ENABLED {
        return true;
    }
    let additional =
        encode_additional(ctx.spec_version, ctx.transaction_version, &ctx.genesis_hash);
    let payload = signing_payload(call, extra, &additional);
    HybridPair::verify(&signature.signature, &payload, &signature.public)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::extrinsic::{
        build_hybrid_signed_extrinsic, extrinsic_hash, signer_account_bytes, SignedExtensionContext,
    };
    use crate::chain::scale_types::{encode_register_miner_call, encode_submit_proof_call};
    use parity_scale_codec::Encode;
    use sp_core::{H256, U256};

    fn proof() -> QuantumProof {
        QuantumProof {
            topology_hash: H256::repeat_byte(1),
            nonce: U256::from(42u64),
            salt: [7u8; 32],
            solutions: vec![vec![0b01], vec![0b10]],
            device_access_time_us: 5,
        }
    }

    fn signed(call: &[u8]) -> (HybridPair, Vec<u8>) {
        let pair = HybridPair::from_string("//Alice", None).expect("alice");
        let ctx = SignedExtensionContext {
            account_nonce: 3,
            genesis_hash: [0x11; 32],
            spec_version: 100,
            transaction_version: 1,
            tip: 0,
        };
        let ext = build_hybrid_signed_extrinsic(&pair, call, &ctx);
        (pair, ext)
    }

    use crate::chain::extrinsic::{
        assemble_signed_extrinsic, encode_additional, signing_payload, RuntimeTxContext,
    };
    use quip_transaction_crypto::HybridTxSignature;

    /// Matches the `SignedExtensionContext` in `signed`.
    const CTX: RuntimeTxContext = RuntimeTxContext {
        genesis_hash: [0x11; 32],
        spec_version: 100,
        transaction_version: 1,
    };

    /// Signed extensions with a chosen era and metadata-hash mode.
    fn extra_with(era: &[u8], mode: u8) -> Vec<u8> {
        let mut extra = era.to_vec();
        extra.extend(Compact(3u32).encode());
        extra.extend(Compact(0u128).encode());
        extra.push(mode);
        extra
    }

    /// A signed extrinsic with chosen extensions and additional data.
    fn signed_with(pair: &HybridPair, call: &[u8], extra: &[u8], additional: &[u8]) -> Vec<u8> {
        let sig = HybridTxSignature::sign(pair, &signing_payload(call, extra, additional));
        assemble_signed_extrinsic(&signer_account_bytes(pair), &sig, extra, call)
    }

    /// `additional` for `CheckMetadataHash` mode Enabled with a made-up hash.
    fn additional_with_metadata_hash() -> Vec<u8> {
        let mut additional = encode_additional(100, 1, &[0x11; 32]);
        let _ = additional.pop(); // Option::None
        additional.push(0x01); // Option::Some
        additional.extend_from_slice(&[0xab; 32]);
        additional
    }

    /// Replace the account field with Bob's account.
    fn with_bob_as_signer(mut ext: Vec<u8>, alice: &HybridPair) -> Vec<u8> {
        let bob = HybridPair::from_string("//Bob", None).expect("bob");
        let at = ext
            .windows(32)
            .position(|w| w == signer_account_bytes(alice))
            .expect("account field");
        let _ = ext.splice(at..at + 32, signer_account_bytes(&bob));
        ext
    }

    /// Re-prefix the body with a compact length off by `delta`.
    fn with_length_off_by(ext: &[u8], delta: i64) -> Vec<u8> {
        let input = &mut &ext[..];
        let len = Compact::<u32>::decode(input).expect("len").0;
        let wrong = u32::try_from(i64::from(len) + delta).expect("len fits");
        let mut out = Compact(wrong).encode();
        out.extend_from_slice(input);
        out
    }

    #[test]
    fn a_short_or_long_compact_length_decodes_to_nothing() {
        let (_, ext) = signed(&encode_submit_proof_call(&proof()));
        assert!(decode_pending_proof(&with_length_off_by(&ext, -1), &CTX).is_none());
        assert!(decode_pending_proof(&with_length_off_by(&ext, 1), &CTX).is_none());
    }

    #[test]
    fn a_signature_for_another_chain_decodes_to_nothing() {
        let (_, ext) = signed(&encode_submit_proof_call(&proof()));
        let other = RuntimeTxContext {
            genesis_hash: [0x22; 32],
            ..CTX
        };
        assert!(decode_pending_proof(&ext, &other).is_none());
    }

    #[test]
    fn a_signature_for_another_runtime_decodes_to_nothing() {
        let (_, ext) = signed(&encode_submit_proof_call(&proof()));
        let upgraded = RuntimeTxContext {
            spec_version: 101,
            ..CTX
        };
        assert!(decode_pending_proof(&ext, &upgraded).is_none());
    }

    #[test]
    fn a_tampered_call_decodes_to_nothing() {
        let (_, mut ext) = signed(&encode_submit_proof_call(&proof()));
        // The last byte belongs to `device_access_time_us`, so the bytes still
        // decode and only the signature catches the change.
        *ext.last_mut().expect("byte") ^= 0x01;
        assert!(decode_pending_proof(&ext, &CTX).is_none());
    }

    #[test]
    fn a_swapped_account_decodes_to_nothing() {
        let (alice, ext) = signed(&encode_submit_proof_call(&proof()));
        assert!(decode_pending_proof(&with_bob_as_signer(ext, &alice), &CTX).is_none());
    }

    #[test]
    fn a_mortal_era_decodes_to_nothing() {
        let alice = HybridPair::from_string("//Alice", None).expect("alice");
        let call = encode_submit_proof_call(&proof());
        let ext = signed_with(
            &alice,
            &call,
            &extra_with(&[0x15, 0x00], 0x00),
            &encode_additional(100, 1, &[0x11; 32]),
        );
        assert!(decode_pending_proof(&ext, &CTX).is_none());
    }

    #[test]
    fn a_metadata_hash_signer_is_accepted_on_its_account_binding() {
        let alice = HybridPair::from_string("//Alice", None).expect("alice");
        let call = encode_submit_proof_call(&proof());
        let ext = signed_with(
            &alice,
            &call,
            &extra_with(&[0x00], 0x01),
            &additional_with_metadata_hash(),
        );
        let pending = decode_pending_proof(&ext, &CTX).expect("decodes");
        assert_eq!(pending.account, signer_account_bytes(&alice));
    }

    #[test]
    fn a_metadata_hash_entry_with_a_foreign_account_decodes_to_nothing() {
        let alice = HybridPair::from_string("//Alice", None).expect("alice");
        let call = encode_submit_proof_call(&proof());
        let ext = signed_with(
            &alice,
            &call,
            &extra_with(&[0x00], 0x01),
            &additional_with_metadata_hash(),
        );
        assert!(decode_pending_proof(&with_bob_as_signer(ext, &alice), &CTX).is_none());
    }

    #[test]
    fn an_unknown_metadata_hash_mode_decodes_to_nothing() {
        let alice = HybridPair::from_string("//Alice", None).expect("alice");
        let call = encode_submit_proof_call(&proof());
        let ext = signed_with(
            &alice,
            &call,
            &extra_with(&[0x00], 0x02),
            &encode_additional(100, 1, &[0x11; 32]),
        );
        assert!(decode_pending_proof(&ext, &CTX).is_none());
    }

    #[test]
    fn a_signed_submit_proof_round_trips() {
        let (pair, ext) = signed(&encode_submit_proof_call(&proof()));
        let pending = decode_pending_proof(&ext, &CTX).expect("decodes");
        assert_eq!(pending.account, signer_account_bytes(&pair));
        assert_eq!(pending.proof, proof());
        assert_eq!(pending.extrinsic_hash, extrinsic_hash(&ext));
    }

    #[test]
    fn another_call_decodes_to_nothing() {
        let (_, ext) = signed(&encode_register_miner_call());
        assert!(decode_pending_proof(&ext, &CTX).is_none());
    }

    #[test]
    fn an_unsigned_extrinsic_decodes_to_nothing() {
        let call = encode_submit_proof_call(&proof());
        let mut body = vec![0x04];
        body.extend_from_slice(&call);
        let mut ext = Compact(u32::try_from(body.len()).expect("len")).encode();
        ext.extend_from_slice(&body);
        assert!(decode_pending_proof(&ext, &CTX).is_none());
    }

    #[test]
    fn trailing_bytes_decode_to_nothing() {
        let (_, mut ext) = signed(&encode_submit_proof_call(&proof()));
        ext.push(0xff);
        assert!(decode_pending_proof(&ext, &CTX).is_none());
    }

    #[test]
    fn truncated_bytes_decode_to_nothing() {
        let (_, ext) = signed(&encode_submit_proof_call(&proof()));
        let cut = ext.len() - 10;
        assert!(decode_pending_proof(ext.get(..cut).expect("prefix"), &CTX).is_none());
    }
}
