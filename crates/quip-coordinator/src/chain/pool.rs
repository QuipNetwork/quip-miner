//! Pending `QuantumPow.submit_proof` extrinsics from the transaction pool.
//!
//! `author_pendingExtrinsics` returns every pool entry as opaque bytes. The
//! coordinator wants only signed `submit_proof` calls, so the decoder here
//! walks the signed v4 envelope the coordinator itself emits
//! (`extrinsic::build_hybrid_signed_extrinsic`) and returns nothing for
//! every other shape. That is not an error: the pool holds every kind of
//! call.

use super::extrinsic::extrinsic_hash;
use super::scale_types::{QuantumProof, QUANTUM_POW_PALLET_INDEX, SUBMIT_PROOF_CALL_INDEX};
use parity_scale_codec::{Compact, Decode, Input};
use quip_transaction_crypto::HybridTxSignature;

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

/// Decode one pool entry.
///
/// `None` for anything that is not a signed v4 `submit_proof`: unsigned
/// extrinsics, other calls, other envelope versions, or bytes with anything
/// left over after the call. The signature is decoded for its length only.
/// The pool already checked it.
#[must_use]
pub fn decode_pending_proof(bytes: &[u8]) -> Option<PendingProof> {
    let input = &mut &bytes[..];
    let _len = Compact::<u32>::decode(input).ok()?;
    if input.read_byte().ok()? != SIGNED_V4 {
        return None;
    }
    if input.read_byte().ok()? != MULTI_ADDRESS_ID {
        return None;
    }
    let account = <[u8; 32]>::decode(input).ok()?;
    let _signature = HybridTxSignature::decode(input).ok()?;
    skip_signed_extensions(input)?;
    if input.read_byte().ok()? != QUANTUM_POW_PALLET_INDEX {
        return None;
    }
    if input.read_byte().ok()? != SUBMIT_PROOF_CALL_INDEX {
        return None;
    }
    let proof = QuantumProof::decode(input).ok()?;
    if input.remaining_len().ok()? != Some(0) {
        return None;
    }
    Some(PendingProof {
        extrinsic_hash: extrinsic_hash(bytes),
        account,
        proof,
    })
}

/// Skip the signed extensions in metadata order, mirroring
/// `extrinsic::encode_extra`: era, nonce, tip, metadata-hash mode.
fn skip_signed_extensions(input: &mut &[u8]) -> Option<()> {
    if input.read_byte().ok()? != ERA_IMMORTAL {
        let _second_era_byte = input.read_byte().ok()?;
    }
    let _nonce = Compact::<u32>::decode(input).ok()?;
    let _tip = Compact::<u128>::decode(input).ok()?;
    let _mode = input.read_byte().ok()?;
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::extrinsic::{
        build_hybrid_signed_extrinsic, extrinsic_hash, signer_account_bytes, SignedExtensionContext,
    };
    use crate::chain::scale_types::{encode_register_miner_call, encode_submit_proof_call};
    use parity_scale_codec::Encode;
    use quip_transaction_crypto::HybridPair;
    use sp_core::Pair as _;
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

    #[test]
    fn a_signed_submit_proof_round_trips() {
        let (pair, ext) = signed(&encode_submit_proof_call(&proof()));
        let pending = decode_pending_proof(&ext).expect("decodes");
        assert_eq!(pending.account, signer_account_bytes(&pair));
        assert_eq!(pending.proof, proof());
        assert_eq!(pending.extrinsic_hash, extrinsic_hash(&ext));
    }

    #[test]
    fn another_call_decodes_to_nothing() {
        let (_, ext) = signed(&encode_register_miner_call());
        assert!(decode_pending_proof(&ext).is_none());
    }

    #[test]
    fn an_unsigned_extrinsic_decodes_to_nothing() {
        let call = encode_submit_proof_call(&proof());
        let mut body = vec![0x04];
        body.extend_from_slice(&call);
        let mut ext = Compact(u32::try_from(body.len()).expect("len")).encode();
        ext.extend_from_slice(&body);
        assert!(decode_pending_proof(&ext).is_none());
    }

    #[test]
    fn trailing_bytes_decode_to_nothing() {
        let (_, mut ext) = signed(&encode_submit_proof_call(&proof()));
        ext.push(0xff);
        assert!(decode_pending_proof(&ext).is_none());
    }

    #[test]
    fn truncated_bytes_decode_to_nothing() {
        let (_, ext) = signed(&encode_submit_proof_call(&proof()));
        let cut = ext.len() - 10;
        assert!(decode_pending_proof(ext.get(..cut).expect("prefix")).is_none());
    }
}
