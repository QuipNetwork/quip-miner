//! Build a chain `QuantumProof` from a coordinator `Proof` + mining context.

use super::scale_types::QuantumProof;
use super::submit::Proof;
use parity_scale_codec::Encode;
use quantum_validation::packed::pack_solution;
use quantum_validation::{derive_nonce, AllowedValueSpec, MilliValue};
use quip_protocol::wire::decode_spins;
use sp_core::{H256, U256};

/// Inputs needed to pack and nonce a proof for submission.
#[derive(Clone, Debug)]
pub struct ProofBuildContext {
    /// Topology hash the proof is mined against.
    pub topology_hash: [u8; 32],
    /// Last winning proof's block hash (nonce derivation input).
    pub last_proof_block_hash: [u8; 32],
    /// Miner identity bytes (nonce derivation input).
    pub miner_identity: [u8; 32],
    /// 32-byte `PoW` salt.
    pub salt: [u8; 32],
    /// Expected spin vector length (topology node count).
    pub num_nodes: usize,
    /// Allowed spin values used when bit-packing solutions.
    pub allowed_spin: AllowedValueSpec<Vec<i32>>,
}

/// Encode a coordinator proof into the on-chain `QuantumProof` SCALE struct.
///
/// Spins are bit-packed under `allowed_spin`. Energies are discarded — the
/// chain recomputes them. Nonce is `derive_nonce(last_proof, miner, salt)`.
///
/// # Errors
/// Returns an error when a solution's spins cannot be decoded, length does
/// not match `num_nodes`, or bit-packing under `allowed_spin` fails.
pub fn build_quantum_proof(proof: &Proof, ctx: &ProofBuildContext) -> Result<QuantumProof, String> {
    let spin_spec = ctx.allowed_spin.as_slice();
    let mut packed = Vec::with_capacity(proof.solutions.len());
    for sol in &proof.solutions {
        let spins = decode_spins(&sol.spins_bytes).map_err(|e| format!("spins: {e}"))?;
        if spins.len() != ctx.num_nodes {
            return Err(format!(
                "solution length {} != topology nodes {}",
                spins.len(),
                ctx.num_nodes
            ));
        }
        let milli: Vec<MilliValue> = spins
            .iter()
            .map(|&s| match s {
                1 => 1000,
                -1 => -1000,
                other => i32::from(other) * 1000,
            })
            .collect();
        let bytes =
            pack_solution(&milli, &spin_spec).map_err(|e| format!("pack_solution: {e:?}"))?;
        packed.push(bytes);
    }

    let nonce: U256 = derive_nonce(&ctx.last_proof_block_hash, &ctx.miner_identity, &ctx.salt);

    Ok(QuantumProof {
        topology_hash: H256::from(ctx.topology_hash),
        nonce,
        salt: ctx.salt,
        solutions: packed,
        device_access_time_us: proof.device_access_time_us,
    })
}

/// SCALE-encode a quantum proof (for tests / debugging).
#[must_use]
pub fn encode_quantum_proof(proof: &QuantumProof) -> Vec<u8> {
    proof.encode()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::submit::Proof;
    use parity_scale_codec::Decode;
    use quip_protocol::wire::encode_spins;

    fn binary_ctx() -> ProofBuildContext {
        ProofBuildContext {
            topology_hash: [0xab; 32],
            last_proof_block_hash: [0x11; 32],
            miner_identity: [0x22; 32],
            salt: [0x33; 32],
            num_nodes: 4,
            allowed_spin: AllowedValueSpec::Set(vec![-1000, 1000]),
        }
    }

    #[test]
    fn packs_binary_spins_and_derives_nonce() {
        let ctx = binary_ctx();
        let proof = Proof {
            job_id: vec![1],
            best_energy_milli: -100,
            diversity_milli: 200,
            n_valid: 1,
            solutions: vec![quip_proto::v1::Solution {
                spins_bytes: encode_spins(&[1, -1, 1, -1]),
                energy_milli: -100,
            }],
            is_pow: true,
            order_id: vec![],
            generation: 1,
            salt: ctx.salt.to_vec(),
            device_access_time_us: 123_456,
        };
        let qp = build_quantum_proof(&proof, &ctx).expect("build");
        assert_eq!(qp.topology_hash, H256::from(ctx.topology_hash));
        assert_eq!(qp.salt, ctx.salt);
        assert_eq!(qp.device_access_time_us, 123_456);
        assert_eq!(qp.solutions.len(), 1);
        // 4 spins * 1 bit = 1 byte
        #[expect(
            clippy::indexing_slicing,
            reason = "single packed solution asserted above"
        )]
        {
            assert_eq!(qp.solutions[0].len(), 1);
        }
        let expected = derive_nonce(&ctx.last_proof_block_hash, &ctx.miner_identity, &ctx.salt);
        assert_eq!(qp.nonce, expected);

        // SCALE roundtrip of the built proof.
        let enc = encode_quantum_proof(&qp);
        let dec = QuantumProof::decode(&mut &enc[..]).expect("decode");
        assert_eq!(dec, qp);
    }
}
