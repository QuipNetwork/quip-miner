//! Coefficient encoding for the problems this coordinator authors.

use quip_proto::v1::{ising_problem, CoefficientEncoding, IsingProblem};
use quip_protocol::wire::{decode_i32_le, encode_i32_le};

/// Integer steps per unit for milli coefficients: the wire `scale` of every
/// problem this coordinator authors.
pub const MILLI_SCALE: u32 = 1000;

/// An `I32` problem at [`MILLI_SCALE`] from milli fields and couplings in
/// graph order. Sampling overrides stay unset.
#[must_use]
pub fn milli_problem(
    graph: Option<ising_problem::Graph>,
    h_milli: &[i32],
    j_milli: &[i32],
) -> IsingProblem {
    IsingProblem {
        graph,
        encoding: CoefficientEncoding::I32 as i32,
        scale: MILLI_SCALE,
        h: encode_i32_le(h_milli),
        j: encode_i32_le(j_milli),
        ..Default::default()
    }
}

/// Decode a problem this coordinator authored back to milli values.
///
/// # Errors
/// Returns a message when the problem is not `I32` at [`MILLI_SCALE`] or an
/// array is not whole `i32` elements. The coordinator never sends another
/// encoding, so either case is a coordinator bug.
pub fn problem_milli(problem: &IsingProblem) -> Result<(Vec<i32>, Vec<i32>), String> {
    if problem.encoding != CoefficientEncoding::I32 as i32 || problem.scale != MILLI_SCALE {
        return Err(format!(
            "expected I32 at scale {MILLI_SCALE}, got encoding {} scale {}",
            problem.encoding, problem.scale
        ));
    }
    let h = decode_i32_le(&problem.h).map_err(|e| format!("h: {e}"))?;
    let j = decode_i32_le(&problem.j).map_err(|e| format!("j: {e}"))?;
    Ok((h, j))
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_proto::v1::CoefficientEncoding;

    #[test]
    fn milli_problem_round_trips_as_i32_at_scale_1000() {
        let p = milli_problem(None, &[1000, -1000, 0], &[500]);
        assert_eq!(p.encoding, CoefficientEncoding::I32 as i32);
        assert_eq!(p.scale, MILLI_SCALE);
        assert_eq!(
            problem_milli(&p).unwrap(),
            (vec![1000, -1000, 0], vec![500])
        );
    }

    #[test]
    fn problem_milli_rejects_another_encoding() {
        let mut p = milli_problem(None, &[1000], &[]);
        p.encoding = CoefficientEncoding::I8 as i32;
        assert!(problem_milli(&p).is_err());
    }

    #[test]
    fn problem_milli_rejects_another_scale() {
        let mut p = milli_problem(None, &[1000], &[]);
        p.scale = 1;
        assert!(problem_milli(&p).is_err());
    }
}
