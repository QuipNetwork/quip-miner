"""FK state vector to Tutte polynomial extraction for the transfer-matrix pipeline.

Converts the final Fortuin-Kasteleyn state vector (from the sweep stage) into
the Tutte polynomial T(G; x, y).

The pipeline works in the (a, b) = (x-1, y-1) basis throughout. After the
sweep, the final boundary vertices are "forgotten" (each block contributes
a*b = (x-1)*(y-1)), the partition states are summed to get the FK partition
function Z, and the FK-to-Tutte prefactor is removed by an exponent shift:

    T(G; x, y) = a^{-k(G)} * b^{-|V|} * Z

In the (a, b) basis this is just subtracting k(G) from every a-exponent and
|V| from every b-exponent. The result is then converted from (a, b) to (x, y)
basis via binomial expansion.

Complexity:
    - Extraction: O(c_m * P) for forgetting + O(terms * degree^2) for basis conversion.
    - End-to-end transfer matrix pipeline: O((n-1) * c_m^2 * P).
"""

from __future__ import annotations

from math import comb
from typing import Dict, List, Tuple

from ..polynomial import TuttePolynomial
from .core import (
    Polynomial,
    enumerate_noncrossing_partitions,
)


def _count_blocks(partition: Tuple[int, ...]) -> int:
    """Count the number of distinct blocks in a canonical partition.

    In canonical form (labels in first-occurrence order starting from 0),
    the block count equals max(labels) + 1.
    """
    if not partition:
        return 0
    return max(partition) + 1


def _convert_ab_to_xy(
    poly_ab: Dict[Tuple[int, int], int],
) -> TuttePolynomial:
    """Convert a polynomial from (a, b) = (x-1, y-1) basis to (x, y) basis.

    Expands each monomial a^i * b^j = (x-1)^i * (y-1)^j using the
    binomial theorem and collects terms in x^p * y^q.

    Args:
        poly_ab: Coefficient dict {(a_pow, b_pow): coeff} in (a, b) basis.

    Returns:
        TuttePolynomial in the standard (x, y) basis.
    """
    coeffs_xy: Dict[Tuple[int, int], int] = {}

    for (a_pow, b_pow), coeff in poly_ab.items():
        for x_pow in range(a_pow + 1):
            binom_x = comb(a_pow, x_pow) * ((-1) ** (a_pow - x_pow))
            for y_pow in range(b_pow + 1):
                binom_y = comb(b_pow, y_pow) * ((-1) ** (b_pow - y_pow))
                contribution = coeff * binom_x * binom_y
                if contribution != 0:
                    coeffs_xy[(x_pow, y_pow)] = (
                        coeffs_xy.get((x_pow, y_pow), 0) + contribution
                    )

    # Remove any terms that cancelled to zero.
    coeffs_xy = {k: v for k, v in coeffs_xy.items() if v != 0}
    return TuttePolynomial.from_coefficients(coeffs_xy)


def extract_tutte_polynomial(
    final_vector: List[Polynomial],
    width: int,
    num_vertices: int,
    num_components: int = 1,
) -> TuttePolynomial:
    """Extract the Tutte polynomial from the final FK state vector.

    After the direct-multiply sweep, each entry in the final state vector
    is a polynomial in the (a, b) = (x-1, y-1) basis representing the
    accumulated FK weight for a particular partition state. Recovery steps:

    1. Forget the final boundary: each block contributes a*b = (x-1)*(y-1).
    2. Sum over all partition states to obtain the FK partition function Z.
    3. Divide Z by a^{k(G)} * b^{|V|} via exponent shift.
    4. Convert from (a, b) basis to (x, y) basis via binomial expansion.

    Args:
        final_vector: State vector of length c_m from direct_multiply(),
            with polynomial entries in the (a, b) basis.
        width: Number of boundary vertices (m) in the final column.
        num_vertices: Total number of vertices |V| in the graph.
        num_components: Number of connected components k(G). Defaults to 1
            (lattice strips are connected).

    Returns:
        The Tutte polynomial T(G; x, y).

    Raises:
        ValueError: If final_vector length does not match c_m, or if the
            exponent shift produces negative exponents.
    """
    partitions = enumerate_noncrossing_partitions(width)
    num_states = len(partitions)

    if len(final_vector) != num_states:
        raise ValueError(
            f"final_vector has {len(final_vector)} entries but width={width} "
            f"requires c_m={num_states} entries"
        )

    # Precompute (a*b)^k for block counts 0..width.
    # In (a, b) basis, (a*b)^k is simply the monomial {(k, k): 1}.
    ab_powers: List[Polynomial] = [
        Polynomial.from_coefficients({(k, k): 1}) for k in range(width + 1)
    ]

    # Forget final boundary and sum over partition states.
    # Each block in the final partition contributes one factor of a*b.
    fk_partition_sum = Polynomial.zero()
    for state_idx, partition in enumerate(partitions):
        if final_vector[state_idx].is_zero():
            continue

        num_blocks = _count_blocks(partition)
        state_contribution = final_vector[state_idx] * ab_powers[num_blocks]
        fk_partition_sum = fk_partition_sum + state_contribution

    # Divide by a^{k(G)} * b^{|V|} via exponent shift.
    a_shift = num_components
    b_shift = num_vertices

    tutte_ab: Dict[Tuple[int, int], int] = {}
    for (a_pow, b_pow), coeff in fk_partition_sum.to_coefficients().items():
        new_a = a_pow - a_shift
        new_b = b_pow - b_shift
        if new_a < 0 or new_b < 0:
            raise ValueError(
                f"Exponent shift produced negative exponent: "
                f"({a_pow},{b_pow}) - ({a_shift},{b_shift}) = "
                f"({new_a},{new_b})."
            )
        tutte_ab[(new_a, new_b)] = coeff

    # Convert from (a, b) basis to (x, y) basis.
    return _convert_ab_to_xy(tutte_ab)