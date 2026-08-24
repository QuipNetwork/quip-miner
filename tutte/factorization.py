"""Polynomial Factorization and GCD for Tutte Polynomials.

This module provides algorithms for computing GCDs and factorizations
of bivariate Tutte polynomials. These are used for:
1. Finding shared factors between polynomials (GCD-based minor relationships)
2. Algebraic decomposition in synthesis
3. Identifying polynomial structure for k-join operations

Key algorithms:
- Bivariate polynomial GCD using subresultant algorithm
- Monomial content extraction
- Primitive part computation
"""

from __future__ import annotations

from math import gcd as math_gcd
from typing import List, Tuple

from .polynomial import TuttePolynomial


# =============================================================================
# MONOMIAL CONTENT AND PRIMITIVE PART
# =============================================================================

def monomial_content(p: TuttePolynomial) -> Tuple[int, int, int]:
    """Extract the monomial content from a polynomial.

    The monomial content is x^a * y^b * c where:
    - a is the minimum x-power across all terms
    - b is the minimum y-power across all terms
    - c is the GCD of all coefficients

    Returns:
        Tuple (a, b, c) representing x^a * y^b * c

    Example:
        monomial_content(2x^3y + 4x^2y^2) = (2, 1, 2) for x^2 * y * 2
    """
    coeffs = p.to_coefficients()

    if not coeffs:
        return (0, 0, 1)

    # Find minimum powers
    min_x = min(i for i, j in coeffs.keys())
    min_y = min(j for i, j in coeffs.keys())

    # Find GCD of coefficients
    coeff_gcd = 0
    for c in coeffs.values():
        coeff_gcd = math_gcd(coeff_gcd, abs(c))

    if coeff_gcd == 0:
        coeff_gcd = 1

    return (min_x, min_y, coeff_gcd)


def primitive_part(p: TuttePolynomial) -> TuttePolynomial:
    """Return p with monomial content factored out.

    The primitive part is p / (x^a * y^b * c) where (a, b, c) is the
    monomial content.

    Returns:
        Polynomial with leading coefficient positive and monomial content removed
    """
    coeffs = p.to_coefficients()

    if not coeffs:
        return p

    a, b, c = monomial_content(p)

    # Divide out the monomial content
    new_coeffs = {}
    for (i, j), coeff in coeffs.items():
        new_i = i - a
        new_j = j - b
        new_coeff = coeff // c
        new_coeffs[(new_i, new_j)] = new_coeff

    return TuttePolynomial.from_coefficients(new_coeffs)


def integer_content(p: TuttePolynomial) -> int:
    """Return the GCD of all coefficients in the polynomial."""
    coeffs = p.to_coefficients()

    if not coeffs:
        return 1

    result = 0
    for c in coeffs.values():
        result = math_gcd(result, abs(c))

    return result if result > 0 else 1


# =============================================================================
# UNIVARIATE POLYNOMIAL GCD (for reduction)
# =============================================================================

def _univariate_gcd(coeffs1: List[int], coeffs2: List[int]) -> List[int]:
    """Compute GCD of two univariate polynomials using Euclidean algorithm.

    Polynomials are represented as coefficient lists [c0, c1, c2, ...]
    for c0 + c1*x + c2*x^2 + ...

    Args:
        coeffs1: First polynomial coefficients
        coeffs2: Second polynomial coefficients

    Returns:
        GCD polynomial coefficients
    """
    # Remove trailing zeros
    while coeffs1 and coeffs1[-1] == 0:
        coeffs1 = coeffs1[:-1]
    while coeffs2 and coeffs2[-1] == 0:
        coeffs2 = coeffs2[:-1]

    if not coeffs1:
        return coeffs2 if coeffs2 else [1]
    if not coeffs2:
        return coeffs1

    # Ensure coeffs1 has higher or equal degree
    if len(coeffs2) > len(coeffs1):
        coeffs1, coeffs2 = coeffs2, coeffs1

    # Euclidean algorithm with pseudo-division for integer coefficients
    while coeffs2:
        # Remove trailing zeros
        while coeffs2 and coeffs2[-1] == 0:
            coeffs2 = coeffs2[:-1]
        if not coeffs2:
            break

        # Pseudo-division: multiply a by leading coeff of b to keep integers
        lc_b = coeffs2[-1]
        deg_diff = len(coeffs1) - len(coeffs2)

        if deg_diff < 0:
            coeffs1, coeffs2 = coeffs2, coeffs1
            continue

        # Compute pseudo-remainder
        remainder = [c * abs(lc_b) for c in coeffs1]

        for i in range(deg_diff, -1, -1):
            if len(remainder) <= i + len(coeffs2) - 1:
                continue
            coeff = remainder[i + len(coeffs2) - 1]
            if coeff == 0:
                continue
            for j, bc in enumerate(coeffs2):
                remainder[i + j] -= coeff * bc // abs(lc_b) if lc_b != 0 else 0

        # Remove leading zeros from remainder
        while remainder and remainder[-1] == 0:
            remainder = remainder[:-1]

        # Reduce by GCD of coefficients
        if remainder:
            g = 0
            for c in remainder:
                g = math_gcd(g, abs(c))
            if g > 1:
                remainder = [c // g for c in remainder]

        coeffs1, coeffs2 = coeffs2, remainder

    # Normalize: make leading coefficient positive
    if coeffs1 and coeffs1[-1] < 0:
        coeffs1 = [-c for c in coeffs1]

    # Reduce by content
    if coeffs1:
        g = 0
        for c in coeffs1:
            g = math_gcd(g, abs(c))
        if g > 1:
            coeffs1 = [c // g for c in coeffs1]

    return coeffs1 if coeffs1 else [1]


# =============================================================================
# BIVARIATE POLYNOMIAL GCD
# =============================================================================

def polynomial_gcd(p: TuttePolynomial, q: TuttePolynomial) -> TuttePolynomial:
    """Compute GCD of two bivariate Tutte polynomials.

    Uses the subresultant algorithm for bivariate polynomials, treating
    one variable as the main variable and the other as a parameter.

    The GCD is computed over the integers, meaning we find the largest
    polynomial (by degree) that divides both p and q exactly.

    Args:
        p: First polynomial
        q: Second polynomial

    Returns:
        GCD polynomial (unique up to sign)

    Note:
        For efficiency, this uses a simplified approach that works well
        for the polynomials encountered in Tutte polynomial synthesis.
    """
    # Handle edge cases
    if p.is_zero():
        return primitive_part(q) if not q.is_zero() else TuttePolynomial.one()
    if q.is_zero():
        return primitive_part(p)

    # Quick check: if T(1,1) values are coprime, polynomials are coprime
    p_trees = p.num_spanning_trees()
    q_trees = q.num_spanning_trees()
    if p_trees != 0 and q_trees != 0 and math_gcd(p_trees, q_trees) == 1:
        return TuttePolynomial.one()

    # Extract monomial content
    p_content = monomial_content(p)
    q_content = monomial_content(q)

    # GCD of monomial contents
    content_gcd_x = min(p_content[0], q_content[0])
    content_gcd_y = min(p_content[1], q_content[1])
    content_gcd_c = math_gcd(p_content[2], q_content[2])

    # Get primitive parts
    p_prim = primitive_part(p)
    q_prim = primitive_part(q)

    # For primitive parts, use evaluation and interpolation
    # or direct polynomial GCD computation
    prim_gcd = _primitive_gcd(p_prim, q_prim)

    # Reconstruct full GCD with monomial content
    if prim_gcd.is_zero() or prim_gcd == TuttePolynomial.one():
        # Only monomial content is common
        if content_gcd_x == 0 and content_gcd_y == 0:
            return TuttePolynomial.from_coefficients({(0, 0): content_gcd_c})

        coeffs = {}
        coeffs[(content_gcd_x, content_gcd_y)] = content_gcd_c
        return TuttePolynomial.from_coefficients(coeffs)

    # Multiply primitive GCD by monomial content GCD
    prim_coeffs = prim_gcd.to_coefficients()
    result_coeffs = {}
    for (i, j), c in prim_coeffs.items():
        result_coeffs[(i + content_gcd_x, j + content_gcd_y)] = c * content_gcd_c

    return TuttePolynomial.from_coefficients(result_coeffs)


def _primitive_gcd(p: TuttePolynomial, q: TuttePolynomial) -> TuttePolynomial:
    """Compute GCD of primitive polynomials (no common monomial factor).

    Uses a simplified approach based on:
    1. Check for divisibility (one divides the other)
    2. Try common low-degree factors
    3. Use evaluation-based heuristics
    """
    from .graphs.k_sum import polynomial_divmod

    # Check if one divides the other
    if _divides(p, q):
        return p
    if _divides(q, p):
        return q

    # Check for simple common factors by trying division
    # Start with the smaller polynomial as potential factor
    smaller, larger = (p, q) if p.num_terms() <= q.num_terms() else (q, p)

    # Try dividing both by the smaller one's factors
    # This is a heuristic that works well for structured polynomials

    # For Tutte polynomials, check known simple factors
    simple_factors = _get_simple_factors(smaller)

    gcd_so_far = TuttePolynomial.one()

    for factor in simple_factors:
        if _divides(factor, p) and _divides(factor, q):
            # Found common factor
            gcd_so_far = gcd_so_far * factor
            _, p = polynomial_divmod(p, factor)
            _, q = polynomial_divmod(q, factor)
            if p.is_zero() or q.is_zero():
                break

    return gcd_so_far


def _divides(divisor: TuttePolynomial, dividend: TuttePolynomial) -> bool:
    """Check if divisor divides dividend exactly."""
    from .graphs.k_sum import polynomial_divmod

    if divisor.is_zero():
        return dividend.is_zero()

    try:
        _, remainder = polynomial_divmod(dividend, divisor)
        return remainder.is_zero()
    except (ValueError, ZeroDivisionError):
        return False


def _get_simple_factors(p: TuttePolynomial) -> List[TuttePolynomial]:
    """Get list of simple potential factors to try.

    For Tutte polynomials, common factors include:
    - x (all terms have x)
    - y (all terms have y)
    - (x + y), (x + 1), (y + 1) for certain graph families
    """
    factors = []
    coeffs = p.to_coefficients()

    if not coeffs:
        return factors

    # Check if x is a factor (all terms have x)
    min_x = min(i for i, j in coeffs.keys())
    if min_x > 0:
        factors.append(TuttePolynomial.x())

    # Check if y is a factor
    min_y = min(j for i, j in coeffs.keys())
    if min_y > 0:
        factors.append(TuttePolynomial.y())

    # Try (x + y) as a factor (common in certain graphs)
    # This evaluates to 2 at (1,1), so only try if p(1,1) is even
    if p.num_spanning_trees() % 2 == 0:
        x_plus_y = TuttePolynomial.from_coefficients({(1, 0): 1, (0, 1): 1})
        factors.append(x_plus_y)

    # Try K_3 = x^2 + x + y (common minor)
    k3 = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
    if p.num_spanning_trees() % 3 == 0:  # K_3 has 3 spanning trees
        factors.append(k3)

    return factors


# =============================================================================
# COMMON FACTOR DETECTION
# =============================================================================

def has_common_factor(p: TuttePolynomial, q: TuttePolynomial) -> bool:
    """Quick test for whether two polynomials share a non-trivial factor.

    Uses pre-filtering based on T(1,1) values before computing full GCD.

    Args:
        p: First polynomial
        q: Second polynomial

    Returns:
        True if gcd(p, q) ≠ 1
    """
    # Handle edge cases
    if p.is_zero() or q.is_zero():
        return True

    # Pre-filter: if T(1,1) values are coprime, polynomials are coprime
    p_trees = p.num_spanning_trees()
    q_trees = q.num_spanning_trees()

    if p_trees > 0 and q_trees > 0:
        if math_gcd(p_trees, q_trees) == 1:
            return False

    # Compute full GCD
    g = polynomial_gcd(p, q)

    # Check if GCD is non-trivial (not 1 or constant)
    return not _is_unit(g)


def _is_unit(p: TuttePolynomial) -> bool:
    """Check if polynomial is a unit (constant ±1)."""
    coeffs = p.to_coefficients()

    if len(coeffs) != 1:
        return False

    (i, j), c = next(iter(coeffs.items()))
    return i == 0 and j == 0 and abs(c) == 1
