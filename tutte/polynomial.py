"""Tutte Polynomial with Bitstring Encoding.

This module provides an immutable TuttePolynomial class that stores coefficients
in a compact bitstring format. The encoding supports both dense and sparse
representations, automatically selecting the more efficient format.

Bitstring Encoding Format:
    Header (variable length):
        [flags: 1 byte]
            bit 0: 0=dense, 1=sparse
            bits 1-7: reserved
        [max_x_degree: varuint]
        [max_y_degree: varuint]

    Dense Body (when sparsity < 70%):
        Coefficients in row-major order: [0,0], [0,1], ..., [0,max_y], [1,0], ...
        Each coefficient as signed varint

    Sparse Body (when sparsity >= 70%):
        [num_terms: varuint]
        For each term:
            [x_power: varuint]
            [y_power: varuint]
            [coefficient: signed varint]
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Tuple, Iterator
from collections import defaultdict


# =============================================================================
# VARINT ENCODING UTILITIES
# =============================================================================

def encode_varuint(n: int) -> bytes:
    """Encode unsigned int: 7 bits per byte, high bit = continuation."""
    if n < 0:
        raise ValueError("varuint cannot encode negative numbers")
    result = []
    while n >= 0x80:
        result.append((n & 0x7F) | 0x80)
        n >>= 7
    result.append(n)
    return bytes(result)


def decode_varuint(data: bytes, offset: int = 0) -> Tuple[int, int]:
    """Decode unsigned varint, returns (value, new_offset)."""
    result = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("Truncated varuint")
        byte = data[offset]
        result |= (byte & 0x7F) << shift
        offset += 1
        if not (byte & 0x80):
            break
        shift += 7
    return result, offset


def encode_varsint(n: int) -> bytes:
    """Encode signed int using zigzag encoding + varuint."""
    # Zigzag: positive n -> 2n, negative n -> -2n - 1
    if n >= 0:
        return encode_varuint(2 * n)
    else:
        return encode_varuint(-2 * n - 1)


def decode_varsint(data: bytes, offset: int = 0) -> Tuple[int, int]:
    """Decode signed varint (zigzag encoded), returns (value, new_offset)."""
    zigzag, new_offset = decode_varuint(data, offset)
    # Reverse zigzag
    if zigzag & 1:
        return -(zigzag >> 1) - 1, new_offset
    else:
        return zigzag >> 1, new_offset


# =============================================================================
# TUTTE POLYNOMIAL CLASS
# =============================================================================

# Cache for common single-term polynomials (one, zero, x, y). Hot-path
# SP reduction and chord-rule iteration request these thousands of
# times each; caching saves the encode/decode roundtrip.
_ONE_TERM_CACHE: Dict[Tuple, 'TuttePolynomial'] = {}


@dataclass(frozen=True)
class TuttePolynomial:
    """Represents a Tutte polynomial as an immutable bitstring.

    The polynomial is stored in a compact binary format that supports
    both dense and sparse representations. Arithmetic operations create
    new immutable instances.

    Example: T = 2x^2 + 3xy + y^2 would be:
        coefficients = {(2, 0): 2, (1, 1): 3, (0, 2): 1}
    """
    _data: bytes

    @classmethod
    def from_coefficients(cls, coeffs: Dict[Tuple[int, int], int]) -> 'TuttePolynomial':
        """Create from coefficient dictionary {(x_power, y_power): coefficient}."""
        # Remove zero coefficients
        coeffs = {k: v for k, v in coeffs.items() if v != 0}

        if not coeffs:
            # Empty polynomial = 0
            return cls(_encode_polynomial({}))

        return cls(_encode_polynomial(coeffs))

    @classmethod
    def from_bytes(cls, data: bytes) -> 'TuttePolynomial':
        """Create from encoded bytes."""
        # Validate by decoding
        _decode_polynomial(data)
        return cls(data)

    @classmethod
    def x(cls, power: int = 1) -> 'TuttePolynomial':
        """Create x^power (single bridge or chain of bridges)."""
        if power < 0:
            raise ValueError("Power must be non-negative")
        if power == 0:
            return cls.one()
        # Hot path: bare x is requested thousands of times during SP
        # reduction and chord-rule iteration. Cache it.
        if power == 1:
            cached = _ONE_TERM_CACHE.get(('x', 1))
            if cached is None:
                cached = cls.from_coefficients({(1, 0): 1})
                _ONE_TERM_CACHE[('x', 1)] = cached
            return cached
        return cls.from_coefficients({(power, 0): 1})

    @classmethod
    def y(cls, power: int = 1) -> 'TuttePolynomial':
        """Create y^power (loops)."""
        if power < 0:
            raise ValueError("Power must be non-negative")
        if power == 0:
            return cls.one()
        if power == 1:
            cached = _ONE_TERM_CACHE.get(('y', 1))
            if cached is None:
                cached = cls.from_coefficients({(0, 1): 1})
                _ONE_TERM_CACHE[('y', 1)] = cached
            return cached
        return cls.from_coefficients({(0, power): 1})

    @classmethod
    def one(cls) -> 'TuttePolynomial':
        """Create constant 1 (empty graph on one vertex)."""
        cached = _ONE_TERM_CACHE.get(('one',))
        if cached is None:
            cached = cls.from_coefficients({(0, 0): 1})
            _ONE_TERM_CACHE[('one',)] = cached
        return cached

    @classmethod
    def zero(cls) -> 'TuttePolynomial':
        """Create constant 0."""
        cached = _ONE_TERM_CACHE.get(('zero',))
        if cached is None:
            cached = cls.from_coefficients({})
            _ONE_TERM_CACHE[('zero',)] = cached
        return cached

    def __add__(self, other: 'TuttePolynomial') -> 'TuttePolynomial':
        """Add two Tutte polynomials."""
        if not isinstance(other, TuttePolynomial):
            return NotImplemented
        result = defaultdict(int, self._coeffs)
        for k, v in other._coeffs.items():
            result[k] += v
        return TuttePolynomial.from_coefficients(dict(result))

    def __sub__(self, other: 'TuttePolynomial') -> 'TuttePolynomial':
        """Subtract two Tutte polynomials."""
        if not isinstance(other, TuttePolynomial):
            return NotImplemented
        result = defaultdict(int, self._coeffs)
        for k, v in other._coeffs.items():
            result[k] -= v
        return TuttePolynomial.from_coefficients(dict(result))

    def __mul__(self, other: 'TuttePolynomial') -> 'TuttePolynomial':
        """Multiply two Tutte polynomials."""
        if not isinstance(other, TuttePolynomial):
            return NotImplemented
        result = defaultdict(int)
        for (i1, j1), c1 in self._coeffs.items():
            for (i2, j2), c2 in other._coeffs.items():
                result[(i1 + i2, j1 + j2)] += c1 * c2
        return TuttePolynomial.from_coefficients(dict(result))

    def __rmul__(self, scalar: int) -> 'TuttePolynomial':
        """Scalar multiplication."""
        if not isinstance(scalar, int):
            return NotImplemented
        return TuttePolynomial.from_coefficients(
            {k: scalar * v for k, v in self._coeffs.items()}
        )

    def __neg__(self) -> 'TuttePolynomial':
        """Negate polynomial."""
        return TuttePolynomial.from_coefficients(
            {k: -v for k, v in self._coeffs.items()}
        )

    def __divmod__(self, other: 'TuttePolynomial') -> Tuple['TuttePolynomial', 'TuttePolynomial']:
        """Divide with remainder: returns (quotient, remainder)."""
        if not isinstance(other, TuttePolynomial):
            return NotImplemented
        from .graphs.k_sum import polynomial_divmod
        return polynomial_divmod(self, other)

    def __floordiv__(self, other: 'TuttePolynomial') -> 'TuttePolynomial':
        """Integer division (exact division, raises if remainder)."""
        if not isinstance(other, TuttePolynomial):
            return NotImplemented
        from .graphs.k_sum import polynomial_divide
        return polynomial_divide(self, other)

    def __mod__(self, other: 'TuttePolynomial') -> 'TuttePolynomial':
        """Remainder after division."""
        if not isinstance(other, TuttePolynomial):
            return NotImplemented
        from .graphs.k_sum import polynomial_divmod
        _, remainder = polynomial_divmod(self, other)
        return remainder

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TuttePolynomial):
            return NotImplemented
        return self._coeffs == other._coeffs

    def __hash__(self) -> int:
        # Hash based on sorted coefficient tuples for consistency
        return hash(tuple(sorted(self._coeffs.items())))

    def __repr__(self) -> str:
        if not self._coeffs:
            return "0"
        terms = []
        for (i, j), c in sorted(self._coeffs.items(), reverse=True):
            if c == 0:
                continue
            term = ""
            if c != 1 or (i == 0 and j == 0):
                if c == -1 and (i > 0 or j > 0):
                    term = "-"
                else:
                    term = str(c)
            if i > 0:
                term += f"x^{i}" if i > 1 else "x"
            if j > 0:
                term += f"y^{j}" if j > 1 else "y"
            terms.append(term)
        return " + ".join(terms).replace("+ -", "- ") if terms else "0"

    def to_bytes(self) -> bytes:
        """Return the encoded bytes representation."""
        return self._data

    def to_coefficients(self) -> Dict[Tuple[int, int], int]:
        """Return coefficient dictionary."""
        return dict(self._coeffs)

    @cached_property
    def _coeffs(self) -> Dict[Tuple[int, int], int]:
        """Decode and cache coefficients."""
        return _decode_polynomial(self._data)

    def evaluate(self, x: float, y: float) -> float:
        """Evaluate the polynomial at (x, y)."""
        return sum(c * (x ** i) * (y ** j)
                   for (i, j), c in self._coeffs.items())

    def evaluate_mod(self, x: int, y: int, p: int) -> int:
        """Evaluate the polynomial at integer (x, y) modulo prime p.

        All arithmetic is integer/modular — no floats. Used as the
        building block for integer-DP + Lagrange-interpolation
        recovery of T(G; x, y) over large graphs.

        Caller passes a prime `p`; we use `pow(base, exp, p)` to keep
        intermediate values bounded even when `x` or `y` is comparable
        to `p`.
        """
        if p <= 0:
            raise ValueError("modulus must be positive")
        x_m = x % p
        y_m = y % p
        result = 0
        for (i, j), c in self._coeffs.items():
            term = (c % p) * pow(x_m, i, p) * pow(y_m, j, p)
            result = (result + term) % p
        return result

    def num_spanning_trees(self) -> int:
        """T(1,1) = number of spanning trees (exact integer sum of coefficients)."""
        return sum(c for c in self._coeffs.values())

    def x_degree(self) -> int:
        """Maximum power of x."""
        if not self._coeffs:
            return 0
        return max(i for i, j in self._coeffs.keys())

    def y_degree(self) -> int:
        """Maximum power of y."""
        if not self._coeffs:
            return 0
        return max(j for i, j in self._coeffs.keys())

    def total_degree(self) -> int:
        """Return the maximum i + j across all terms."""
        if not self._coeffs:
            return 0
        return max(i + j for i, j in self._coeffs.keys())

    def num_terms(self) -> int:
        """Number of non-zero terms."""
        return len(self._coeffs)

    def terms(self) -> Iterator[Tuple[int, int, int]]:
        """Iterate over (x_power, y_power, coefficient) tuples."""
        for (i, j), c in self._coeffs.items():
            yield i, j, c

    def is_zero(self) -> bool:
        """Check if polynomial is zero."""
        return len(self._coeffs) == 0

    def coefficient(self, x_power: int, y_power: int) -> int:
        """Get coefficient of x^x_power * y^y_power."""
        return self._coeffs.get((x_power, y_power), 0)


# =============================================================================
# ENCODING / DECODING FUNCTIONS
# =============================================================================

# Flag bits
FLAG_SPARSE = 0x01

# Sparsity threshold for switching to sparse encoding
SPARSE_THRESHOLD = 0.70


def _encode_polynomial(coeffs: Dict[Tuple[int, int], int]) -> bytes:
    """Encode polynomial coefficients to bytes."""
    if not coeffs:
        # Empty polynomial: use sparse format with 0 terms
        result = bytearray()
        result.append(FLAG_SPARSE)  # sparse format
        result.extend(encode_varuint(0))  # max_x = 0
        result.extend(encode_varuint(0))  # max_y = 0
        result.extend(encode_varuint(0))  # num_terms = 0
        return bytes(result)

    max_x = max(i for i, j in coeffs.keys())
    max_y = max(j for i, j in coeffs.keys())

    # Calculate sparsity
    total_cells = (max_x + 1) * (max_y + 1)
    num_terms = len(coeffs)
    sparsity = 1.0 - (num_terms / total_cells) if total_cells > 0 else 0

    if sparsity >= SPARSE_THRESHOLD:
        return _encode_sparse(coeffs, max_x, max_y)
    else:
        return _encode_dense(coeffs, max_x, max_y)


def _encode_dense(coeffs: Dict[Tuple[int, int], int], max_x: int, max_y: int) -> bytes:
    """Encode using dense format (row-major order)."""
    result = bytearray()

    # Header
    result.append(0)  # flags: dense
    result.extend(encode_varuint(max_x))
    result.extend(encode_varuint(max_y))

    # Coefficients in row-major order
    for i in range(max_x + 1):
        for j in range(max_y + 1):
            c = coeffs.get((i, j), 0)
            result.extend(encode_varsint(c))

    return bytes(result)


def _encode_sparse(coeffs: Dict[Tuple[int, int], int], max_x: int, max_y: int) -> bytes:
    """Encode using sparse format (term list)."""
    result = bytearray()

    # Header
    result.append(FLAG_SPARSE)  # flags: sparse
    result.extend(encode_varuint(max_x))
    result.extend(encode_varuint(max_y))

    # Number of terms
    result.extend(encode_varuint(len(coeffs)))

    # Each term
    for (i, j), c in sorted(coeffs.items()):
        result.extend(encode_varuint(i))
        result.extend(encode_varuint(j))
        result.extend(encode_varsint(c))

    return bytes(result)


def _decode_polynomial(data: bytes) -> Dict[Tuple[int, int], int]:
    """Decode bytes to coefficient dictionary."""
    if not data:
        return {}

    offset = 0

    # Read flags
    flags = data[offset]
    offset += 1

    # Read max degrees
    max_x, offset = decode_varuint(data, offset)
    max_y, offset = decode_varuint(data, offset)

    if flags & FLAG_SPARSE:
        return _decode_sparse(data, offset, max_x, max_y)
    else:
        return _decode_dense(data, offset, max_x, max_y)


def _decode_dense(data: bytes, offset: int, max_x: int, max_y: int) -> Dict[Tuple[int, int], int]:
    """Decode dense format."""
    coeffs = {}
    for i in range(max_x + 1):
        for j in range(max_y + 1):
            c, offset = decode_varsint(data, offset)
            if c != 0:
                coeffs[(i, j)] = c
    return coeffs


def _decode_sparse(data: bytes, offset: int, max_x: int, max_y: int) -> Dict[Tuple[int, int], int]:
    """Decode sparse format."""
    num_terms, offset = decode_varuint(data, offset)

    coeffs = {}
    for _ in range(num_terms):
        i, offset = decode_varuint(data, offset)
        j, offset = decode_varuint(data, offset)
        c, offset = decode_varsint(data, offset)
        if c != 0:
            coeffs[(i, j)] = c

    return coeffs


# =============================================================================
# KNOWN TUTTE POLYNOMIALS FOR COMMON GRAPH FAMILIES
# =============================================================================

KNOWN_POLYNOMIALS = {
    # Path graphs P_n (n nodes, n-1 edges): T = x^{n-1}
    'P2': lambda: TuttePolynomial.x(1),
    'P3': lambda: TuttePolynomial.x(2),
    'P4': lambda: TuttePolynomial.x(3),
    'P5': lambda: TuttePolynomial.x(4),

    # Cycle graphs C_n: T = x^{n-1} + x^{n-2} + ... + x + y
    'C3': lambda: TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1}),
    'C4': lambda: TuttePolynomial.from_coefficients({(3, 0): 1, (2, 0): 1, (1, 0): 1, (0, 1): 1}),
    'C5': lambda: TuttePolynomial.from_coefficients({(4, 0): 1, (3, 0): 1, (2, 0): 1, (1, 0): 1, (0, 1): 1}),

    # Complete graphs K_n
    'K2': lambda: TuttePolynomial.x(1),
    'K3': lambda: TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1}),
    'K4': lambda: TuttePolynomial.from_coefficients({
        (3, 0): 1, (2, 0): 3, (1, 1): 4, (1, 0): 2, (0, 1): 2, (0, 2): 3, (0, 3): 1
    }),
}


def get_known_polynomial(name: str) -> TuttePolynomial:
    """Get a known polynomial by name."""
    if name not in KNOWN_POLYNOMIALS:
        raise KeyError(f"Unknown polynomial: {name}")
    return KNOWN_POLYNOMIALS[name]()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cycle_polynomial(n: int) -> TuttePolynomial:
    """Generate Tutte polynomial for cycle graph C_n.

    T(C_n) = x^{n-1} + x^{n-2} + ... + x + y  (for n >= 3)
    """
    if n < 3:
        raise ValueError("Cycle must have at least 3 vertices")
    coeffs = {(i, 0): 1 for i in range(1, n)}
    coeffs[(0, 1)] = 1
    return TuttePolynomial.from_coefficients(coeffs)


def path_polynomial(n: int) -> TuttePolynomial:
    """Generate Tutte polynomial for path graph P_n.

    T(P_n) = x^{n-1} (n nodes = n-1 edges, all bridges)
    """
    if n < 1:
        raise ValueError("Path must have at least 1 vertex")
    if n == 1:
        return TuttePolynomial.one()
    return TuttePolynomial.x(n - 1)
