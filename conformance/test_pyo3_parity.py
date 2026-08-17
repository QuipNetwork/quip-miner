"""Golden-vector parity for the PyO3-backed quip_proto primitives.

Runs the same conformance/golden_vectors.json cases through the compiled
quip_proto.scoring / wire that the Rust golden tests consume. Because the
binding is the Rust code, the two cannot diverge.
"""

import json
from pathlib import Path

import pytest

from quip_proto._core import scoring, wire

GOLDEN = json.loads(
    (Path(__file__).resolve().parents[1] / "conformance/golden_vectors.json").read_text()
)


def test_energy_matches_golden():
    for c in GOLDEN["energy"]:
        spins = c["spins"]
        h = [x / 1000 for x in c["h_milli"]]
        j = [x / 1000 for x in c["j_milli"]]
        edges = [tuple(e) for e in c["edges"]]
        assert scoring.energy_milli(spins, h, j, edges) == c["energy_milli"]


def test_diversity_matches_golden():
    for c in GOLDEN["diversity"]:
        assert abs(scoring.set_diversity(c["solutions"]) - c["diversity"]) < 1e-12


def test_wire_roundtrip():
    assert wire.decode_i32_le(wire.encode_i32_le([1, -2, 3])) == [1, -2, 3]
    assert wire.decode_spins(wire.encode_spins([1, -1, 1])) == [1, -1, 1]


def test_decode_i32_bad_length_raises():
    with pytest.raises(ValueError):
        wire.decode_i32_le(b"\x00\x00\x00")


def test_decode_spins_bad_byte_raises():
    # 0x00 is neither 0x01 (+1) nor 0xFF (-1): the PyO3 wire.decode_spins must
    # surface WireError::BadSpinByte as a Python ValueError, not silently accept.
    with pytest.raises(ValueError):
        wire.decode_spins(b"\x00")


def test_wire_empty_payload_roundtrip():
    assert wire.decode_i32_le(wire.encode_i32_le([])) == []
    assert wire.decode_spins(wire.encode_spins([])) == []


def test_wire_i32_bounds_roundtrip():
    # i32::MIN / i32::MAX are load-bearing edge values for the LE codec.
    vals = [-2147483648, -1, 0, 1, 2147483647]
    assert wire.decode_i32_le(wire.encode_i32_le(vals)) == vals


def test_positive_sign_convention():
    # spins [+1,-1]; h=[1.0,-0.5]; edge (0,1) J=2.0 -> E = 1 + 0.5 - 2.0 = -0.5 -> -500
    assert scoring.energy_milli([1, -1], [1.0, -0.5], [2.0], [(0, 1)]) == -500
