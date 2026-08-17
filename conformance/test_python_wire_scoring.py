import pytest
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from quip_proto import wire, scoring  # noqa: E402

GOLDEN = json.loads((ROOT / "conformance" / "golden_vectors.json").read_text())


def test_wire_roundtrip():
    assert wire.encode_i32_le([-1000, 0, 1000]) == bytes([0x18, 0xFC, 0xFF, 0xFF, 0, 0, 0, 0, 0xE8, 3, 0, 0])
    assert wire.decode_i32_le(wire.encode_i32_le([-1000, 0, 1000])) == [-1000, 0, 1000]
    assert wire.encode_spins([1, -1, 1]) == bytes([0x01, 0xFF, 0x01])
    assert wire.decode_spins(bytes([0x01, 0xFF, 0x01])) == [1, -1, 1]


def test_energy_matches_golden():
    for c in GOLDEN["energy"]:
        h = [v / 1000.0 for v in c["h_milli"]]
        j = [v / 1000.0 for v in c["j_milli"]]
        edges = [tuple(e) for e in c["edges"]]
        assert scoring.energy_milli(c["spins"], h, j, edges) == c["energy_milli"]


def test_truncation_matches_golden():
    # The golden `truncation` section (added by a Task-4 fix) pins truncation
    # toward zero cross-language; cases are chosen so int() != round(). Python's
    # int(e*1000) must match the committed (truncated) energy_milli, and the
    # matching Rust test (golden_scoring.rs) asserts the identical values.
    for c in GOLDEN["truncation"]:
        assert int(c["energy"] * 1000) == c["energy_milli"]


def test_diversity_matches_golden():
    for c in GOLDEN["diversity"]:
        assert abs(scoring.set_diversity(c["solutions"]) - c["diversity"]) < 1e-9


def test_energy_oob_edge_is_skipped_not_raising():
    # edge (0, 5) references node 5, out of range for a 2-spin problem; must be
    # skipped like a length-mismatched h/j entry, not raise IndexError.
    # E = (1*1) + (1*-1) = 0 -> 0 milli
    assert scoring.energy_milli([1, -1], [1.0, 1.0], [1.0], [(0, 5)]) == 0


def test_energy_truncates_toward_zero():
    # E = 0.0015 -> int(1.5) -> 1 (truncation, not round to 2); exercises the
    # truncation branch through scoring.energy_milli itself, mirroring Rust's
    # energy_truncates_toward_zero test (golden `energy` cases are all
    # integer-valued so this path is otherwise never hit via the function).
    assert scoring.energy_milli([1], [0.0015], [], []) == 1


def test_energy_milli_saturates_at_i64_boundary():
    # Python's int(e*1000) is unbounded and even raises OverflowError for very
    # large e; Rust's `(e * 1000.0) as i64` saturates. Python must replicate
    # Rust's saturating cast so cross-language scores agree at the boundary.
    i64_max = (1 << 63) - 1
    assert scoring.energy_milli([1], [1e16], [], []) == i64_max
    # Must not raise (Python's bare int() would raise OverflowError here).
    assert scoring.energy_milli([1], [1e308], [], []) == i64_max


def test_energy_milli_saturates_at_negative_i64_boundary():
    i64_min = -(1 << 63)
    assert scoring.energy_milli([1], [-1e16], [], []) == i64_min


def test_energy_milli_rejects_negative_edge_index():
    # scoring is now the PyO3 binding to Rust, whose edge indices are usize and
    # cannot be negative. A negative index is a type error, not a silent skip —
    # stricter than the old pure-Python guard and unreachable from the wire
    # (which decodes u32 -> non-negative). This enforces Rust's contract.
    with pytest.raises((OverflowError, ValueError)):
        scoring.energy_milli([1, -1], [], [1.0], [(-1, 0)])


def test_set_diversity_zero_width_solutions_is_zero():
    # Zero-length solution vectors must not raise ZeroDivisionError; the
    # shared reference and Rust both return 0.0 for this case.
    assert scoring.set_diversity([[], []]) == 0.0
