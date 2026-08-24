"""Binary Encoding/Decoding for Rainbow Table and Contraction Cache.

This module provides efficient binary serialization of the rainbow table,
supporting both v1 (legacy) and v2 (with canonical keys and minor relationships)
formats, as well as multigraph and contraction cache tables.
"""

from __future__ import annotations

from typing import Dict, List

from ..polynomial import TuttePolynomial, encode_varuint, decode_varuint
from .core import RainbowTable, MinorEntry


# =============================================================================
# BINARY ENCODING
# =============================================================================

def encode_rainbow_table_binary(table: RainbowTable) -> bytes:
    """Encode rainbow table to compact binary v2 format.

    v2 Format:
        Header:
            [magic: 4 bytes]    = "RTBL"
            [version: 1 byte]   = 2
            [flags: 1 byte]     bit 0: has_minor_rels, bit 1: structural_minors_computed
            [num_entries: varuint]

        Entry Section (per entry):
            [canonical_key: 32 bytes]       <- raw SHA256
            [name_len: varuint] [name: bytes]
            [node_count: varuint]
            [edge_count: varuint]
            [spanning_trees: varuint]
            [poly_len: varuint] [poly_bytes: bytes]

        Minor Relationships Section (if flag bit 0):
            [num_majors: varuint]
            Per major:
                [major_index: varuint]
                [num_minors: varuint]
                [minor_index: varuint] x num_minors
    """
    import hashlib

    result = bytearray()

    # Magic header
    result.extend(b"RTBL")
    result.append(2)  # version

    # Flags
    has_minors = bool(table.minor_relationships)
    flags = 0
    if has_minors:
        flags |= 0x01
    if table._structural_minors_computed:
        flags |= 0x02
    result.append(flags)

    # Number of entries
    entries_ordered = list(table.entries.items())
    result.extend(encode_varuint(len(entries_ordered)))

    # Build key -> index map for minor relationships
    key_to_index: Dict[str, int] = {}
    for idx, (key, _entry) in enumerate(entries_ordered):
        key_to_index[key] = idx

    # Entry section
    for key, entry in entries_ordered:
        # Canonical key as raw 32-byte SHA256
        result.extend(bytes.fromhex(key))

        # Name
        name_bytes = entry.name.encode('utf-8')
        result.extend(encode_varuint(len(name_bytes)))
        result.extend(name_bytes)

        # Metadata
        result.extend(encode_varuint(entry.node_count))
        result.extend(encode_varuint(entry.edge_count))
        result.extend(encode_varuint(entry.spanning_trees))

        # Polynomial as bitstring
        poly_bytes = entry.polynomial.to_bytes()
        result.extend(encode_varuint(len(poly_bytes)))
        result.extend(poly_bytes)

    # Minor relationships section
    if has_minors:
        # Filter to only majors with valid indices
        valid_majors = []
        for major_key, minor_keys in table.minor_relationships.items():
            if major_key not in key_to_index:
                continue
            valid_minor_indices = []
            for mk in minor_keys:
                if mk in key_to_index:
                    valid_minor_indices.append(key_to_index[mk])
            if valid_minor_indices:
                valid_majors.append((key_to_index[major_key], valid_minor_indices))

        result.extend(encode_varuint(len(valid_majors)))
        for major_idx, minor_indices in valid_majors:
            result.extend(encode_varuint(major_idx))
            result.extend(encode_varuint(len(minor_indices)))
            for mi in minor_indices:
                result.extend(encode_varuint(mi))

    return bytes(result)


def decode_rainbow_table_binary(data: bytes) -> RainbowTable:
    """Decode binary rainbow table (supports v1 and v2 formats).

    Returns a fully populated RainbowTable with entries and minor_relationships.
    """
    offset = 0

    # Magic header
    if data[offset:offset + 4] != b"RTBL":
        raise ValueError("Invalid magic header -- not a rainbow table binary")
    offset += 4

    version = data[offset]
    offset += 1

    if version == 1:
        return _decode_binary_v1(data, offset)
    elif version == 2:
        return _decode_binary_v2(data, offset)
    else:
        raise ValueError(f"Unsupported binary version: {version}")


def _decode_binary_v1(data: bytes, offset: int) -> RainbowTable:
    """Decode v1 binary format (no canonical keys, no minor relationships)."""
    table = RainbowTable()

    num_entries, offset = decode_varuint(data, offset)

    for _ in range(num_entries):
        # Name
        name_len, offset = decode_varuint(data, offset)
        name = data[offset:offset + name_len].decode('utf-8')
        offset += name_len

        # Metadata
        node_count, offset = decode_varuint(data, offset)
        edge_count, offset = decode_varuint(data, offset)
        spanning_trees, offset = decode_varuint(data, offset)

        # Polynomial
        poly_len, offset = decode_varuint(data, offset)
        poly_bytes = data[offset:offset + poly_len]
        offset += poly_len
        polynomial = TuttePolynomial.from_bytes(poly_bytes)

        # v1 has no canonical key stored -- we can't reconstruct it without
        # the graph, so use a placeholder based on name
        import hashlib
        canonical_key = hashlib.sha256(f"v1:{name}".encode()).hexdigest()

        entry = MinorEntry(
            name=name,
            polynomial=polynomial,
            node_count=node_count,
            edge_count=edge_count,
            canonical_key=canonical_key,
            spanning_trees=spanning_trees,
            num_terms=polynomial.num_terms(),
        )

        table.entries[canonical_key] = entry
        table.name_index[name] = canonical_key

    table._sort_by_complexity()
    return table


def _decode_binary_v2(data: bytes, offset: int) -> RainbowTable:
    """Decode v2 binary format (with canonical keys and minor relationships)."""
    table = RainbowTable()

    # Flags
    flags = data[offset]
    offset += 1
    has_minors = bool(flags & 0x01)
    table._structural_minors_computed = bool(flags & 0x02)

    # Number of entries
    num_entries, offset = decode_varuint(data, offset)

    # Read entries, build index -> key mapping
    index_to_key: List[str] = []

    for _ in range(num_entries):
        # Canonical key: 32 raw bytes -> hex string
        canonical_key = data[offset:offset + 32].hex()
        offset += 32

        # Name
        name_len, offset = decode_varuint(data, offset)
        name = data[offset:offset + name_len].decode('utf-8')
        offset += name_len

        # Metadata
        node_count, offset = decode_varuint(data, offset)
        edge_count, offset = decode_varuint(data, offset)
        spanning_trees, offset = decode_varuint(data, offset)

        # Polynomial
        poly_len, offset = decode_varuint(data, offset)
        poly_bytes = data[offset:offset + poly_len]
        offset += poly_len
        polynomial = TuttePolynomial.from_bytes(poly_bytes)

        entry = MinorEntry(
            name=name,
            polynomial=polynomial,
            node_count=node_count,
            edge_count=edge_count,
            canonical_key=canonical_key,
            spanning_trees=spanning_trees,
            num_terms=polynomial.num_terms(),
        )

        table.entries[canonical_key] = entry
        table.name_index[name] = canonical_key
        index_to_key.append(canonical_key)

    # Minor relationships section
    if has_minors:
        num_majors, offset = decode_varuint(data, offset)
        for _ in range(num_majors):
            major_idx, offset = decode_varuint(data, offset)
            num_minors, offset = decode_varuint(data, offset)
            minor_keys = []
            for _ in range(num_minors):
                minor_idx, offset = decode_varuint(data, offset)
                if minor_idx < len(index_to_key):
                    minor_keys.append(index_to_key[minor_idx])
            if major_idx < len(index_to_key) and minor_keys:
                table.minor_relationships[index_to_key[major_idx]] = minor_keys

    table._sort_by_complexity()
    return table


def load_binary_rainbow_table(path: str) -> RainbowTable:
    """Load rainbow table from binary file."""
    with open(path, 'rb') as f:
        data = f.read()
    return decode_rainbow_table_binary(data)


def save_binary_rainbow_table(table: RainbowTable, path: str) -> int:
    """Save rainbow table to binary format, return size in bytes."""
    data = encode_rainbow_table_binary(table)
    with open(path, 'wb') as f:
        f.write(data)
    return len(data)


# =============================================================================
# MULTIGRAPH LOOKUP TABLE BINARY ENCODING
# =============================================================================

def encode_multigraph_lookup_table(cache: Dict[str, 'TuttePolynomial']) -> bytes:
    """Encode multigraph lookup table to compact binary format.

    Format:
        Header:
            [magic: 4 bytes]    = "MGLT"
            [version: 1 byte]   = 1
            [num_entries: varuint]

        Entry Section (per entry):
            [canonical_key: 32 bytes]       <- raw SHA256
            [poly_len: varuint] [poly_bytes: bytes]
    """
    result = bytearray()

    # Magic header
    result.extend(b"MGLT")
    result.append(1)  # version

    # Number of entries
    result.extend(encode_varuint(len(cache)))

    # Entry section
    for key, poly in cache.items():
        # Canonical key as raw 32-byte SHA256
        result.extend(bytes.fromhex(key))

        # Polynomial as binary
        poly_bytes = poly.to_bytes()
        result.extend(encode_varuint(len(poly_bytes)))
        result.extend(poly_bytes)

    return bytes(result)


def decode_multigraph_lookup_table(data: bytes) -> Dict[str, 'TuttePolynomial']:
    """Decode multigraph lookup table from binary format.

    Returns dict of canonical_key -> TuttePolynomial.
    """
    from ..polynomial import TuttePolynomial

    offset = 0

    # Magic header
    if data[offset:offset + 4] != b"MGLT":
        raise ValueError("Invalid magic header -- not a multigraph lookup table binary")
    offset += 4

    version = data[offset]
    offset += 1
    if version != 1:
        raise ValueError(f"Unsupported multigraph lookup table version: {version}")

    # Number of entries
    num_entries, offset = decode_varuint(data, offset)

    cache: Dict[str, 'TuttePolynomial'] = {}
    for _ in range(num_entries):
        # Canonical key: 32 raw bytes -> hex string
        canonical_key = data[offset:offset + 32].hex()
        offset += 32

        # Polynomial
        poly_len, offset = decode_varuint(data, offset)
        poly_bytes = data[offset:offset + poly_len]
        offset += poly_len
        polynomial = TuttePolynomial.from_bytes(poly_bytes)

        cache[canonical_key] = polynomial

    return cache


def save_multigraph_lookup_table(cache: Dict[str, 'TuttePolynomial'], path: str) -> int:
    """Save multigraph lookup table to binary format, return size in bytes."""
    data = encode_multigraph_lookup_table(cache)
    with open(path, 'wb') as f:
        f.write(data)
    return len(data)


def load_multigraph_lookup_table(path: str) -> Dict[str, 'TuttePolynomial']:
    """Load multigraph lookup table from binary file."""
    with open(path, 'rb') as f:
        data = f.read()
    return decode_multigraph_lookup_table(data)


# =============================================================================
# MERGER LOOKUP TABLE — chord-junction mergers T(G ∪_{V_T} G)
# =============================================================================
#
# Stores precomputed merger polynomials indexed by (base_canonical_key, V_T).
# Schema mirrors `multigraph_lookup_table` (raw 32-byte canonical keys +
# binary polynomial) but adds the V_T tuple, source/family metadata, and a
# secondary merger canonical key. See `tutte/lookup/merger.py` for the
# data model. Magic header `"MRGT"`, version `1`.


def _encode_optional_string(value) -> bytes:
    """Encode an Optional[str] as ``[len varuint][utf8 bytes]``.

    Empty/None values are stored as length 0; the decoder distinguishes
    None from "" via a presence flag prepended by the caller.
    """
    if value is None:
        return encode_varuint(0)
    raw = value.encode('utf-8')
    return encode_varuint(len(raw)) + raw


def _decode_optional_string(data: bytes, offset: int):
    length, offset = decode_varuint(data, offset)
    if length == 0:
        return None, offset
    raw = data[offset:offset + length].decode('utf-8')
    return raw, offset + length


def encode_merger_lookup_table(table) -> bytes:
    """Encode a `MergerTable` to compact binary format.

    Format:
        Header:
            [magic: 4 bytes]   = "MRGT"
            [version: 1 byte]  = 1
            [num_entries: varuint]

        Entry Section (per entry):
            [base_canonical_key: 32 bytes raw]
            [num_v_t: varuint] [v_t[0]: varuint] ... [v_t[k-1]: varuint]
            [merger_present: 1 byte] (0/1)
            [merger_canonical_key: 32 bytes raw]   (only if merger_present)
            [base_name: optional-string]
            [family_tag: optional-string]
            [base_node_count: varuint]
            [base_edge_count: varuint]
            [merger_node_count: varuint]
            [merger_edge_count: varuint]
            [poly_len: varuint] [poly_bytes: bytes]

    `MergerTable` is imported lazily to avoid a circular import with
    `tutte.lookup.merger`, which depends on this module via `binary.py`
    exporting `load_merger_lookup_table`.
    """
    result = bytearray()

    result.extend(b"MRGT")
    result.append(1)  # version
    result.extend(encode_varuint(len(table.by_source)))

    for entry in table.by_source.values():
        # Base canonical key — 32 raw bytes
        result.extend(bytes.fromhex(entry.base_canonical_key))

        # V_T tuple
        result.extend(encode_varuint(len(entry.v_t)))
        for v in entry.v_t:
            result.extend(encode_varuint(v))

        # Optional merger canonical key (32 raw bytes if present)
        if entry.merger_canonical_key is not None:
            result.append(1)
            result.extend(bytes.fromhex(entry.merger_canonical_key))
        else:
            result.append(0)

        # Optional strings
        result.extend(_encode_optional_string(entry.base_name))
        result.extend(_encode_optional_string(entry.family_tag))

        # Counters
        result.extend(encode_varuint(entry.base_node_count))
        result.extend(encode_varuint(entry.base_edge_count))
        result.extend(encode_varuint(entry.merger_node_count))
        result.extend(encode_varuint(entry.merger_edge_count))

        # Polynomial
        poly_bytes = entry.polynomial.to_bytes()
        result.extend(encode_varuint(len(poly_bytes)))
        result.extend(poly_bytes)

    return bytes(result)


def decode_merger_lookup_table(data: bytes):
    """Decode a `MergerTable` from binary format. See encoder for schema."""
    from ..polynomial import TuttePolynomial
    from .merger import MergerEntry, MergerTable

    offset = 0
    if data[offset:offset + 4] != b"MRGT":
        raise ValueError("Invalid magic header -- not a merger lookup table binary")
    offset += 4

    version = data[offset]
    offset += 1
    if version != 1:
        raise ValueError(f"Unsupported merger lookup table version: {version}")

    num_entries, offset = decode_varuint(data, offset)

    table = MergerTable()
    for _ in range(num_entries):
        # Base canonical key
        base_canonical_key = data[offset:offset + 32].hex()
        offset += 32

        # V_T tuple
        v_t_len, offset = decode_varuint(data, offset)
        v_t_items = []
        for _ in range(v_t_len):
            v, offset = decode_varuint(data, offset)
            v_t_items.append(v)
        v_t = tuple(v_t_items)

        # Optional merger canonical key
        merger_present = data[offset]
        offset += 1
        if merger_present:
            merger_canonical_key = data[offset:offset + 32].hex()
            offset += 32
        else:
            merger_canonical_key = None

        # Optional strings
        base_name, offset = _decode_optional_string(data, offset)
        family_tag, offset = _decode_optional_string(data, offset)

        # Counters
        base_node_count, offset = decode_varuint(data, offset)
        base_edge_count, offset = decode_varuint(data, offset)
        merger_node_count, offset = decode_varuint(data, offset)
        merger_edge_count, offset = decode_varuint(data, offset)

        # Polynomial
        poly_len, offset = decode_varuint(data, offset)
        poly_bytes = data[offset:offset + poly_len]
        offset += poly_len
        polynomial = TuttePolynomial.from_bytes(poly_bytes)

        entry = MergerEntry(
            base_canonical_key=base_canonical_key,
            v_t=v_t,
            polynomial=polynomial,
            merger_canonical_key=merger_canonical_key,
            base_name=base_name,
            family_tag=family_tag,
            base_node_count=base_node_count,
            base_edge_count=base_edge_count,
            merger_node_count=merger_node_count,
            merger_edge_count=merger_edge_count,
        )
        table.add_entry(entry)

    return table


def save_merger_lookup_table(table, path: str) -> int:
    """Save a `MergerTable` to a binary file. Returns size in bytes."""
    blob = encode_merger_lookup_table(table)
    with open(path, 'wb') as f:
        f.write(blob)
    return len(blob)


def load_merger_lookup_table(path: str):
    """Load a `MergerTable` from a binary file."""
    with open(path, 'rb') as f:
        data = f.read()
    return decode_merger_lookup_table(data)


