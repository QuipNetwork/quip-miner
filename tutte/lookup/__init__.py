"""Rainbow table lookup system subpackage."""

from .core import (
    MinorEntry,
    GCDMinorIndex,
    RainbowTable,
    load_default_table,
    load_default_multigraph_table,
    save_default_multigraph_table,
)
from .merger import (
    MergerEntry,
    MergerTable,
    VTTuple,
    load_default_merger_table,
    save_default_merger_table,
)
from .binary import (
    encode_rainbow_table_binary,
    decode_rainbow_table_binary,
    load_binary_rainbow_table,
    save_binary_rainbow_table,
    encode_merger_lookup_table,
    decode_merger_lookup_table,
    load_merger_lookup_table,
    save_merger_lookup_table,
)
from .bootstrap import (
    build_basic_table,
    sympy_to_tutte,
)
# Re-export is_graph_minor here for backward compat (was in rainbow_table)
from ..graphs.minor import is_graph_minor
