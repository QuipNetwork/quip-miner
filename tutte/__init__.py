"""Tutte polynomial synthesis library."""

# Foundational types
from .polynomial import TuttePolynomial, encode_varuint, decode_varuint
from .graph import (
    Graph, MultiGraph, CellSignature, NodeSignature,
    complete_graph, cycle_graph, path_graph, star_graph, wheel_graph,
    petersen_graph, grid_graph, cut_vertex_join, disjoint_union,
)
from .factorization import (
    polynomial_gcd, has_common_factor, monomial_content,
    primitive_part,
)
from .validation import (
    verify_spanning_trees, count_spanning_trees_kirchhoff,
    compute_tutte_networkx,
)

# Graph algorithms
from .graphs import (
    is_series_parallel, compute_sp_tutte_if_applicable,
    polynomial_divmod, polynomial_divide,
    is_graph_minor,
)

# Lookup / rainbow table
from .lookup import (
    RainbowTable, MinorEntry, GCDMinorIndex, load_default_table,
    encode_rainbow_table_binary, decode_rainbow_table_binary,
    load_binary_rainbow_table, save_binary_rainbow_table,
    build_basic_table, sympy_to_tutte,
)

# Synthesis engine
from .synthesis import (
    SynthesisEngine, SynthesisResult, BaseMultigraphSynthesizer, UnionFind,
    synthesize, compute_tutte_polynomial,
)