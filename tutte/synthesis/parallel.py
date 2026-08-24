"""Process-level parallelism for Tutte polynomial synthesis.

Parallelizes the two independent recursive calls in _batch_reduce_parallel:
when reducing k parallel edges between u,v, T(G₀) and T(G_c) are independent
and can be computed on separate cores.

Uses ProcessPoolExecutor with 2 workers. Each worker gets its own
SynthesisEngine with a snapshot of the main cache. New cache entries
discovered by workers are merged back into the main engine.
"""

from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from ..graph import MultiGraph
    from ..polynomial import TuttePolynomial
    from .base import BaseMultigraphSynthesizer


# Module-level worker state (initialized per-process)
_worker_engine = None


def _init_worker(cache_snapshot_bytes: bytes, verbose: bool) -> None:
    """Initialize worker process with its own SynthesisEngine."""
    global _worker_engine
    from .engine import SynthesisEngine
    _worker_engine = SynthesisEngine(verbose=verbose)
    cache_snapshot = pickle.loads(cache_snapshot_bytes)
    _worker_engine._multigraph_cache.update(cache_snapshot)
    _worker_engine._fast_hash_set_complete = False
    _worker_engine._in_worker = True  # Prevent nested parallelism


def _synthesize_worker(args: bytes) -> bytes:
    """Worker function: synthesize a multigraph and return result + new cache entries.

    Args/returns are pickled bytes to avoid repeated pickling of the same objects.
    """
    mg, max_depth, skip_minor_search = pickle.loads(args)
    pre_keys = set(_worker_engine._multigraph_cache.keys())
    poly = _worker_engine._synthesize_multigraph(mg, max_depth, skip_minor_search)
    new_entries = {k: v for k, v in _worker_engine._multigraph_cache.items()
                   if k not in pre_keys}
    return pickle.dumps((poly, new_entries))


# Module-level pool (lazy-initialized, reused across calls)
_pool = None
_pool_cache_size = 0  # Track cache size at pool creation to know when to refresh


def _get_or_create_pool(engine: BaseMultigraphSynthesizer) -> ProcessPoolExecutor:
    """Get or create the worker pool, refreshing if cache has grown significantly."""
    global _pool, _pool_cache_size

    current_cache_size = len(engine._multigraph_cache)

    if _pool is None or current_cache_size > _pool_cache_size * 2 + 100:
        # Shut down old pool if exists
        if _pool is not None:
            _pool.shutdown(wait=False)

        cache_bytes = pickle.dumps(dict(engine._multigraph_cache))
        _pool = ProcessPoolExecutor(
            max_workers=2,
            initializer=_init_worker,
            initargs=(cache_bytes, engine.verbose),
        )
        _pool_cache_size = current_cache_size

    return _pool


def parallel_synthesize_pair(
    engine: BaseMultigraphSynthesizer,
    mg1: MultiGraph,
    mg2: MultiGraph,
    max_depth: int,
    skip_minor_search: bool,
) -> Tuple[TuttePolynomial, TuttePolynomial]:
    """Synthesize two multigraphs in parallel on separate processes.

    Args:
        engine: The main synthesis engine (for cache snapshot and merging)
        mg1: First multigraph to synthesize
        mg2: Second multigraph to synthesize
        max_depth: Maximum recursion depth
        skip_minor_search: Whether to skip expensive minor search

    Returns:
        Tuple of (poly1, poly2) — the Tutte polynomials for mg1 and mg2
    """
    pool = _get_or_create_pool(engine)

    args1 = pickle.dumps((mg1, max_depth, skip_minor_search))
    args2 = pickle.dumps((mg2, max_depth, skip_minor_search))

    f1 = pool.submit(_synthesize_worker, args1)
    f2 = pool.submit(_synthesize_worker, args2)

    poly1, cache1 = pickle.loads(f1.result())
    poly2, cache2 = pickle.loads(f2.result())

    # Merge worker-discovered cache entries back into main engine
    engine._merge_worker_cache(cache1)
    engine._merge_worker_cache(cache2)

    return poly1, poly2


def shutdown_pool() -> None:
    """Shut down the worker pool. Call when done with parallel synthesis."""
    global _pool, _pool_cache_size
    if _pool is not None:
        _pool.shutdown(wait=True)
        _pool = None
        _pool_cache_size = 0
