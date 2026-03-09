"""Vector store factory for PugBrain.

Creates the appropriate vector storage backend based on configuration.
Priority: ruvector (Rust) > numpy (fallback).

Usage:
    from neural_memory.storage.vector import create_vector_store

    # Auto-detect best available backend
    store = create_vector_store(dimension=768)

    # Force specific backend
    store = create_vector_store(backend="ruvector", dimension=768)
    store = create_vector_store(backend="numpy", dimension=768)
"""

from __future__ import annotations

import logging
from typing import Any

from neural_memory.storage.vector.base import VectorStore

logger = logging.getLogger("pugbrain.vector.factory")

_VALID_BACKENDS = {"ruvector", "numpy", "auto"}


def create_vector_store(
    backend: str = "auto",
    dimension: int = 768,
    persist_dir: str | None = None,
    **kwargs: Any,
) -> VectorStore:
    """Create a vector store backend.

    PugBrain uses ruvector (Rust-based HNSW) as the primary vector backend.
    Falls back to numpy brute-force for environments without Rust binaries.

    Args:
        backend: Backend name — "ruvector", "numpy", or "auto" (try ruvector first).
        dimension: Vector dimensionality.
        persist_dir: Directory for persistent storage (None = in-memory).
        **kwargs: Additional backend-specific arguments.

    Returns:
        VectorStore instance (not yet initialized — call await store.initialize()).

    Raises:
        ValueError: If backend is unknown.
        ImportError: If the requested backend is not available.
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"PugBrain: Unknown vector backend '{backend}'. "
            f"Valid options: {', '.join(sorted(_VALID_BACKENDS))}. Gâu! 🐶"
        )

    if backend == "auto":
        return _create_auto(dimension, persist_dir, **kwargs)
    elif backend == "ruvector":
        return _create_ruvector(dimension, persist_dir, **kwargs)
    else:
        return _create_numpy(dimension, persist_dir)


def _create_auto(
    dimension: int,
    persist_dir: str | None,
    **kwargs: Any,
) -> VectorStore:
    """Try ruvector first, fall back to numpy."""
    try:
        import ruvector  # noqa: F401

        logger.info("PugBrain Vector: Using ruvector (Rust HNSW) backend 🦀")
        return _create_ruvector(dimension, persist_dir, **kwargs)
    except ImportError:
        logger.info(
            "PugBrain Vector: ruvector not installed, using numpy fallback. "
            "Install ruvector for 10-100x faster vector search! Gâu gâu! 🐶"
        )
        return _create_numpy(dimension, persist_dir)


def _create_ruvector(
    dimension: int,
    persist_dir: str | None,
    **kwargs: Any,
) -> VectorStore:
    """Create a RuVector store."""
    from neural_memory.storage.vector.ruvector_store import RuVectorStore

    return RuVectorStore(
        dimension=dimension,
        persist_dir=persist_dir,
        metric=kwargs.get("metric", "cosine"),
        ef_construction=kwargs.get("ef_construction", 200),
        m=kwargs.get("m", 16),
    )


def _create_numpy(
    dimension: int,
    persist_dir: str | None,
) -> VectorStore:
    """Create a NumPy fallback store."""
    from neural_memory.storage.vector.numpy_store import NumpyVectorStore

    return NumpyVectorStore(
        dimension=dimension,
        persist_dir=persist_dir,
    )


def get_available_backends() -> list[str]:
    """List available vector backends on this system.

    Returns:
        List of backend names that can be used.
    """
    available = ["numpy"]  # Always available (numpy is a dependency)

    try:
        import ruvector  # noqa: F401

        available.insert(0, "ruvector")
    except ImportError:
        pass

    return available
