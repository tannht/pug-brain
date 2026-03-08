"""Vector storage backends for PugBrain semantic search.

Provides a unified interface for vector similarity search with
pluggable backends. RuVector (Rust-based) is the primary backend.

Usage:
    from neural_memory.storage.vector import create_vector_store

    store = create_vector_store(backend="ruvector", dimension=768)
    await store.initialize()
    await store.upsert("neuron-1", embedding_vector, {"content": "hello"})
    results = await store.search(query_vector, top_k=10)
"""

from neural_memory.storage.vector.base import VectorSearchResult, VectorStore
from neural_memory.storage.vector.factory import create_vector_store

__all__ = ["VectorSearchResult", "VectorStore", "create_vector_store"]
