"""RuVector backend — Rust-based high-performance vector storage for PugBrain.

RuVector provides HNSW-based approximate nearest neighbor search
with optional persistence to disk. It is the primary vector backend
for PugBrain, replacing FAISS/ChromaDB with a lightweight Rust library.

When the `ruvector` package is not installed, the factory will
transparently fall back to the NumPy backend.

Expected ruvector API (Rust bindings via PyO3):
    import ruvector
    index = ruvector.HnswIndex(dimension=768, metric="cosine")
    index.add(id, vector)
    results = index.search(vector, k=10)
    index.save(path)
    index = ruvector.HnswIndex.load(path)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from neural_memory.storage.vector.base import VectorSearchResult, VectorStore

logger = logging.getLogger("pugbrain.vector.ruvector")


class RuVectorStore(VectorStore):
    """Rust-based vector storage using the ruvector HNSW library.

    This is PugBrain's primary vector backend. It provides:
    - HNSW (Hierarchical Navigable Small World) indexing
    - Cosine and L2 distance metrics
    - Persistent storage to disk
    - Thread-safe concurrent access via Rust's ownership model

    Args:
        dimension: Vector dimensionality (e.g., 768 for MiniLM, 3072 for text-embedding-3-large).
        persist_dir: Directory for persistent storage. None = in-memory only.
        metric: Distance metric — "cosine" (default) or "l2".
        ef_construction: HNSW build-time quality param (higher = better index, slower build).
        m: HNSW max connections per node (higher = better recall, more memory).
    """

    def __init__(
        self,
        dimension: int = 768,
        persist_dir: str | None = None,
        metric: str = "cosine",
        ef_construction: int = 200,
        m: int = 16,
    ) -> None:
        self._dimension = dimension
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._metric = metric
        self._ef_construction = ef_construction
        self._m = m
        self._index: Any = None
        self._metadata: dict[str, dict[str, Any]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    async def initialize(self) -> None:
        """Initialize the RuVector index, loading from disk if available."""
        self._loop = asyncio.get_event_loop()

        try:
            import ruvector

            # Try to load existing index
            if self._persist_dir and (self._persist_dir / "index.ruv").exists():
                logger.info("PugBrain RuVector: Loading index from %s", self._persist_dir)
                self._index = await asyncio.to_thread(
                    ruvector.HnswIndex.load,
                    str(self._persist_dir / "index.ruv"),
                )
                # Load metadata sidecar
                meta_path = self._persist_dir / "metadata.json"
                if meta_path.exists():
                    self._metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                logger.info(
                    "PugBrain RuVector: Creating new HNSW index (dim=%d, metric=%s)",
                    self._dimension,
                    self._metric,
                )
                self._index = ruvector.HnswIndex(
                    dimension=self._dimension,
                    metric=self._metric,
                    ef_construction=self._ef_construction,
                    m=self._m,
                )
        except ImportError:
            raise ImportError(
                "PugBrain RuVector backend requires the 'ruvector' package. "
                "Install with: pip install ruvector\n"
                "Or use backend='numpy' for a pure-Python fallback. Gâu gâu! 🐶"
            )

    async def upsert(
        self,
        vector_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a vector in the HNSW index."""
        if self._index is None:
            raise RuntimeError("PugBrain RuVector: Index not initialized. Call initialize() first.")

        await asyncio.to_thread(self._index.upsert, vector_id, embedding)
        if metadata:
            self._metadata[vector_id] = metadata
        elif vector_id not in self._metadata:
            self._metadata[vector_id] = {}

    async def upsert_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Batch insert/update vectors."""
        if self._index is None:
            raise RuntimeError("PugBrain RuVector: Index not initialized.")

        await asyncio.to_thread(self._index.upsert_batch, ids, embeddings)
        for i, vid in enumerate(ids):
            self._metadata[vid] = metadatas[i] if metadatas and i < len(metadatas) else {}

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using HNSW."""
        if self._index is None:
            raise RuntimeError("PugBrain RuVector: Index not initialized.")

        # Over-fetch if filtering to ensure we get enough results
        fetch_k = top_k * 3 if filter_metadata else top_k

        raw_results = await asyncio.to_thread(self._index.search, query_embedding, fetch_k)

        results: list[VectorSearchResult] = []
        for rid, score in raw_results:
            meta = self._metadata.get(rid, {})

            # Apply metadata filter
            if filter_metadata:
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    continue

            results.append(VectorSearchResult(id=rid, score=float(score), metadata=meta))

            if len(results) >= top_k:
                break

        return results

    async def delete(self, vector_id: str) -> bool:
        """Delete a vector from the index."""
        if self._index is None:
            return False

        try:
            await asyncio.to_thread(self._index.delete, vector_id)
            self._metadata.pop(vector_id, None)
            return True
        except (KeyError, ValueError):
            return False

    async def delete_batch(self, vector_ids: list[str]) -> int:
        """Delete multiple vectors."""
        count = 0
        for vid in vector_ids:
            if await self.delete(vid):
                count += 1
        return count

    async def count(self) -> int:
        """Get total vector count."""
        if self._index is None:
            return 0
        return await asyncio.to_thread(lambda: self._index.count())

    async def get(self, vector_id: str) -> tuple[list[float], dict[str, Any]] | None:
        """Get a vector and its metadata."""
        if self._index is None:
            return None
        try:
            vec = await asyncio.to_thread(self._index.get, vector_id)
            return (vec, self._metadata.get(vector_id, {}))
        except (KeyError, ValueError):
            return None

    async def flush(self) -> None:
        """Persist index and metadata to disk."""
        if self._index is None or self._persist_dir is None:
            return

        self._persist_dir.mkdir(parents=True, exist_ok=True)

        await asyncio.to_thread(self._index.save, str(self._persist_dir / "index.ruv"))

        meta_path = self._persist_dir / "metadata.json"
        meta_json = json.dumps(self._metadata, ensure_ascii=False)
        await asyncio.to_thread(meta_path.write_text, meta_json, "utf-8")

        logger.debug(
            "PugBrain RuVector: Flushed %d vectors to %s", len(self._metadata), self._persist_dir
        )

    async def close(self) -> None:
        """Flush and close."""
        await self.flush()
        self._index = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def backend_name(self) -> str:
        return "ruvector"
