"""NumPy fallback vector store for PugBrain.

Pure-Python vector storage using NumPy for cosine similarity.
Used as a fallback when the ruvector Rust package is unavailable.
Suitable for development, testing, and small-scale deployments.

Performance: O(n) brute-force search. Fine for < 100k vectors.
For production with > 100k vectors, use the ruvector backend.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from neural_memory.storage.vector.base import VectorSearchResult, VectorStore

logger = logging.getLogger("pugbrain.vector.numpy")


class NumpyVectorStore(VectorStore):
    """Pure-Python vector store using NumPy brute-force search.

    Fallback backend for environments without ruvector.
    Stores vectors in a numpy array with cosine similarity search.

    Args:
        dimension: Vector dimensionality.
        persist_dir: Directory for persistent storage (npz format).
    """

    def __init__(
        self,
        dimension: int = 768,
        persist_dir: str | None = None,
    ) -> None:
        self._dimension = dimension
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._metadata: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        """Load existing data from disk if available."""
        if self._persist_dir:
            data_path = self._persist_dir / "vectors.npz"
            meta_path = self._persist_dir / "metadata.json"

            if data_path.exists():
                logger.info("PugBrain NumPy: Loading vectors from %s", data_path)
                data = np.load(data_path, allow_pickle=True)
                self._vectors = data["vectors"]
                self._ids = data["ids"].tolist()

                if meta_path.exists():
                    self._metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                logger.info(
                    "PugBrain NumPy: Loaded %d vectors (dim=%d)",
                    len(self._ids),
                    self._dimension,
                )
                return

        self._vectors = np.empty((0, self._dimension), dtype=np.float32)
        self._ids = []
        logger.info(
            "PugBrain NumPy: Initialized empty store (dim=%d)",
            self._dimension,
        )

    async def upsert(
        self,
        vector_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a vector."""
        vec = np.array(embedding, dtype=np.float32)

        # Normalize for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if vector_id in self._ids:
            idx = self._ids.index(vector_id)
            self._vectors[idx] = vec  # type: ignore[index]
        else:
            self._ids.append(vector_id)
            if self._vectors is None or len(self._vectors) == 0:
                self._vectors = vec.reshape(1, -1)
            else:
                self._vectors = np.vstack([self._vectors, vec.reshape(1, -1)])

        self._metadata[vector_id] = metadata or {}

    async def upsert_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Batch insert/update."""
        for i, (vid, emb) in enumerate(zip(ids, embeddings, strict=False)):
            meta = metadatas[i] if metadatas and i < len(metadatas) else None
            await self.upsert(vid, emb, meta)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Brute-force cosine similarity search."""
        if self._vectors is None or len(self._ids) == 0:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Cosine similarity (vectors are pre-normalized)
        similarities = self._vectors @ query

        # Sort by similarity descending
        sorted_indices = np.argsort(-similarities)

        results: list[VectorSearchResult] = []
        for idx in sorted_indices:
            vid = self._ids[idx]
            meta = self._metadata.get(vid, {})

            if filter_metadata:
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    continue

            results.append(
                VectorSearchResult(
                    id=vid,
                    score=float(similarities[idx]),
                    metadata=meta,
                )
            )
            if len(results) >= top_k:
                break

        return results

    async def delete(self, vector_id: str) -> bool:
        """Delete a vector."""
        if vector_id not in self._ids:
            return False

        idx = self._ids.index(vector_id)
        self._ids.pop(idx)
        if self._vectors is not None:
            self._vectors = np.delete(self._vectors, idx, axis=0)
        self._metadata.pop(vector_id, None)
        return True

    async def delete_batch(self, vector_ids: list[str]) -> int:
        """Delete multiple vectors."""
        count = 0
        for vid in vector_ids:
            if await self.delete(vid):
                count += 1
        return count

    async def count(self) -> int:
        """Get total vector count."""
        return len(self._ids)

    async def get(self, vector_id: str) -> tuple[list[float], dict[str, Any]] | None:
        """Get a vector and its metadata."""
        if vector_id not in self._ids:
            return None
        idx = self._ids.index(vector_id)
        vec = self._vectors[idx].tolist() if self._vectors is not None else []
        return (vec, self._metadata.get(vector_id, {}))

    async def flush(self) -> None:
        """Persist to disk."""
        if self._persist_dir is None or self._vectors is None:
            return

        self._persist_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            self._persist_dir / "vectors.npz",
            vectors=self._vectors,
            ids=np.array(self._ids),
        )

        meta_path = self._persist_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(self._metadata, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("PugBrain NumPy: Flushed %d vectors", len(self._ids))

    async def close(self) -> None:
        """Flush and clean up."""
        await self.flush()
        self._vectors = None
        self._ids = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def backend_name(self) -> str:
        return "numpy"
