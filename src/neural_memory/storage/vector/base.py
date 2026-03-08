"""Abstract base class for vector storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorSearchResult:
    """Result from a vector similarity search.

    Attributes:
        id: The vector/document ID.
        score: Similarity score (higher = more similar, 0.0-1.0 for cosine).
        metadata: Optional metadata dict stored alongside the vector.
    """

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract interface for vector storage backends.

    All PugBrain vector backends must implement this interface.
    The primary backend is RuVector (Rust-based), with fallback
    to a pure-Python NumPy implementation for environments where
    Rust binaries are unavailable.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store (create indexes, load data, etc.)."""
        ...

    @abstractmethod
    async def upsert(
        self,
        vector_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a vector.

        Args:
            vector_id: Unique identifier for this vector.
            embedding: The embedding vector (list of floats).
            metadata: Optional metadata to store alongside the vector.
        """
        ...

    @abstractmethod
    async def upsert_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Batch insert or update vectors.

        Args:
            ids: List of vector IDs.
            embeddings: List of embedding vectors.
            metadatas: Optional list of metadata dicts.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter (exact match).

        Returns:
            List of VectorSearchResult, sorted by score descending.
        """
        ...

    @abstractmethod
    async def delete(self, vector_id: str) -> bool:
        """Delete a vector by ID.

        Args:
            vector_id: The vector ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def delete_batch(self, vector_ids: list[str]) -> int:
        """Delete multiple vectors by ID.

        Args:
            vector_ids: List of vector IDs to delete.

        Returns:
            Number of vectors actually deleted.
        """
        ...

    @abstractmethod
    async def count(self) -> int:
        """Get the total number of vectors stored.

        Returns:
            Total vector count.
        """
        ...

    @abstractmethod
    async def get(self, vector_id: str) -> tuple[list[float], dict[str, Any]] | None:
        """Get a vector and its metadata by ID.

        Args:
            vector_id: The vector ID.

        Returns:
            Tuple of (embedding, metadata) or None if not found.
        """
        ...

    async def close(self) -> None:  # noqa: B027
        """Close the vector store and release resources."""

    async def flush(self) -> None:  # noqa: B027
        """Flush any pending writes to persistent storage."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of vectors in this store."""
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable name of the backend (e.g., 'ruvector', 'numpy')."""
        ...
