"""Protocol definition for external source adapters."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from neural_memory.integration.models import (
    ExternalRecord,
    SourceCapability,
    SourceSystemType,
)


@runtime_checkable
class SourceAdapter(Protocol):
    """Protocol defining the interface for external source adapters.

    Each adapter connects to one external memory system and normalizes
    its records into ExternalRecord instances.
    """

    @property
    def system_type(self) -> SourceSystemType:
        """Category of this source system (e.g., vector_db, graph_store)."""
        ...

    @property
    def system_name(self) -> str:
        """Unique name of this source system (e.g., 'chromadb', 'mem0')."""
        ...

    @property
    def capabilities(self) -> frozenset[SourceCapability]:
        """Set of capabilities this adapter supports."""
        ...

    async def fetch_all(
        self,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """Fetch all records from the external source.

        Args:
            collection: Optional collection/namespace filter
            limit: Optional maximum number of records to fetch

        Returns:
            List of normalized ExternalRecord instances
        """
        ...

    async def fetch_since(
        self,
        since: datetime,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """Fetch records modified since a given timestamp.

        Args:
            since: Only fetch records modified after this timestamp
            collection: Optional collection/namespace filter
            limit: Optional maximum number of records to fetch

        Returns:
            List of normalized ExternalRecord instances
        """
        ...

    async def health_check(self) -> dict[str, Any]:
        """Verify connection to the external source.

        Returns:
            Dict with at least 'healthy' (bool) and 'message' (str) keys
        """
        ...

    # Write operations for bidirectional sync (optional)

    async def create_record(
        self,
        content: str,
        *,
        collection: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: set[str] | None = None,
    ) -> str | None:
        """Create a new record in the external source.

        Args:
            content: The text content to store
            collection: Optional collection/namespace
            metadata: Optional metadata dict
            tags: Optional set of tags

        Returns:
            The ID of the created record, or None if not supported
        """
        ...

    async def update_record(
        self,
        record_id: str,
        content: str | None = None,
        *,
        collection: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: set[str] | None = None,
    ) -> bool:
        """Update an existing record in the external source.

        Args:
            record_id: The ID of the record to update
            content: Optional new content
            collection: Optional collection/namespace
            metadata: Optional updated metadata
            tags: Optional updated tags

        Returns:
            True if update succeeded, False otherwise
        """
        ...

    async def delete_record(
        self,
        record_id: str,
        *,
        collection: str | None = None,
    ) -> bool:
        """Delete a record from the external source.

        Args:
            record_id: The ID of the record to delete
            collection: Optional collection/namespace

        Returns:
            True if deletion succeeded, False otherwise
        """
        ...
