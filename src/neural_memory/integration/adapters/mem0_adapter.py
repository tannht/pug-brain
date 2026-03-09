"""Mem0 source adapter for importing memories."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

from neural_memory.integration.models import (
    ExternalRecord,
    SourceCapability,
    SourceSystemType,
)

from neural_memory.utils.timeutils import utcnow

# Default implementations for optional write methods
_NOT_IMPLEMENTED_MSG = "{} adapter does not support this operation"

logger = logging.getLogger(__name__)

# Tag constraints (consistent with _remember handler in server.py)
_MAX_TAGS = 50
_MAX_TAG_LEN = 100


def _parse_mem0_records(
    memories: list[dict[str, Any]] | dict[str, Any],
    *,
    source_system: str,
    user_id: str | None,
    agent_id: str | None,
    limit: int | None,
) -> list[ExternalRecord]:
    """Parse raw Mem0 response into ExternalRecord list.

    Shared by both Platform and Self-hosted adapters.
    """
    records: list[ExternalRecord] = []
    items = memories if isinstance(memories, list) else memories.get("results", [])

    for mem in items:
        if limit is not None and len(records) >= limit:
            break

        mem_id = mem.get("id", "")
        content = mem.get("memory", "") or mem.get("text", "")
        if not content:
            continue

        metadata = mem.get("metadata", {}) or {}

        created_at = utcnow()
        if "created_at" in mem:
            try:
                created_at = datetime.fromisoformat(str(mem["created_at"]).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        updated_at = None
        if "updated_at" in mem:
            try:
                updated_at = datetime.fromisoformat(str(mem["updated_at"]).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        source_type = mem.get("category", metadata.get("type", "memory"))

        tags: set[str] = set()
        if "categories" in mem:
            for cat in mem["categories"]:
                if isinstance(cat, str) and len(cat) <= _MAX_TAG_LEN and len(tags) < _MAX_TAGS:
                    tags.add(cat)
        if user_id:
            tags.add(f"user:{user_id}")
        if agent_id:
            tags.add(f"agent:{agent_id}")

        record = ExternalRecord.create(
            id=str(mem_id),
            source_system=source_system,
            content=content,
            source_collection=user_id or agent_id or "default",
            created_at=created_at,
            updated_at=updated_at,
            source_type=source_type,
            metadata=metadata,
            tags=tags,
        )
        records.append(record)

    return records


class _BaseMem0Adapter:
    """Base class for Mem0 adapters (Platform and Self-hosted).

    Subclasses must override ``_get_client()`` and ``system_name``.
    """

    def __init__(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        self._user_id = user_id
        self._agent_id = agent_id
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize the Mem0 client. Must be overridden."""
        raise NotImplementedError

    @property
    def system_type(self) -> SourceSystemType:
        return SourceSystemType.MEMORY_LAYER

    @property
    def system_name(self) -> str:
        raise NotImplementedError

    @property
    def capabilities(self) -> frozenset[SourceCapability]:
        return frozenset(
            {
                SourceCapability.FETCH_ALL,
                SourceCapability.FETCH_SINCE,
                SourceCapability.FETCH_METADATA,
                SourceCapability.HEALTH_CHECK,
                SourceCapability.CREATE_RECORD,
                SourceCapability.UPDATE_RECORD,
                SourceCapability.DELETE_RECORD,
            }
        )

    async def fetch_all(
        self,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """Fetch all memories from Mem0."""
        client = self._get_client()

        kwargs: dict[str, Any] = {}
        if self._user_id:
            kwargs["user_id"] = self._user_id
        if self._agent_id:
            kwargs["agent_id"] = self._agent_id

        memories = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: client.get_all(**kwargs),
        )

        return _parse_mem0_records(
            memories,
            source_system=self.system_name,
            user_id=self._user_id,
            agent_id=self._agent_id,
            limit=limit,
        )

    async def fetch_since(
        self,
        since: datetime,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """Fetch memories modified since a given timestamp.

        Uses Mem0's v2 filtering API to filter by updated_at or created_at.
        Falls back to created_at if updated_at is not available.

        Args:
            since: Only fetch records modified after this timestamp
            collection: Optional collection/namespace filter (not used by Mem0)
            limit: Optional maximum number of records to fetch

        Returns:
            List of normalized ExternalRecord instances
        """
        client = self._get_client()

        # Format timestamp for Mem0 API (ISO 8601)
        since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Build filter for updated_at >= since (or created_at as fallback)
        # Mem0 v2 API uses {"AND": [{"updated_at": {"gte": "..."}}]} format
        filter_obj: dict[str, Any] = {
            "AND": [
                {"updated_at": {"gte": since_str}},
            ]
        }

        kwargs: dict[str, Any] = {"filters": filter_obj}
        if self._user_id:
            kwargs["user_id"] = self._user_id
        if self._agent_id:
            kwargs["agent_id"] = self._agent_id

        try:
            memories = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.get_all(**kwargs),
            )
        except Exception as e:
            # Some Mem0 versions may not support filters parameter
            # Fall back to fetch_all and filter in Python
            logger.debug(
                "%s filter query failed, falling back to client-side filtering: %s",
                self.system_name,
                e,
            )
            all_records = await self.fetch_all(collection=collection, limit=None)
            filtered = [r for r in all_records if r.updated_at and r.updated_at >= since]
            if limit is not None:
                return filtered[:limit]
            return filtered

        return _parse_mem0_records(
            memories,
            source_system=self.system_name,
            user_id=self._user_id,
            agent_id=self._agent_id,
            limit=limit,
        )

    async def health_check(self) -> dict[str, Any]:
        """Check Mem0 connectivity."""
        try:
            client = self._get_client()
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.get_all(user_id=self._user_id or "healthcheck", limit=1),
            )
            return {
                "healthy": True,
                "message": f"{self.system_name} connected successfully",
                "system": self.system_name,
            }
        except Exception:
            logger.debug("%s health check failed", self.system_name, exc_info=True)
            return {
                "healthy": False,
                "message": f"{self.system_name} connection failed",
                "system": self.system_name,
            }

    # Write operations for bidirectional sync

    async def create_record(
        self,
        content: str,
        *,
        collection: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: set[str] | None = None,
    ) -> str | None:
        """Create a new memory in Mem0.

        Args:
            content: The text content to store
            collection: Optional collection/namespace (maps to user_id or agent_id)
            metadata: Optional metadata dict
            tags: Optional set of tags (mapped to Mem0 categories)

        Returns:
            The ID of the created memory, or None if creation failed
        """
        client = self._get_client()

        # Build kwargs for Mem0 add() method
        kwargs: dict[str, Any] = {"messages": [{"role": "user", "content": content}]}
        if self._user_id:
            kwargs["user_id"] = self._user_id
        elif collection:
            kwargs["user_id"] = collection
        if self._agent_id:
            kwargs["agent_id"] = self._agent_id
        if metadata:
            kwargs["metadata"] = metadata
        if tags:
            # Mem0 uses "categories" for tags
            kwargs["categories"] = list(tags)

        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.add(**kwargs),
            )
            # Mem0 returns {"results": [{"id": "..."}]}
            if isinstance(result, dict) and "results" in result:
                results = result["results"]
                if results and isinstance(results, list):
                    return str(results[0].get("id", ""))
            return None
        except Exception:
            logger.error("%s create_record failed", self.system_name, exc_info=True)
            return None

    async def update_record(
        self,
        record_id: str,
        content: str | None = None,
        *,
        collection: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: set[str] | None = None,
    ) -> bool:
        """Update an existing memory in Mem0.

        Note: Mem0's update API requires the memory_id and new content.
        Tags/categories and metadata updates may not be supported directly.

        Args:
            record_id: The ID of the memory to update
            content: Optional new content
            collection: Optional collection/namespace (not used by Mem0 update)
            metadata: Optional updated metadata
            tags: Optional updated tags

        Returns:
            True if update succeeded, False otherwise
        """
        if not content:
            # Mem0 update requires content
            logger.warning("%s update_record requires content", self.system_name)
            return False

        client = self._get_client()

        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.update(memory_id=record_id, data=content),
            )
            return True
        except Exception:
            logger.error("%s update_record failed for %s", self.system_name, record_id, exc_info=True)
            return False

    async def delete_record(
        self,
        record_id: str,
        *,
        collection: str | None = None,
    ) -> bool:
        """Delete a memory from Mem0.

        Args:
            record_id: The ID of the memory to delete
            collection: Optional collection/namespace (not used by Mem0 delete)

        Returns:
            True if deletion succeeded, False otherwise
        """
        client = self._get_client()

        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.delete(memory_id=record_id),
            )
            return True
        except Exception:
            logger.error("%s delete_record failed for %s", self.system_name, record_id, exc_info=True)
            return False


class Mem0Adapter(_BaseMem0Adapter):
    """Adapter for Mem0 Platform (cloud API, requires API key).

    Usage:
        adapter = Mem0Adapter(api_key="...", user_id="alice")
        records = await adapter.fetch_all()
    """

    def __init__(
        self,
        api_key: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(user_id=user_id, agent_id=agent_id)
        self._api_key = api_key

    def _get_client(self) -> Any:
        """Lazy-initialize Mem0 Platform client."""
        if self._client is None:
            from mem0 import MemoryClient

            api_key = self._api_key or os.environ.get("MEM0_API_KEY")
            if not api_key:
                msg = (
                    "Mem0 API key required. Provide via api_key parameter "
                    "or MEM0_API_KEY environment variable."
                )
                raise ValueError(msg)

            self._client = MemoryClient(api_key=api_key)

        return self._client

    @property
    def system_name(self) -> str:
        return "mem0"


class Mem0SelfHostedAdapter(_BaseMem0Adapter):
    """Adapter for self-hosted Mem0 using ``from mem0 import Memory`` (no API key).

    Usage:
        adapter = Mem0SelfHostedAdapter(user_id="alice")
        records = await adapter.fetch_all()
    """

    def __init__(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(user_id=user_id, agent_id=agent_id)
        self._mem0_config = config

    def _get_client(self) -> Any:
        """Lazy-initialize self-hosted Mem0 Memory instance."""
        if self._client is None:
            from mem0 import Memory

            if self._mem0_config:
                self._client = Memory.from_config(self._mem0_config)
            else:
                self._client = Memory()

        return self._client

    @property
    def system_name(self) -> str:
        return "mem0_self_hosted"
