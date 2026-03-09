"""SyncEngine — orchestrates the full import pipeline."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Coroutine
from typing import TYPE_CHECKING, Any

from neural_memory.integration.mapper import RecordMapper
from neural_memory.integration.models import (
    ExportResult,
    ExternalRecord,
    ExternalRelationship,
    ImportResult,
    SourceCapability,
    SyncState,
)
from neural_memory.utils.timeutils import utcnow

_MAX_SYNC_LIMIT = 10000

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.integration.adapter import SourceAdapter
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Callback type: (records_processed, total_records, current_record_id)
# Can be sync or async
ProgressCallback = Callable[[int, int, str], None] | Callable[[int, int, str], Coroutine[Any, Any, None]]


class SyncEngine:
    """Orchestrates importing records from external sources into NeuralMemory.

    The engine:
    1. Connects to source via SourceAdapter
    2. Fetches records (full or incremental based on SyncState)
    3. Maps each record through RecordMapper
    4. Persists to NeuralStorage with batch commits
    5. Creates cross-record relationship synapses
    6. Tracks sync state for incremental updates
    """

    def __init__(
        self,
        storage: NeuralStorage,
        config: BrainConfig,
        batch_size: int = 50,
    ) -> None:
        self._storage = storage
        self._config = config
        self._batch_size = batch_size
        self._mapper = RecordMapper(storage, config)

    async def sync(
        self,
        adapter: SourceAdapter,
        collection: str | None = None,
        sync_state: SyncState | None = None,
        progress_callback: ProgressCallback | None = None,
        limit: int | None = None,
    ) -> tuple[ImportResult, SyncState]:
        """Execute a sync from an external source.

        Args:
            adapter: The source adapter to fetch from
            collection: Optional collection filter
            sync_state: Previous sync state for incremental sync
            progress_callback: Optional progress callback
            limit: Optional maximum records to import

        Returns:
            Tuple of (ImportResult, updated SyncState)
        """
        start_time = time.monotonic()
        source_name = adapter.system_name
        collection_name = collection or "default"
        limit = min(limit, _MAX_SYNC_LIMIT) if limit else _MAX_SYNC_LIMIT

        if sync_state is None:
            sync_state = SyncState(
                source_system=source_name,
                source_collection=collection_name,
            )

        # Fetch records (incremental or full)
        records = await self._fetch_records(
            adapter=adapter,
            collection=collection,
            sync_state=sync_state,
            limit=limit,
        )

        total_records = len(records)
        logger.info(
            "Fetched %d records from %s/%s",
            total_records,
            source_name,
            collection_name,
        )

        # Enable batch mode
        self._storage.disable_auto_save()

        imported_count = 0
        skipped_count = 0
        failed_count = 0
        errors: list[str] = []
        fibers_created: list[str] = []
        record_to_fiber: dict[str, str] = {}
        all_relationships: list[ExternalRelationship] = []

        try:
            for i, record in enumerate(records):
                try:
                    if not record.content or not record.content.strip():
                        skipped_count += 1
                        continue

                    if await self._is_already_imported(record):
                        skipped_count += 1
                        continue

                    result = await self._mapper.map_record(record)

                    fibers_created.append(result.encoding_result.fiber.id)
                    record_to_fiber[record.id] = result.encoding_result.fiber.id
                    imported_count += 1

                    if record.relationships:
                        all_relationships.extend(record.relationships)

                    if progress_callback is not None:
                        progress_callback(i + 1, total_records, record.id)

                    # Batch commit
                    if (i + 1) % self._batch_size == 0:
                        await self._storage.batch_save()

                except Exception:
                    failed_count += 1
                    logger.warning("Failed to import record %s", record.id, exc_info=True)
                    errors.append(f"Failed to import record {record.id}")

            # Second pass: create relationship synapses
            if all_relationships:
                try:
                    await self._mapper.create_relationship_synapses(
                        record_to_fiber=record_to_fiber,
                        relationships=all_relationships,
                    )
                except Exception:
                    logger.warning("Relationship synapse creation failed", exc_info=True)
                    errors.append("Failed to create relationship synapses")

            # Final commit
            await self._storage.batch_save()
        finally:
            self._storage.enable_auto_save()

        duration = time.monotonic() - start_time

        updated_state = sync_state.with_update(
            last_sync_at=utcnow(),
            records_imported=sync_state.records_imported + imported_count,
            last_record_id=records[-1].id if records else sync_state.last_record_id,
        )

        import_result = ImportResult(
            source_system=source_name,
            source_collection=collection_name,
            records_fetched=total_records,
            records_imported=imported_count,
            records_skipped=skipped_count,
            records_failed=failed_count,
            errors=tuple(errors),
            duration_seconds=round(duration, 2),
            fibers_created=tuple(fibers_created),
        )

        logger.info(
            "Import complete: %d imported, %d skipped, %d failed in %.1fs",
            imported_count,
            skipped_count,
            failed_count,
            duration,
        )

        return import_result, updated_state

    async def _fetch_records(
        self,
        adapter: SourceAdapter,
        collection: str | None,
        sync_state: SyncState,
        limit: int | None,
    ) -> list[ExternalRecord]:
        """Fetch records, choosing incremental or full based on capabilities."""
        if (
            sync_state.last_sync_at is not None
            and SourceCapability.FETCH_SINCE in adapter.capabilities
        ):
            return await adapter.fetch_since(
                since=sync_state.last_sync_at,
                collection=collection,
                limit=limit,
            )

        return await adapter.fetch_all(
            collection=collection,
            limit=limit,
        )

    async def _is_already_imported(self, record: ExternalRecord) -> bool:
        """Check if a record has already been imported."""
        existing = await self._storage.find_neurons(
            content_exact=record.content,
            limit=1,
        )
        if not existing:
            return False

        for neuron in existing:
            if (
                neuron.metadata.get("import_source") == record.source_system
                and neuron.metadata.get("import_record_id") == record.id
            ):
                return True

        return False

    async def health_check(self, adapter: SourceAdapter) -> dict[str, Any]:
        """Run health check on an adapter."""
        if SourceCapability.HEALTH_CHECK not in adapter.capabilities:
            return {
                "healthy": False,
                "message": f"Adapter {adapter.system_name} does not support health checks",
            }
        return await adapter.health_check()

    async def export(
        self,
        adapter: SourceAdapter,
        collection: str | None = None,
        progress_callback: ProgressCallback | None = None,
        limit: int | None = None,
        filter_fn: Callable[[Any], bool] | None = None,
    ) -> ExportResult:
        """Export PugBrain memories to an external source.

        This enables bidirectional sync by pushing local memories to
        external systems like Mem0.

        Args:
            adapter: The target adapter to export to
            collection: Optional collection/namespace in target
            progress_callback: Optional progress callback
            limit: Optional maximum records to export
            filter_fn: Optional filter function to select which memories to export

        Returns:
            ExportResult with export statistics
        """
        from neural_memory.integration.models import ExportResult

        start_time = time.monotonic()
        target_name = adapter.system_name
        collection_name = collection or "default"

        # Check if adapter supports create operation
        if SourceCapability.CREATE_RECORD not in adapter.capabilities:
            return ExportResult(
                target_system=target_name,
                target_collection=collection_name,
                records_exported=0,
                records_skipped=0,
                records_failed=0,
                errors=(f"Adapter {target_name} does not support create_record",),
                duration_seconds=0.0,
            )

        # Fetch local fibers to export
        fibers = await self._storage.get_fibers(limit=limit or _MAX_SYNC_LIMIT)
        total_fibers = len(fibers)

        logger.info(
            "Exporting %d fibers to %s/%s",
            total_fibers,
            target_name,
            collection_name,
        )

        exported_count = 0
        skipped_count = 0
        failed_count = 0
        errors: list[str] = []
        exported_ids: list[tuple[str, str]] = []

        for i, fiber in enumerate(fibers):
            try:
                # Apply filter if provided
                if filter_fn is not None and not filter_fn(fiber):
                    skipped_count += 1
                    continue

                # Skip already exported records (check metadata)
                if fiber.metadata.get("exported_to") == target_name:
                    skipped_count += 1
                    continue

                # Get fiber content from primary neuron
                neurons_dict = await self._storage.get_neurons_batch(list(fiber.neuron_ids))
                if not neurons_dict:
                    skipped_count += 1
                    continue

                # Combine content from neurons
                neurons = list(neurons_dict.values())
                content_parts = [n.content for n in neurons if n.content]
                if not content_parts:
                    skipped_count += 1
                    continue

                content = "\n".join(content_parts)

                # Build metadata from fiber
                export_metadata = {
                    "fiber_id": fiber.id,
                    "fiber_tags": list(fiber.tags) if fiber.tags else [],
                    "source": "pugbrain",
                }

                # Create record in external system
                external_id = await adapter.create_record(
                    content=content,
                    collection=collection,
                    metadata=export_metadata,
                    tags=set(fiber.tags) if fiber.tags else None,
                )

                if external_id:
                    exported_count += 1
                    exported_ids.append((fiber.id, external_id))
                    logger.debug(
                        "Exported fiber %s to %s as %s", fiber.id, target_name, external_id
                    )
                else:
                    failed_count += 1
                    errors.append(f"Failed to export fiber {fiber.id}: no ID returned")

                if progress_callback is not None:
                    progress_callback(i + 1, total_fibers, fiber.id)

            except Exception:
                failed_count += 1
                logger.warning("Failed to export fiber %s", fiber.id, exc_info=True)
                errors.append(f"Failed to export fiber {fiber.id}")

        duration = time.monotonic() - start_time

        export_result = ExportResult(
            target_system=target_name,
            target_collection=collection_name,
            records_exported=exported_count,
            records_skipped=skipped_count,
            records_failed=failed_count,
            errors=tuple(errors),
            duration_seconds=round(duration, 2),
            exported_ids=tuple(exported_ids),
        )

        logger.info(
            "Export complete: %d exported, %d skipped, %d failed in %.1fs",
            exported_count,
            skipped_count,
            failed_count,
            duration,
        )

        return export_result
