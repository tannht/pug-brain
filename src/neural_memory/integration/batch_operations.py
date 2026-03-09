"""Batch operations for import/export with progress tracking and error recovery.

Provides utilities for large-scale synchronization between PugBrain and
external memory systems with features like:
- Progress tracking and callbacks
- Error recovery (resume from checkpoint)
- Rate limiting
- Cancellation support
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from neural_memory.integration.models import ExportResult, ImportResult
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.integration.adapter import SourceAdapter
    from neural_memory.integration.sync_engine import SyncEngine

logger = logging.getLogger(__name__)

# Default rate limits
_DEFAULT_REQUESTS_PER_SECOND = 10
_DEFAULT_BATCH_SIZE = 50

# Callback types
ProgressCallback = Callable[[int, int, str], None]  # (current, total, current_id)
StatusCallback = Callable[[str, dict[str, Any]], None]  # (status, metadata)


class BatchOperationStatus(StrEnum):
    """Status of a batch operation."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchCheckpoint:
    """Checkpoint for resumable batch operations.

    Attributes:
        operation_id: Unique identifier for this operation
        operation_type: "import" or "export"
        source_system: Source/target system name
        collection: Collection name
        started_at: When the operation started
        last_record_id: ID of the last processed record
        processed_count: Number of records processed so far
        failed_count: Number of failed records
        status: Current operation status
        metadata: Additional checkpoint data
    """

    operation_id: str
    operation_type: str
    source_system: str
    collection: str
    started_at: datetime
    last_record_id: str | None = None
    processed_count: int = 0
    failed_count: int = 0
    status: BatchOperationStatus = BatchOperationStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint to dictionary.

        Converts Path objects in metadata to strings for JSON compatibility.
        """
        # Convert Path objects to strings for JSON serialization
        serializable_metadata = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.metadata.items()
        }

        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "source_system": self.source_system,
            "collection": self.collection,
            "started_at": self.started_at.isoformat(),
            "last_record_id": self.last_record_id,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "status": self.status.value,
            "metadata": serializable_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchCheckpoint:
        """Deserialize checkpoint from dictionary.

        Handles invalid status values gracefully by defaulting to PENDING.
        """
        # Parse status with error handling
        status_str = data.get("status", "pending")
        try:
            status = BatchOperationStatus(status_str)
        except (ValueError, KeyError):
            # Invalid status - default to PENDING
            logger.warning("Invalid status '%s' in checkpoint, defaulting to PENDING", status_str)
            status = BatchOperationStatus.PENDING

        # Convert Path objects in metadata to strings for JSON compatibility
        metadata = data.get("metadata", {})
        if metadata:
            metadata = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in metadata.items()
            }

        return cls(
            operation_id=data["operation_id"],
            operation_type=data["operation_type"],
            source_system=data["source_system"],
            collection=data["collection"],
            started_at=datetime.fromisoformat(data["started_at"]),
            last_record_id=data.get("last_record_id"),
            processed_count=data.get("processed_count", 0),
            failed_count=data.get("failed_count", 0),
            status=status,
            metadata=metadata,
        )


@dataclass
class BatchConfig:
    """Configuration for batch operations.

    Attributes:
        batch_size: Number of records per batch
        requests_per_second: Rate limit for API calls
        max_retries: Maximum retry attempts per record
        retry_delay_seconds: Delay between retries
        checkpoint_interval: Save checkpoint every N records
        checkpoint_path: Path to save checkpoint files
    """

    batch_size: int = _DEFAULT_BATCH_SIZE
    requests_per_second: float = _DEFAULT_REQUESTS_PER_SECOND
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    checkpoint_interval: int = 100
    checkpoint_path: Path | None = None


class BatchOperationManager:
    """Manages batch import/export operations with progress tracking and recovery.

    Usage:
        manager = BatchOperationManager(sync_engine, config)

        # Import with progress tracking
        result = await manager.import_with_progress(
            adapter=mem0_adapter,
            collection="user123",
            on_progress=lambda c, t, id: print(f"{c}/{t}: {id}"),
        )

        # Export with checkpointing
        checkpoint = await manager.export_with_checkpoint(
            adapter=mem0_adapter,
            collection="user123",
            checkpoint_path=Path("export_checkpoint.json"),
        )
    """

    def __init__(
        self,
        sync_engine: SyncEngine,
        config: BatchConfig | None = None,
    ) -> None:
        self._engine = sync_engine
        self._config = config or BatchConfig()
        self._cancelled = False
        self._paused = False

    def cancel(self) -> None:
        """Cancel the current operation."""
        self._cancelled = True

    def pause(self) -> None:
        """Pause the current operation."""
        self._paused = True

    def resume(self) -> None:
        """Resume a paused operation."""
        self._paused = False

    async def import_with_progress(
        self,
        adapter: SourceAdapter,
        collection: str | None = None,
        limit: int | None = None,
        on_progress: ProgressCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> ImportResult:
        """Run import with progress tracking and rate limiting.

        Args:
            adapter: Source adapter to import from
            collection: Optional collection filter
            limit: Optional maximum records to import
            on_progress: Optional progress callback
            on_status: Optional status callback

        Returns:
            ImportResult with import statistics

        Raises:
            asyncio.CancelledError: If operation is cancelled
        """
        self._cancelled = False
        self._paused = False

        operation_id = f"import_{adapter.system_name}_{utcnow().strftime('%Y%m%d_%H%M%S')}"

        if on_status:
            on_status("started", {"operation_id": operation_id})

        # Rate limiter
        rate_limiter = _RateLimiter(self._config.requests_per_second)

        # Wrap progress callback with rate limiting
        tracked_progress: list[int] = [0]

        async def tracked_callback(current: int, total: int, record_id: str) -> None:
            tracked_progress[0] = current

            # Check for cancellation
            if self._cancelled:
                raise asyncio.CancelledError("Operation cancelled by user")

            # Check for pause
            while self._paused:
                await asyncio.sleep(0.5)
                if self._cancelled:
                    raise asyncio.CancelledError("Operation cancelled by user")

            # Apply rate limiting
            await rate_limiter.acquire()

            if on_progress:
                on_progress(current, total, record_id)

        async def sync_with_cancel_check() -> ImportResult:
            """Wrapper that periodically checks cancellation during sync."""
            # Create a task for the sync operation
            sync_task = asyncio.create_task(
                self._engine.sync(
                    adapter=adapter,
                    collection=collection,
                    limit=limit,
                    progress_callback=tracked_callback if on_progress or self._config.requests_per_second > 0 else None,
                )
            )

            # Wait for sync to complete or be cancelled
            while not sync_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(sync_task), timeout=0.1)
                except TimeoutError:
                    # Check cancellation flag periodically
                    if self._cancelled:
                        sync_task.cancel()
                        try:
                            await sync_task
                        except asyncio.CancelledError:
                            # Re-raise with our custom message
                            raise asyncio.CancelledError("Operation cancelled")
                    # Continue waiting
                    pass

            result = await sync_task
            return result[0]  # sync returns (ImportResult, SyncState)

        try:
            result = await sync_with_cancel_check()

            if on_status:
                on_status("completed", {
                    "operation_id": operation_id,
                    "records_imported": result.records_imported,
                    "duration": result.duration_seconds,
                })

            return result

        except asyncio.CancelledError:
            if on_status:
                on_status("cancelled", {
                    "operation_id": operation_id,
                    "processed": tracked_progress[0],
                })
            raise
        except Exception as e:
            if on_status:
                on_status("failed", {
                    "operation_id": operation_id,
                    "error": str(e),
                    "processed": tracked_progress[0],
                })
            raise

    async def export_with_checkpoint(
        self,
        adapter: SourceAdapter,
        collection: str | None = None,
        limit: int | None = None,
        checkpoint_path: Path | None = None,
        resume_from: Path | None = None,
        on_progress: ProgressCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> tuple[ExportResult, BatchCheckpoint | None]:
        """Run export with checkpointing for resumable operations.

        Args:
            adapter: Target adapter to export to
            collection: Optional collection in target
            limit: Optional maximum records to export
            checkpoint_path: Path to save checkpoint (for resume)
            resume_from: Path to existing checkpoint to resume from
            on_progress: Optional progress callback
            on_status: Optional status callback

        Returns:
            Tuple of (ExportResult, final checkpoint or None)
        """
        self._cancelled = False
        self._paused = False

        # Load existing checkpoint if resuming
        checkpoint: BatchCheckpoint | None = None
        if resume_from and resume_from.exists():
            try:
                data = json.loads(resume_from.read_text())
                checkpoint = BatchCheckpoint.from_dict(data)
                logger.info(
                    "Resuming export from checkpoint: %s records already processed",
                    checkpoint.processed_count,
                )
            except Exception:
                logger.warning("Failed to load checkpoint, starting fresh", exc_info=True)

        if checkpoint is None:
            checkpoint = BatchCheckpoint(
                operation_id=f"export_{adapter.system_name}_{utcnow().strftime('%Y%m%d_%H%M%S')}",
                operation_type="export",
                source_system=adapter.system_name,
                collection=collection or "default",
                started_at=utcnow(),
                status=BatchOperationStatus.RUNNING,
            )

        if on_status:
            on_status("started", {"operation_id": checkpoint.operation_id})

        rate_limiter = _RateLimiter(self._config.requests_per_second)
        checkpoint_counter = 0

        def save_checkpoint() -> None:
            """Save checkpoint to file.

            Handles I/O errors gracefully by logging and continuing.
            """
            if checkpoint_path and checkpoint:
                try:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint_path.write_text(json.dumps(checkpoint.to_dict(), indent=2))
                except (OSError, IOError) as e:
                    # Log error but don't fail the operation
                    logger.warning("Failed to save checkpoint to %s: %s", checkpoint_path, e)

        tracked_progress: list[int] = [checkpoint.processed_count]

        def tracked_callback(current: int, total: int, record_id: str) -> None:
            nonlocal checkpoint_counter

            tracked_progress[0] = current

            if checkpoint:
                checkpoint.last_record_id = record_id
                checkpoint.processed_count = current

                # Save checkpoint periodically
                checkpoint_counter += 1
                if checkpoint_counter >= self._config.checkpoint_interval:
                    save_checkpoint()
                    checkpoint_counter = 0

            if on_progress:
                on_progress(current, total, record_id)

        try:
            # Run export
            result = await self._engine.export(
                adapter=adapter,
                collection=collection,
                limit=limit,
                progress_callback=tracked_callback,
            )

            if checkpoint:
                checkpoint.status = BatchOperationStatus.COMPLETED
                save_checkpoint()

            if on_status:
                on_status("completed", {
                    "operation_id": checkpoint.operation_id if checkpoint else None,
                    "records_exported": result.records_exported,
                    "duration": result.duration_seconds,
                })

            return result, checkpoint

        except asyncio.CancelledError:
            if checkpoint:
                checkpoint.status = BatchOperationStatus.CANCELLED
                save_checkpoint()
            if on_status:
                on_status("cancelled", {
                    "operation_id": checkpoint.operation_id if checkpoint else None,
                    "processed": tracked_progress[0],
                })
            raise
        except Exception as e:
            if checkpoint:
                checkpoint.status = BatchOperationStatus.FAILED
                checkpoint.metadata["error"] = str(e)
                save_checkpoint()
            if on_status:
                on_status("failed", {
                    "operation_id": checkpoint.operation_id if checkpoint else None,
                    "error": str(e),
                    "processed": tracked_progress[0],
                })
            raise

    @staticmethod
    def load_checkpoint(path: Path) -> BatchCheckpoint | None:
        """Load a checkpoint from file.

        Args:
            path: Path to checkpoint file

        Returns:
            BatchCheckpoint or None if file doesn't exist or is invalid
        """
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return BatchCheckpoint.from_dict(data)
        except Exception:
            logger.warning("Failed to load checkpoint from %s", path, exc_info=True)
            return None


class _RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, requests_per_second: float) -> None:
        self._min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0
        self._last_time: float = 0

    async def acquire(self) -> None:
        """Wait until a request can be made within rate limits."""
        if self._min_interval == 0:
            return

        now = asyncio.get_running_loop().time()
        elapsed = now - self._last_time
        wait_time = self._min_interval - elapsed

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        self._last_time = asyncio.get_running_loop().time()
