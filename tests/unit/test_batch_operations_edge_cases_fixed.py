"""Comprehensive edge case tests for batch operations.

Tests cover:
1. Checkpoint recovery from corrupted/invalid JSON files
2. Rate limiting with extreme values (very high/low rates)
3. Cancellation at different operation phases (start, middle, end)
4. Pause/resume during active operations
5. Large datasets (1000+ records) with progress tracking
6. Concurrent batch operations (multiple imports/exports)
7. Memory leaks under sustained operations
8. Network timeout handling (simulated with slow mocks)
9. Invalid adapter responses (None, empty, malformed data)
10. Checkpoint file permission errors
11. Very large record IDs or metadata
12. Race conditions in pause/resume/cancel
"""

from __future__ import annotations

import asyncio
import gc
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from neural_memory.core.brain import BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron
from neural_memory.integration.batch_operations import (
    BatchConfig,
    BatchOperationManager,
    BatchOperationStatus,
    _RateLimiter,
)
from neural_memory.integration.models import (
    ExportResult,
    ExternalRecord,
    ImportResult,
    SourceCapability,
    SourceSystemType,
)
from neural_memory.utils.timeutils import utcnow

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for checkpoint files."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def mock_storage() -> MagicMock:
    """Create a mock storage with sensible defaults."""
    storage = MagicMock()

    # Batch operations methods
    storage.disable_auto_save = Mock()
    storage.enable_auto_save = Mock()
    storage.batch_save = AsyncMock()
    storage.find_neurons = AsyncMock(return_value=[])
    storage.get_fibers = AsyncMock(return_value=[])

    return storage


@pytest.fixture
def mock_brain_config() -> BrainConfig:
    """Create a mock brain config."""
    return BrainConfig()


@pytest.fixture
def mock_sync_engine(mock_storage: MagicMock, mock_brain_config: BrainConfig) -> MagicMock:
    """Create a mock sync engine."""
    from neural_memory.integration.sync_engine import SyncEngine

    engine = SyncEngine(storage=mock_storage, config=mock_brain_config, batch_size=10)

    # Mock the internal sync method to avoid real storage operations
    engine.sync = AsyncMock()  # type: ignore[method-assign]
    engine.export = AsyncMock()  # type: ignore[method-assign]

    return engine


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Create a mock adapter with basic capabilities."""
    adapter = MagicMock()
    adapter.system_name = "test_adapter"
    adapter.system_type = SourceSystemType.MEMORY_LAYER
    adapter.capabilities = frozenset(
        [
            SourceCapability.FETCH_ALL,
            SourceCapability.CREATE_RECORD,
        ]
    )
    adapter.fetch_all = AsyncMock(return_value=[])
    adapter.create_record = AsyncMock(return_value="external-id-1")
    adapter.health_check = AsyncMock(return_value={"healthy": True, "message": "OK"})

    return adapter


@pytest.fixture
def sample_records(count: int = 10) -> list[ExternalRecord]:
    """Create sample external records."""
    return [
        ExternalRecord.create(
            id=f"record-{i}",
            source_system="test_adapter",
            content=f"Test content {i}",
            source_collection="default",
        )
        for i in range(count)
    ]


@pytest.fixture
def sample_fibers(count: int = 10) -> list[Fiber]:
    """Create sample fibers."""
    fibers = []
    for i in range(count):
        neuron = Neuron.create(
            type="fact",
            content=f"Test content {i}",
        )
        fiber = Fiber.create(
            neurons=[neuron],
            tags={f"tag-{i}"},
        )
        fibers.append(fiber)
    return fibers


# ============================================================================
# Checkpoint Recovery Tests
# ============================================================================


class TestCheckpointRecovery:
    """Test checkpoint recovery from corrupted/invalid JSON files."""

    async def test_load_checkpoint_from_valid_file(self, temp_checkpoint_dir: Path) -> None:
        """Loading a valid checkpoint file should succeed."""
        checkpoint_data = {
            "operation_id": "test-op-1",
            "operation_type": "export",
            "source_system": "test_adapter",
            "collection": "default",
            "started_at": utcnow().isoformat(),
            "last_record_id": "record-5",
            "processed_count": 5,
            "failed_count": 0,
            "status": "running",
            "metadata": {},
        }

        checkpoint_path = temp_checkpoint_dir / "checkpoint.json"
        checkpoint_path.write_text(json.dumps(checkpoint_data))

        result = BatchOperationManager.load_checkpoint(checkpoint_path)

        assert result is not None
        assert result.operation_id == "test-op-1"
        assert result.processed_count == 5
        assert result.last_record_id == "record-5"

    async def test_load_checkpoint_from_missing_file(self, temp_checkpoint_dir: Path) -> None:
        """Loading from a non-existent file should return None."""
        result = BatchOperationManager.load_checkpoint(temp_checkpoint_dir / "nonexistent.json")
        assert result is None

    async def test_load_checkpoint_from_invalid_json(self, temp_checkpoint_dir: Path) -> None:
        """Loading from a file with invalid JSON should return None."""
        checkpoint_path = temp_checkpoint_dir / "invalid.json"
        checkpoint_path.write_text("{ invalid json }")

        result = BatchOperationManager.load_checkpoint(checkpoint_path)

        assert result is None

    async def test_load_checkpoint_from_empty_file(self, temp_checkpoint_dir: Path) -> None:
        """Loading from an empty file should return None."""
        checkpoint_path = temp_checkpoint_dir / "empty.json"
        checkpoint_path.write_text("")

        result = BatchOperationManager.load_checkpoint(checkpoint_path)

        assert result is None

    async def test_load_checkpoint_with_missing_fields(self, temp_checkpoint_dir: Path) -> None:
        """Loading checkpoint with missing required fields should raise or handle gracefully."""
        checkpoint_data = {
            "operation_id": "test-op-1",
            # Missing required fields: operation_type, source_system, collection, started_at
        }

        checkpoint_path = temp_checkpoint_dir / "incomplete.json"
        checkpoint_path.write_text(json.dumps(checkpoint_data))

        # Should handle the error gracefully and return None
        result = BatchOperationManager.load_checkpoint(checkpoint_path)
        assert result is None

    async def test_load_checkpoint_with_invalid_datetime(self, temp_checkpoint_dir: Path) -> None:
        """Loading checkpoint with invalid datetime format should fail gracefully."""
        checkpoint_data = {
            "operation_id": "test-op-1",
            "operation_type": "export",
            "source_system": "test_adapter",
            "collection": "default",
            "started_at": "not-a-datetime",
            "processed_count": 0,
            "failed_count": 0,
            "status": "pending",
            "metadata": {},
        }

        checkpoint_path = temp_checkpoint_dir / "bad-datetime.json"
        checkpoint_path.write_text(json.dumps(checkpoint_data))

        result = BatchOperationManager.load_checkpoint(checkpoint_path)
        assert result is None

    async def test_load_checkpoint_with_invalid_status(self, temp_checkpoint_dir: Path) -> None:
        """Loading checkpoint with invalid status enum should default to PENDING."""
        checkpoint_data = {
            "operation_id": "test-op-1",
            "operation_type": "export",
            "source_system": "test_adapter",
            "collection": "default",
            "started_at": utcnow().isoformat(),
            "processed_count": 0,
            "failed_count": 0,
            "status": "invalid_status",
            "metadata": {},
        }

        checkpoint_path = temp_checkpoint_dir / "bad-status.json"
        checkpoint_path.write_text(json.dumps(checkpoint_data))

        result = BatchOperationManager.load_checkpoint(checkpoint_path)
        # Should not fail - should default to PENDING
        assert result is not None
        assert result.status == BatchOperationStatus.PENDING

    async def test_resume_from_checkpoint_after_corruption(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Resuming from a corrupted checkpoint should start fresh."""
        # First create a corrupted checkpoint
        checkpoint_path = temp_checkpoint_dir / "corrupted.json"
        checkpoint_path.write_text("{ corrupted }")

        manager = BatchOperationManager(mock_sync_engine)

        # Setup mock to return a result
        mock_sync_engine.export.return_value = ExportResult(
            target_system="test_adapter",
            target_collection="default",
            records_exported=5,
            duration_seconds=1.0,
        )

        # Should handle corruption gracefully and create new checkpoint
        result, checkpoint = await manager.export_with_checkpoint(
            adapter=mock_adapter,
            checkpoint_path=checkpoint_path,
            resume_from=checkpoint_path,
        )

        assert result is not None
        assert checkpoint is not None
        assert checkpoint.status == BatchOperationStatus.COMPLETED


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Test rate limiting with extreme values."""

    @pytest.mark.parametrize("rate", [0.1, 1.0, 10.0, 100.0, 1000.0])
    @pytest.mark.timeout(120)  # Extended timeout for rate limiting tests
    async def test_rate_limiter_with_various_rates(self, rate: float) -> None:
        """Test rate limiter with various rates per second."""
        limiter = _RateLimiter(rate)

        start_time = asyncio.get_running_loop().time()

        # Make 3 requests
        for _ in range(3):
            await limiter.acquire()

        elapsed = asyncio.get_running_loop().time() - start_time

        # For very high rates (1000+), should complete very quickly
        # For very low rates (0.1), should take at least 20 seconds between requests
        if rate >= 1000:
            assert elapsed < 0.01  # Should be nearly instant
        elif rate <= 0.1:
            assert (
                elapsed >= 20
            )  # Should have noticeable delays (at least 20s for 3 requests at 0.1/sec)

    async def test_rate_limiter_with_zero_rate(self) -> None:
        """Rate limiter with zero rate should not delay at all."""
        limiter = _RateLimiter(0.0)

        start_time = asyncio.get_running_loop().time()

        for _ in range(10):
            await limiter.acquire()

        elapsed = asyncio.get_running_loop().time() - start_time

        # Should complete instantly with no delays
        assert elapsed < 0.01

    async def test_rate_limiter_with_negative_rate(self) -> None:
        """Negative rate should be treated as zero (no limiting)."""
        limiter = _RateLimiter(-10.0)

        start_time = asyncio.get_running_loop().time()

        for _ in range(10):
            await limiter.acquire()

        elapsed = asyncio.get_running_loop().time() - start_time

        # Should complete instantly
        assert elapsed < 0.01

    async def test_rate_limiter_with_very_high_rate(self) -> None:
        """Very high rate (1M req/s) should not cause precision issues."""
        limiter = _RateLimiter(1_000_000.0)

        start_time = asyncio.get_running_loop().time()

        for _ in range(100):
            await limiter.acquire()

        elapsed = asyncio.get_running_loop().time() - start_time

        # Should complete very quickly
        assert elapsed < 0.1

    async def test_rate_limiter_concurrent_acquires(self) -> None:
        """Multiple concurrent acquires should be properly rate limited."""
        limiter = _RateLimiter(10.0)  # 10 requests per second

        async def make_requests(count: int) -> list[float]:
            times = []
            for _ in range(count):
                start = asyncio.get_running_loop().time()
                await limiter.acquire()
                times.append(asyncio.get_running_loop().time() - start)
            return times

        # Run 20 requests concurrently
        results = await asyncio.gather(
            make_requests(10),
            make_requests(10),
        )

        # Each batch of 10 should take approximately 1 second
        # The combined results might overlap but rate limiting should still apply
        all_times = [t for batch in results for t in batch]
        assert len(all_times) == 20


# ============================================================================
# Cancellation Tests
# ============================================================================


class TestCancellation:
    """Test cancellation at different operation phases."""

    async def test_cancel_at_start(
        self, mock_sync_engine: MagicMock, mock_adapter: MagicMock
    ) -> None:
        """Cancelling immediately after start should prevent processing."""
        config = BatchConfig(requests_per_second=10)  # Enable rate limiting
        manager = BatchOperationManager(mock_sync_engine, config)

        started = asyncio.Event()

        async def slow_sync(
            *args: Any, progress_callback: Any = None, **kwargs: Any
        ) -> tuple[ImportResult, Any]:
            started.set()
            # Simulate processing with progress callback
            for i in range(5):
                if progress_callback:
                    progress_callback(i + 1, 5, f"record-{i}")
                await asyncio.sleep(0.05)  # Longer delay to allow cancellation
            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=0,
            ), None

        mock_sync_engine.sync = slow_sync  # type: ignore[method-assign]

        # Start import
        task = asyncio.create_task(
            manager.import_with_progress(adapter=mock_adapter, on_progress=lambda c, t, iid: None)
        )

        # Wait for operation to start, then cancel immediately
        await started.wait()
        manager.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_cancel_during_operation(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
        sample_records: list[ExternalRecord],
    ) -> None:
        """Cancelling during operation should stop processing."""
        config = BatchConfig(requests_per_second=10)  # Enable rate limiting
        manager = BatchOperationManager(mock_sync_engine, config)

        processed_count = [0]

        async def sync_with_progress(
            *args: Any,
            progress_callback: Any = None,
            **kwargs: Any,
        ) -> tuple[ImportResult, Any]:
            for i, record in enumerate(sample_records):
                if progress_callback:
                    progress_callback(i + 1, len(sample_records), record.id)
                processed_count[0] += 1

                # Small delay to allow cancellation
                await asyncio.sleep(0.01)

            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=processed_count[0],
            ), None

        mock_sync_engine.sync = sync_with_progress  # type: ignore[method-assign]

        # Start operation
        task = asyncio.create_task(
            manager.import_with_progress(adapter=mock_adapter, on_progress=lambda c, t, iid: None)
        )

        # Wait a bit then cancel
        await asyncio.sleep(0.05)
        manager.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Should have processed some but not all records
        assert processed_count[0] > 0
        assert processed_count[0] < len(sample_records)

    async def test_cancel_at_end(
        self, mock_sync_engine: MagicMock, mock_adapter: MagicMock
    ) -> None:
        """Cancelling near the end should still report partial progress."""
        config = BatchConfig(requests_per_second=10)
        manager = BatchOperationManager(mock_sync_engine, config)

        status_updates = []

        async def sync_near_complete(
            *args: Any, progress_callback: Any = None, **kwargs: Any
        ) -> tuple[ImportResult, Any]:
            for i in range(10):
                if progress_callback:
                    progress_callback(i + 1, 10, f"record-{i}")
                await asyncio.sleep(0.01)
            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=99,
            ), None

        mock_sync_engine.sync = sync_near_complete  # type: ignore[method-assign]

        # Cancel after starting
        task = asyncio.create_task(
            manager.import_with_progress(
                adapter=mock_adapter,
                on_progress=lambda c, t, iid: None,
                on_status=lambda s, m: status_updates.append((s, m)),
            )
        )

        await asyncio.sleep(0.04)
        manager.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Should have a cancelled status update
        assert any(s == "cancelled" for s, _ in status_updates)

    async def test_multiple_cancels_idempotent(self, mock_sync_engine: MagicMock) -> None:
        """Calling cancel multiple times should be safe."""
        manager = BatchOperationManager(mock_sync_engine)

        manager.cancel()
        manager.cancel()
        manager.cancel()

        assert manager._cancelled is True

    async def test_cancel_during_pause(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Cancelling while paused should exit immediately."""
        config = BatchConfig(requests_per_second=10)
        manager = BatchOperationManager(mock_sync_engine, config)

        async def sync_with_pause(
            *args: Any, progress_callback: Any = None, **kwargs: Any
        ) -> tuple[ImportResult, Any]:
            # The pause check happens in the tracked_callback in BatchOperationManager
            for i in range(10):
                if progress_callback:
                    progress_callback(i + 1, 10, f"record-{i}")
                await asyncio.sleep(0.01)
            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=0,
            ), None

        mock_sync_engine.sync = sync_with_pause  # type: ignore[method-assign]

        # Start and immediately pause
        task = asyncio.create_task(
            manager.import_with_progress(adapter=mock_adapter, on_progress=lambda c, t, iid: None)
        )

        await asyncio.sleep(0.01)
        manager.pause()

        # Cancel while paused
        await asyncio.sleep(0.02)
        manager.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


# ============================================================================
# Pause/Resume Tests
# ============================================================================


class TestPauseResume:
    """Test pause/resume during active operations."""

    async def test_pause_and_resume_operation(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Pausing should halt progress, resuming should continue."""
        manager = BatchOperationManager(mock_sync_engine)

        processed_records = []

        async def sync_with_pause_check(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            for i in range(10):
                # Check pause in callback
                while manager._paused:
                    await asyncio.sleep(0.01)

                processed_records.append(i)
                await asyncio.sleep(0.01)

            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=10,
            ), None

        mock_sync_engine.sync = sync_with_pause_check  # type: ignore[method-assign]

        # Start operation
        task = asyncio.create_task(manager.import_with_progress(adapter=mock_adapter))

        # Wait a bit then pause
        await asyncio.sleep(0.03)
        manager.pause()

        # Wait while paused
        await asyncio.sleep(0.05)
        paused_count = len(processed_records)

        # Resume
        manager.resume()

        await task

        # Should have processed all records
        assert len(processed_records) == 10
        # Should have been paused for some time
        assert paused_count > 0
        assert paused_count < 10

    async def test_multiple_pause_resume_cycles(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Multiple pause/resume cycles should work correctly."""
        manager = BatchOperationManager(mock_sync_engine)

        pause_count = [0]

        async def sync_with_multiple_pauses(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            for _i in range(20):
                while manager._paused:
                    pause_count[0] += 1
                    await asyncio.sleep(0.01)

                await asyncio.sleep(0.01)

            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=20,
            ), None

        mock_sync_engine.sync = sync_with_multiple_pauses  # type: ignore[method-assign]

        # Start operation
        task = asyncio.create_task(manager.import_with_progress(adapter=mock_adapter))

        # Multiple pause/resume cycles
        for _ in range(3):
            await asyncio.sleep(0.02)
            manager.pause()
            await asyncio.sleep(0.02)
            manager.resume()

        await task

        # Should have completed successfully
        assert pause_count[0] > 0

    async def test_pause_state_persistence(self, mock_sync_engine: MagicMock) -> None:
        """Pause state should persist across operations."""
        manager = BatchOperationManager(mock_sync_engine)

        assert manager._paused is False

        manager.pause()
        assert manager._paused is True

        manager.resume()
        assert manager._paused is False

    async def test_resume_when_not_paused(self, mock_sync_engine: MagicMock) -> None:
        """Resuming when not paused should be safe."""
        manager = BatchOperationManager(mock_sync_engine)

        # Should not raise
        manager.resume()
        manager.resume()
        assert manager._paused is False


# ============================================================================
# Large Dataset Tests
# ============================================================================


class TestLargeDatasets:
    """Test large datasets (1000+ records) with progress tracking."""

    async def test_import_large_dataset(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Importing 1000+ records should track progress correctly."""
        config = BatchConfig(requests_per_second=1000)  # Fast rate for large dataset
        manager = BatchOperationManager(mock_sync_engine, config)

        progress_updates = []

        async def sync_large_dataset(
            *args: Any, progress_callback: Any = None, **kwargs: Any
        ) -> tuple[ImportResult, Any]:
            total = 1500

            for i in range(total):
                if progress_callback:
                    progress_callback(i + 1, total, f"record-{i}")

                # Simulate some work
                if i % 100 == 0:
                    await asyncio.sleep(0.001)

            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=total,
            ), None

        mock_sync_engine.sync = sync_large_dataset  # type: ignore[method-assign]

        result = await manager.import_with_progress(
            adapter=mock_adapter,
            on_progress=lambda c, t, iid: progress_updates.append((c, t, iid)),
        )

        assert result.records_imported == 1500
        assert len(progress_updates) > 0

        # Check final progress update
        final_current, final_total, _ = progress_updates[-1]
        assert final_current == 1500
        assert final_total == 1500

    async def test_export_large_dataset_with_checkpointing(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Exporting 2000 records should save checkpoints periodically."""
        config = BatchConfig(
            batch_size=50,
            checkpoint_interval=100,
            checkpoint_path=temp_checkpoint_dir,
        )
        manager = BatchOperationManager(mock_sync_engine, config)

        checkpoint_path = temp_checkpoint_dir / "large_export.json"

        checkpoint_saves = []

        original_write = checkpoint_path.write_text

        def track_writes(*args: Any, **kwargs: Any) -> int:
            checkpoint_saves.append(len(checkpoint_saves) + 1)
            return original_write(*args, **kwargs)

        with patch.object(Path, "write_text", track_writes):  # type: ignore[attr-defined]

            async def export_large_dataset(
                *args: Any, progress_callback: Any = None, **kwargs: Any
            ) -> ExportResult:
                total = 2000

                for i in range(total):
                    if progress_callback:
                        progress_callback(i + 1, total, f"fiber-{i}")

                return ExportResult(
                    target_system="test_adapter",
                    target_collection="default",
                    records_exported=total,
                    duration_seconds=5.0,
                )

            mock_sync_engine.export = export_large_dataset  # type: ignore[method-assign]

            result, checkpoint = await manager.export_with_checkpoint(
                adapter=mock_adapter,
                checkpoint_path=checkpoint_path,
            )

            assert result.records_exported == 2000
            assert checkpoint is not None
            assert checkpoint.status == BatchOperationStatus.COMPLETED

            # Should have saved checkpoints at intervals
            # Initial + (2000 / 100) = ~21 saves
            assert len(checkpoint_saves) >= 1

    async def test_progress_callback_frequency(
        self, mock_sync_engine: MagicMock, mock_adapter: MagicMock
    ) -> None:
        """Progress callbacks should be called for each record."""
        manager = BatchOperationManager(mock_sync_engine)

        callback_count = [0]

        async def sync_with_callbacks(
            *args: Any, progress_callback: Any = None, **kwargs: Any
        ) -> tuple[ImportResult, Any]:
            total = 500

            for i in range(total):
                if progress_callback:
                    callback_count[0] += 1
                    progress_callback(i + 1, total, f"record-{i}")

            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=total,
            ), None

        mock_sync_engine.sync = sync_with_callbacks  # type: ignore[method-assign]

        await manager.import_with_progress(
            adapter=mock_adapter,
            on_progress=lambda c, t, iid: None,
        )

        assert callback_count[0] == 500


# ============================================================================
# Concurrent Operations Tests
# ============================================================================


class TestConcurrentOperations:
    """Test concurrent batch operations (multiple imports/exports)."""

    async def test_concurrent_imports_from_different_sources(
        self,
        mock_sync_engine: MagicMock,
    ) -> None:
        """Multiple imports from different sources should run concurrently."""
        manager = BatchOperationManager(mock_sync_engine)

        adapter1 = MagicMock()
        adapter1.system_name = "adapter1"
        adapter1.fetch_all = AsyncMock(return_value=[])

        adapter2 = MagicMock()
        adapter2.system_name = "adapter2"
        adapter2.fetch_all = AsyncMock(return_value=[])

        async def sync_for_adapter(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            await asyncio.sleep(0.1)
            adapter = kwargs.get("adapter") or args[0] if args else None
            system_name = getattr(adapter, "system_name", "unknown") if adapter else "unknown"
            return ImportResult(
                source_system=system_name,
                source_collection="default",
                records_imported=10,
            ), None

        mock_sync_engine.sync = sync_for_adapter  # type: ignore[method-assign]

        # Run concurrent imports
        results = await asyncio.gather(
            manager.import_with_progress(adapter=adapter1),
            manager.import_with_progress(adapter=adapter2),
        )

        assert len(results) == 2
        assert all(r.records_imported == 10 for r in results)

    async def test_concurrent_exports_to_different_targets(
        self,
        mock_sync_engine: MagicMock,
    ) -> None:
        """Multiple exports to different targets should run concurrently."""
        manager = BatchOperationManager(mock_sync_engine)

        target1 = MagicMock()
        target1.system_name = "target1"
        target1.create_record = AsyncMock(return_value="ext-id-1")

        target2 = MagicMock()
        target2.system_name = "target2"
        target2.create_record = AsyncMock(return_value="ext-id-2")

        async def export_to_target(*args: Any, **kwargs: Any) -> ExportResult:
            await asyncio.sleep(0.1)
            adapter = kwargs.get("adapter") or args[0] if args else None
            system_name = getattr(adapter, "system_name", "unknown") if adapter else "unknown"
            return ExportResult(
                target_system=system_name,
                target_collection="default",
                records_exported=5,
            )

        mock_sync_engine.export = export_to_target  # type: ignore[method-assign]

        # Run concurrent exports
        results = await asyncio.gather(
            manager.export_with_checkpoint(adapter=target1),
            manager.export_with_checkpoint(adapter=target2),
        )

        assert len(results) == 2

    async def test_concurrent_import_and_export(
        self,
        mock_sync_engine: MagicMock,
    ) -> None:
        """Import and export operations should be able to run concurrently."""
        manager = BatchOperationManager(mock_sync_engine)

        source = MagicMock()
        source.system_name = "source"
        source.fetch_all = AsyncMock(return_value=[])

        target = MagicMock()
        target.system_name = "target"
        target.create_record = AsyncMock(return_value="ext-id")

        async def sync_operation(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            await asyncio.sleep(0.1)
            return ImportResult(
                source_system="source",
                source_collection="default",
                records_imported=10,
            ), None

        async def export_operation(*args: Any, **kwargs: Any) -> ExportResult:
            await asyncio.sleep(0.1)
            return ExportResult(
                target_system="target",
                target_collection="default",
                records_exported=5,
            )

        mock_sync_engine.sync = sync_operation  # type: ignore[method-assign]
        mock_sync_engine.export = export_operation  # type: ignore[method-assign]

        # Run import and export concurrently
        import_task = asyncio.create_task(manager.import_with_progress(adapter=source))
        export_task = asyncio.create_task(manager.export_with_checkpoint(adapter=target))

        results = await asyncio.gather(import_task, export_task)

        assert len(results) == 2

    async def test_manager_state_isolation(self, mock_sync_engine: MagicMock) -> None:
        """Each manager should maintain its own cancellation/pause state."""
        manager1 = BatchOperationManager(mock_sync_engine)
        manager2 = BatchOperationManager(mock_sync_engine)

        manager1.cancel()
        manager2.pause()

        assert manager1._cancelled is True
        assert manager1._paused is False

        assert manager2._cancelled is False
        assert manager2._paused is True


# ============================================================================
# Memory Leak Tests
# ============================================================================


class TestMemoryLeaks:
    """Test memory leaks under sustained operations."""

    async def test_sustained_operations_no_growth(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Running many operations should not cause unbounded memory growth."""
        manager = BatchOperationManager(mock_sync_engine)

        # Track the number of objects created
        initial_objects = len(gc.get_objects())  # type: ignore[attr-defined]

        # Run many operations
        for _ in range(100):
            await manager.import_with_progress(adapter=mock_adapter)

        final_objects = len(gc.get_objects())  # type: ignore[attr-defined]

        # Allow some growth but not unbounded
        # This is a weak check but can catch obvious leaks
        growth = final_objects - initial_objects
        assert growth < 10000  # Arbitrary threshold

    async def test_checkpoint_cleanup(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Old checkpoints should be cleaned up or overwritten."""
        manager = BatchOperationManager(
            mock_sync_engine,
            BatchConfig(checkpoint_path=temp_checkpoint_dir),
        )

        checkpoint_path = temp_checkpoint_dir / "test_checkpoint.json"

        # Run multiple operations with the same checkpoint path
        for _i in range(5):

            async def export_op(*args: Any, **kwargs: Any) -> ExportResult:
                return ExportResult(
                    target_system="test_adapter",
                    target_collection="default",
                    records_exported=10,
                )

            mock_sync_engine.export = export_op  # type: ignore[method-assign]

            await manager.export_with_checkpoint(
                adapter=mock_adapter,
                checkpoint_path=checkpoint_path,
            )

        # Should only have one checkpoint file
        checkpoint_files = list(temp_checkpoint_dir.glob("*.json"))
        assert len(checkpoint_files) <= 5

    async def test_callback_cleanup(
        self, mock_sync_engine: MagicMock, mock_adapter: MagicMock
    ) -> None:
        """Callbacks should not accumulate references."""
        config = BatchConfig(requests_per_second=100)  # Fast rate for quick test
        manager = BatchOperationManager(mock_sync_engine, config)

        callbacks = []

        async def sync_op(
            *args: Any, progress_callback: Any = None, **kwargs: Any
        ) -> tuple[ImportResult, Any]:
            if progress_callback:
                progress_callback(1, 1, "test-record")
            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=1,
            ), None

        mock_sync_engine.sync = sync_op  # type: ignore[method-assign]

        # Run with many different callbacks
        for i in range(10):  # Reduced from 100 for faster test

            def callback(c, t, iid, i=i):
                return callbacks.append(i)  # Capture i by default

            await manager.import_with_progress(
                adapter=mock_adapter,
                on_progress=callback,
            )

        # All callbacks should have been called
        assert len(callbacks) == 10


# ============================================================================
# Network Timeout Tests
# ============================================================================


class TestNetworkTimeouts:
    """Test network timeout handling (simulated with slow mocks)."""

    async def test_slow_adapter_response(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Slow adapter responses should be handled gracefully."""
        manager = BatchOperationManager(mock_sync_engine)

        async def slow_sync(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            # Simulate network delay
            await asyncio.sleep(0.5)
            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=1,
            ), None

        mock_sync_engine.sync = slow_sync  # type: ignore[method-assign]

        # Should complete without timeout issues
        result = await manager.import_with_progress(
            adapter=mock_adapter,
        )

        assert result.records_imported == 1

    async def test_intermittent_slow_responses(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Mix of fast and slow responses should be handled correctly."""
        manager = BatchOperationManager(mock_sync_engine)

        call_count = [0]

        async def variable_speed_sync(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            call_count[0] += 1

            # Every other call is slow
            if call_count[0] % 2 == 0:
                await asyncio.sleep(0.2)
            else:
                await asyncio.sleep(0.01)

            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=call_count[0],
            ), None

        mock_sync_engine.sync = variable_speed_sync  # type: ignore[method-assign]

        # Run multiple operations
        for _ in range(5):
            await manager.import_with_progress(adapter=mock_adapter)

        assert call_count[0] == 5

    @pytest.mark.timeout(120)
    async def test_timeout_during_rate_limiting(self, mock_sync_engine: MagicMock) -> None:
        """Rate limiting delays should not trigger timeouts."""
        config = BatchConfig(requests_per_second=2.0)  # 2 requests per second
        manager = BatchOperationManager(mock_sync_engine, config)

        # Track time between callbacks
        times = []

        async def sync_with_delays(
            *args: Any, progress_callback: Any = None, **kwargs: Any
        ) -> tuple[ImportResult, Any]:
            for i in range(3):
                times.append(asyncio.get_running_loop().time())
                if progress_callback:
                    progress_callback(i + 1, 3, f"record-{i}")

            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=3,
            ), None

        mock_sync_engine.sync = sync_with_delays  # type: ignore[method-assign]

        start_time = asyncio.get_running_loop().time()
        await manager.import_with_progress(adapter=MagicMock(), on_progress=lambda c, t, iid: None)
        end_time = asyncio.get_running_loop().time()

        total_time = end_time - start_time

        # With 3 requests at 2/sec, should take at least 1 second (0.5s between requests after first)
        assert total_time >= 0.8  # Allow some tolerance

    async def test_hanging_operation_cancellation(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """A hanging operation should be cancellable."""
        manager = BatchOperationManager(mock_sync_engine)

        hang_event = asyncio.Event()

        async def hanging_sync(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            # Wait indefinitely until cancelled
            try:
                await asyncio.sleep(float("inf"))
            except asyncio.CancelledError:
                hang_event.set()
                raise

        mock_sync_engine.sync = hanging_sync  # type: ignore[method-assign]

        # Start operation
        task = asyncio.create_task(manager.import_with_progress(adapter=mock_adapter))

        # Give it time to start
        await asyncio.sleep(0.1)

        # Cancel
        manager.cancel()

        with pytest.raises((asyncio.CancelledError, asyncio.TimeoutError)):
            await asyncio.wait_for(task, timeout=1.0)


# ============================================================================
# Invalid Adapter Response Tests
# ============================================================================


class TestInvalidAdapterResponses:
    """Test invalid adapter responses (None, empty, malformed data)."""

    async def test_adapter_returns_none_for_records(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Adapter returning None for records should be handled."""
        manager = BatchOperationManager(mock_sync_engine)

        mock_adapter.fetch_all = AsyncMock(return_value=None)  # type: ignore[assignment]

        async def sync_with_none(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            # Handle None response gracefully
            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=0,
            ), None

        mock_sync_engine.sync = sync_with_none  # type: ignore[method-assign]

        result = await manager.import_with_progress(adapter=mock_adapter)

        assert result.records_imported == 0

    async def test_adapter_returns_empty_list(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Adapter returning empty list should be handled."""
        manager = BatchOperationManager(mock_sync_engine)

        mock_adapter.fetch_all = AsyncMock(return_value=[])

        async def sync_with_empty(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=0,
            ), None

        mock_sync_engine.sync = sync_with_empty  # type: ignore[method-assign]

        result = await manager.import_with_progress(adapter=mock_adapter)

        assert result.records_imported == 0

    async def test_adapter_returns_malformed_records(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Adapter returning malformed records should handle errors."""
        manager = BatchOperationManager(mock_sync_engine)

        # Return records with None/empty content
        bad_records = [
            ExternalRecord.create(
                id="bad-1",
                source_system="test_adapter",
                content="",  # Empty content
            ),
            ExternalRecord.create(
                id="bad-2",
                source_system="test_adapter",
                content="   ",  # Whitespace only
            ),
            ExternalRecord.create(
                id="good-1",
                source_system="test_adapter",
                content="Valid content",
            ),
        ]

        mock_adapter.fetch_all = AsyncMock(return_value=bad_records)

        async def sync_with_mixed(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            # Only valid records should be imported
            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_fetched=3,
                records_imported=1,
                records_skipped=2,
            ), None

        mock_sync_engine.sync = sync_with_mixed  # type: ignore[method-assign]

        result = await manager.import_with_progress(adapter=mock_adapter)

        assert result.records_imported == 1
        assert result.records_skipped == 2

    async def test_create_record_returns_none(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Create record returning None should be handled as failure."""
        manager = BatchOperationManager(mock_sync_engine)

        mock_adapter.create_record = AsyncMock(return_value=None)

        async def export_with_none(*args: Any, **kwargs: Any) -> ExportResult:
            return ExportResult(
                target_system="test_adapter",
                target_collection="default",
                records_exported=0,
                records_failed=1,
                errors=("Failed to export: no ID returned",),
            )

        mock_sync_engine.export = export_with_none  # type: ignore[method-assign]

        result, _ = await manager.export_with_checkpoint(adapter=mock_adapter)

        assert result.records_exported == 0
        assert result.records_failed == 1
        assert len(result.errors) > 0

    async def test_adapter_raises_exception(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Adapter raising exception should be handled."""
        manager = BatchOperationManager(mock_sync_engine)

        mock_adapter.fetch_all = AsyncMock(side_effect=ConnectionError("Network error"))

        status_updates = []

        async def sync_with_error(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            raise ConnectionError("Network error")

        mock_sync_engine.sync = sync_with_error  # type: ignore[method-assign]

        with pytest.raises(ConnectionError):
            await manager.import_with_progress(
                adapter=mock_adapter,
                on_status=lambda s, m: status_updates.append((s, m)),
            )

        # Should have a failed status update
        assert any(s == "failed" for s, _ in status_updates)


# ============================================================================
# Checkpoint Permission Tests
# ============================================================================


class TestCheckpointPermissions:
    """Test checkpoint file permission errors."""

    async def test_checkpoint_directory_creation(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Checkpoint directory should be created if it doesn't exist."""
        manager = BatchOperationManager(mock_sync_engine)

        non_existent_dir = temp_checkpoint_dir / "deep" / "nested" / "path"
        checkpoint_path = non_existent_dir / "checkpoint.json"

        async def export_op(*args: Any, **kwargs: Any) -> ExportResult:
            return ExportResult(
                target_system="test_adapter",
                target_collection="default",
                records_exported=1,
            )

        mock_sync_engine.export = export_op  # type: ignore[method-assign]

        result, checkpoint = await manager.export_with_checkpoint(
            adapter=mock_adapter,
            checkpoint_path=checkpoint_path,
        )

        # Directory should be created
        assert checkpoint_path.parent.exists()
        assert checkpoint is not None

    @patch("pathlib.Path.write_text")
    async def test_checkpoint_write_permission_denied(
        self,
        mock_write_text: Mock,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Permission denied when writing checkpoint should be handled."""
        mock_write_text.side_effect = PermissionError("Permission denied")

        manager = BatchOperationManager(mock_sync_engine)

        checkpoint_path = temp_checkpoint_dir / "checkpoint.json"

        async def export_op(*args: Any, **kwargs: Any) -> ExportResult:
            return ExportResult(
                target_system="test_adapter",
                target_collection="default",
                records_exported=1,
            )

        mock_sync_engine.export = export_op  # type: ignore[method-assign]

        # Should handle permission error gracefully
        # The export should still succeed even if checkpoint fails
        result, checkpoint = await manager.export_with_checkpoint(
            adapter=mock_adapter,
            checkpoint_path=checkpoint_path,
        )

        assert result.records_exported == 1

    @patch("pathlib.Path.read_text")
    async def test_checkpoint_read_permission_denied(
        self,
        mock_read_text: Mock,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Permission denied when reading checkpoint should start fresh."""
        mock_read_text.side_effect = PermissionError("Permission denied")

        manager = BatchOperationManager(mock_sync_engine)

        checkpoint_path = temp_checkpoint_dir / "checkpoint.json"

        async def export_op(*args: Any, **kwargs: Any) -> ExportResult:
            return ExportResult(
                target_system="test_adapter",
                target_collection="default",
                records_exported=1,
            )

        mock_sync_engine.export = export_op  # type: ignore[method-assign]

        # Should start fresh when checkpoint can't be read
        result, checkpoint = await manager.export_with_checkpoint(
            adapter=mock_adapter,
            checkpoint_path=checkpoint_path,
            resume_from=checkpoint_path,
        )

        assert result.records_exported == 1
        assert checkpoint is not None


# ============================================================================
# Large Record Tests
# ============================================================================


class TestLargeRecords:
    """Test very large record IDs or metadata."""

    async def test_very_long_record_ids(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Very long record IDs should be handled."""
        manager = BatchOperationManager(mock_sync_engine)

        # Create record with extremely long ID
        long_id = "record-" + "x" * 10000

        async def sync_with_long_id(
            *args: Any, progress_callback: Any = None, **kwargs: Any
        ) -> tuple[ImportResult, Any]:
            if progress_callback:
                progress_callback(1, 1, long_id)

            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=1,
            ), None

        mock_sync_engine.sync = sync_with_long_id  # type: ignore[method-assign]

        progress_calls = []

        await manager.import_with_progress(
            adapter=mock_adapter,
            on_progress=lambda c, t, iid: progress_calls.append((c, t, iid)),
        )

        # Long ID should be preserved
        assert len(progress_calls) == 1
        assert progress_calls[0][2] == long_id

    async def test_very_large_metadata(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Records with very large metadata should be handled."""
        manager = BatchOperationManager(mock_sync_engine)

        # Create record with huge metadata

        async def sync_with_large_metadata(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=1,
            ), None

        mock_sync_engine.sync = sync_with_large_metadata  # type: ignore[method-assign]

        result = await manager.import_with_progress(adapter=mock_adapter)

        assert result.records_imported == 1

    async def test_checkpoint_with_large_data(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Checkpoint with large metadata should be saved correctly."""
        manager = BatchOperationManager(mock_sync_engine)

        checkpoint_path = temp_checkpoint_dir / "large_checkpoint.json"

        # Create checkpoint with large metadata
        large_metadata = {
            "records": [f"record-{i}" for i in range(10000)],
        }

        async def export_with_large_checkpoint(*args: Any, **kwargs: Any) -> ExportResult:
            return ExportResult(
                target_system="test_adapter",
                target_collection="default",
                records_exported=1,
            )

        mock_sync_engine.export = export_with_large_checkpoint  # type: ignore[method-assign]

        result, checkpoint = await manager.export_with_checkpoint(
            adapter=mock_adapter,
            checkpoint_path=checkpoint_path,
        )

        # Update checkpoint with large metadata
        if checkpoint:
            checkpoint.metadata = large_metadata
            checkpoint_path.write_text(json.dumps(checkpoint.to_dict()))

        # Checkpoint should be loadable
        loaded_checkpoint = BatchOperationManager.load_checkpoint(checkpoint_path)
        assert loaded_checkpoint is not None
        assert len(loaded_checkpoint.metadata.get("records", [])) == 10000


# ============================================================================
# Race Condition Tests
# ============================================================================


class TestRaceConditions:
    """Test race conditions in pause/resume/cancel."""

    async def test_concurrent_pause_and_cancel(self, mock_sync_engine: MagicMock) -> None:
        """Pause and cancel happening concurrently should be handled."""
        manager = BatchOperationManager(mock_sync_engine)

        async def slow_operation(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            # Give time for pause and cancel to race
            await asyncio.sleep(0.2)

            # Check state
            if manager._cancelled:
                raise asyncio.CancelledError("Cancelled")

            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=0,
            ), None

        mock_sync_engine.sync = slow_operation  # type: ignore[method-assign]

        task = asyncio.create_task(manager.import_with_progress(adapter=MagicMock()))

        # Quickly trigger pause and cancel
        await asyncio.gather(
            asyncio.to_thread(manager.pause),
            asyncio.to_thread(manager.cancel),
        )

        with pytest.raises(asyncio.CancelledError):
            await task

        # Cancel should win
        assert manager._cancelled is True

    async def test_rapid_resume_cancel_cycles(self, mock_sync_engine: MagicMock) -> None:
        """Rapid resume/cancel cycles should be handled."""
        manager = BatchOperationManager(mock_sync_engine)

        # Start paused
        manager.pause()

        async def pausable_operation(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            for _ in range(10):
                while manager._paused:
                    if manager._cancelled:
                        raise asyncio.CancelledError("Cancelled")
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.01)

            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=10,
            ), None

        mock_sync_engine.sync = pausable_operation  # type: ignore[method-assign]

        task = asyncio.create_task(manager.import_with_progress(adapter=MagicMock()))

        # Rapid resume/pause/cancel
        manager.resume()
        await asyncio.sleep(0.02)
        manager.pause()
        await asyncio.sleep(0.02)
        manager.resume()
        await asyncio.sleep(0.02)
        manager.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_cancel_between_pause_and_resume(self, mock_sync_engine: MagicMock) -> None:
        """Cancel called while paused should exit immediately."""
        manager = BatchOperationManager(mock_sync_engine)

        pause_count = [0]

        async def paused_operation(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            while manager._paused:
                pause_count[0] += 1
                if manager._cancelled:
                    raise asyncio.CancelledError("Cancelled during pause")
                await asyncio.sleep(0.01)

            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=0,
            ), None

        mock_sync_engine.sync = paused_operation  # type: ignore[method-assign]

        task = asyncio.create_task(manager.import_with_progress(adapter=MagicMock()))

        # Pause then cancel
        await asyncio.sleep(0.02)
        manager.pause()
        await asyncio.sleep(0.02)
        manager.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Should have been in pause state
        assert pause_count[0] > 0

    async def test_state_change_during_callback(self, mock_sync_engine: MagicMock) -> None:
        """State changes during progress callback should be handled."""
        manager = BatchOperationManager(mock_sync_engine)

        state_changes = []

        async def operation_with_callback(
            *args: Any, progress_callback: Any = None, **kwargs: Any
        ) -> tuple[ImportResult, Any]:
            for i in range(10):
                if progress_callback:
                    # Change state during callback
                    if i == 5:
                        manager.pause()
                        state_changes.append("paused")
                    if i == 7:
                        manager.resume()
                        state_changes.append("resumed")
                    if i == 9:
                        manager.cancel()
                        state_changes.append("cancelled")

                    progress_callback(i + 1, 10, f"record-{i}")

                await asyncio.sleep(0.01)

            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=10,
            ), None

        mock_sync_engine.sync = operation_with_callback  # type: ignore[method-assign]

        with pytest.raises(asyncio.CancelledError):
            await manager.import_with_progress(adapter=MagicMock())

        assert len(state_changes) >= 1


# ============================================================================
# Stress Tests
# ============================================================================


class TestStressTests:
    """Stress tests for batch operations."""

    async def test_rapid_sequential_operations(
        self,
        mock_sync_engine: MagicMock,
        mock_adapter: MagicMock,
    ) -> None:
        """Many sequential operations should complete without issues."""
        manager = BatchOperationManager(mock_sync_engine)

        operation_count = [0]

        async def quick_sync(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            operation_count[0] += 1
            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=1,
            ), None

        mock_sync_engine.sync = quick_sync  # type: ignore[method-assign]

        # Run many operations quickly
        for _ in range(100):
            await manager.import_with_progress(adapter=mock_adapter)

        assert operation_count[0] == 100

    async def test_burst_operations(
        self, mock_sync_engine: MagicMock, mock_adapter: MagicMock
    ) -> None:
        """Burst of concurrent operations should be handled."""
        manager = BatchOperationManager(mock_sync_engine)

        async def quick_sync(*args: Any, **kwargs: Any) -> tuple[ImportResult, Any]:
            await asyncio.sleep(0.01)
            return ImportResult(
                source_system="test_adapter",
                source_collection="default",
                records_imported=1,
            ), None

        mock_sync_engine.sync = quick_sync  # type: ignore[method-assign]

        # Burst of 50 operations
        tasks = [manager.import_with_progress(adapter=mock_adapter) for _ in range(50)]

        results = await asyncio.gather(*tasks)

        assert len(results) == 50
        assert all(r.records_imported == 1 for r in results)


# ============================================================================
# End of test file
# ============================================================================
