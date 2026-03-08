"""External source integration layer for PugBrain.

Import memories from competing systems (ChromaDB, Mem0, Graphiti, etc.)
into PugBrain's neuron/synapse/fiber graph.
"""

from neural_memory.integration.adapter import SourceAdapter
from neural_memory.integration.batch_operations import (
    BatchCheckpoint,
    BatchConfig,
    BatchOperationManager,
    BatchOperationStatus,
)
from neural_memory.integration.mapper import MappingResult, RecordMapper
from neural_memory.integration.models import (
    ExportResult,
    ExternalRecord,
    ExternalRelationship,
    ImportResult,
    SourceCapability,
    SourceSystemType,
    SyncState,
)
from neural_memory.integration.sync_engine import SyncEngine

__all__ = [
    "BatchCheckpoint",
    "BatchConfig",
    "BatchOperationManager",
    "BatchOperationStatus",
    "ExportResult",
    "ExternalRecord",
    "ExternalRelationship",
    "ImportResult",
    "MappingResult",
    "RecordMapper",
    "SourceAdapter",
    "SourceCapability",
    "SourceSystemType",
    "SyncEngine",
    "SyncState",
]
