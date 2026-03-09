"""Configuration management for PugBrain."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _detect_vector_backend() -> str:
    """Prefer RuVector if available, otherwise fall back to numpy."""
    try:
        import ruvector  # type: ignore[import-untyped]  # noqa: F401

        return "ruvector"
    except ImportError:
        return "numpy"


@dataclass
class Config:
    """
    Application configuration.

    Loaded from environment variables with sensible defaults.
    """

    # Server settings
    host: str = "127.0.0.1"
    port: int = 18790
    debug: bool = False

    # Storage settings
    storage_backend: str = "sqlite"  # Default to sqlite for persistence
    sqlite_path: str | None = None
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_username: str | None = None
    falkordb_password: str | None = None
    vector_backend: str = field(default_factory=_detect_vector_backend)

    # Brain defaults
    default_decay_rate: float = 0.1
    default_activation_threshold: float = 0.2
    default_max_spread_hops: int = 4
    default_max_context_tokens: int = 1500

    # CORS settings
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:*", "http://127.0.0.1:*"]
    )

    # Trusted networks (CIDR notation, e.g. "172.16.0.0/12,192.168.0.0/16")
    # Allows non-localhost requests from these networks (for Docker/container deployments)
    trusted_networks: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables."""

        def get_bool(key: str, default: bool) -> bool:
            value = os.getenv(key)
            if value is None:
                return default
            return value.lower() in ("true", "1", "yes")

        def get_int(key: str, default: int) -> int:
            value = os.getenv(key)
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                return default

        def get_float(key: str, default: float) -> float:
            value = os.getenv(key)
            if value is None:
                return default
            try:
                return float(value)
            except ValueError:
                return default

        def get_list(key: str, default: list[str]) -> list[str]:
            value = os.getenv(key)
            if value is None:
                return default
            return [s.strip() for s in value.split(",")]

        # Define default home directory for PugBrain
        pug_home = os.path.expanduser(os.getenv("PUGBRAIN_HOME", "~/.pugbrain"))
        os.makedirs(pug_home, exist_ok=True)

        default_sqlite_path = os.path.join(pug_home, "brain.db")

        return cls(
            host=os.getenv("NEURAL_MEMORY_HOST", "127.0.0.1"),
            port=get_int("NEURAL_MEMORY_PORT", 18790),
            debug=get_bool("NEURAL_MEMORY_DEBUG", False),
            storage_backend=os.getenv("NEURAL_MEMORY_STORAGE", "sqlite"),
            sqlite_path=os.getenv("NEURAL_MEMORY_SQLITE_PATH", default_sqlite_path),
            falkordb_host=os.getenv("NEURAL_MEMORY_FALKORDB_HOST", "localhost"),
            falkordb_port=get_int("NEURAL_MEMORY_FALKORDB_PORT", 6379),
            falkordb_username=os.getenv("NEURAL_MEMORY_FALKORDB_USERNAME"),
            falkordb_password=os.getenv("NEURAL_MEMORY_FALKORDB_PASSWORD"),
            vector_backend=os.getenv("NEURAL_MEMORY_VECTOR_BACKEND", _detect_vector_backend()),
            default_decay_rate=get_float("NEURAL_MEMORY_DECAY_RATE", 0.1),
            default_activation_threshold=get_float("NEURAL_MEMORY_ACTIVATION_THRESHOLD", 0.2),
            default_max_spread_hops=get_int("NEURAL_MEMORY_MAX_SPREAD_HOPS", 4),
            default_max_context_tokens=get_int("NEURAL_MEMORY_MAX_CONTEXT_TOKENS", 1500),
            cors_origins=get_list(
                "NEURAL_MEMORY_CORS_ORIGINS",
                ["http://localhost:*", "http://127.0.0.1:*"],
            ),
            trusted_networks=get_list(
                "NEURAL_MEMORY_TRUSTED_NETWORKS",
                [],
            ),
        )


# Singleton config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
