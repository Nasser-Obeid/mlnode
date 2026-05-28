"""MLnode error hierarchy.

MLnodeError
├── ValidationError
│   ├── CycleDetectedError
│   └── ShapeMismatchError
├── CustomLayerError
└── SchemaVersionError
"""

from __future__ import annotations


class MLnodeError(Exception):
    """Base error for all MLnode exceptions."""


class ValidationError(MLnodeError):
    """Raised when graph validation fails."""

    def __init__(self, message: str, *, node_id: str | None = None) -> None:
        self.node_id = node_id
        super().__init__(message)


class CycleDetectedError(ValidationError):
    """Raised when the graph contains a cycle or self-loop."""

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        cycle_str = " → ".join(cycle)
        super().__init__(f"Cycle detected: {cycle_str}")


class ShapeMismatchError(ValidationError):
    """Raised when tensor shapes are incompatible at a node boundary."""

    def __init__(
        self,
        message: str,
        *,
        node_id: str,
        expected_shape: tuple | None = None,
        actual_shape: tuple | None = None,
        source_node_id: str | None = None,
    ) -> None:
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        self.source_node_id = source_node_id
        super().__init__(message, node_id=node_id)


class CustomLayerError(MLnodeError):
    """Raised when a custom layer fails validation."""

    def __init__(self, message: str, *, node_id: str, original_error: Exception | None = None) -> None:
        self.node_id = node_id
        self.original_error = original_error
        super().__init__(message)


class SchemaVersionError(MLnodeError):
    """Raised when the JSON schema version is unsupported."""

    def __init__(self, found_version: str, supported_versions: list[str]) -> None:
        self.found_version = found_version
        self.supported_versions = supported_versions
        supported_str = ", ".join(supported_versions)
        super().__init__(
            f"Schema v{found_version} not supported by MLnode v1.0. "
            f"Supported versions: {supported_str}"
        )
