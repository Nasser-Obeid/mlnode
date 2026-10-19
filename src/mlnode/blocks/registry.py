"""Block registry — stores reusable block definitions by name.

Allows blocks to be defined once and referenced by multiple nodes
via the 'block_ref' field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlnode.errors import MLnodeError
from mlnode.schema.models import BlockDefinition


class BlockRegistry:
    """Registry for reusable block definitions.

    Usage:
        registry = BlockRegistry()
        registry.register("resnet_block", block_def)
        registry.register_from_file("transformer_block", "blocks/transformer.json")

        # During graph parsing, block_ref nodes are resolved from this registry.
    """

    def __init__(self) -> None:
        self._blocks: dict[str, BlockDefinition] = {}

    def register(self, name: str, block: BlockDefinition) -> None:
        """Register a block definition by name."""
        if name in self._blocks:
            raise MLnodeError(f"Block '{name}' is already registered.")
        self._blocks[name] = block

    def register_from_dict(self, name: str, data: dict[str, Any]) -> None:
        """Register a block from a dictionary."""
        block = BlockDefinition.model_validate(data)
        self.register(name, block)

    def register_from_file(self, name: str, path: str | Path) -> None:
        """Register a block from a JSON file."""
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        self.register_from_dict(name, data)

    def get(self, name: str) -> BlockDefinition:
        """Retrieve a block definition by name."""
        if name not in self._blocks:
            raise MLnodeError(
                f"Block '{name}' not found in registry. "
                f"Available: {sorted(self._blocks.keys())}"
            )
        return self._blocks[name]

    def has(self, name: str) -> bool:
        """Check if a block is registered."""
        return name in self._blocks

    @property
    def names(self) -> list[str]:
        """List all registered block names."""
        return sorted(self._blocks.keys())

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        return f"BlockRegistry(blocks={self.names})"


# Global default registry
_DEFAULT_REGISTRY = BlockRegistry()


def get_default_registry() -> BlockRegistry:
    """Return the global default block registry."""
    return _DEFAULT_REGISTRY
