"""MLnodeModel — top-level API (spec §7).

Pipeline: JSON → Parser → Validator → Executor → nn.Module

    model = MLnodeModel.load("resnet.mlnode.json")
    result = model.validate()
    if result.valid:
        module = model.build()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch.nn as nn

from mlnode.executor.builder import ModuleBuilder
from mlnode.parser.dag import DAGParser, ParsedGraph
from mlnode.schema.models import GraphSchema
from mlnode.schema.types import ValidationResult
from mlnode.validator.shapes import ShapeValidator


class MLnodeModel:
    """High-level interface for loading, validating, and building MLnode models.

    Usage:
        model = MLnodeModel.load("model.mlnode.json")
        result = model.validate()       # static shape check
        module = model.build()          # nn.Module
        model.save("model_copy.mlnode.json")
    """

    def __init__(
        self, graph: ParsedGraph, *, original_data: dict[str, Any] | None = None
    ) -> None:
        self._graph = graph
        # Preserve the original JSON data so save() doesn't silently flatten blocks.
        self._original_data = original_data

    @classmethod
    def load(
        cls, path: str | Path, block_registry: Any = None, layer_registry: Any = None
    ) -> MLnodeModel:
        """Load a MLnode model from a .mlnode.json file.

        Files are read as UTF-8 on every platform (not the Windows locale code page).
        """
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        graph = DAGParser.parse_dict(data, block_registry, layer_registry)
        return cls(graph, original_data=data)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], block_registry: Any = None, layer_registry: Any = None
    ) -> MLnodeModel:
        """Create a MLnode model from a dictionary."""
        graph = DAGParser.parse_dict(data, block_registry, layer_registry)
        return cls(graph, original_data=data)

    @property
    def schema(self) -> GraphSchema:
        """Access the underlying graph schema."""
        return self._graph.schema

    @property
    def graph(self) -> ParsedGraph:
        """Access the parsed graph."""
        return self._graph

    def validate(self) -> ValidationResult:
        """Run static shape validation (spec §3).

        Returns a ValidationResult with all errors collected,
        rather than failing on the first one.
        """
        validator = ShapeValidator(self._graph)
        return validator.validate()

    def build(self) -> nn.Module:
        """Build an executable nn.Module from the graph (spec §2).

        Calls validate() first and raises if invalid.
        """
        result = self.validate()
        if not result.valid:
            error_msgs = [e.message for e in result.errors]
            from mlnode.errors import ValidationError

            raise ValidationError(
                f"Cannot build: {len(result.errors)} validation error(s):\n"
                + "\n".join(f"  - {msg}" for msg in error_msgs)
            )
        return ModuleBuilder.build(self._graph)

    def save(self, path: str | Path, *, expanded: bool = False) -> None:
        """Save the model schema to a .mlnode.json file.

        Args:
            path: Output file path.
            expanded: If True, save the block-expanded (flat) schema.
                      If False (default), save the original schema with blocks intact.
        """
        path = Path(path)
        if expanded or self._original_data is None:
            data = self._graph.schema.model_dump(mode="json")
        else:
            data = self._original_data
        with path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    def __repr__(self) -> str:
        name = self._graph.schema.metadata.name or "unnamed"
        n_nodes = len(self._graph.schema.nodes)
        n_edges = len(self._graph.schema.edges)
        return f"MLnodeModel(name='{name}', nodes={n_nodes}, edges={n_edges})"
