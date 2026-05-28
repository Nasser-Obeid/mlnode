"""Pydantic models for the MLnode v1.0 JSON schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from mlnode.schema.types import (
    CORE_LAYER_TYPES,
    REDUCTION_OPS,
    Dim,
    ReductionType,
)

# Schema versions this parser supports.
SUPPORTED_VERSIONS = ["1.0"]


class MetadataSchema(BaseModel):
    """Optional model metadata."""

    name: str | None = None
    description: str | None = None
    author: str | None = None


class EdgeSchema(BaseModel):
    """A single edge in the DAG (§1.3 — uniform 4-field schema)."""

    source: str
    source_port: str | None = None
    target: str
    target_port: str | None = None


class BlockDefinition(BaseModel):
    """An inline sub-graph definition for a Block node.

    A block encapsulates a reusable sub-graph with a single input and output.
    When expanded, internal node IDs are namespaced: {block_node_id}/{internal_id}.
    """

    nodes: list[NodeSchema]
    edges: list[EdgeSchema]
    input_node: str
    output_node: str

    @model_validator(mode="after")
    def validate_block(self) -> BlockDefinition:
        node_ids = {n.id for n in self.nodes}
        if self.input_node not in node_ids:
            raise ValueError(
                f"Block input_node '{self.input_node}' not found in block nodes."
            )
        if self.output_node not in node_ids:
            raise ValueError(
                f"Block output_node '{self.output_node}' not found in block nodes."
            )
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f"Block edge source '{edge.source}' not in block nodes.")
            if edge.target not in node_ids:
                raise ValueError(f"Block edge target '{edge.target}' not in block nodes.")
        return self


class NodeSchema(BaseModel):
    """A single node in the DAG."""

    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)

    # Multi-output ports (§1.4).
    outputs: list[str] | None = None

    # Weight sharing (§1.5).
    shared_id: str | None = None

    # Fallback reduction for non-reduction-op nodes receiving >1 input (§1.2).
    reduction: ReductionType | None = None
    reduction_dim: int | None = None

    # HuggingFace fields (§5).
    hf_model: str | None = None
    hf_config: str | None = None

    # Custom layer path.
    path: str | None = None

    # Block fields — inline definition or registry reference.
    block: BlockDefinition | None = None
    block_ref: str | None = None

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Node id cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_node_type(self) -> NodeSchema:
        """Ensure the node type is valid or is a recognized special type."""
        t = self.type

        # Custom layers
        if t == "custom":
            if not self.path:
                raise ValueError(f"Node '{self.id}': custom type requires 'path' field")
            return self

        # Block nodes
        if t == "Block":
            if self.block is None and self.block_ref is None:
                raise ValueError(
                    f"Node '{self.id}': Block type requires either 'block' "
                    f"(inline definition) or 'block_ref' (registry reference)."
                )
            return self

        # HuggingFace nodes
        if self.hf_model is not None or self.hf_config is not None:
            return self

        # Core types
        if t not in CORE_LAYER_TYPES:
            raise ValueError(
                f"Node '{self.id}': unknown type '{t}'. "
                f"Use a core type, 'custom' with a path, or set hf_model/hf_config."
            )

        # Reduction field not allowed on reduction ops (implicit semantics).
        if t in REDUCTION_OPS and self.reduction is not None:
            raise ValueError(
                f"Node '{self.id}': '{t}' has implicit reduction semantics. "
                f"Remove the 'reduction' field."
            )

        return self


class GraphSchema(BaseModel):
    """Top-level MLnode graph (§1)."""

    mlnode_schema_version: str
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)
    input_shape: list[Dim]
    output_node: str
    nodes: list[NodeSchema]
    edges: list[EdgeSchema]

    @field_validator("mlnode_schema_version")
    @classmethod
    def check_version(cls, v: str) -> str:
        if v not in SUPPORTED_VERSIONS:
            from mlnode.errors import SchemaVersionError

            raise SchemaVersionError(v, SUPPORTED_VERSIONS)
        return v

    @model_validator(mode="after")
    def validate_graph_basics(self) -> GraphSchema:
        """Structural checks that don't require building the DAG."""
        node_ids = {n.id for n in self.nodes}

        # output_node must exist
        if self.output_node not in node_ids:
            raise ValueError(
                f"output_node '{self.output_node}' not found in nodes. "
                f"Available: {sorted(node_ids)}"
            )

        # At least one Input node
        input_nodes = [n for n in self.nodes if n.type == "Input"]
        if not input_nodes:
            raise ValueError("Graph must have at least one Input node.")

        # No duplicate node IDs
        if len(node_ids) != len(self.nodes):
            seen: set[str] = set()
            dupes: set[str] = set()
            for n in self.nodes:
                if n.id in seen:
                    dupes.add(n.id)
                seen.add(n.id)
            raise ValueError(f"Duplicate node IDs: {sorted(dupes)}")

        # All edge endpoints reference existing nodes
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f"Edge source '{edge.source}' not found in nodes.")
            if edge.target not in node_ids:
                raise ValueError(f"Edge target '{edge.target}' not found in nodes.")

        # Weight sharing: shared nodes must have identical type and params (§1.5)
        shared_groups: dict[str, list[NodeSchema]] = {}
        for n in self.nodes:
            if n.shared_id is not None:
                shared_groups.setdefault(n.shared_id, []).append(n)

        for shared_id, group in shared_groups.items():
            ref = group[0]
            for other in group[1:]:
                if other.type != ref.type or other.params != ref.params:
                    raise ValueError(
                        f"Weight-sharing group '{shared_id}': nodes '{ref.id}' and "
                        f"'{other.id}' must have identical type and params."
                    )

        return self

    @property
    def input_nodes(self) -> list[NodeSchema]:
        """Return Input nodes in declaration order (§1.1)."""
        return [n for n in self.nodes if n.type == "Input"]
