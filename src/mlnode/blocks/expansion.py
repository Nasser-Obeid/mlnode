"""Block expansion — flattens Block nodes into the parent graph.

When a Block node is encountered, its internal sub-graph is inlined
into the parent graph with namespaced IDs: {block_node_id}/{internal_id}.

The block's input node is replaced by the incoming edges to the Block node.
The block's output node's output becomes the Block node's output.

Expansion happens recursively: blocks can contain other blocks.
"""

from __future__ import annotations

from typing import Any

from mlnode.errors import MLnodeError
from mlnode.schema.models import (
    BlockDefinition,
    EdgeSchema,
    GraphSchema,
    NodeSchema,
)

# Maximum nesting depth to prevent infinite recursion
MAX_BLOCK_DEPTH = 10


def expand_blocks(
    schema: GraphSchema,
    registry: Any | None = None,
    _depth: int = 0,
) -> GraphSchema:
    """Expand all Block nodes in the graph into flat nodes and edges.

    Args:
        schema: The graph schema potentially containing Block nodes.
        registry: Optional BlockRegistry for resolving block_ref references.
        _depth: Current recursion depth (internal).

    Returns:
        A new GraphSchema with all Block nodes expanded.
    """
    if _depth > MAX_BLOCK_DEPTH:
        raise MLnodeError(
            f"Block nesting depth exceeded {MAX_BLOCK_DEPTH}. "
            f"Check for circular block references."
        )

    block_nodes = [n for n in schema.nodes if n.type == "Block"]
    if not block_nodes:
        return schema  # Nothing to expand

    # Work on copies
    new_nodes: list[NodeSchema] = []
    new_edges: list[EdgeSchema] = []
    block_node_ids = {n.id for n in block_nodes}

    # Keep all non-block nodes
    for node in schema.nodes:
        if node.type != "Block":
            new_nodes.append(node)

    # Keep edges that don't touch block nodes; we'll reconnect them
    edges_into_blocks: dict[str, list[EdgeSchema]] = {bid: [] for bid in block_node_ids}
    edges_from_blocks: dict[str, list[EdgeSchema]] = {bid: [] for bid in block_node_ids}
    # Edges where BOTH source and target are blocks
    inter_block_edges: list[EdgeSchema] = []

    for edge in schema.edges:
        src_is_block = edge.source in block_node_ids
        tgt_is_block = edge.target in block_node_ids

        if src_is_block and tgt_is_block:
            inter_block_edges.append(edge)
        elif tgt_is_block:
            edges_into_blocks[edge.target].append(edge)
        elif src_is_block:
            edges_from_blocks[edge.source].append(edge)
        else:
            new_edges.append(edge)

    # Expand each block
    for block_node in block_nodes:
        block_def = _resolve_block_def(block_node, registry)
        prefix = block_node.id

        # Namespace internal node IDs
        internal_nodes, internal_edges = _namespace_block(
            block_def, prefix
        )

        # Add internal nodes (excluding the block's input node — it's virtual)
        ns_input_id = f"{prefix}/{block_def.input_node}"
        ns_output_id = f"{prefix}/{block_def.output_node}"

        for inode in internal_nodes:
            if inode.id == ns_input_id:
                continue  # Skip the block's input node; replaced by external edges
            new_nodes.append(inode)

        # Add internal edges, remapping references to the block's input node
        for iedge in internal_edges:
            if iedge.source == ns_input_id:
                continue  # These will be replaced by external edges below
            new_edges.append(iedge)

        # Connect external edges coming INTO the block → first internal node(s)
        # (edges that targeted the block node now target the first internal
        # nodes after the block input node)
        first_internal_targets = [
            iedge.target for iedge in internal_edges
            if iedge.source == ns_input_id
        ]

        for ext_edge in edges_into_blocks[block_node.id]:
            for int_target in first_internal_targets:
                # Find the internal edge to copy port info
                matching_internal = [
                    ie for ie in internal_edges
                    if ie.source == ns_input_id and ie.target == int_target
                ]
                target_port = matching_internal[0].target_port if matching_internal else None
                new_edges.append(EdgeSchema(
                    source=ext_edge.source,
                    source_port=ext_edge.source_port,
                    target=int_target,
                    target_port=target_port,
                ))

        # Connect block's output node → external edges going FROM the block
        for ext_edge in edges_from_blocks[block_node.id]:
            new_edges.append(EdgeSchema(
                source=ns_output_id,
                source_port=ext_edge.source_port,
                target=ext_edge.target,
                target_port=ext_edge.target_port,
            ))

    # Handle inter-block edges: source block's output → target block's first internal nodes
    block_lookup = {n.id: n for n in block_nodes}
    for edge in inter_block_edges:
        src_block = block_lookup[edge.source]
        tgt_block = block_lookup[edge.target]
        src_def = _resolve_block_def(src_block, registry)
        tgt_def = _resolve_block_def(tgt_block, registry)

        src_output_id = f"{edge.source}/{src_def.output_node}"
        tgt_input_id = f"{edge.target}/{tgt_def.input_node}"

        # Find target block's first internal edges (from its input node)
        _, tgt_internal_edges = _namespace_block(tgt_def, edge.target)
        first_targets = [
            ie.target for ie in tgt_internal_edges
            if ie.source == tgt_input_id
        ]

        for int_target in first_targets:
            matching = [
                ie for ie in tgt_internal_edges
                if ie.source == tgt_input_id and ie.target == int_target
            ]
            target_port = matching[0].target_port if matching else None
            new_edges.append(EdgeSchema(
                source=src_output_id,
                source_port=edge.source_port,
                target=int_target,
                target_port=target_port,
            ))

    # Update output_node if it was a block
    output_node = schema.output_node
    if output_node in block_node_ids:
        block_node = next(n for n in block_nodes if n.id == output_node)
        block_def = _resolve_block_def(block_node, registry)
        output_node = f"{block_node.id}/{block_def.output_node}"

    # Build expanded schema
    expanded = GraphSchema(
        mlnode_schema_version=schema.mlnode_schema_version,
        metadata=schema.metadata,
        input_shape=schema.input_shape,
        output_node=output_node,
        nodes=new_nodes,
        edges=new_edges,
    )

    # Recurse for nested blocks
    return expand_blocks(expanded, registry, _depth + 1)


def _resolve_block_def(
    node: NodeSchema, registry: Any | None
) -> BlockDefinition:
    """Get the BlockDefinition from either inline or registry reference."""
    if node.block is not None:
        return node.block

    if node.block_ref is not None:
        if registry is None:
            from mlnode.blocks.registry import get_default_registry
            registry = get_default_registry()
        return registry.get(node.block_ref)

    raise MLnodeError(
        f"Block node '{node.id}' has neither 'block' nor 'block_ref'."
    )


def _namespace_block(
    block_def: BlockDefinition, prefix: str
) -> tuple[list[NodeSchema], list[EdgeSchema]]:
    """Create namespaced copies of a block's nodes and edges.

    All IDs are prefixed with '{prefix}/'.
    """
    nodes: list[NodeSchema] = []
    for node in block_def.nodes:
        # Deep copy and update the ID
        node_data = node.model_dump()
        node_data["id"] = f"{prefix}/{node.id}"

        # If it's an Input node inside a block, keep it as Input type
        # (it'll be removed during expansion, just used for wiring)
        nodes.append(NodeSchema.model_validate(node_data))

    edges: list[EdgeSchema] = []
    for edge in block_def.edges:
        edges.append(EdgeSchema(
            source=f"{prefix}/{edge.source}",
            source_port=edge.source_port,
            target=f"{prefix}/{edge.target}",
            target_port=edge.target_port,
        ))

    return nodes, edges
