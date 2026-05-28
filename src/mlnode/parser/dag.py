"""DAG parser: builds a NetworkX digraph from a validated GraphSchema.

Responsibilities (spec §2):
- Build directed graph from nodes and edges.
- Detect cycles and self-loops → CycleDetectedError.
- Compute topological execution order.
- Resolve edge ports into the store key format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from mlnode.errors import CycleDetectedError, ValidationError
from mlnode.schema.models import EdgeSchema, GraphSchema, NodeSchema


@dataclass
class ParsedEdge:
    """An edge resolved to store-key references."""

    source: str
    source_port: str | None
    target: str
    target_port: str | None

    @property
    def source_key(self) -> str:
        """Key used in the tensor store for the source output."""
        if self.source_port:
            return f"{self.source}:{self.source_port}"
        return self.source

    @property
    def target_key(self) -> str | None:
        """Named input port on the target, or None for default input."""
        return self.target_port


@dataclass
class ParsedNode:
    """A node enriched with parsed edge information."""

    schema: NodeSchema
    incoming_edges: list[ParsedEdge] = field(default_factory=list)
    outgoing_edges: list[ParsedEdge] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.schema.id

    @property
    def type(self) -> str:
        return self.schema.type


@dataclass
class ParsedGraph:
    """The fully parsed DAG, ready for validation and execution."""

    schema: GraphSchema
    nodes: dict[str, ParsedNode]
    execution_order: list[str]
    nx_graph: nx.DiGraph

    @property
    def input_nodes(self) -> list[ParsedNode]:
        return [self.nodes[n.id] for n in self.schema.input_nodes]


class DAGParser:
    """Parses a MLnode JSON graph into a validated DAG structure."""

    @staticmethod
    def parse_file(path: str | Path, block_registry: Any | None = None) -> ParsedGraph:
        """Load and parse a .mlnode.json file."""
        path = Path(path)
        with path.open() as f:
            data = json.load(f)
        return DAGParser.parse_dict(data, block_registry)

    @staticmethod
    def parse_dict(data: dict[str, Any], block_registry: Any | None = None) -> ParsedGraph:
        """Parse a graph from a dictionary."""
        schema = GraphSchema.model_validate(data)
        return DAGParser.parse_schema(schema, block_registry)

    @staticmethod
    def parse_schema(schema: GraphSchema, block_registry: Any | None = None) -> ParsedGraph:
        """Parse a validated GraphSchema into a DAG.

        Block nodes are expanded into flat nodes before DAG construction.
        """
        # Expand blocks before building the graph
        has_blocks = any(n.type == "Block" for n in schema.nodes)
        if has_blocks:
            from mlnode.blocks.expansion import expand_blocks
            schema = expand_blocks(schema, block_registry)

        # Build node lookup
        nodes: dict[str, ParsedNode] = {}
        for node_schema in schema.nodes:
            nodes[node_schema.id] = ParsedNode(schema=node_schema)

        # Build NetworkX graph and populate edges on nodes
        G = nx.DiGraph()
        G.add_nodes_from(nodes.keys())

        for edge_schema in schema.edges:
            parsed_edge = ParsedEdge(
                source=edge_schema.source,
                source_port=edge_schema.source_port,
                target=edge_schema.target,
                target_port=edge_schema.target_port,
            )

            # Self-loop check
            if edge_schema.source == edge_schema.target:
                raise CycleDetectedError([edge_schema.source, edge_schema.source])

            G.add_edge(edge_schema.source, edge_schema.target)
            nodes[edge_schema.source].outgoing_edges.append(parsed_edge)
            nodes[edge_schema.target].incoming_edges.append(parsed_edge)

        # Cycle detection
        if not nx.is_directed_acyclic_graph(G):
            try:
                cycle = nx.find_cycle(G, orientation="original")
                cycle_nodes = [u for u, _, _ in cycle] + [cycle[-1][1]]
            except nx.NetworkXNoCycle:
                cycle_nodes = ["unknown"]
            raise CycleDetectedError(cycle_nodes)

        # Topological sort
        execution_order = list(nx.topological_sort(G))

        # Validate output_node is reachable from at least one input
        input_ids = {n.id for n in schema.input_nodes}
        reachable = set()
        for input_id in input_ids:
            reachable.update(nx.descendants(G, input_id))
            reachable.add(input_id)

        if schema.output_node not in reachable:
            raise ValidationError(
                f"Output node '{schema.output_node}' is not reachable from any Input node.",
                node_id=schema.output_node,
            )

        # Validate multi-output port references
        for node_id, parsed_node in nodes.items():
            for edge in parsed_node.outgoing_edges:
                if edge.source_port is not None:
                    node_outputs = parsed_node.schema.outputs or []
                    if edge.source_port not in node_outputs:
                        raise ValidationError(
                            f"Edge references source_port '{edge.source_port}' on node "
                            f"'{node_id}', but node declares outputs: {node_outputs}",
                            node_id=node_id,
                        )

        return ParsedGraph(
            schema=schema,
            nodes=nodes,
            execution_order=execution_order,
            nx_graph=G,
        )
