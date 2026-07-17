"""Utility functions for MLnode."""

from __future__ import annotations

from typing import TypeVar

from mlnode.errors import MLnodeError
from mlnode.schema.types import MULTI_INPUT_LAYERS

T = TypeVar("T")


def resolve_multi_inputs(
    node_id: str,
    node_type: str,
    edge_ports: list[str | None],
    values: list[T],
) -> list[T]:
    """Order incoming values for a multi-positional-input layer.

    ``values`` are the incoming tensors (or shapes) in edge declaration order;
    ``edge_ports`` are the matching ``target_port`` labels (None if unlabeled).

    Labeled edges are mapped onto the layer's port list from
    MULTI_INPUT_LAYERS (e.g. ("tgt", "memory") for TransformerDecoder).
    Unlabeled edges are assigned to the ports in declaration order.

    MultiheadAttention allows fewer inputs than ports:
    1 input → query=key=value, 2 inputs → query, key=value.
    All other multi-input layers require exactly one value per port.
    """
    ports = MULTI_INPUT_LAYERS[node_type]
    by_port: dict[str, T] = {}

    labeled = [p for p in edge_ports if p is not None]
    if labeled:
        if len(labeled) != len(edge_ports):
            raise MLnodeError(
                f"Node '{node_id}' ({node_type}): mix of labeled and unlabeled "
                f"input edges. Label all edges with target_port "
                f"({', '.join(ports)}) or none."
            )
        for port, value in zip(edge_ports, values):
            if port not in ports:
                raise MLnodeError(
                    f"Node '{node_id}' ({node_type}): unknown target_port "
                    f"'{port}'. Valid ports: {', '.join(ports)}."
                )
            if port in by_port:
                raise MLnodeError(
                    f"Node '{node_id}' ({node_type}): target_port '{port}' "
                    f"received more than one edge."
                )
            by_port[port] = value  # type: ignore[index]
    else:
        if len(values) > len(ports):
            raise MLnodeError(
                f"Node '{node_id}' ({node_type}): expected at most "
                f"{len(ports)} inputs ({', '.join(ports)}), got {len(values)}."
            )
        for port, value in zip(ports, values):
            by_port[port] = value

    # MultiheadAttention: allow self-attention shorthand.
    if node_type == "MultiheadAttention":
        if "query" not in by_port:
            raise MLnodeError(
                f"Node '{node_id}' (MultiheadAttention): a 'query' input is required."
            )
        if "key" not in by_port and "value" not in by_port:
            by_port["key"] = by_port["query"]
            by_port["value"] = by_port["query"]
        elif "key" not in by_port:
            by_port["key"] = by_port["value"]
        elif "value" not in by_port:
            by_port["value"] = by_port["key"]
    else:
        missing = [p for p in ports if p not in by_port]
        if missing:
            raise MLnodeError(
                f"Node '{node_id}' ({node_type}): missing input(s) "
                f"{', '.join(missing)}. Expected {len(ports)} inputs "
                f"({', '.join(ports)})."
            )

    return [by_port[p] for p in ports]
