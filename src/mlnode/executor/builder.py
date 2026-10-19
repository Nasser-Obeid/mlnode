"""Module builder — converts a ParsedGraph into an executable nn.Module (spec §2).

Pipeline: ParsedGraph → MLnodeModule(nn.Module)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mlnode.errors import MLnodeError
from mlnode.layers.registry import build_layer
from mlnode.parser.dag import ParsedGraph, ParsedNode
from mlnode.schema.types import (
    ELEMENTWISE_OPS,
    MULTI_INPUT_LAYERS,
    REDUCTION_OPS,
    TENSOR_OPS,
    ReductionType,
)
from mlnode.utils import resolve_multi_inputs


class MLnodeModule(nn.Module):
    """A dynamically constructed nn.Module from a MLnode DAG."""

    def __init__(self, graph: ParsedGraph) -> None:
        super().__init__()
        self.graph = graph
        self._output_node_id = graph.schema.output_node
        self._execution_order = graph.execution_order
        self._nodes = graph.nodes

        # Build layers and register as submodules.
        self._layers = nn.ModuleDict()
        self._shared_layers: dict[str, nn.Module] = {}

        for node_id in self._execution_order:
            node = self._nodes[node_id]

            if node.type == "Input":
                continue

            # Skip reduction/tensor ops that don't have learnable parameters
            if node.type in REDUCTION_OPS or node.type in TENSOR_OPS:
                continue

            # Weight sharing
            shared_id = node.schema.shared_id
            if shared_id is not None and shared_id in self._shared_layers:
                self._layers[node_id] = self._shared_layers[shared_id]
                continue

            # Build the layer — JSON custom layers become nested MLnodeModules
            if node_id in graph.custom_layers:
                layer = MLnodeModule(graph.custom_layers[node_id])
            else:
                layer = build_layer(node.schema)
            self._layers[node_id] = layer

            if shared_id is not None:
                self._shared_layers[shared_id] = layer

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        """Execute the DAG (spec §2)."""
        input_nodes = self.graph.input_nodes
        if len(args) != len(input_nodes):
            raise MLnodeError(
                f"Expected {len(input_nodes)} input(s), got {len(args)}. "
                f"Input nodes: {[n.id for n in input_nodes]}"
            )

        store: dict[str, torch.Tensor] = {}
        for i, inp_node in enumerate(input_nodes):
            store[inp_node.id] = args[i]

        for node_id in self._execution_order:
            node = self._nodes[node_id]

            if node.type == "Input":
                continue

            # Gather inputs
            incoming = []
            for edge in node.incoming_edges:
                key = edge.source_key
                if key not in store:
                    raise MLnodeError(
                        f"Node '{node_id}': missing tensor for '{key}' in store."
                    )
                incoming.append(store[key])

            # Compute result
            result = self._execute_node(node, incoming)

            # Store outputs
            if node.schema.outputs:
                if isinstance(result, (tuple, list)):
                    for j, port_name in enumerate(node.schema.outputs):
                        store[f"{node_id}:{port_name}"] = result[j]
                else:
                    # Single result with declared ports — error
                    if len(node.schema.outputs) > 1:
                        raise MLnodeError(
                            f"Node '{node_id}' declares {len(node.schema.outputs)} output ports "
                            f"but forward returned a single tensor, not a sequence."
                        )
                    store[f"{node_id}:{node.schema.outputs[0]}"] = result
            else:
                store[node_id] = result

        if self._output_node_id not in store:
            raise MLnodeError(
                f"Output node '{self._output_node_id}' not found in tensor store."
            )
        return store[self._output_node_id]

    def _execute_node(
        self, node: ParsedNode, incoming: list[torch.Tensor]
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Execute a single node."""
        node_type = node.type
        params = node.schema.params

        # Elementwise reduction ops
        if node_type in ELEMENTWISE_OPS:
            result = incoming[0]
            for t in incoming[1:]:
                if node_type == "Add":
                    result = result + t
                elif node_type == "Subtract":
                    result = result - t
                elif node_type == "Multiply":
                    result = result * t
                elif node_type == "Divide":
                    result = result / t
            return result

        # Concat / Stack
        if node_type == "Concat":
            dim = params.get("dim", -1)
            return torch.cat(incoming, dim=dim)

        if node_type == "Stack":
            dim = params.get("dim", 0)
            return torch.stack(incoming, dim=dim)

        # Split
        if node_type == "Split":
            sections = params.get("split_size_or_sections", 2)
            dim = params.get("dim", 0)
            return torch.split(incoming[0], sections, dim=dim)

        # Tensor ops without learnable params
        if node_type == "Flatten":
            start_dim = params.get("start_dim", 1)
            end_dim = params.get("end_dim", -1)
            return torch.flatten(incoming[0], start_dim, end_dim)

        if node_type == "Reshape":
            shape = params["shape"]
            return incoming[0].reshape(*shape)

        if node_type == "View":
            shape = params["shape"]
            return incoming[0].view(*shape)

        if node_type == "Permute":
            dims = params["dims"]
            return incoming[0].permute(*dims)

        if node_type == "Transpose":
            dim0 = params.get("dim0", 0)
            dim1 = params.get("dim1", 1)
            return incoming[0].transpose(dim0, dim1)

        if node_type == "Squeeze":
            dim = params.get("dim", None)
            return incoming[0].squeeze(dim) if dim is not None else incoming[0].squeeze()

        if node_type == "Unsqueeze":
            dim = params["dim"]
            return incoming[0].unsqueeze(dim)

        # Layers with several positional inputs (Transformer, TransformerDecoder,
        # MultiheadAttention, Bilinear, …): order tensors by target_port and
        # pass them as separate arguments — never reduced.
        if node_type in MULTI_INPUT_LAYERS:
            layer = self._layers[node.id] if node.id in self._layers else None
            if layer is None:
                raise MLnodeError(
                    f"Node '{node.id}' ({node_type}): no layer found. "
                    f"This type may not be registered."
                )
            edge_ports = [e.target_port for e in node.incoming_edges]
            args = resolve_multi_inputs(node.id, node_type, edge_ports, incoming)
            result = layer(*args)
            if isinstance(result, tuple) and node_type in _TUPLE_OUTPUT_LAYERS:
                result = result[0]
            return result

        # JSON custom layers with several Input nodes: bind by target_port
        # (= the layer's Input node ids) or declaration order.
        if node.id in self.graph.custom_layers:
            ports = tuple(n.id for n in self.graph.custom_layers[node.id].input_nodes)
            if len(ports) > 1:
                edge_ports = [e.target_port for e in node.incoming_edges]
                args = resolve_multi_inputs(
                    node.id, node_type, edge_ports, incoming, ports=ports
                )
                return self._layers[node.id](*args)

        # Fallback reduction for non-reduction-op nodes with >1 input
        if len(incoming) > 1:
            reduction = node.schema.reduction
            if reduction is None:
                raise MLnodeError(
                    f"Node '{node.id}' received {len(incoming)} inputs but has no "
                    f"reduction strategy. Set 'reduction' field."
                )
            reduction_dim = node.schema.reduction_dim if node.schema.reduction_dim is not None else -1
            if reduction == ReductionType.CONCAT:
                combined = torch.cat(incoming, dim=reduction_dim)
            elif reduction == ReductionType.ADD:
                combined = sum(incoming[1:], incoming[0])
            elif reduction == ReductionType.STACK:
                combined = torch.stack(incoming, dim=reduction_dim)
            else:
                raise MLnodeError(f"Unknown reduction type: {reduction}")
            incoming = [combined]

        # Standard layer forward
        layer = self._layers[node.id] if node.id in self._layers else None
        if layer is None:
            raise MLnodeError(
                f"Node '{node.id}' ({node_type}): no layer found. "
                f"This type may not be registered."
            )
        result = layer(incoming[0])

        # Some layers return tuples; extract the primary output tensor.
        # RNN/LSTM/GRU → (output, hidden_state), MHA → (attn_output, attn_weights)
        if isinstance(result, tuple) and node_type in _TUPLE_OUTPUT_LAYERS:
            result = result[0]

        return result


# Layer types whose forward() returns a tuple — we extract index 0.
_TUPLE_OUTPUT_LAYERS = frozenset({
    "RNN", "LSTM", "GRU", "MultiheadAttention",
})


class ModuleBuilder:
    """Builds an nn.Module from a ParsedGraph."""

    @staticmethod
    def build(graph: ParsedGraph) -> nn.Module:
        """Construct the executable MLnodeModule."""
        return MLnodeModule(graph)
