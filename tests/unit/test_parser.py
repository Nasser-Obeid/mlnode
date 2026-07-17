"""Tests for DAG parser — topo sort, cycle detection, port validation."""

from __future__ import annotations

import pytest

from mlnode.errors import CycleDetectedError, ValidationError
from mlnode.parser.dag import DAGParser
from tests.fixtures.graphs import (
    CYCLE_GRAPH,
    MINIMAL_LINEAR,
    MULTI_INPUT_GRAPH,
    RESNET_SKIP_BLOCK,
    SELF_LOOP_GRAPH,
    SPLIT_GRAPH,
)


class TestDAGParser:
    def test_minimal_linear(self) -> None:
        graph = DAGParser.parse_dict(MINIMAL_LINEAR)
        assert graph.execution_order == ["input", "fc"]
        assert len(graph.nodes) == 2

    def test_resnet_topo_order(self) -> None:
        graph = DAGParser.parse_dict(RESNET_SKIP_BLOCK)
        order = graph.execution_order
        # Input must come first
        assert order[0] == "input"
        # conv1 before conv2
        assert order.index("conv1") < order.index("conv2")
        # Both conv2 and input before skip (Add)
        assert order.index("conv2") < order.index("skip")
        # skip before relu
        assert order.index("skip") < order.index("relu")

    def test_cycle_detected(self) -> None:
        with pytest.raises(CycleDetectedError):
            DAGParser.parse_dict(CYCLE_GRAPH)

    def test_self_loop_detected(self) -> None:
        with pytest.raises(CycleDetectedError):
            DAGParser.parse_dict(SELF_LOOP_GRAPH)

    def test_multi_input_graph(self) -> None:
        graph = DAGParser.parse_dict(MULTI_INPUT_GRAPH)
        input_nodes = graph.input_nodes
        assert len(input_nodes) == 2
        assert input_nodes[0].id == "enc_input"
        assert input_nodes[1].id == "dec_input"

    def test_split_graph_port_edges(self) -> None:
        graph = DAGParser.parse_dict(SPLIT_GRAPH)
        split_node = graph.nodes["split"]
        assert len(split_node.outgoing_edges) == 2
        ports = {e.source_port for e in split_node.outgoing_edges}
        assert ports == {"left", "right"}

    def test_invalid_source_port(self) -> None:
        import copy
        data = copy.deepcopy(SPLIT_GRAPH)
        # Reference a port that doesn't exist
        data["edges"][1]["source_port"] = "nonexistent"
        with pytest.raises(ValidationError, match="source_port"):
            DAGParser.parse_dict(data)

    def test_unreachable_output_node(self) -> None:
        import copy
        data = copy.deepcopy(MINIMAL_LINEAR)
        # Add a disconnected node and set it as output
        data["nodes"].append(
            {"id": "orphan", "type": "Linear", "params": {"in_features": 5, "out_features": 3}}
        )
        data["output_node"] = "orphan"
        with pytest.raises(ValidationError, match="not reachable"):
            DAGParser.parse_dict(data)
