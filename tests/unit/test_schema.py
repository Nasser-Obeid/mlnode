"""Tests for Pydantic schema models."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError as PydanticValidationError

from mlnode.errors import SchemaVersionError
from mlnode.schema.models import EdgeSchema, GraphSchema, NodeSchema
from tests.fixtures.graphs import (
    BAD_VERSION_GRAPH,
    MINIMAL_LINEAR,
    RESNET_SKIP_BLOCK,
    WEIGHT_SHARING_GRAPH,
)


class TestNodeSchema:
    def test_valid_core_type(self) -> None:
        node = NodeSchema(id="fc", type="Linear", params={"in_features": 10, "out_features": 5})
        assert node.id == "fc"
        assert node.type == "Linear"

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(PydanticValidationError, match="empty"):
            NodeSchema(id="  ", type="Linear", params={})

    def test_unknown_type_rejected(self) -> None:
        with pytest.raises(PydanticValidationError, match="unknown type"):
            NodeSchema(id="x", type="NotALayer", params={})

    def test_custom_requires_path(self) -> None:
        with pytest.raises(PydanticValidationError, match="path"):
            NodeSchema(id="x", type="custom", params={})

    def test_custom_with_path_ok(self) -> None:
        node = NodeSchema(id="x", type="custom", path="my.module.Layer", params={})
        assert node.path == "my.module.Layer"

    def test_reduction_on_reduction_op_rejected(self) -> None:
        with pytest.raises(PydanticValidationError, match="implicit reduction"):
            NodeSchema(id="x", type="Add", params={}, reduction="add")

    def test_hf_model_node(self) -> None:
        node = NodeSchema(id="bert", type="Input", hf_model="bert-base-uncased", params={})
        assert node.hf_model == "bert-base-uncased"


class TestEdgeSchema:
    def test_basic_edge(self) -> None:
        edge = EdgeSchema(source="a", target="b")
        assert edge.source_port is None
        assert edge.target_port is None

    def test_edge_with_ports(self) -> None:
        edge = EdgeSchema(source="split", source_port="left", target="fc", target_port=None)
        assert edge.source_port == "left"


class TestGraphSchema:
    def test_valid_minimal(self) -> None:
        graph = GraphSchema.model_validate(MINIMAL_LINEAR)
        assert graph.mlnode_schema_version == "1.0"
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_valid_resnet(self) -> None:
        graph = GraphSchema.model_validate(RESNET_SKIP_BLOCK)
        assert graph.metadata.name == "ResNet Block"
        assert len(graph.nodes) == 5

    def test_bad_version_raises(self) -> None:
        with pytest.raises(SchemaVersionError):
            GraphSchema.model_validate(BAD_VERSION_GRAPH)

    def test_missing_output_node(self) -> None:
        data = copy.deepcopy(MINIMAL_LINEAR)
        data["output_node"] = "nonexistent"
        with pytest.raises(PydanticValidationError, match="not found"):
            GraphSchema.model_validate(data)

    def test_no_input_node(self) -> None:
        data = copy.deepcopy(MINIMAL_LINEAR)
        data["nodes"] = [
            {"id": "fc", "type": "Linear", "params": {"in_features": 10, "out_features": 5}},
        ]
        data["output_node"] = "fc"
        data["edges"] = []
        with pytest.raises(PydanticValidationError, match="Input node"):
            GraphSchema.model_validate(data)

    def test_duplicate_node_ids(self) -> None:
        data = copy.deepcopy(MINIMAL_LINEAR)
        data["nodes"].append(
            {"id": "input", "type": "Input", "params": {}}
        )
        with pytest.raises(PydanticValidationError, match="Duplicate"):
            GraphSchema.model_validate(data)

    def test_edge_references_missing_node(self) -> None:
        data = copy.deepcopy(MINIMAL_LINEAR)
        data["edges"].append(
            {"source": "fc", "target": "nonexistent"}
        )
        with pytest.raises(PydanticValidationError, match="not found"):
            GraphSchema.model_validate(data)

    def test_weight_sharing_valid(self) -> None:
        graph = GraphSchema.model_validate(WEIGHT_SHARING_GRAPH)
        shared = [n for n in graph.nodes if n.shared_id == "shared_fc"]
        assert len(shared) == 2

    def test_weight_sharing_mismatched_params(self) -> None:
        data = copy.deepcopy(WEIGHT_SHARING_GRAPH)
        # Change fc2 to have different out_features
        data["nodes"][2]["params"]["out_features"] = 20
        with pytest.raises(PydanticValidationError, match="identical type and params"):
            GraphSchema.model_validate(data)

    def test_input_nodes_order(self) -> None:
        from tests.fixtures.graphs import MULTI_INPUT_GRAPH
        graph = GraphSchema.model_validate(MULTI_INPUT_GRAPH)
        inputs = graph.input_nodes
        assert inputs[0].id == "enc_input"
        assert inputs[1].id == "dec_input"
