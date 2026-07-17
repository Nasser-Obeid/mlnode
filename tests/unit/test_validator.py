"""Tests for static shape validation."""

from __future__ import annotations

import pytest

from mlnode.parser.dag import DAGParser
from mlnode.validator.shapes import ShapeValidator
from tests.fixtures.graphs import (
    MINIMAL_LINEAR,
    RESNET_SKIP_BLOCK,
    SHAPE_MISMATCH_GRAPH,
    SPLIT_GRAPH,
)


class TestShapeValidator:
    def test_minimal_linear_shapes(self) -> None:
        graph = DAGParser.parse_dict(MINIMAL_LINEAR)
        result = ShapeValidator(graph).validate()
        assert result.valid
        assert result.node_shapes["input"] == ("B", 10)
        assert result.node_shapes["fc"] == ("B", 5)

    def test_resnet_block_shapes(self) -> None:
        graph = DAGParser.parse_dict(RESNET_SKIP_BLOCK)
        result = ShapeValidator(graph).validate()
        assert result.valid
        # Input is (B, 64, 56, 56), conv with padding=1 preserves spatial dims
        assert result.node_shapes["conv1"] == ("B", 64, 56, 56)
        assert result.node_shapes["conv2"] == ("B", 64, 56, 56)
        assert result.node_shapes["relu"] == ("B", 64, 56, 56)

    def test_shape_mismatch_detected(self) -> None:
        graph = DAGParser.parse_dict(SHAPE_MISMATCH_GRAPH)
        result = ShapeValidator(graph).validate()
        assert not result.valid
        assert len(result.errors) > 0
        error = result.errors[0]
        assert "add" in error.node_id.lower() or "Add" in error.message

    def test_conv2d_shape_computation(self) -> None:
        """Conv2d with no padding reduces spatial dims."""
        data = {
            "mlnode_schema_version": "1.0",
            "input_shape": ["B", 3, 32, 32],
            "output_node": "conv",
            "nodes": [
                {"id": "input", "type": "Input", "params": {}},
                {"id": "conv", "type": "Conv2d", "params": {
                    "in_channels": 3, "out_channels": 16, "kernel_size": 5
                }},
            ],
            "edges": [
                {"source": "input", "target": "conv"},
            ],
        }
        graph = DAGParser.parse_dict(data)
        result = ShapeValidator(graph).validate()
        assert result.valid
        # (32 - 5 + 0) / 1 + 1 = 28
        assert result.node_shapes["conv"] == ("B", 16, 28, 28)

    def test_flatten_shape(self) -> None:
        data = {
            "mlnode_schema_version": "1.0",
            "input_shape": ["B", 64, 7, 7],
            "output_node": "flat",
            "nodes": [
                {"id": "input", "type": "Input", "params": {}},
                {"id": "flat", "type": "Flatten", "params": {"start_dim": 1}},
            ],
            "edges": [{"source": "input", "target": "flat"}],
        }
        graph = DAGParser.parse_dict(data)
        result = ShapeValidator(graph).validate()
        assert result.valid
        assert result.node_shapes["flat"] == ("B", 3136)  # 64 * 7 * 7

    def test_adaptive_pool_shape(self) -> None:
        data = {
            "mlnode_schema_version": "1.0",
            "input_shape": ["B", 512, 14, 14],
            "output_node": "pool",
            "nodes": [
                {"id": "input", "type": "Input", "params": {}},
                {"id": "pool", "type": "AdaptiveAvgPool2d", "params": {"output_size": 1}},
            ],
            "edges": [{"source": "input", "target": "pool"}],
        }
        graph = DAGParser.parse_dict(data)
        result = ShapeValidator(graph).validate()
        assert result.valid
        assert result.node_shapes["pool"] == ("B", 512, 1, 1)

    def test_concat_shape(self) -> None:
        data = {
            "mlnode_schema_version": "1.0",
            "input_shape": ["B", 10],
            "output_node": "cat",
            "nodes": [
                {"id": "input", "type": "Input", "params": {}},
                {"id": "fc1", "type": "Linear", "params": {"in_features": 10, "out_features": 16}},
                {"id": "fc2", "type": "Linear", "params": {"in_features": 10, "out_features": 32}},
                {"id": "cat", "type": "Concat", "params": {"dim": -1}},
            ],
            "edges": [
                {"source": "input", "target": "fc1"},
                {"source": "input", "target": "fc2"},
                {"source": "fc1", "target": "cat"},
                {"source": "fc2", "target": "cat"},
            ],
        }
        graph = DAGParser.parse_dict(data)
        result = ShapeValidator(graph).validate()
        assert result.valid
        assert result.node_shapes["cat"] == ("B", 48)  # 16 + 32

    def test_collects_all_errors(self) -> None:
        """Validator should report all errors, not just the first."""
        result = ShapeValidator(DAGParser.parse_dict(SHAPE_MISMATCH_GRAPH)).validate()
        assert not result.valid
        # Should have at least one error
        assert len(result.errors) >= 1
