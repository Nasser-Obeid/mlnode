"""Tests for custom layer validation (spec §4)."""

from __future__ import annotations

import pytest

from mlnode.errors import CustomLayerError
from mlnode.layers.custom import (
    CustomLayerResult,
    clear_cache,
    validate_custom_layer,
    resolve_custom_class,
)
from mlnode import MLnodeModel


class TestResolveCustomClass:
    def test_valid_import(self) -> None:
        cls = resolve_custom_class("tests.fixtures.custom_layers.ScaleLayer", "test")
        from tests.fixtures.custom_layers import ScaleLayer
        assert cls is ScaleLayer

    def test_invalid_path_format(self) -> None:
        with pytest.raises(CustomLayerError, match="invalid custom path"):
            resolve_custom_class("NoDotsHere", "test")

    def test_module_not_found(self) -> None:
        with pytest.raises(CustomLayerError, match="could not import"):
            resolve_custom_class("nonexistent.module.Layer", "test")

    def test_class_not_found(self) -> None:
        with pytest.raises(CustomLayerError, match="not found in"):
            resolve_custom_class("tests.fixtures.custom_layers.DoesNotExist", "test")

    def test_not_nn_module(self) -> None:
        with pytest.raises(CustomLayerError, match="not a subclass of nn.Module"):
            resolve_custom_class("tests.fixtures.custom_layers.BadLayer", "test")


class TestValidateCustomLayer:
    def setup_method(self) -> None:
        clear_cache()

    def test_scale_layer_validation(self) -> None:
        result = validate_custom_layer(
            "tests.fixtures.custom_layers.ScaleLayer",
            {"scale": 2.0},
            ("B", 10),
            "test_node",
        )
        assert isinstance(result, CustomLayerResult)
        assert result.output_shape == ("B", 10)  # passthrough shape

    def test_bottleneck_layer_validation(self) -> None:
        result = validate_custom_layer(
            "tests.fixtures.custom_layers.BottleneckProjection",
            {"in_features": 32, "bottleneck": 8},
            ("B", 32),
            "test_node",
        )
        assert result.output_shape == ("B", 32)  # projects back to in_features

    def test_caching(self) -> None:
        r1 = validate_custom_layer(
            "tests.fixtures.custom_layers.ScaleLayer",
            {"scale": 1.0},
            ("B", 10),
            "node1",
        )
        r2 = validate_custom_layer(
            "tests.fixtures.custom_layers.ScaleLayer",
            {"scale": 1.0},
            ("B", 10),
            "node2",
        )
        assert r1.instance is r2.instance  # Same cached instance

    def test_forward_crash_caught(self) -> None:
        with pytest.raises(CustomLayerError, match="dummy forward failed"):
            validate_custom_layer(
                "tests.fixtures.custom_layers.CrashyLayer",
                {},
                ("B", 10),
                "test_node",
            )

    def test_bad_init_params(self) -> None:
        with pytest.raises(CustomLayerError, match="failed to instantiate"):
            validate_custom_layer(
                "tests.fixtures.custom_layers.BadInitLayer",
                {"required_param": "wrong"},
                ("B", 10),
                "test_node",
            )

    def test_symbolic_shape_preserved(self) -> None:
        result = validate_custom_layer(
            "tests.fixtures.custom_layers.ScaleLayer",
            {"scale": 1.0},
            ("B", "Seq", 64),
            "test_node",
        )
        # B and Seq should be preserved as symbolic
        assert result.output_shape[0] == "B"
        assert result.output_shape[1] == "Seq"
        assert result.output_shape[2] == 64


class TestCustomLayerInGraph:
    """Test custom layers used inside a full graph."""

    def test_custom_layer_in_graph(self) -> None:
        graph = {
            "mlnode_schema_version": "1.0",
            "input_shape": ["B", 10],
            "output_node": "scaled",
            "nodes": [
                {"id": "input", "type": "Input", "params": {}},
                {"id": "scaled", "type": "custom",
                 "path": "tests.fixtures.custom_layers.ScaleLayer",
                 "params": {"scale": 3.0}},
            ],
            "edges": [{"source": "input", "target": "scaled"}],
        }
        model = MLnodeModel.from_dict(graph)
        result = model.validate()
        assert result.valid

        import torch
        module = model.build()
        x = torch.ones(4, 10)
        out = module(x)
        assert out.shape == (4, 10)

    def test_custom_bottleneck_in_graph(self) -> None:
        graph = {
            "mlnode_schema_version": "1.0",
            "input_shape": ["B", 32],
            "output_node": "bottleneck",
            "nodes": [
                {"id": "input", "type": "Input", "params": {}},
                {"id": "bottleneck", "type": "custom",
                 "path": "tests.fixtures.custom_layers.BottleneckProjection",
                 "params": {"in_features": 32, "bottleneck": 8}},
            ],
            "edges": [{"source": "input", "target": "bottleneck"}],
        }
        model = MLnodeModel.from_dict(graph)
        module = model.build()

        import torch
        x = torch.randn(2, 32)
        out = module(x)
        assert out.shape == (2, 32)
