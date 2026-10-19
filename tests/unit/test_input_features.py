"""Tests for static input-width checks (in_features, in_channels, …) and
fallback-reduction shape simulation on core layers."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mlnode import MLnodeModel


def _chain(input_shape: list[Any], *layers: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    nodes = [{"id": "input", "type": "Input"}]
    edges = []
    prev = "input"
    for i, (node_type, params) in enumerate(layers):
        node_id = f"n{i}"
        nodes.append({"id": node_id, "type": node_type, "params": params})
        edges.append({"source": prev, "target": node_id})
        prev = node_id
    return {
        "mlnode_schema_version": "1.0",
        "input_shape": input_shape,
        "output_node": prev,
        "nodes": nodes,
        "edges": edges,
    }


def _runtime_fails(data: dict[str, Any], *inputs: torch.Tensor) -> bool:
    """Build without validate() and check PyTorch rejects the input."""
    from mlnode.executor.builder import ModuleBuilder
    from mlnode.parser.dag import DAGParser

    module = ModuleBuilder.build(DAGParser.parse_dict(data))
    try:
        module(*inputs)
    except RuntimeError:
        return True
    return False


class TestInputFeatureChecks:
    def test_linear_mismatch(self) -> None:
        data = _chain(["B", 12], ("Linear", {"in_features": 10, "out_features": 4}))
        result = MLnodeModel.from_dict(data).validate()
        assert not result.valid
        error = result.errors[0]
        assert error.message == (
            "Node 'n0' (Linear, in_features=10) expects (B, 10), got (B, 12) from 'input'"
        )
        assert error.expected_shape == ("B", 10)
        assert error.actual_shape == ("B", 12)
        assert error.source_node_id == "input"
        assert _runtime_fails(data, torch.randn(2, 12))

    def test_conv_mismatch_after_previous_layer(self) -> None:
        data = _chain(
            ["B", 3, 32, 32],
            ("Conv2d", {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "padding": 1}),
            ("Conv2d", {"in_channels": 3, "out_channels": 64, "kernel_size": 3}),
        )
        result = MLnodeModel.from_dict(data).validate()
        assert [e.node_id for e in result.errors] == ["n1"]
        assert "in_channels=3" in result.errors[0].message
        assert result.errors[0].source_node_id == "n0"
        assert _runtime_fails(data, torch.randn(2, 3, 32, 32))

    @pytest.mark.parametrize(
        ("input_shape", "layer"),
        [
            (["B", 8, 16], ("BatchNorm1d", {"num_features": 4})),
            (["B", 8, 4, 4], ("BatchNorm2d", {"num_features": 4})),
            (["B", 8, 4, 4], ("GroupNorm", {"num_groups": 2, "num_channels": 4})),
            (["B", 5, 8], ("LSTM", {"input_size": 4, "hidden_size": 3, "batch_first": True})),
            (["B", 8], ("GRUCell", {"input_size": 4, "hidden_size": 3})),
            (["B", 5, 8], ("LayerNorm", {"normalized_shape": 4})),
            (["B", 5, 8], ("LayerNorm", {"normalized_shape": [4, 8]})),
            (["B", 3, 8, 8], ("ConvTranspose2d",
                              {"in_channels": 4, "out_channels": 2, "kernel_size": 2})),
        ],
    )
    def test_mismatch_detected_and_fails_at_runtime(
        self, input_shape: list[Any], layer: tuple[str, dict[str, Any]]
    ) -> None:
        data = _chain(input_shape, layer)
        result = MLnodeModel.from_dict(data).validate()
        assert not result.valid
        assert "expects" in result.errors[0].message
        concrete = [2 if d == "B" else d for d in input_shape]
        assert _runtime_fails(data, torch.randn(*concrete))

    @pytest.mark.parametrize(
        ("input_shape", "layer"),
        [
            (["B", 10], ("Linear", {"in_features": 10, "out_features": 4})),
            (["B", "D"], ("Linear", {"in_features": 10, "out_features": 4})),
            (["B", "C", 8, 8], ("Conv2d", {"in_channels": 3, "out_channels": 4, "kernel_size": 3})),
            (["B", 5, 8], ("LayerNorm", {"normalized_shape": [5, 8]})),
            (["B", 5, "D"], ("LayerNorm", {"normalized_shape": 8})),
            (["B", 8, 4], ("InstanceNorm1d", {"num_features": 3})),  # warns only in torch
        ],
    )
    def test_matching_or_symbolic_passes(
        self, input_shape: list[Any], layer: tuple[str, dict[str, Any]]
    ) -> None:
        result = MLnodeModel.from_dict(_chain(input_shape, layer)).validate()
        assert result.valid, result.errors


class TestFallbackReduction:
    @staticmethod
    def _two_into_linear(reduction: str | None, in_features: int) -> dict[str, Any]:
        node: dict[str, Any] = {
            "id": "fc", "type": "Linear",
            "params": {"in_features": in_features, "out_features": 3},
        }
        if reduction:
            node["reduction"] = reduction
        return {
            "mlnode_schema_version": "1.0",
            "input_shape": ["B", 8],
            "output_node": "fc",
            "nodes": [
                {"id": "a", "type": "Input"},
                {"id": "b", "type": "Input"},
                node,
            ],
            "edges": [{"source": "a", "target": "fc"}, {"source": "b", "target": "fc"}],
        }

    def test_concat_widens_input(self) -> None:
        model = MLnodeModel.from_dict(self._two_into_linear("concat", 16))
        result = model.validate()
        assert result.valid, result.errors
        assert model.build()(torch.randn(2, 8), torch.randn(2, 8)).shape == (2, 3)

    def test_concat_with_wrong_width_is_caught(self) -> None:
        result = MLnodeModel.from_dict(self._two_into_linear("concat", 8)).validate()
        assert "expects (B, 8), got (B, 16)" in result.errors[0].message

    def test_add_keeps_width(self) -> None:
        assert MLnodeModel.from_dict(self._two_into_linear("add", 8)).validate().valid

    def test_multiple_inputs_without_reduction(self) -> None:
        result = MLnodeModel.from_dict(self._two_into_linear(None, 8)).validate()
        assert "no reduction strategy" in result.errors[0].message
