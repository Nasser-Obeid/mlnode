"""Integration tests — full pipeline: JSON → validate → build → forward."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from mlnode import MLnodeModel
from mlnode.errors import CycleDetectedError, SchemaVersionError, ValidationError
from tests.fixtures.graphs import (
    BAD_VERSION_GRAPH,
    CYCLE_GRAPH,
    MINIMAL_LINEAR,
    MULTI_INPUT_GRAPH,
    RESNET_SKIP_BLOCK,
    SPLIT_GRAPH,
    WEIGHT_SHARING_GRAPH,
)


class TestMLnodeModelAPI:
    """Test the public MLnodeModel API (spec §7)."""

    def test_from_dict(self) -> None:
        model = MLnodeModel.from_dict(MINIMAL_LINEAR)
        assert repr(model) == "MLnodeModel(name='Minimal Linear', nodes=2, edges=1)"

    def test_load_and_save(self, tmp_path: Path) -> None:
        # Save
        model = MLnodeModel.from_dict(MINIMAL_LINEAR)
        save_path = tmp_path / "model.mlnode.json"
        model.save(save_path)
        assert save_path.exists()

        # Reload
        loaded = MLnodeModel.load(save_path)
        assert loaded.schema.metadata.name == "Minimal Linear"
        assert len(loaded.schema.nodes) == 2

    def test_validate_valid_graph(self) -> None:
        model = MLnodeModel.from_dict(MINIMAL_LINEAR)
        result = model.validate()
        assert result.valid

    def test_validate_catches_shape_mismatch(self) -> None:
        from tests.fixtures.graphs import SHAPE_MISMATCH_GRAPH
        model = MLnodeModel.from_dict(SHAPE_MISMATCH_GRAPH)
        result = model.validate()
        assert not result.valid

    def test_build_raises_on_invalid(self) -> None:
        from tests.fixtures.graphs import SHAPE_MISMATCH_GRAPH
        model = MLnodeModel.from_dict(SHAPE_MISMATCH_GRAPH)
        with pytest.raises(ValidationError, match="validation error"):
            model.build()

    def test_cycle_rejected_at_parse(self) -> None:
        with pytest.raises(CycleDetectedError):
            MLnodeModel.from_dict(CYCLE_GRAPH)

    def test_bad_version_rejected(self) -> None:
        with pytest.raises(SchemaVersionError, match="v1.0.0"):
            MLnodeModel.from_dict(BAD_VERSION_GRAPH)


class TestEndToEnd:
    """Full forward pass tests."""

    def test_minimal_linear_forward(self) -> None:
        model = MLnodeModel.from_dict(MINIMAL_LINEAR)
        module = model.build()
        x = torch.randn(4, 10)
        out = module(x)
        assert out.shape == (4, 5)

    def test_resnet_skip_forward(self) -> None:
        model = MLnodeModel.from_dict(RESNET_SKIP_BLOCK)
        module = model.build()
        x = torch.randn(2, 64, 56, 56)
        out = module(x)
        assert out.shape == (2, 64, 56, 56)

    def test_multi_input_forward(self) -> None:
        model = MLnodeModel.from_dict(MULTI_INPUT_GRAPH)
        module = model.build()
        x1 = torch.randn(4, 32)
        x2 = torch.randn(4, 32)
        out = module(x1, x2)
        assert out.shape == (4, 16)

    def test_split_and_concat_forward(self) -> None:
        model = MLnodeModel.from_dict(SPLIT_GRAPH)
        module = model.build()
        x = torch.randn(4, 64)
        out = module(x)
        assert out.shape == (4, 10)

    def test_weight_sharing_same_module(self) -> None:
        model = MLnodeModel.from_dict(WEIGHT_SHARING_GRAPH)
        module = model.build()
        # fc1 and fc2 should be the exact same nn.Module instance
        fc1 = module._layers["fc1"]
        fc2 = module._layers["fc2"]
        assert fc1 is fc2

    def test_weight_sharing_forward(self) -> None:
        model = MLnodeModel.from_dict(WEIGHT_SHARING_GRAPH)
        module = model.build()
        x = torch.randn(4, 10)
        out = module(x)
        assert out.shape == (4, 10)

    def test_gradient_flows(self) -> None:
        """Verify gradients propagate through the DAG."""
        model = MLnodeModel.from_dict(RESNET_SKIP_BLOCK)
        module = model.build()
        x = torch.randn(2, 64, 56, 56, requires_grad=True)
        out = module(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
