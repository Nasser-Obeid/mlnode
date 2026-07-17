"""Tests for blocks — expansion, registry, and end-to-end execution."""

from __future__ import annotations

import pytest
import torch

from mlnode import MLnodeModel, BlockRegistry
from mlnode.blocks.expansion import expand_blocks
from mlnode.blocks.registry import BlockRegistry as BR
from mlnode.errors import MLnodeError
from mlnode.schema.models import BlockDefinition, GraphSchema
from tests.fixtures.graphs import (
    BLOCK_REF_GRAPH,
    CHAINED_BLOCKS_GRAPH,
    INLINE_BLOCK_GRAPH,
    MLP_BLOCK_GRAPH,
    MLP_BLOCK_DEF,
    RESNET_BLOCK_DEF,
)


class TestBlockDefinition:
    def test_valid_block_def(self) -> None:
        block = BlockDefinition.model_validate(RESNET_BLOCK_DEF)
        assert block.input_node == "block_in"
        assert block.output_node == "block_relu"
        assert len(block.nodes) == 8

    def test_invalid_input_node(self) -> None:
        data = {**MLP_BLOCK_DEF, "input_node": "nonexistent"}
        with pytest.raises(Exception, match="not found"):
            BlockDefinition.model_validate(data)

    def test_invalid_output_node(self) -> None:
        data = {**MLP_BLOCK_DEF, "output_node": "nonexistent"}
        with pytest.raises(Exception, match="not found"):
            BlockDefinition.model_validate(data)


class TestBlockRegistry:
    def test_register_and_get(self) -> None:
        registry = BR()
        block = BlockDefinition.model_validate(RESNET_BLOCK_DEF)
        registry.register("resnet_block", block)
        assert registry.has("resnet_block")
        assert registry.get("resnet_block") is block

    def test_duplicate_register_raises(self) -> None:
        registry = BR()
        block = BlockDefinition.model_validate(RESNET_BLOCK_DEF)
        registry.register("resnet_block", block)
        with pytest.raises(MLnodeError, match="already registered"):
            registry.register("resnet_block", block)

    def test_get_missing_raises(self) -> None:
        registry = BR()
        with pytest.raises(MLnodeError, match="not found"):
            registry.get("nonexistent")

    def test_register_from_dict(self) -> None:
        registry = BR()
        registry.register_from_dict("mlp", MLP_BLOCK_DEF)
        assert registry.has("mlp")
        assert len(registry) == 1

    def test_names(self) -> None:
        registry = BR()
        registry.register_from_dict("b_block", MLP_BLOCK_DEF)
        registry.register_from_dict("a_block", RESNET_BLOCK_DEF)
        assert registry.names == ["a_block", "b_block"]


class TestBlockExpansion:
    def test_inline_block_expansion(self) -> None:
        schema = GraphSchema.model_validate(INLINE_BLOCK_GRAPH)
        expanded = expand_blocks(schema)

        # Block node should be gone
        node_types = {n.type for n in expanded.nodes}
        assert "Block" not in node_types

        # Internal nodes should be namespaced
        node_ids = {n.id for n in expanded.nodes}
        assert "res1/block_conv1" in node_ids
        assert "res1/block_bn1" in node_ids
        assert "res1/block_relu" in node_ids

        # Block's input node should NOT be present (it's virtual)
        assert "res1/block_in" not in node_ids

    def test_chained_blocks_expansion(self) -> None:
        schema = GraphSchema.model_validate(CHAINED_BLOCKS_GRAPH)
        expanded = expand_blocks(schema)

        node_ids = {n.id for n in expanded.nodes}
        # Both blocks should be expanded with their own namespace
        assert "res1/block_conv1" in node_ids
        assert "res2/block_conv1" in node_ids
        assert "res1/block_relu" in node_ids
        assert "res2/block_relu" in node_ids

    def test_block_ref_expansion(self) -> None:
        registry = BR()
        registry.register_from_dict("resnet_block", RESNET_BLOCK_DEF)

        schema = GraphSchema.model_validate(BLOCK_REF_GRAPH)
        expanded = expand_blocks(schema, registry)

        node_ids = {n.id for n in expanded.nodes}
        assert "res1/block_conv1" in node_ids
        assert "Block" not in {n.type for n in expanded.nodes}

    def test_output_node_remapped(self) -> None:
        schema = GraphSchema.model_validate(INLINE_BLOCK_GRAPH)
        expanded = expand_blocks(schema)

        # output_node should be updated to the namespaced output
        assert expanded.output_node == "res1/block_relu"

    def test_mlp_block_expansion(self) -> None:
        schema = GraphSchema.model_validate(MLP_BLOCK_GRAPH)
        expanded = expand_blocks(schema)

        node_ids = {n.id for n in expanded.nodes}
        assert "block1/mlp_fc1" in node_ids
        assert "block1/mlp_relu" in node_ids
        assert "block1/mlp_out" in node_ids
        assert "block1/mlp_in" not in node_ids  # Input removed
        assert expanded.output_node == "block1/mlp_out"


class TestBlockEndToEnd:
    """Full pipeline tests: Block JSON → validate → build → forward."""

    def test_mlp_block_forward(self) -> None:
        model = MLnodeModel.from_dict(MLP_BLOCK_GRAPH)
        result = model.validate()
        assert result.valid

        module = model.build()
        x = torch.randn(4, 32)
        out = module(x)
        assert out.shape == (4, 32)

    def test_inline_resnet_block_forward(self) -> None:
        model = MLnodeModel.from_dict(INLINE_BLOCK_GRAPH)
        result = model.validate()
        assert result.valid

        module = model.build()
        x = torch.randn(2, 64, 56, 56)
        out = module(x)
        assert out.shape == (2, 64, 56, 56)

    def test_chained_blocks_forward(self) -> None:
        model = MLnodeModel.from_dict(CHAINED_BLOCKS_GRAPH)
        result = model.validate()
        assert result.valid

        module = model.build()
        x = torch.randn(2, 64, 56, 56)
        out = module(x)
        assert out.shape == (2, 64, 56, 56)

    def test_block_ref_forward(self) -> None:
        registry = BR()
        registry.register_from_dict("resnet_block", RESNET_BLOCK_DEF)

        model = MLnodeModel.from_dict(BLOCK_REF_GRAPH, block_registry=registry)
        result = model.validate()
        assert result.valid

        module = model.build()
        x = torch.randn(2, 64, 56, 56)
        out = module(x)
        assert out.shape == (2, 64, 56, 56)

    def test_gradients_through_block(self) -> None:
        model = MLnodeModel.from_dict(INLINE_BLOCK_GRAPH)
        module = model.build()
        x = torch.randn(2, 64, 56, 56, requires_grad=True)
        out = module(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_block_node_missing_def_and_ref(self) -> None:
        """Block node with neither block nor block_ref should fail."""
        from pydantic import ValidationError as PydanticValidationError
        graph = {
            "mlnode_schema_version": "1.0",
            "input_shape": ["B", 10],
            "output_node": "b1",
            "nodes": [
                {"id": "input", "type": "Input", "params": {}},
                {"id": "b1", "type": "Block", "params": {}},
            ],
            "edges": [{"source": "input", "target": "b1"}],
        }
        with pytest.raises(PydanticValidationError, match="Block type requires"):
            MLnodeModel.from_dict(graph)
