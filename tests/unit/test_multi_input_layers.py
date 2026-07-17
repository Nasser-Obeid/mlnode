"""End-to-end tests for multi-input layers and the transformer family.

Covers: registry construction, port-ordered execution, shape validation,
and the vision-op fixes (PixelShuffle/PixelUnshuffle run as real layers).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mlnode.model import MLnodeModel


def _model(input_shape, nodes, edges, output_node):
    return MLnodeModel.from_dict({
        "mlnode_schema_version": "1.0",
        "input_shape": input_shape,
        "output_node": output_node,
        "nodes": [{"id": "input", "type": "Input", "params": {}}, *nodes],
        "edges": edges,
    })


class TestTransformerEncoder:
    def test_build_and_forward(self) -> None:
        model = _model(
            ["B", 10, 32],
            [{"id": "enc", "type": "TransformerEncoder",
              "params": {"d_model": 32, "nhead": 4, "dim_feedforward": 64,
                         "batch_first": True, "num_layers": 2, "norm": True}}],
            [{"source": "input", "target": "enc"}],
            "enc",
        )
        assert model.validate().valid
        module = model.build()
        out = module(torch.randn(3, 10, 32))
        assert out.shape == (3, 10, 32)
        # num_layers is respected
        enc = module._layers["enc"]
        assert isinstance(enc, nn.TransformerEncoder)
        assert enc.num_layers == 2
        assert enc.norm is not None

    def test_d_model_mismatch_caught_statically(self) -> None:
        model = _model(
            ["B", 10, 16],
            [{"id": "enc", "type": "TransformerEncoder",
              "params": {"d_model": 32, "nhead": 4, "batch_first": True}}],
            [{"source": "input", "target": "enc"}],
            "enc",
        )
        result = model.validate()
        assert not result.valid
        assert "d_model" in result.errors[0].message


class TestTransformerDecoder:
    def test_build_and_forward_with_ports(self) -> None:
        # tgt comes from the input; memory from an encoder branch.
        model = _model(
            ["B", 10, 32],
            [
                {"id": "enc", "type": "TransformerEncoder",
                 "params": {"d_model": 32, "nhead": 4, "batch_first": True,
                            "num_layers": 1}},
                {"id": "dec", "type": "TransformerDecoder",
                 "params": {"d_model": 32, "nhead": 4, "batch_first": True,
                            "num_layers": 1}},
            ],
            [
                {"source": "input", "target": "enc"},
                # Declared memory-first on purpose: ports must win over order.
                {"source": "enc", "target": "dec", "target_port": "memory"},
                {"source": "input", "target": "dec", "target_port": "tgt"},
            ],
            "dec",
        )
        assert model.validate().valid
        out = model.build()(torch.randn(3, 10, 32))
        assert out.shape == (3, 10, 32)

    def test_unlabeled_edges_use_declaration_order(self) -> None:
        # First edge → tgt, second edge → memory.
        model = _model(
            ["B", 6, 16],
            [
                {"id": "proj", "type": "Linear",
                 "params": {"in_features": 16, "out_features": 16}},
                {"id": "dec", "type": "TransformerDecoderLayer",
                 "params": {"d_model": 16, "nhead": 2, "batch_first": True}},
            ],
            [
                {"source": "input", "target": "dec"},
                {"source": "proj", "target": "dec"},
                {"source": "input", "target": "proj"},
            ],
            "dec",
        )
        assert model.validate().valid
        out = model.build()(torch.randn(2, 6, 16))
        assert out.shape == (2, 6, 16)


class TestFullTransformer:
    def test_src_tgt_ports(self) -> None:
        # Two Input nodes: src and tgt sequences of different lengths.
        model = MLnodeModel.from_dict({
            "mlnode_schema_version": "1.0",
            "input_shape": ["B", 8, 16],
            "output_node": "xf",
            "nodes": [
                {"id": "src_in", "type": "Input", "params": {}},
                {"id": "tgt_in", "type": "Input", "params": {}},
                {"id": "xf", "type": "Transformer",
                 "params": {"d_model": 16, "nhead": 2, "num_encoder_layers": 1,
                            "num_decoder_layers": 1, "dim_feedforward": 32,
                            "batch_first": True}},
            ],
            "edges": [
                {"source": "src_in", "target": "xf", "target_port": "src"},
                {"source": "tgt_in", "target": "xf", "target_port": "tgt"},
            ],
        })
        assert model.validate().valid
        out = model.build()(torch.randn(2, 8, 16), torch.randn(2, 8, 16))
        assert out.shape == (2, 8, 16)


class TestMultiheadAttention:
    def test_self_attention_single_input(self) -> None:
        model = _model(
            ["B", 5, 16],
            [{"id": "attn", "type": "MultiheadAttention",
              "params": {"embed_dim": 16, "num_heads": 2, "batch_first": True}}],
            [{"source": "input", "target": "attn"}],
            "attn",
        )
        assert model.validate().valid
        out = model.build()(torch.randn(2, 5, 16))
        assert out.shape == (2, 5, 16)

    def test_cross_attention_with_ports(self) -> None:
        model = MLnodeModel.from_dict({
            "mlnode_schema_version": "1.0",
            "input_shape": ["B", 5, 16],
            "output_node": "attn",
            "nodes": [
                {"id": "q_in", "type": "Input", "params": {}},
                {"id": "kv_in", "type": "Input", "params": {}},
                {"id": "attn", "type": "MultiheadAttention",
                 "params": {"embed_dim": 16, "num_heads": 2, "batch_first": True}},
            ],
            "edges": [
                {"source": "q_in", "target": "attn", "target_port": "query"},
                {"source": "kv_in", "target": "attn", "target_port": "key"},
                {"source": "kv_in", "target": "attn", "target_port": "value"},
            ],
        })
        assert model.validate().valid
        out = model.build()(torch.randn(2, 5, 16), torch.randn(2, 5, 16))
        assert out.shape == (2, 5, 16)


class TestBilinear:
    def test_two_inputs(self) -> None:
        model = MLnodeModel.from_dict({
            "mlnode_schema_version": "1.0",
            "input_shape": ["B", 8],
            "output_node": "bil",
            "nodes": [
                {"id": "a", "type": "Input", "params": {}},
                {"id": "b", "type": "Input", "params": {}},
                {"id": "bil", "type": "Bilinear",
                 "params": {"in1_features": 8, "in2_features": 8,
                            "out_features": 4}},
            ],
            "edges": [
                {"source": "a", "target": "bil", "target_port": "input1"},
                {"source": "b", "target": "bil", "target_port": "input2"},
            ],
        })
        assert model.validate().valid
        out = model.build()(torch.randn(3, 8), torch.randn(3, 8))
        assert out.shape == (3, 4)


class TestVisionOps:
    def test_pixel_shuffle_actually_shuffles(self) -> None:
        model = _model(
            ["B", 4, 8, 8],
            [{"id": "ps", "type": "PixelShuffle", "params": {"upscale_factor": 2}}],
            [{"source": "input", "target": "ps"}],
            "ps",
        )
        assert model.validate().valid
        out = model.build()(torch.randn(2, 4, 8, 8))
        assert out.shape == (2, 1, 16, 16)  # not identity

    def test_pixel_unshuffle_param_name(self) -> None:
        model = _model(
            ["B", 1, 16, 16],
            [{"id": "pu", "type": "PixelUnshuffle",
              "params": {"downscale_factor": 2}}],
            [{"source": "input", "target": "pu"}],
            "pu",
        )
        assert model.validate().valid
        out = model.build()(torch.randn(2, 1, 16, 16))
        assert out.shape == (2, 4, 8, 8)


class TestRegistryFixes:
    def test_rrelu_maps_to_rrelu(self) -> None:
        from mlnode.layers.registry import _LAYER_REGISTRY
        layer = _LAYER_REGISTRY["RReLU"]({})
        assert isinstance(layer, nn.RReLU)

    def test_lazy_batchnorm_registered(self) -> None:
        from mlnode.layers.registry import _LAYER_REGISTRY
        for name in ("LazyBatchNorm1d", "LazyBatchNorm2d", "LazyBatchNorm3d"):
            assert name in _LAYER_REGISTRY

    def test_rnnbase_removed(self) -> None:
        from mlnode.layers.registry import _LAYER_REGISTRY
        from mlnode.schema.types import CORE_LAYER_TYPES
        assert "RNNBase" not in _LAYER_REGISTRY
        assert "RNNBase" not in CORE_LAYER_TYPES

    def test_lazy_linear_shape_rule(self) -> None:
        model = _model(
            ["B", 32],
            [{"id": "fc", "type": "LazyLinear", "params": {"out_features": 7}}],
            [{"source": "input", "target": "fc"}],
            "fc",
        )
        result = model.validate()
        assert result.valid
        assert result.node_shapes["fc"] == ("B", 7)
        out = model.build()(torch.randn(2, 32))
        assert out.shape == (2, 7)


class TestPortErrors:
    def test_unknown_port_rejected(self) -> None:
        model = _model(
            ["B", 6, 16],
            [{"id": "dec", "type": "TransformerDecoderLayer",
              "params": {"d_model": 16, "nhead": 2, "batch_first": True}}],
            [
                {"source": "input", "target": "dec", "target_port": "tgt"},
                {"source": "input", "target": "dec", "target_port": "bogus"},
            ],
            "dec",
        )
        result = model.validate()
        assert not result.valid
        assert "bogus" in result.errors[0].message

    def test_duplicate_port_rejected(self) -> None:
        model = _model(
            ["B", 6, 16],
            [{"id": "dec", "type": "TransformerDecoderLayer",
              "params": {"d_model": 16, "nhead": 2, "batch_first": True}}],
            [
                {"source": "input", "target": "dec", "target_port": "tgt"},
                {"source": "input", "target": "dec", "target_port": "tgt"},
            ],
            "dec",
        )
        result = model.validate()
        assert not result.valid
        assert "more than one edge" in result.errors[0].message
