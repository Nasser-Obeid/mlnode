"""Tests for expanded shape transforms — Conv1d/3d, ConvTranspose, sequence, attention, tensor ops."""

from __future__ import annotations

import pytest

from mlnode.parser.dag import DAGParser
from mlnode.validator.shapes import ShapeValidator


def _make_graph(input_shape, node_type, node_params, node_id="layer"):
    """Helper to create a minimal graph for shape testing."""
    return {
        "mlnode_schema_version": "1.0",
        "input_shape": input_shape,
        "output_node": node_id,
        "nodes": [
            {"id": "input", "type": "Input", "params": {}},
            {"id": node_id, "type": node_type, "params": node_params},
        ],
        "edges": [{"source": "input", "target": node_id}],
    }


def _validate(data):
    graph = DAGParser.parse_dict(data)
    return ShapeValidator(graph).validate()


class TestConv1dShape:
    def test_basic(self) -> None:
        r = _validate(_make_graph(
            ["B", 3, 100], "Conv1d",
            {"in_channels": 3, "out_channels": 16, "kernel_size": 5}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 16, 96)  # 100 - 5 + 1

    def test_with_padding(self) -> None:
        r = _validate(_make_graph(
            ["B", 3, 100], "Conv1d",
            {"in_channels": 3, "out_channels": 16, "kernel_size": 5, "padding": 2}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 16, 100)  # same padding

    def test_with_stride(self) -> None:
        r = _validate(_make_graph(
            ["B", 3, 100], "Conv1d",
            {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "stride": 2}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 16, 49)  # floor((100-3)/2)+1


class TestConv3dShape:
    def test_basic(self) -> None:
        r = _validate(_make_graph(
            ["B", 3, 16, 16, 16], "Conv3d",
            {"in_channels": 3, "out_channels": 8, "kernel_size": 3}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 8, 14, 14, 14)


class TestConvTransposeShape:
    def test_conv_transpose_1d(self) -> None:
        r = _validate(_make_graph(
            ["B", 16, 10], "ConvTranspose1d",
            {"in_channels": 16, "out_channels": 8, "kernel_size": 3, "stride": 2}
        ))
        assert r.valid
        # (10-1)*2 - 0 + 1*(3-1) + 0 + 1 = 21
        assert r.node_shapes["layer"] == ("B", 8, 21)

    def test_conv_transpose_2d(self) -> None:
        r = _validate(_make_graph(
            ["B", 16, 7, 7], "ConvTranspose2d",
            {"in_channels": 16, "out_channels": 8, "kernel_size": 4, "stride": 2, "padding": 1}
        ))
        assert r.valid
        # (7-1)*2 - 2 + 1*(4-1) + 0 + 1 = 14
        assert r.node_shapes["layer"] == ("B", 8, 14, 14)

    def test_conv_transpose_3d(self) -> None:
        r = _validate(_make_graph(
            ["B", 16, 4, 4, 4], "ConvTranspose3d",
            {"in_channels": 16, "out_channels": 8, "kernel_size": 3, "stride": 2}
        ))
        assert r.valid
        # (4-1)*2 - 0 + 1*(3-1) + 0 + 1 = 9
        assert r.node_shapes["layer"] == ("B", 8, 9, 9, 9)


class TestPoolingShapes:
    def test_maxpool1d(self) -> None:
        r = _validate(_make_graph(
            ["B", 64, 100], "MaxPool1d",
            {"kernel_size": 2}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 64, 50)

    def test_avgpool1d(self) -> None:
        r = _validate(_make_graph(
            ["B", 64, 100], "AvgPool1d",
            {"kernel_size": 4, "stride": 4}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 64, 25)

    def test_maxpool3d(self) -> None:
        r = _validate(_make_graph(
            ["B", 64, 8, 8, 8], "MaxPool3d",
            {"kernel_size": 2}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 64, 4, 4, 4)

    def test_adaptive_pool1d(self) -> None:
        r = _validate(_make_graph(
            ["B", 64, 100], "AdaptiveAvgPool1d",
            {"output_size": 1}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 64, 1)

    def test_adaptive_pool3d(self) -> None:
        r = _validate(_make_graph(
            ["B", 64, 8, 8, 8], "AdaptiveAvgPool3d",
            {"output_size": 1}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 64, 1, 1, 1)


class TestNormShapes:
    def test_batchnorm1d(self) -> None:
        r = _validate(_make_graph(["B", 64, 100], "BatchNorm1d", {"num_features": 64}))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 64, 100)

    def test_batchnorm3d(self) -> None:
        r = _validate(_make_graph(
            ["B", 64, 4, 4, 4], "BatchNorm3d", {"num_features": 64}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 64, 4, 4, 4)

    def test_groupnorm(self) -> None:
        r = _validate(_make_graph(
            ["B", 64, 8, 8], "GroupNorm", {"num_groups": 8, "num_channels": 64}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 64, 8, 8)


class TestSequenceShapes:
    def test_embedding(self) -> None:
        r = _validate(_make_graph(
            ["B", "Seq"], "Embedding",
            {"num_embeddings": 10000, "embedding_dim": 256}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", "Seq", 256)

    def test_rnn(self) -> None:
        r = _validate(_make_graph(
            ["B", "Seq", 64], "RNN",
            {"input_size": 64, "hidden_size": 128}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", "Seq", 128)

    def test_rnn_bidirectional(self) -> None:
        r = _validate(_make_graph(
            ["B", "Seq", 64], "RNN",
            {"input_size": 64, "hidden_size": 128, "bidirectional": True}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", "Seq", 256)

    def test_lstm(self) -> None:
        r = _validate(_make_graph(
            ["B", "Seq", 64], "LSTM",
            {"input_size": 64, "hidden_size": 128}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", "Seq", 128)

    def test_gru_bidirectional(self) -> None:
        r = _validate(_make_graph(
            ["B", "Seq", 64], "GRU",
            {"input_size": 64, "hidden_size": 64, "bidirectional": True}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", "Seq", 128)


class TestAttentionShapes:
    def test_multihead_attention(self) -> None:
        r = _validate(_make_graph(
            ["B", "Seq", 512], "MultiheadAttention",
            {"embed_dim": 512, "num_heads": 8}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", "Seq", 512)

    def test_transformer_encoder_layer(self) -> None:
        r = _validate(_make_graph(
            ["B", "Seq", 512], "TransformerEncoderLayer",
            {"d_model": 512, "nhead": 8}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", "Seq", 512)

    def test_transformer_decoder_layer(self) -> None:
        # A decoder layer needs two inputs: tgt and memory (via target_port).
        r = _validate({
            "mlnode_schema_version": "1.0",
            "input_shape": ["B", "Seq", 512],
            "output_node": "layer",
            "nodes": [
                {"id": "input", "type": "Input", "params": {}},
                {"id": "layer", "type": "TransformerDecoderLayer",
                 "params": {"d_model": 512, "nhead": 8}},
            ],
            "edges": [
                {"source": "input", "target": "layer", "target_port": "tgt"},
                {"source": "input", "target": "layer", "target_port": "memory"},
            ],
        })
        assert r.valid
        assert r.node_shapes["layer"] == ("B", "Seq", 512)

    def test_transformer_decoder_layer_missing_memory_is_invalid(self) -> None:
        r = _validate(_make_graph(
            ["B", "Seq", 512], "TransformerDecoderLayer",
            {"d_model": 512, "nhead": 8}
        ))
        assert not r.valid
        assert "memory" in r.errors[0].message


class TestTensorOpShapes:
    def test_reshape(self) -> None:
        r = _validate(_make_graph(
            ["B", 64, 7, 7], "Reshape",
            {"shape": [-1, 3136]}
        ))
        assert r.valid
        # -1 inferred from total elements

    def test_permute(self) -> None:
        r = _validate(_make_graph(
            ["B", "Seq", 512], "Permute",
            {"dims": [1, 0, 2]}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("Seq", "B", 512)

    def test_transpose(self) -> None:
        r = _validate(_make_graph(
            ["B", "Seq", 512], "Transpose",
            {"dim0": 0, "dim1": 1}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("Seq", "B", 512)

    def test_unsqueeze(self) -> None:
        r = _validate(_make_graph(
            ["B", 10], "Unsqueeze",
            {"dim": 1}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 1, 10)

    def test_squeeze(self) -> None:
        r = _validate(_make_graph(
            ["B", 1, 10], "Squeeze",
            {"dim": 1}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 10)

    def test_squeeze_no_dim(self) -> None:
        r = _validate(_make_graph(
            ["B", 1, 10, 1], "Squeeze",
            {}
        ))
        assert r.valid
        assert r.node_shapes["layer"] == ("B", 10)

    def test_split_uneven_sections(self) -> None:
        """Split with different section sizes should give per-port shapes."""
        data = {
            "mlnode_schema_version": "1.0",
            "input_shape": ["B", 100],
            "output_node": "fc",
            "nodes": [
                {"id": "input", "type": "Input", "params": {}},
                {"id": "split", "type": "Split",
                 "params": {"split_size_or_sections": [60, 40], "dim": -1},
                 "outputs": ["big", "small"]},
                {"id": "fc", "type": "Linear", "params": {"in_features": 60, "out_features": 10}},
            ],
            "edges": [
                {"source": "input", "target": "split"},
                {"source": "split", "source_port": "big", "target": "fc"},
            ],
        }
        r = _validate(data)
        assert r.valid
        assert r.node_shapes["split:big"] == ("B", 60)
        assert r.node_shapes["split:small"] == ("B", 40)


class TestStackShape:
    @pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
    def test_matches_torch(self, dim: int) -> None:
        import torch

        data = {
            "mlnode_schema_version": "1.0",
            "input_shape": [2, 5],
            "output_node": "stack",
            "nodes": [
                {"id": "input", "type": "Input", "params": {}},
                {"id": "stack", "type": "Stack", "params": {"dim": dim}},
            ],
            "edges": [
                {"source": "input", "target": "stack"},
                {"source": "input", "target": "stack"},
                {"source": "input", "target": "stack"},
            ],
        }
        r = _validate(data)
        assert r.valid
        x = torch.zeros(2, 5)
        assert r.node_shapes["stack"] == tuple(torch.stack([x, x, x], dim=dim).shape)
