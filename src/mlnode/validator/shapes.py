"""Static shape validator (spec §3).

Simulates tensor shapes through the DAG without GPU memory.
Each layer type registers a shape transform rule.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from mlnode.errors import MLnodeError
from mlnode.parser.dag import ParsedGraph, ParsedNode
from mlnode.schema.types import (
    ELEMENTWISE_OPS,
    MULTI_INPUT_LAYERS,
    Dim,
    Shape,
    ShapeError,
    ValidationResult,
)
from mlnode.utils import resolve_multi_inputs

# Type alias for shape transform functions.
# Takes (input_shapes, node_params) → output_shape
ShapeTransformFn = Callable[[list[Shape], dict[str, Any]], Shape]

# Global registry of shape transform rules.
_SHAPE_REGISTRY: dict[str, ShapeTransformFn] = {}


def register_shape(node_type: str) -> Callable[[ShapeTransformFn], ShapeTransformFn]:
    """Decorator to register a shape transform for a node type."""

    def decorator(fn: ShapeTransformFn) -> ShapeTransformFn:
        _SHAPE_REGISTRY[node_type] = fn
        return fn

    return decorator


def get_shape_registry() -> dict[str, ShapeTransformFn]:
    """Return a copy of the shape registry (for custom layer probing)."""
    return dict(_SHAPE_REGISTRY)


def register_custom_shape(node_type: str, fn: ShapeTransformFn) -> None:
    """Register a shape transform for a custom or block node type."""
    _SHAPE_REGISTRY[node_type] = fn


def _resolve_dim(dim: Dim) -> int | None:
    """Return int if concrete, None if symbolic."""
    return dim if isinstance(dim, int) else None


def _to_tuple(val: Any, ndim: int) -> tuple:
    """Normalize scalar-or-tuple param to an ndim-tuple."""
    if isinstance(val, (list, tuple)):
        return tuple(val)
    return (val,) * ndim


def _conv_output_dim(
    input_dim: Dim, kernel: int, stride: int, padding: int, dilation: int
) -> Dim:
    """⌊(input + 2*padding - dilation*(kernel-1) - 1) / stride⌋ + 1"""
    resolved = _resolve_dim(input_dim)
    if resolved is None:
        return input_dim
    return math.floor((resolved + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1


def _conv_transpose_output_dim(
    input_dim: Dim, kernel: int, stride: int, padding: int, dilation: int, output_padding: int,
) -> Dim:
    """(input - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1"""
    resolved = _resolve_dim(input_dim)
    if resolved is None:
        return input_dim
    return (resolved - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1



# ══════════════════════════════════════════════
#  I/O Layer
# ══════════════════════════════════════════════

@register_shape("Input")
def _input_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    raise NotImplementedError("Input shape is injected by the validator, not computed.")



# ══════════════════════════════════════════════
#  Convolutions Layers
# ══════════════════════════════════════════════

@register_shape("Conv1d")
def _conv1d_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]  # (B, C, L)
    out_ch = params["out_channels"]
    k = _to_tuple(params["kernel_size"], 1)
    s = _to_tuple(params.get("stride", 1), 1)
    p = _to_tuple(params.get("padding", 0), 1)
    d = _to_tuple(params.get("dilation", 1), 1)
    l_out = _conv_output_dim(inp[2], k[0], s[0], p[0], d[0])
    return (inp[0], out_ch, l_out)


@register_shape("Conv2d")
def _conv2d_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]  # (B, C, H, W)
    out_ch = params["out_channels"]
    k = _to_tuple(params["kernel_size"], 2)
    s = _to_tuple(params.get("stride", 1), 2)
    p = _to_tuple(params.get("padding", 0), 2)
    d = _to_tuple(params.get("dilation", 1), 2)
    h_out = _conv_output_dim(inp[2], k[0], s[0], p[0], d[0])
    w_out = _conv_output_dim(inp[3], k[1], s[1], p[1], d[1])
    return (inp[0], out_ch, h_out, w_out)


@register_shape("Conv3d")
def _conv3d_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]  # (B, C, D, H, W)
    out_ch = params["out_channels"]
    k = _to_tuple(params["kernel_size"], 3)
    s = _to_tuple(params.get("stride", 1), 3)
    p = _to_tuple(params.get("padding", 0), 3)
    d = _to_tuple(params.get("dilation", 1), 3)
    d_out = _conv_output_dim(inp[2], k[0], s[0], p[0], d[0])
    h_out = _conv_output_dim(inp[3], k[1], s[1], p[1], d[1])
    w_out = _conv_output_dim(inp[4], k[2], s[2], p[2], d[2])
    return (inp[0], out_ch, d_out, h_out, w_out)


@register_shape("ConvTranspose1d")
def _conv_t1d_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]
    out_ch = params["out_channels"]
    k = _to_tuple(params["kernel_size"], 1)
    s = _to_tuple(params.get("stride", 1), 1)
    p = _to_tuple(params.get("padding", 0), 1)
    d = _to_tuple(params.get("dilation", 1), 1)
    op = _to_tuple(params.get("output_padding", 0), 1)
    l_out = _conv_transpose_output_dim(inp[2], k[0], s[0], p[0], d[0], op[0])
    return (inp[0], out_ch, l_out)


@register_shape("ConvTranspose2d")
def _conv_t2d_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]
    out_ch = params["out_channels"]
    k = _to_tuple(params["kernel_size"], 2)
    s = _to_tuple(params.get("stride", 1), 2)
    p = _to_tuple(params.get("padding", 0), 2)
    d = _to_tuple(params.get("dilation", 1), 2)
    op = _to_tuple(params.get("output_padding", 0), 2)
    h_out = _conv_transpose_output_dim(inp[2], k[0], s[0], p[0], d[0], op[0])
    w_out = _conv_transpose_output_dim(inp[3], k[1], s[1], p[1], d[1], op[1])
    return (inp[0], out_ch, h_out, w_out)


@register_shape("ConvTranspose3d")
def _conv_t3d_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]
    out_ch = params["out_channels"]
    k = _to_tuple(params["kernel_size"], 3)
    s = _to_tuple(params.get("stride", 1), 3)
    p = _to_tuple(params.get("padding", 0), 3)
    d = _to_tuple(params.get("dilation", 1), 3)
    op = _to_tuple(params.get("output_padding", 0), 3)
    d_out = _conv_transpose_output_dim(inp[2], k[0], s[0], p[0], d[0], op[0])
    h_out = _conv_transpose_output_dim(inp[3], k[1], s[1], p[1], d[1], op[1])
    w_out = _conv_transpose_output_dim(inp[4], k[2], s[2], p[2], d[2], op[2])
    return (inp[0], out_ch, d_out, h_out, w_out)


@register_shape("LazyConv1d")
@register_shape("LazyConv2d")
@register_shape("LazyConv3d")
@register_shape("LazyConvTranspose1d")
@register_shape("LazyConvTranspose2d")
@register_shape("LazyConvTranspose3d")
@register_shape("Unfold")
@register_shape("Fold")
def _conv_layer_notimplemented_shape(inputs: list[Shape], params: dict[str, Any])-> Shape:
    raise NotImplementedError("work in progress")



# ══════════════════════════════════════════════
#  Pooling Layers
# ══════════════════════════════════════════════

@register_shape("MaxPool1d")
@register_shape("AvgPool1d")
def _pool1d_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]  # (B, C, L)
    k = _to_tuple(params["kernel_size"], 1)
    s = _to_tuple(params.get("stride", k), 1)
    p = _to_tuple(params.get("padding", 0), 1)
    l_out = _conv_output_dim(inp[2], k[0], s[0], p[0], 1)
    return (inp[0], inp[1], l_out)


@register_shape("MaxPool2d")
@register_shape("AvgPool2d")
def _pool2d_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]
    k = _to_tuple(params["kernel_size"], 2)
    s = _to_tuple(params.get("stride", k), 2)
    p = _to_tuple(params.get("padding", 0), 2)
    h_out = _conv_output_dim(inp[2], k[0], s[0], p[0], 1)
    w_out = _conv_output_dim(inp[3], k[1], s[1], p[1], 1)
    return (inp[0], inp[1], h_out, w_out)


@register_shape("MaxPool3d")
@register_shape("AvgPool3d")
def _pool3d_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]
    k = _to_tuple(params["kernel_size"], 3)
    s = _to_tuple(params.get("stride", k), 3)
    p = _to_tuple(params.get("padding", 0), 3)
    d_out = _conv_output_dim(inp[2], k[0], s[0], p[0], 1)
    h_out = _conv_output_dim(inp[3], k[1], s[1], p[1], 1)
    w_out = _conv_output_dim(inp[4], k[2], s[2], p[2], 1)
    return (inp[0], inp[1], d_out, h_out, w_out)


@register_shape("AdaptiveAvgPool1d")
@register_shape("AdaptiveMaxPool1d")
def _adaptive_pool1d_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]
    return (inp[0], inp[1], params["output_size"])


@register_shape("AdaptiveAvgPool2d")
@register_shape("AdaptiveMaxPool2d")
def _adaptive_pool2d_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]
    out = _to_tuple(params["output_size"], 2)
    return (inp[0], inp[1], out[0], out[1])


@register_shape("AdaptiveAvgPool3d")
@register_shape("AdaptiveMaxPool3d")
def _adaptive_pool3d_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]
    out = _to_tuple(params["output_size"], 3)
    return (inp[0], inp[1], out[0], out[1], out[2])


@register_shape("MaxUnpool1d")
@register_shape("MaxUnpool2d")
@register_shape("MaxUnpool3d")
@register_shape("FractionalMaxPool2d")
@register_shape("FractionalMaxPool3d")
@register_shape("LPPool1d")
@register_shape("LPPool2d")
@register_shape("LPPool3d")
def _pool_layer_notimplemented_shape(inputs: list[Shape], params: dict[str, Any])-> Shape:
    raise NotImplementedError("work in progress")



# ══════════════════════════════════════════════
#  Non-linear Activations (weighted sum, nonlinearity) & Others
# ══════════════════════════════════════════════

@register_shape("ELU")
@register_shape("Hardshrink")
@register_shape("Hardsigmoid")
@register_shape("Hardtanh")
@register_shape("Hardswish")
@register_shape("LeakyReLU")
@register_shape("LogSigmoid")
@register_shape("PReLU")
@register_shape("ReLU")
@register_shape("ReLU6")
@register_shape("RReLU")
@register_shape("SELU")
@register_shape("CELU")
@register_shape("GELU")
@register_shape("Sigmoid")
@register_shape("SiLU")
@register_shape("Mish")
@register_shape("Softplus")
@register_shape("Softshrink")
@register_shape("Softsign")
@register_shape("Tanh")
@register_shape("Tanhshrink")
@register_shape("Threshold")
@register_shape("Softmin")
@register_shape("Softmax")
@register_shape("Softmax2d")
@register_shape("LogSoftmax")
def _passthrough_activation_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    return inputs[0]

@register_shape("MultiheadAttention")
def _mha_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    # Inputs arrive port-ordered (query, key, value); output follows the query.
    # Assumes batch_first=True: (B, Seq_q, D) → (B, Seq_q, embed_dim)
    query = inputs[0]
    embed_dim = params["embed_dim"]
    q_dim = _resolve_dim(query[-1])
    if q_dim is not None and q_dim != embed_dim:
        raise ValueError(
            f"query last dim {q_dim} does not match embed_dim {embed_dim}"
        )
    return (*query[:-1], embed_dim)

@register_shape("GLU")
@register_shape("AdaptiveLogSoftmaxWithLoss")
def _non_linear_notimplemented_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    raise NotImplementedError



# ══════════════════════════════════════════════
#  Normalization — all passthrough
# ══════════════════════════════════════════════

@register_shape("BatchNorm1d")
@register_shape("BatchNorm2d")
@register_shape("BatchNorm3d")
@register_shape("GroupNorm")
@register_shape("SyncBatchNorm")
@register_shape("InstanceNorm1d")
@register_shape("InstanceNorm2d")
@register_shape("InstanceNorm3d")
@register_shape("LayerNorm")
@register_shape("LocalResponseNorm")
@register_shape("RMSNorm")
def _norm_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    return inputs[0]


@register_shape("LazyBatchNorm1d")
@register_shape("LazyBatchNorm2d")
@register_shape("LazyBatchNorm3d")
@register_shape("LazyInstanceNorm1d")
@register_shape("LazyInstanceNorm2d")
@register_shape("LazyInstanceNorm3d")
def _lazy_norm_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    # Lazy norm layers infer num_features at runtime; shape is unchanged.
    return inputs[0]



# ══════════════════════════════════════════════
#  Recurrent Layers
# ══════════════════════════════════════════════

@register_shape("RNN")
@register_shape("GRU")
def _rnn_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]  # (B, Seq, input_size) → (B, Seq, hidden_size * num_directions)
    hidden = params["hidden_size"]
    bidir = params.get("bidirectional", False)
    d = 2 if bidir else 1
    return (inp[0], inp[1], hidden * d)


@register_shape("LSTM")
def _lstm_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]
    hidden = params["hidden_size"]
    bidir = params.get("bidirectional", False)
    d = 2 if bidir else 1
    return (inp[0], inp[1], hidden * d)


@register_shape("GRUCell")
@register_shape("RNNCell")
def _cell_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]
    hidden = params["hidden_size"]
    return (inp[0], hidden)


@register_shape("LSTMCell")
def _rnn_notimplemented_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    raise NotImplementedError



# ══════════════════════════════════════════════
#  Transformer Layers
# ══════════════════════════════════════════════


def _check_d_model(shape: Shape, d_model: int, role: str) -> None:
    dim = _resolve_dim(shape[-1])
    if dim is not None and dim != d_model:
        raise ValueError(
            f"{role} last dim {dim} does not match d_model {d_model}"
        )


@register_shape("TransformerEncoder")
@register_shape("TransformerEncoderLayer")
def _transformer_encoder_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    # (B, Seq, d_model) → unchanged (assumes batch_first=True)
    src = inputs[0]
    _check_d_model(src, params["d_model"], "src")
    return src


@register_shape("TransformerDecoder")
@register_shape("TransformerDecoderLayer")
def _transformer_decoder_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    # Port-ordered inputs (tgt, memory) → output follows tgt.
    d_model = params["d_model"]
    for role, shape in zip(("tgt", "memory"), inputs):
        _check_d_model(shape, d_model, role)
    return inputs[0]


@register_shape("Transformer")
def _transformer_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    # Port-ordered inputs (src, tgt) → output follows tgt.
    d_model = params.get("d_model", 512)
    for role, shape in zip(("src", "tgt"), inputs):
        _check_d_model(shape, d_model, role)
    return inputs[-1]




# ══════════════════════════════════════════════
#  Linear Layers
# ══════════════════════════════════════════════

@register_shape("Identity")
def _identity_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    return inputs[0]


@register_shape("Linear")
def _linear_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]
    return (*inp[:-1], params["out_features"])


@register_shape("Bilinear")
def _bilinear_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    # Port-ordered inputs (input1, input2); both share leading dims.
    in1 = inputs[0]
    for role, key, shape in (
        ("input1", "in1_features", inputs[0]),
        ("input2", "in2_features", inputs[-1]),
    ):
        expected = params.get(key)
        actual = _resolve_dim(shape[-1])
        if expected is not None and actual is not None and actual != expected:
            raise ValueError(
                f"{role} last dim {actual} does not match {key} {expected}"
            )
    return (*in1[:-1], params["out_features"])


@register_shape("LazyLinear")
def _lazy_linear_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    # in_features is inferred at runtime; output dim is known statically.
    inp = inputs[0]
    return (*inp[:-1], params["out_features"])



# ══════════════════════════════════════════════
#  Dropout Layers
# ══════════════════════════════════════════════

@register_shape("Dropout")
@register_shape("Dropout1d")
@register_shape("Dropout2d")
@register_shape("Dropout3d")
@register_shape("AlphaDropout")
@register_shape("FeatureAlphaDropout")
def _passthrough_dropout_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    return inputs[0]



# ══════════════════════════════════════════════
#  Sparse Layers
# ══════════════════════════════════════════════

@register_shape("Embedding")
@register_shape("EmbeddingBag")
def _embedding_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]  # (B, Seq) int indices → (B, Seq, embedding_dim)
    return (*inp, params["embedding_dim"])



# ══════════════════════════════════════════════
#  Vision Layers
# ══════════════════════════════════════════════

@register_shape("PixelShuffle")
def _pixel_shuffle_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]  # (B, C, H, W)
    r = params["upscale_factor"]
    return (inp[0], inp[1] // (r * r), inp[2] * r, inp[3] * r)


@register_shape("PixelUnshuffle")
def _pixel_unshuffle_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]  # (B, C, H, W)
    r = params["downscale_factor"]
    return (inp[0], inp[1] * (r * r), inp[2] // r, inp[3] // r)


# Upsample / UpsamplingNearest2d / UpsamplingBilinear2d: not yet supported
# (also absent from CORE_LAYER_TYPES — add shape rules when they land).



# ══════════════════════════════════════════════
#  Reduction ops
# ══════════════════════════════════════════════

@register_shape("Add")
@register_shape("Subtract")
@register_shape("Multiply")
@register_shape("Divide")
def _elementwise_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    return inputs[0]


@register_shape("Stack")
def _stack_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    dim = params.get("dim", 0)
    ref = list(inputs[0])
    ref.insert(dim, len(inputs))
    return tuple(ref)


@register_shape("Concat")
def _concat_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    dim = params.get("dim", -1)
    ref = inputs[0]
    if dim < 0:
        dim = len(ref) + dim
    result = list(ref)
    concat_sum = _resolve_dim(ref[dim]) or 0
    for inp in inputs[1:]:
        d = _resolve_dim(inp[dim])
        if d is not None and isinstance(concat_sum, int):
            concat_sum += d
        else:
            concat_sum = "*"  # type: ignore[assignment]
    result[dim] = concat_sum
    return tuple(result)



# ══════════════════════════════════════════════
#  Tensor manipulation ops
# ══════════════════════════════════════════════

@register_shape("Flatten")
def _flatten_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]
    start_dim = params.get("start_dim", 1)
    end_dim = params.get("end_dim", -1)
    if end_dim < 0:
        end_dim = len(inp) + end_dim
    flat_dims = inp[start_dim : end_dim + 1]
    concrete = [_resolve_dim(d) for d in flat_dims]
    if all(c is not None for c in concrete):
        flat_size = math.prod(c for c in concrete if c is not None)
        return (*inp[:start_dim], flat_size, *inp[end_dim + 1 :])
    return (*inp[:start_dim], "*", *inp[end_dim + 1 :])


@register_shape("Reshape")
def _reshape_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    shape = params["shape"]
    inp = inputs[0]
    result: list[Dim] = []
    for i, s in enumerate(shape):
        if s == -1:
            total_in = 1
            all_concrete = True
            for d in inp:
                r = _resolve_dim(d)
                if r is not None:
                    total_in *= r
                else:
                    all_concrete = False
                    break
            if all_concrete:
                other_prod = 1
                for o in shape:
                    if o != -1:
                        other_prod *= abs(o)
                result.append(total_in // other_prod)
            else:
                result.append("*")
        elif s == 0:
            result.append(inp[i] if i < len(inp) else "*")
        else:
            result.append(s)
    return tuple(result)


@register_shape("View")
def _view_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    return _reshape_shape(inputs, params)


@register_shape("Permute")
def _permute_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]
    dims = params["dims"]
    return tuple(inp[d] for d in dims)


@register_shape("Transpose")
def _transpose_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = list(inputs[0])
    dim0 = params.get("dim0", 0)
    dim1 = params.get("dim1", 1)
    if dim0 < 0:
        dim0 = len(inp) + dim0
    if dim1 < 0:
        dim1 = len(inp) + dim1
    inp[dim0], inp[dim1] = inp[dim1], inp[dim0]
    return tuple(inp)


@register_shape("Unsqueeze")
def _unsqueeze_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = list(inputs[0])
    dim = params["dim"]
    if dim < 0:
        dim = len(inp) + 1 + dim
    inp.insert(dim, 1)
    return tuple(inp)


@register_shape("Squeeze")
def _squeeze_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = list(inputs[0])
    dim = params.get("dim", None)
    if dim is not None:
        if dim < 0:
            dim = len(inp) + dim
        resolved = _resolve_dim(inp[dim])
        if resolved == 1:
            inp.pop(dim)
        return tuple(inp)
    return tuple(d for d in inp if _resolve_dim(d) != 1)


@register_shape("Split")
def _split_shape(inputs: list[Shape], params: dict[str, Any]) -> Shape:
    inp = inputs[0]
    dim = params.get("dim", 0)
    if dim < 0:
        dim = len(inp) + dim
    sections = params.get("split_size_or_sections", 2)
    if isinstance(sections, list):
        result = list(inp)
        result[dim] = sections[0]
        return tuple(result)
    result = list(inp)
    resolved = _resolve_dim(inp[dim])
    result[dim] = resolved // sections if resolved is not None else "*"
    return tuple(result)


# ══════════════════════════════════════════════
#  Validator
# ══════════════════════════════════════════════


class ShapeValidator:
    """Validates tensor shapes through the DAG statically (spec §3)."""

    def __init__(self, graph: ParsedGraph) -> None:
        self.graph = graph

    def validate(self) -> ValidationResult:
        """Run shape propagation and return all errors."""
        result = ValidationResult()
        store: dict[str, Shape] = {}
        input_shape = tuple(self.graph.schema.input_shape)

        # Seed input nodes
        for input_node in self.graph.input_nodes:
            store[input_node.id] = input_shape
            result.node_shapes[input_node.id] = input_shape

        # Walk execution order
        for node_id in self.graph.execution_order:
            node = self.graph.nodes[node_id]

            if node.type == "Input":
                continue

            # Gather input shapes
            input_shapes = self._gather_inputs(node, store, result)
            if input_shapes is None:
                continue

            # Validate elementwise ops require identical shapes
            if len(input_shapes) > 1 and node.type in ELEMENTWISE_OPS:
                ref = input_shapes[0]
                for i, s in enumerate(input_shapes[1:], 1):
                    if s != ref:
                        result.add_error(ShapeError(
                            node_id=node_id,
                            message=(
                                f"Node '{node_id}' ({node.type}): all inputs must have "
                                f"identical shapes. Input 0 has {ref}, input {i} has {s}."
                            ),
                            expected_shape=ref,
                            actual_shape=s,
                        ))

            # Compute output shape
            transform = _SHAPE_REGISTRY.get(node.type)
            if transform is not None:
                try:
                    out_shape = transform(input_shapes, node.schema.params)
                    store[node_id] = out_shape
                    result.node_shapes[node_id] = out_shape

                    if node.schema.outputs:
                        self._store_port_shapes(node, out_shape, store, result)
                except Exception as e:
                    result.add_error(ShapeError(
                        node_id=node_id,
                        message=f"Node '{node_id}': shape computation failed: {e}",
                    ))
            else:
                if len(input_shapes) == 1:
                    store[node_id] = input_shapes[0]
                    result.node_shapes[node_id] = input_shapes[0]
                else:
                    result.add_error(ShapeError(
                        node_id=node_id,
                        message=(
                            f"Node '{node_id}': no shape transform registered for "
                            f"type '{node.type}' and multiple inputs received."
                        ),
                    ))

        return result

    def _gather_inputs(
        self, node: ParsedNode, store: dict[str, Shape], result: ValidationResult,
    ) -> list[Shape] | None:
        shapes: list[Shape] = []
        for edge in node.incoming_edges:
            key = edge.source_key
            if key not in store:
                result.add_error(ShapeError(
                    node_id=node.id,
                    message=(
                        f"Node '{node.id}': missing input from '{key}'. "
                        f"Source node may have failed validation."
                    ),
                    source_node_id=edge.source,
                ))
                return None
            shapes.append(store[key])

        # Mirror the executor: order multi-positional inputs by target_port
        # so shape rules see (tgt, memory), (src, tgt), (query, key, value)…
        if node.type in MULTI_INPUT_LAYERS and shapes:
            edge_ports = [e.target_port for e in node.incoming_edges]
            try:
                shapes = resolve_multi_inputs(node.id, node.type, edge_ports, shapes)
            except MLnodeError as e:
                result.add_error(ShapeError(node_id=node.id, message=str(e)))
                return None

        return shapes

    def _store_port_shapes(
        self, node: ParsedNode, out_shape: Shape,
        store: dict[str, Shape], result: ValidationResult,
    ) -> None:
        """Store per-port shapes for multi-output nodes like Split."""
        assert node.schema.outputs is not None
        dim = node.schema.params.get("dim", 0)
        sections = node.schema.params.get("split_size_or_sections", None)

        if node.type == "Split" and isinstance(sections, list):
            norm_dim = dim if dim >= 0 else len(out_shape) + dim
            for i, port_name in enumerate(node.schema.outputs):
                port_key = f"{node.id}:{port_name}"
                port_shape = list(out_shape)
                port_shape[norm_dim] = sections[i]
                store[port_key] = tuple(port_shape)
                result.node_shapes[port_key] = tuple(port_shape)
        else:
            for port_name in node.schema.outputs:
                port_key = f"{node.id}:{port_name}"
                store[port_key] = out_shape
                result.node_shapes[port_key] = out_shape
