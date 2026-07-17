"""Core type definitions for MLnode schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union

# A dimension is either a symbolic string ("B", "Seq", "D", etc.) or a concrete int.
Dim = Union[str, int]

# A shape is a tuple of dimensions.
Shape = tuple[Dim, ...]

# Symbolic dimension names recognized by the validator.
SYMBOLIC_DIMS = frozenset({"B", "C", "H", "W", "Seq", "D", "*"})


class ReductionType(str, Enum):
    """Reduction strategies for multi-input nodes."""

    ADD = "add"
    CONCAT = "concat"
    STACK = "stack"


# Node types that have implicit reduction semantics (§1.2).
REDUCTION_OPS = frozenset({"Add", "Subtract", "Multiply", "Divide", "Concat", "Stack",})

# Elementwise reduction ops — all inputs must have identical shapes.
ELEMENTWISE_OPS = frozenset({"Add", "Subtract", "Multiply", "Divide"})

# Layers whose forward() takes several positional tensor arguments.
# Maps type → ordered input port names. Edges may set target_port to one of
# these names; unlabeled edges fall back to declaration order.
MULTI_INPUT_LAYERS: dict[str, tuple[str, ...]] = {
    "Transformer": ("src", "tgt"),
    "TransformerDecoder": ("tgt", "memory"),
    "TransformerDecoderLayer": ("tgt", "memory"),
    "MultiheadAttention": ("query", "key", "value"),
    "Bilinear": ("input1", "input2"),
}

# Tensor manipulation ops that don't have learnable parameters.
TENSOR_OPS = frozenset({
    "Flatten", "Reshape", "Permute", "Transpose", "View",
    "Unsqueeze", "Squeeze", "Split", 
})

# All supported core node types.
CORE_LAYER_TYPES = frozenset({
    "Input",

    #Convolution Layers
    "Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    "LazyConv1d", "LazyConv2d", "LazyConv3d",
    "LazyConvTranspose1d", "LazyConvTranspose2d", "LazyConvTranspose3d",
    "Unfold", "Fold",
    
    # Pooling layers
    "MaxPool1d", "MaxPool2d", "MaxPool3d",
    "MaxUnpool1d", "MaxUnpool2d", "MaxUnpool3d",
    "AvgPool1d", "AvgPool2d", "AvgPool3d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "FractionalMaxPool2d", "FractionalMaxPool3d",
    "LPPool1d", "LPPool2d", "LPPool3d",
    "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",

    # Non-linear Activations (weighted sum, nonlinearity)
    "ELU", "Hardshrink", "Hardsigmoid", "Hardtanh", "Hardswish",
    "LeakyReLU", "LogSigmoid", "PReLU", "ReLU6", "RReLU", "MultiheadAttention",
    "SELU", "CELU", "SiLU", "Mish", "Softplus", "Softshrink", "Softsign",
    "ReLU", "GELU", "Sigmoid", "Tanh",
    "Tanhshrink", "Threshold", "GLU",
    
    # Non-linear Activations (other)
    "Softmin", "Softmax", "Softmax2d", "LogSoftmax", "AdaptiveLogSoftmaxWithLoss",
    
    # Normalization Layers
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "LazyBatchNorm1d", "LazyBatchNorm2d", "LazyBatchNorm3d",
    "GroupNorm", "SyncBatchNorm",
    "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", 
    "LazyInstanceNorm1d", "LazyInstanceNorm2d", "LazyInstanceNorm3d",
    "LayerNorm", "LocalResponseNorm", "RMSNorm",
    
    # Recurrent Layers
    # (RNNBase is intentionally excluded: it is an abstract base with no forward())
    "RNN", "LSTM", "GRU",
    "RNNCell", "LSTMCell", "GRUCell",

    # Transformer Layers
    "Transformer", "TransformerEncoder", "TransformerDecoder",
    "TransformerEncoderLayer", "TransformerDecoderLayer",
    
    # Linear
    "Identity", "Linear", "Bilinear", "LazyLinear",
    
    # Dropout Layers
    "Dropout",
    "Dropout1d", "Dropout2d", "Dropout3d", 
    "AlphaDropout", "FeatureAlphaDropout",
    
    # Sparse Layers
    "Embedding", "EmbeddingBag",

    # Composable blocks
    "Block",
    # Reduction ops (elementwise: Add/Subtract/Multiply/Divide)
    "Add", "Subtract", "Multiply", "Divide", "Concat", "Stack",
    # Vision ops
    "PixelShuffle", "PixelUnshuffle",

    # Tensor ops
    "Flatten", "Reshape", "Permute", "Transpose", "View",
    "Unsqueeze", "Squeeze", "Split", 
    #
    #to be added
    #
    #"ChannelShuffle",
    #"Upsample", "UpsamplingNearest2d", "UpsamplingBilinear2d",
    #
    # Padding Layers
    #"ReflectionPad1d", "ReflectionPad2d", "ReflectionPad3d",
    #"ReplicationPad1d", "ReplicationPad2d", "ReplicationPad3d",
    #"ZeroPad1d", "ZeroPad2d", "ZeroPad3d",
    #"ConstantPad1d", "ConstantPad2d", "ConstantPad3d",
    #"CircularPad1d", "CircularPad2d", "CircularPad3d",

})


@dataclass
class ShapeError:
    """A single shape validation error."""

    node_id: str
    message: str
    expected_shape: Shape | None = None
    actual_shape: Shape | None = None
    source_node_id: str | None = None


@dataclass
class ValidationResult:
    """Result of graph validation (§7 API).

    Collects all errors rather than failing on the first one,
    so users can fix multiple issues in a single pass.
    """

    valid: bool = True
    errors: list[ShapeError] = field(default_factory=list)
    node_shapes: dict[str, Shape] = field(default_factory=dict)

    def add_error(self, error: ShapeError) -> None:
        self.valid = False
        self.errors.append(error)
