"""Layer registry — maps VPL node type strings to nn.Module constructors.

No exec() / eval() / compile() — spec §6 principle 6.
"""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from mlnode.errors import CustomLayerError, MLnodeError
from mlnode.schema.models import NodeSchema
from mlnode.schema.types import Shape

# Registry: type name → callable that takes params dict → nn.Module
_LAYER_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_layer(
    type_name: str,
) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator to register a layer constructor."""

    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        _LAYER_REGISTRY[type_name] = fn
        return fn

    return decorator


def _build_from_torch(cls: type[nn.Module], params: dict[str, Any]) -> nn.Module:
    """Instantiate a standard torch.nn module from params."""
    return cls(**params)


# ──────────────────────────────────────────────
# Register all core PyTorch layers
# ──────────────────────────────────────────────

_TORCH_LAYER_MAP: dict[str, type[nn.Module]] = {
    
    # Convolution Layers
    "Conv1d": nn.Conv1d,
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "ConvTranspose1d": nn.ConvTranspose1d,
    "ConvTranspose2d": nn.ConvTranspose2d,
    "ConvTranspose3d": nn.ConvTranspose3d,
    "LazyConv1d": nn.LazyConv1d, 
    "LazyConv2d": nn.LazyConv2d, 
    "LazyConv3d": nn.LazyConv3d,
    "LazyConvTranspose1d": nn.LazyConvTranspose1d, 
    "LazyConvTranspose2d": nn.LazyConvTranspose2d, 
    "LazyConvTranspose3d": nn.LazyConvTranspose3d,
    "Unfold": nn.Unfold, 
    "Fold": nn.Fold,

    # Pooling Layers
    "MaxPool1d": nn.MaxPool1d, 
    "MaxPool2d": nn.MaxPool2d, 
    "MaxPool3d": nn.MaxPool3d,
    "MaxUnpool1d": nn.MaxUnpool1d, 
    "MaxUnpool2d": nn.MaxUnpool2d, 
    "MaxUnpool3d": nn.MaxUnpool3d,
    "AvgPool1d": nn.AvgPool1d, 
    "AvgPool2d": nn.AvgPool2d, 
    "AvgPool3d": nn.AvgPool3d,
    "AdaptiveAvgPool1d": nn.AdaptiveAvgPool1d, 
    "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d, 
    "AdaptiveAvgPool3d": nn.AdaptiveAvgPool3d,
    "FractionalMaxPool2d": nn.FractionalMaxPool2d, 
    "FractionalMaxPool3d": nn.FractionalMaxPool3d,
    "LPPool1d": nn.LPPool1d, 
    "LPPool2d": nn.LPPool2d, 
    "LPPool3d": nn.LPPool3d,
    "AdaptiveMaxPool1d": nn.AdaptiveMaxPool1d,
    "AdaptiveMaxPool2d": nn.AdaptiveMaxPool2d,
    "AdaptiveMaxPool3d": nn.AdaptiveMaxPool3d,

    # Non-linear Activations (weighted sum, nonlinearity)
    "ELU": nn.ELU, 
    "Hardshrink": nn.Hardshrink, 
    "Hardsigmoid": nn.Hardsigmoid, 
    "Hardtanh": nn.Hardtanh, 
    "Hardswish": nn.Hardswish,
    "LeakyReLU": nn.LeakyReLU,
    "LogSigmoid": nn.LogSigmoid,
    "PReLU": nn.PReLU,
    "ReLU6": nn.ReLU6,
    "RReLU": nn.RReLU,
    "MultiheadAttention": nn.MultiheadAttention,
    "SELU": nn.SELU, 
    "CELU": nn.CELU, 
    "SiLU": nn.SiLU, 
    "Mish": nn.Mish, 
    "Softplus": nn.Softplus, 
    "Softshrink": nn.Softshrink, 
    "Softsign": nn.Softsign,
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "Tanhshrink": nn.Tanhshrink, 
    "Threshold": nn.Threshold, 
    "GLU": nn.GLU,
    
     # Non-linear Activations (other)
    "Softmin": nn.Softmin, 
    "Softmax": nn.Softmax, 
    "Softmax2d": nn.Softmax2d, 
    "LogSoftmax": nn.LogSoftmax, 
    "AdaptiveLogSoftmaxWithLoss": nn.AdaptiveLogSoftmaxWithLoss,
    
    # Normalization Layers
    "BatchNorm1d": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm3d": nn.BatchNorm3d,
    "LazyBatchNorm1d": nn.LazyBatchNorm1d,
    "LazyBatchNorm2d": nn.LazyBatchNorm2d,
    "LazyBatchNorm3d": nn.LazyBatchNorm3d,
    "LayerNorm": nn.LayerNorm,
    "GroupNorm": nn.GroupNorm,
    "SyncBatchNorm": nn.SyncBatchNorm,
    "InstanceNorm1d": nn.InstanceNorm1d, 
    "InstanceNorm2d": nn.InstanceNorm2d, 
    "InstanceNorm3d": nn.InstanceNorm3d, 
    "LazyInstanceNorm1d": nn.LazyInstanceNorm1d, 
    "LazyInstanceNorm2d": nn.LazyInstanceNorm2d,
    "LazyInstanceNorm3d": nn.LazyInstanceNorm3d,
    "LocalResponseNorm": nn.LocalResponseNorm, 
    "RMSNorm": nn.RMSNorm,
    
    # Recurrent Layers
    # (RNNBase excluded: abstract base class without forward())
    "RNN": nn.RNN,
    "LSTM": nn.LSTM,
    "GRU": nn.GRU,
    "RNNCell": nn.RNNCell,
    "LSTMCell": nn.LSTMCell,
    "GRUCell": nn.GRUCell,

    # Transformer Layers
    # TransformerEncoder/TransformerDecoder need composed construction —
    # they are registered below with dedicated builders.
    "Transformer": nn.Transformer,
    "TransformerEncoderLayer": nn.TransformerEncoderLayer,
    "TransformerDecoderLayer": nn.TransformerDecoderLayer,

    # Linear
    "Linear": nn.Linear,
    "Identity": nn.Identity, 
    "Bilinear": nn.Bilinear, 
    "LazyLinear": nn.LazyLinear,

    # Dropout Layers
    "Dropout": nn.Dropout,
    "Dropout1d": nn.Dropout1d, 
    "Dropout2d": nn.Dropout2d, 
    "Dropout3d": nn.Dropout3d, 
    "AlphaDropout": nn.AlphaDropout, 
    "FeatureAlphaDropout": nn.FeatureAlphaDropout,

    # Sparse Layers
    "Embedding": nn.Embedding,
    "EmbeddingBag": nn.EmbeddingBag,

    # Vision Layers
    "PixelShuffle": nn.PixelShuffle,          # params: upscale_factor
    "PixelUnshuffle": nn.PixelUnshuffle,      # params: downscale_factor
}

# Auto-register all torch layers
for _name, _cls in _TORCH_LAYER_MAP.items():
    _LAYER_REGISTRY[_name] = lambda params, cls=_cls: _build_from_torch(cls, params)


# ──────────────────────────────────────────────
# Composed transformer stacks
#
# nn.TransformerEncoder/Decoder take a *layer instance* + num_layers, which
# cannot come straight from JSON params. These builders accept the flat
# TransformerEncoderLayer/DecoderLayer params plus:
#   num_layers (int, default 6) — how many stacked layers
#   norm       (bool, default False) — append a final LayerNorm(d_model)
# The same flat params are what mlnode-editor emits for these node types.
# ──────────────────────────────────────────────

@register_layer("TransformerEncoder")
def _build_transformer_encoder(params: dict[str, Any]) -> nn.Module:
    p = dict(params)
    num_layers = p.pop("num_layers", 6)
    final_norm = p.pop("norm", False)
    layer = nn.TransformerEncoderLayer(**p)
    norm = nn.LayerNorm(layer.self_attn.embed_dim) if final_norm else None
    return nn.TransformerEncoder(layer, num_layers=num_layers, norm=norm)


@register_layer("TransformerDecoder")
def _build_transformer_decoder(params: dict[str, Any]) -> nn.Module:
    p = dict(params)
    num_layers = p.pop("num_layers", 6)
    final_norm = p.pop("norm", False)
    layer = nn.TransformerDecoderLayer(**p)
    norm = nn.LayerNorm(layer.self_attn.embed_dim) if final_norm else None
    return nn.TransformerDecoder(layer, num_layers=num_layers, norm=norm)



def build_layer(node: NodeSchema, input_shape: Shape | None = None) -> nn.Module:
    """Build an nn.Module from a NodeSchema.

    Handles four cases:
    1. Core registered types → direct construction
    2. HuggingFace models → delegated to hf loader
    3. Custom layers → import, validate with dummy forward, cache
    4. Block nodes → handled by the executor (not here)
    """
    # HuggingFace
    if node.hf_model is not None:
        from mlnode.huggingface.loader import load_hf_pretrained
        return load_hf_pretrained(node.hf_model)

    if node.hf_config is not None:
        from mlnode.huggingface.loader import load_hf_from_config
        return load_hf_from_config(node.hf_config)

    # Custom layer — full spec §4 validation
    if node.type == "custom":
        return _build_custom_layer(node, input_shape)

    # Core registered type
    constructor = _LAYER_REGISTRY.get(node.type)
    if constructor is None:
        raise MLnodeError(
            f"Node '{node.id}': no layer registered for type '{node.type}'. "
            f"Available types: {sorted(_LAYER_REGISTRY.keys())}"
        )

    try:
        return constructor(node.params)
    except Exception as e:
        raise MLnodeError(
            f"Node '{node.id}': failed to construct {node.type} with params "
            f"{node.params}: {e}"
        ) from e


def _build_custom_layer(node: NodeSchema, input_shape: Shape | None = None) -> nn.Module:
    """Import, validate, and instantiate a custom layer (spec §4).

    If input_shape is provided, performs full dummy-forward validation.
    Otherwise falls back to import + instantiate only.
    """
    path = node.path
    if path is None:
        # JSON layers ('layer' field) are resolved by the parser and built by
        # the executor as nested modules; they never reach build_layer().
        raise CustomLayerError(
            f"Node '{node.id}': custom type requires 'path' field. JSON layers "
            f"('layer') are built from a parsed graph, not by build_layer().",
            node_id=node.id,
        )

    if input_shape is not None:
        # Full validation with dummy forward (spec §4 steps 1–5)
        from mlnode.layers.custom import validate_custom_layer
        result = validate_custom_layer(path, node.params, input_shape, node.id)
        return result.instance

    # Fallback: import + instantiate without probing
    from mlnode.layers.custom import resolve_custom_class
    cls = resolve_custom_class(path, node.id)

    try:
        return cls(**node.params)
    except Exception as e:
        raise CustomLayerError(
            f"Node '{node.id}': failed to instantiate '{path}' with params "
            f"{node.params}: {e}",
            node_id=node.id,
            original_error=e,
        ) from e
