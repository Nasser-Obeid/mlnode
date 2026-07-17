"""HuggingFace model loading (spec §5).

| Field     | Behavior                                      |
|-----------|-----------------------------------------------|
| hf_model  | AutoModel.from_pretrained → loads weights     |
| hf_config | AutoConfig → ModelClass(config) → random init |
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from mlnode.errors import MLnodeError


def _import_transformers() -> Any:
    """transformers is an optional dependency (the [hf] extra)."""
    try:
        import transformers
    except ImportError as e:
        raise MLnodeError(
            "HuggingFace nodes (hf_model / hf_config) require the 'transformers' "
            'package. Install it with: pip install "mlnode[hf]"'
        ) from e
    return transformers


def load_hf_pretrained(model_name: str) -> nn.Module:
    """Load a pretrained HuggingFace model.

    Args:
        model_name: HuggingFace model identifier (e.g., "bert-base-uncased").

    Returns:
        The loaded model as an nn.Module.
    """
    transformers = _import_transformers()

    model = transformers.AutoModel.from_pretrained(model_name)
    return model  # type: ignore[return-value]


def load_hf_from_config(config_name: str) -> nn.Module:
    """Load a randomly initialized model from a HuggingFace config.

    Args:
        config_name: Config class name (e.g., "GPT2Config") or model identifier.

    Returns:
        A randomly initialized model as an nn.Module.
    """
    transformers = _import_transformers()

    config = transformers.AutoConfig.from_pretrained(config_name)
    model = transformers.AutoModel.from_config(config)
    return model  # type: ignore[return-value]
