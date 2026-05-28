"""Custom layer validation (spec §4).

1. Verify class inherits nn.Module.
2. Instantiate with provided params.
3. Run dummy forward with torch.zeros(2, *expected_shape).
4. Catch exceptions → wrap in CustomLayerError with JSON node reference.
5. Cache result. No exec() / eval() ever.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from mlnode.errors import CustomLayerError
from mlnode.schema.types import Dim, Shape

logger = logging.getLogger(__name__)

DUMMY_BATCH_SIZE = 2


@dataclass(frozen=True)
class CustomLayerResult:
    """Cached result of custom layer validation."""

    module_class: type[nn.Module]
    params: dict[str, Any]
    input_shape: Shape
    output_shape: Shape
    instance: nn.Module


# Global validation cache: (import_path, frozen_params) → result
_VALIDATION_CACHE: dict[tuple[str, tuple], CustomLayerResult] = {}


def clear_cache() -> None:
    """Clear the custom layer validation cache."""
    _VALIDATION_CACHE.clear()


def _freeze_params(params: dict[str, Any]) -> tuple:
    """Convert params dict to a hashable tuple for cache key."""
    items = []
    for k, v in sorted(params.items()):
        if isinstance(v, (list, tuple)):
            items.append((k, tuple(v)))
        elif isinstance(v, dict):
            items.append((k, _freeze_params(v)))
        else:
            items.append((k, v))
    return tuple(items)


def _concretize_shape(shape: Shape) -> tuple[int, ...]:
    """Convert symbolic shape to concrete dims for dummy tensor creation.

    Symbolic dims get reasonable defaults: B=2, Seq=16, C/D/H/W=8.
    """
    defaults = {"B": DUMMY_BATCH_SIZE, "Seq": 16, "C": 8, "H": 8, "W": 8, "D": 8, "*": 4}
    result = []
    for d in shape:
        if isinstance(d, int):
            result.append(d)
        elif isinstance(d, str):
            result.append(defaults.get(d, 4))
        else:
            result.append(4)
    return tuple(result)


def resolve_custom_class(import_path: str, node_id: str) -> type[nn.Module]:
    """Import and validate a custom layer class (steps 1).

    No exec() / eval() — uses importlib only.
    """
    parts = import_path.rsplit(".", 1)
    if len(parts) != 2:
        raise CustomLayerError(
            f"Node '{node_id}': invalid custom path '{import_path}'. "
            f"Expected 'module.path.ClassName'.",
            node_id=node_id,
        )

    module_path, class_name = parts

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise CustomLayerError(
            f"Node '{node_id}': could not import module '{module_path}': {e}",
            node_id=node_id,
            original_error=e,
        ) from e

    cls = getattr(module, class_name, None)
    if cls is None:
        raise CustomLayerError(
            f"Node '{node_id}': class '{class_name}' not found in '{module_path}'.",
            node_id=node_id,
        )

    # Step 1: verify nn.Module
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise CustomLayerError(
            f"Node '{node_id}': '{import_path}' is not a subclass of nn.Module.",
            node_id=node_id,
        )

    return cls


def validate_custom_layer(
    import_path: str,
    params: dict[str, Any],
    input_shape: Shape,
    node_id: str,
) -> CustomLayerResult:
    """Full custom layer validation (spec §4, steps 1-5).

    Returns a cached CustomLayerResult with the probed output shape.
    """
    # Step 5: check cache
    cache_key = (import_path, _freeze_params(params))
    if cache_key in _VALIDATION_CACHE:
        cached = _VALIDATION_CACHE[cache_key]
        if cached.input_shape == input_shape:
            return cached

    # Step 1: import and verify
    cls = resolve_custom_class(import_path, node_id)

    # Step 2: instantiate
    try:
        instance = cls(**params)
    except Exception as e:
        raise CustomLayerError(
            f"Node '{node_id}': failed to instantiate '{import_path}' "
            f"with params {params}: {e}",
            node_id=node_id,
            original_error=e,
        ) from e

    # Step 3: dummy forward with torch.zeros(2, *expected_shape)
    concrete_shape = _concretize_shape(input_shape)
    dummy_input = torch.zeros(*concrete_shape)

    try:
        instance.eval()
        with torch.no_grad():
            dummy_output = instance(dummy_input)
    except Exception as e:
        raise CustomLayerError(
            f"Node '{node_id}': dummy forward failed for '{import_path}' "
            f"with input shape {concrete_shape}: {e}",
            node_id=node_id,
            original_error=e,
        ) from e

    # Infer output shape, mapping concrete dims back to symbolic where possible
    if isinstance(dummy_output, torch.Tensor):
        output_shape = _infer_symbolic_output_shape(
            input_shape, concrete_shape, tuple(dummy_output.shape)
        )
    else:
        raise CustomLayerError(
            f"Node '{node_id}': custom layer '{import_path}' returned "
            f"{type(dummy_output).__name__}, expected torch.Tensor.",
            node_id=node_id,
        )

    # Step 5: cache result
    result = CustomLayerResult(
        module_class=cls,
        params=params,
        input_shape=input_shape,
        output_shape=output_shape,
        instance=instance,
    )
    _VALIDATION_CACHE[cache_key] = result

    logger.info(
        "Custom layer '%s' validated: %s → %s",
        import_path, input_shape, output_shape,
    )
    return result


def _infer_symbolic_output_shape(
    symbolic_input: Shape,
    concrete_input: tuple[int, ...],
    concrete_output: tuple[int, ...],
) -> Shape:
    """Map concrete output dims back to symbolic names where they match input dims.

    If an output dim equals a concrete input dim that was symbolic,
    restore the symbol. Otherwise keep the concrete value.
    """
    # Build reverse map: concrete_value → symbolic_name (from input)
    concrete_to_sym: dict[int, Dim] = {}
    for sym, conc in zip(symbolic_input, concrete_input):
        if isinstance(sym, str):
            concrete_to_sym[conc] = sym

    result: list[Dim] = []
    for d in concrete_output:
        if d in concrete_to_sym:
            result.append(concrete_to_sym[d])
        else:
            result.append(d)
    return tuple(result)
