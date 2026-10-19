"""MLnode — Visual Programming Language for Machine Learning.

JSON-described DAG → executable PyTorch nn.Module.
"""

__version__ = "1.1.0"

from mlnode.blocks.registry import BlockRegistry
from mlnode.errors import (
    CustomLayerError,
    CycleDetectedError,
    SchemaVersionError,
    ShapeMismatchError,
    ValidationError,
    MLnodeError,
)
from mlnode.layers.json_layers import LayerRegistry
from mlnode.model import MLnodeModel
from mlnode.schema.models import BlockDefinition, LayerDefinition
from mlnode.schema.types import ValidationResult

__all__ = [
    "MLnodeModel",
    "BlockDefinition",
    "BlockRegistry",
    "LayerDefinition",
    "LayerRegistry",
    "ValidationResult",
    "MLnodeError",
    "ValidationError",
    "CycleDetectedError",
    "ShapeMismatchError",
    "CustomLayerError",
    "SchemaVersionError",
]
