"""JSON-defined custom layers.

A custom layer can be written in JSON instead of Python: a named,
parameterized sub-graph built as its own nested nn.Module.

    "custom_layers": {
      "SqueezeExcite": {
        "params": {"channels": {"type": "int"}, "reduction": {"type": "int", "default": 16}},
        "nodes": [
          {"id": "x", "type": "Input"},
          {"id": "fc1", "type": "Conv2d",
           "params": {"in_channels": "$channels",
                      "out_channels": "${channels // reduction}", "kernel_size": 1}},
          ...
        ],
        "edges": [...],
        "output_node": "scale"
      }
    }

    {"id": "se1", "type": "custom", "layer": "SqueezeExcite", "params": {"channels": 64}}

Definitions resolve from the model file's "custom_layers" first, then from a
LayerRegistry (the process-wide default if none is passed).

Param references inside a layer's node params:
    "$name"      → the param's value (any JSON type)
    "${expr}"    → arithmetic over params: + - * / // % **, unary -, parentheses,
                   min(), max(), abs(), int(), float(), round()
    "$$text"     → the literal string "$text"

Expressions are evaluated by walking a whitelisted AST — no exec() / eval().
"""

from __future__ import annotations

import ast
import json
import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlnode.errors import CustomLayerError, MLnodeError
from mlnode.schema.models import (
    GraphSchema,
    LayerDefinition,
    MetadataSchema,
    NodeSchema,
    ParamSpec,
)

# Maximum nesting depth of JSON layers using other JSON layers.
MAX_LAYER_DEPTH = 10


class LayerRegistry:
    """Registry for JSON custom layer definitions.

    Usage:
        registry = LayerRegistry()
        registry.register_from_file("SqueezeExcite", "layers/se.json")
        registry.register_library("layers/vision.json")   # {"custom_layers": {...}}

        model = MLnodeModel.load("model.mlnode.json", layer_registry=registry)
    """

    def __init__(self) -> None:
        self._layers: dict[str, LayerDefinition] = {}

    def register(self, name: str, layer: LayerDefinition) -> None:
        """Register a layer definition by name."""
        if name in self._layers:
            raise MLnodeError(f"Custom layer '{name}' is already registered.")
        self._layers[name] = layer

    def register_from_dict(self, name: str, data: dict[str, Any]) -> None:
        """Register a layer from a dictionary."""
        self.register(name, LayerDefinition.model_validate(data))

    def register_from_file(self, name: str, path: str | Path) -> None:
        """Register a single layer definition from a JSON file."""
        self.register_from_dict(name, _read_json(path))

    def register_library(self, path: str | Path) -> list[str]:
        """Register every layer in a {"custom_layers": {name: definition}} JSON file.

        Returns the registered names.
        """
        data = _read_json(path)
        layers = data.get("custom_layers")
        if not isinstance(layers, dict):
            raise MLnodeError(
                f"Layer library '{path}' must be a JSON object with a "
                f"'custom_layers' mapping of name → definition."
            )
        for name, definition in layers.items():
            self.register_from_dict(name, definition)
        return sorted(layers)

    def get(self, name: str) -> LayerDefinition:
        """Retrieve a layer definition by name."""
        if name not in self._layers:
            raise MLnodeError(
                f"Custom layer '{name}' not found in registry. "
                f"Available: {sorted(self._layers.keys())}"
            )
        return self._layers[name]

    def has(self, name: str) -> bool:
        """Check if a layer is registered."""
        return name in self._layers

    @property
    def names(self) -> list[str]:
        """List all registered layer names."""
        return sorted(self._layers.keys())

    def __len__(self) -> int:
        return len(self._layers)

    def __repr__(self) -> str:
        return f"LayerRegistry(layers={self.names})"


# Global default registry
_DEFAULT_REGISTRY = LayerRegistry()


def get_default_layer_registry() -> LayerRegistry:
    """Return the global default custom layer registry."""
    return _DEFAULT_REGISTRY


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise MLnodeError(f"'{path}' must contain a JSON object.")
    return data


# ──────────────────────────────────────────────
# Resolution and instantiation
# ──────────────────────────────────────────────


def resolve_layer_definition(
    node: NodeSchema,
    inline: dict[str, LayerDefinition],
    registry: LayerRegistry | None,
) -> LayerDefinition:
    """Find a node's layer definition: inline custom_layers first, then the registry."""
    assert node.layer is not None
    if node.layer in inline:
        return inline[node.layer]
    if registry is None:
        registry = get_default_layer_registry()
    if registry.has(node.layer):
        return registry.get(node.layer)
    available = sorted(set(inline) | set(registry.names))
    raise CustomLayerError(
        f"Node '{node.id}': custom layer '{node.layer}' is not defined. "
        f"Add it to 'custom_layers' or register it in a LayerRegistry. "
        f"Available: {available}",
        node_id=node.id,
    )


def instantiate_layer(
    node: NodeSchema,
    definition: LayerDefinition,
    parent: GraphSchema,
) -> GraphSchema:
    """Bind a node's params into its layer definition and return the sub-graph.

    The sub-graph inherits the parent's inline custom_layers so layers may use
    other layers. Its input_shape is a placeholder: the validator feeds the
    actual incoming shapes.
    """
    params = bind_params(node, definition)

    nodes: list[NodeSchema] = []
    for inner in definition.nodes:
        data = inner.model_dump()
        try:
            data["params"] = substitute(data["params"], params)
        except ValueError as e:
            raise CustomLayerError(
                f"Node '{node.id}' (layer '{node.layer}'): in inner node "
                f"'{inner.id}': {e}",
                node_id=node.id,
                original_error=e,
            ) from e
        try:
            nodes.append(NodeSchema.model_validate(data))
        except ValueError as e:
            raise CustomLayerError(
                f"Node '{node.id}' (layer '{node.layer}'): inner node "
                f"'{inner.id}' is invalid after param substitution: {e}",
                node_id=node.id,
                original_error=e,
            ) from e

    return GraphSchema(
        mlnode_schema_version=parent.mlnode_schema_version,
        metadata=MetadataSchema(name=node.layer, description=definition.description),
        input_shape=parent.input_shape,
        output_node=definition.output_node,
        nodes=nodes,
        edges=definition.edges,
        custom_layers=parent.custom_layers,
    )


_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "str": lambda v: isinstance(v, str),
    "list": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
    "any": lambda v: True,
}


def bind_params(node: NodeSchema, definition: LayerDefinition) -> dict[str, Any]:
    """Merge node params over declared defaults and type-check them."""
    unknown = sorted(set(node.params) - set(definition.params))
    if unknown:
        raise CustomLayerError(
            f"Node '{node.id}' (layer '{node.layer}'): unknown param(s) {unknown}. "
            f"Declared params: {sorted(definition.params)}",
            node_id=node.id,
        )

    bound: dict[str, Any] = {}
    for name, spec in definition.params.items():
        if name in node.params:
            value = node.params[name]
        elif spec.required:
            raise CustomLayerError(
                f"Node '{node.id}' (layer '{node.layer}'): missing required param '{name}'.",
                node_id=node.id,
            )
        else:
            value = spec.default
        _check_param_type(node, name, spec, value)
        bound[name] = value
    return bound


def _check_param_type(node: NodeSchema, name: str, spec: ParamSpec, value: Any) -> None:
    # A null default means "optional, no value" and is passed through unchecked.
    if value is None and not spec.required:
        return
    if not _TYPE_CHECKS[spec.type](value):
        raise CustomLayerError(
            f"Node '{node.id}' (layer '{node.layer}'): param '{name}' expects "
            f"{spec.type}, got {type(value).__name__} ({value!r}).",
            node_id=node.id,
        )


# ──────────────────────────────────────────────
# Param substitution
# ──────────────────────────────────────────────


def substitute(value: Any, params: dict[str, Any]) -> Any:
    """Recursively replace "$name" / "${expr}" references in a JSON value."""
    if isinstance(value, dict):
        return {k: substitute(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, params) for v in value]
    if not isinstance(value, str) or not value.startswith("$"):
        return value
    if value.startswith("$$"):
        return value[1:]
    if value.startswith("${") and value.endswith("}"):
        return evaluate_expression(value[2:-1], params)
    name = value[1:]
    if name not in params:
        raise ValueError(f"unknown param reference '{value}'. Params: {sorted(params)}")
    return params[name]


_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "min": min,
    "max": max,
    "abs": abs,
    "int": int,
    "float": float,
    "round": round,
}

# Guards against expressions like "${10 ** 10 ** 10}" hanging the parser.
_MAX_EXPONENT = 64


def evaluate_expression(expr: str, params: dict[str, Any]) -> Any:
    """Evaluate a restricted arithmetic expression over params.

    Parses with ast and walks only whitelisted node types — never eval().
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid expression '${{{expr}}}': {e.msg}") from e
    try:
        return _eval_node(tree.body, params)
    except ValueError as e:
        raise ValueError(f"in expression '${{{expr}}}': {e}") from e
    except (ArithmeticError, TypeError) as e:
        raise ValueError(f"expression '${{{expr}}}' failed: {e}") from e


def _eval_node(node: ast.AST, params: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in params:
            raise ValueError(f"unknown param '{node.id}'. Params: {sorted(params)}")
        return params[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left, params)
        right = _eval_node(node.right, params)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent {right} exceeds the limit of {_MAX_EXPONENT}")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, params))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        args = [_eval_node(a, params) for a in node.args]
        return _FUNCTIONS[node.func.id](*args)
    raise ValueError(
        f"unsupported syntax '{ast.unparse(node)}'. Allowed: numbers, param names, "
        f"+ - * / // % **, parentheses, {', '.join(_FUNCTIONS)}()"
    )
