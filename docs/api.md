# Python API Reference

Public API (everything importable from the top-level `mlnode` package):

```python
from mlnode import (
    MLnodeModel,        # top-level entry point
    BlockRegistry,      # named, reusable block definitions
    BlockDefinition,    # Pydantic model of a block sub-graph
    LayerRegistry,      # named JSON custom layer definitions
    LayerDefinition,    # Pydantic model of a JSON custom layer
    ValidationResult,   # returned by validate()
    MLnodeError, ValidationError, CycleDetectedError,
    ShapeMismatchError, CustomLayerError, SchemaVersionError,
)
```

Lower-level modules (`mlnode.parser`, `mlnode.validator`, `mlnode.executor`, `mlnode.layers`, `mlnode.utils`) are importable and documented below, but their surface is more likely to change.

---

## MLnodeModel

`mlnode/model.py` — ties the pipeline together: JSON → Parser → Validator → Executor → `nn.Module`.

```python
class MLnodeModel:
    @classmethod
    def load(cls, path: str | Path, block_registry=None, layer_registry=None) -> MLnodeModel
    @classmethod
    def from_dict(cls, data: dict, block_registry=None, layer_registry=None) -> MLnodeModel
    def validate(self) -> ValidationResult
    def build(self) -> nn.Module
    def save(self, path: str | Path, *, expanded: bool = False) -> None

    schema: GraphSchema      # property — the (block-expanded) Pydantic schema
    graph: ParsedGraph       # property — the parsed DAG
```

- **`load` / `from_dict`** parse and structurally validate immediately: schema errors, unknown types, cycles, unreachable outputs, and bad port references raise here — not later. `block_registry` resolves `block_ref` nodes and `layer_registry` resolves JSON custom layer names not defined in the file's own `custom_layers`; when omitted, the global default registries are used. Files are read as UTF-8 on every platform.
- **`validate`** runs static shape checking and never raises for shape problems — inspect the returned `ValidationResult`.
- **`build`** calls `validate()` first and raises `ValidationError` (message lists every error) if invalid; otherwise returns an executable `MLnodeModule`.
- **`save`** writes UTF-8 JSON with `\n` line endings on every platform. By default it writes the original data with `Block` nodes and `custom_layers` intact; `expanded=True` writes the flattened graph (also used automatically when the model wasn't created from raw data).

## ValidationResult and ShapeError

`mlnode/schema/types.py` — dataclasses:

```python
@dataclass
class ValidationResult:
    valid: bool = True
    errors: list[ShapeError]
    node_shapes: dict[str, Shape]     # includes per-port keys like "split:left"

@dataclass
class ShapeError:
    node_id: str
    message: str
    expected_shape: Shape | None
    actual_shape: Shape | None
    source_node_id: str | None
```

`Shape` is `tuple[str | int, ...]`; `Dim` is `str | int`.

## Errors

`mlnode/errors.py`:

```
MLnodeError                      # base class for everything
├── ValidationError              # .node_id
│   ├── CycleDetectedError       # .cycle: list[str]
│   └── ShapeMismatchError       # .expected_shape, .actual_shape, .source_node_id
├── CustomLayerError             # .node_id, .original_error
└── SchemaVersionError           # .found_version, .supported_versions
```

## BlockRegistry and BlockDefinition

`mlnode/blocks/registry.py`, `mlnode/schema/models.py`:

```python
registry = BlockRegistry()
registry.register(name, block_def)                  # raises if name exists
registry.register_from_dict(name, {...})
registry.register_from_file(name, "block.json")
registry.get(name)          # BlockDefinition; raises MLnodeError if missing
registry.has(name)          # bool
registry.names              # sorted list[str]
len(registry)

from mlnode.blocks.registry import get_default_registry   # the global fallback
```

`BlockDefinition` is a Pydantic model with `nodes`, `edges`, `input_node`, `output_node`; it validates that the named input/output nodes and all edge endpoints exist inside the block. See [blocks.md](blocks.md).

## Schema Models

`mlnode/schema/models.py` — Pydantic models mirroring the JSON format: `GraphSchema` (top level, `input_nodes` property returns `Input` nodes in declaration order), `NodeSchema`, `EdgeSchema`, `MetadataSchema`, `BlockDefinition`. `SUPPORTED_VERSIONS = ["1.0"]`.

`mlnode/schema/types.py` — constants the rest of the system keys off:

| Constant | Contents |
|----------|----------|
| `CORE_LAYER_TYPES` | Every accepted node type string |
| `REDUCTION_OPS` | `Add`, `Subtract`, `Multiply`, `Divide`, `Concat`, `Stack` |
| `ELEMENTWISE_OPS` | `Add`, `Subtract`, `Multiply`, `Divide` |
| `TENSOR_OPS` | `Flatten`, `Reshape`, `Permute`, `Transpose`, `View`, `Unsqueeze`, `Squeeze`, `Split` |
| `MULTI_INPUT_LAYERS` | type → ordered port names (`Transformer` → `("src", "tgt")`, …) |
| `SYMBOLIC_DIMS` | `B C H W Seq D *` |

## Parser

`mlnode/parser/dag.py`:

```python
DAGParser.parse_file(path, block_registry=None) -> ParsedGraph
DAGParser.parse_dict(data, block_registry=None) -> ParsedGraph
DAGParser.parse_schema(schema, block_registry=None) -> ParsedGraph
```

Expands blocks, builds a NetworkX `DiGraph`, rejects self-loops/cycles, topologically sorts, verifies the output is reachable from an input, and checks `source_port` references.

`ParsedGraph` carries `schema`, `nodes: dict[str, ParsedNode]`, `execution_order: list[str]`, `nx_graph`, and an `input_nodes` property. `ParsedNode` wraps a `NodeSchema` with `incoming_edges` / `outgoing_edges` (`ParsedEdge`, whose `source_key` property is the tensor-store key, e.g. `"split:left"`).

## LayerRegistry and LayerDefinition

`mlnode/layers/json_layers.py`, `mlnode/schema/models.py` — see [custom-layers.md](custom-layers.md#json-layers):

```python
registry = LayerRegistry()
registry.register(name, LayerDefinition.model_validate({...}))
registry.register_from_dict(name, data)
registry.register_from_file(name, path)      # one definition per file
registry.register_library(path) -> list[str] # {"custom_layers": {name: definition}}
registry.get(name) -> LayerDefinition        # MLnodeError if missing
registry.has(name) -> bool
registry.names -> list[str]

mlnode.layers.get_default_layer_registry() -> LayerRegistry   # process-wide default
```

Registering the same name twice raises `MLnodeError`. `LayerDefinition` fields: `params: dict[str, ParamSpec]`, `nodes`, `edges`, `output_node`, `description`; `.input_ports` lists the Input node IDs in order. `ParamSpec` has `type`, `default`, `description`, and `.required` (true when no `default` key was given).

After parsing, `model.graph.custom_layers` maps each JSON layer node's ID to its parsed sub-graph (`ParsedGraph`, params already substituted).

## Validator

`mlnode/validator/shapes.py`:

```python
ShapeValidator(graph, input_shapes=None).validate() -> ValidationResult

register_shape(node_type)              # decorator — add a shape rule
register_custom_shape(node_type, fn)   # imperative version
get_shape_registry()                   # copy of the rule table
```

Rules have signature `(input_shapes: list[Shape], params: dict) -> Shape`. `input_shapes` optionally gives one shape per `Input` node (used for JSON layer sub-graphs); by default every Input gets the schema's `input_shape`. See [validation.md](validation.md).

## Executor

`mlnode/executor/builder.py`:

```python
ModuleBuilder.build(graph) -> nn.Module    # returns an MLnodeModule
```

`MLnodeModule` is the generated `nn.Module`. Behavior worth knowing:

- `forward(*args)` requires exactly one tensor per `Input` node (declaration order) and raises `MLnodeError` otherwise.
- Layers are registered in an `nn.ModuleDict` keyed by node ID, so `state_dict()` keys are stable and readable. Weight-shared nodes point at the same module instance.
- Reduction ops, tensor ops, and the fallback `reduction` field are executed inline (`torch.cat`, `torch.stack`, elementwise operators, `reshape`, …) — they own no parameters.
- Multi-input layers get their arguments ordered by `target_port` via `resolve_multi_inputs`.
- JSON custom layer nodes are built as nested `MLnodeModule`s; multi-input layers receive one tensor per Input node.
- Tuple-returning layers (`RNN`, `LSTM`, `GRU`, `MultiheadAttention`) have their primary tensor extracted (`result[0]`) unless the node declares `outputs` ports.

## Layers

`mlnode/layers/registry.py`:

```python
build_layer(node_schema, input_shape=None) -> nn.Module
register_layer(type_name)     # decorator for custom constructors
```

Handles core types (direct `torch.nn` construction from `params`), HuggingFace nodes, and custom layers. `TransformerEncoder`/`TransformerDecoder` have dedicated builders that accept flat layer params plus `num_layers` (default 6) and `norm` (bool → final `LayerNorm(d_model)`).

`mlnode/layers/custom.py` — see [custom-layers.md](custom-layers.md):

```python
resolve_custom_class(import_path, node_id) -> type[nn.Module]
validate_custom_layer(import_path, params, input_shape, node_id) -> CustomLayerResult
clear_cache()
```

## HuggingFace

`mlnode/huggingface/loader.py` — see [huggingface.md](huggingface.md):

```python
load_hf_pretrained(model_name) -> nn.Module    # AutoModel.from_pretrained
load_hf_from_config(config_name) -> nn.Module  # AutoConfig → AutoModel.from_config
```

## Utils

`mlnode/utils/__init__.py`:

```python
resolve_multi_inputs(node_id, node_type, edge_ports, values, ports=None) -> list
```

Orders incoming values (tensors or shapes) onto a multi-input layer's port list (`ports`, or the type's entry in `MULTI_INPUT_LAYERS`); enforces the labeling rules described in [json-schema.md](json-schema.md#multi-input-layers-and-target_port). Used by both the executor and the validator so the two always agree.
