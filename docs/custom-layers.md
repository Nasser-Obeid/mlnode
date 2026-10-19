# Custom Layers

MLnode supports two kinds of custom layer, both used as `"type": "custom"` nodes:

| | JSON layers (`layer`) | Python layers (`path`) |
|---|---|---|
| Defined in | JSON — in the model file or a layer library | A Python `nn.Module` subclass |
| Requires schema | `1.1` | `1.0`+ |
| Built as | A nested MLnode graph (its own `nn.Module`) | Your class, instantiated with `params` |
| Static shape validation | Full, through the layer's sub-graph | Passthrough unless you probe or register a rule |
| Editable in mlnode-editor / shareable without code | Yes | No — the class must be importable |
| Arbitrary Python in `forward` | No — composed from MLnode node types | Yes |

Reach for a JSON layer whenever the layer can be composed from existing node types; fall back to a Python class for anything else.

## JSON Layers

A JSON layer is a named, parameterized sub-graph. Define it once under `custom_layers`, then use it from any number of nodes with different params. Implementation: `src/mlnode/layers/json_layers.py`. A complete runnable example: [`examples/models/custom_layers.mlnode.json`](../examples/models/custom_layers.mlnode.json).

```json
{
  "mlnode_schema_version": "1.1",
  "input_shape": ["B", 3, 32, 32],
  "output_node": "head",
  "custom_layers": {
    "ConvBNAct": {
      "description": "Conv2d → BatchNorm2d → ReLU with 'same' padding",
      "params": {
        "in_channels":  {"type": "int"},
        "out_channels": {"type": "int"},
        "kernel_size":  {"type": "int", "default": 3},
        "stride":       {"type": "int", "default": 1}
      },
      "nodes": [
        {"id": "x",    "type": "Input"},
        {"id": "conv", "type": "Conv2d", "params": {
          "in_channels": "$in_channels", "out_channels": "$out_channels",
          "kernel_size": "$kernel_size", "stride": "$stride",
          "padding": "${kernel_size // 2}", "bias": false
        }},
        {"id": "bn",   "type": "BatchNorm2d", "params": {"num_features": "$out_channels"}},
        {"id": "act",  "type": "ReLU"}
      ],
      "edges": [
        {"source": "x", "target": "conv"},
        {"source": "conv", "target": "bn"},
        {"source": "bn", "target": "act"}
      ],
      "output_node": "act"
    }
  },
  "nodes": [
    {"id": "input", "type": "Input"},
    {"id": "stem",  "type": "custom", "layer": "ConvBNAct", "params": {"in_channels": 3, "out_channels": 32}},
    {"id": "down",  "type": "custom", "layer": "ConvBNAct", "params": {"in_channels": 32, "out_channels": 64, "stride": 2}},
    ...
  ]
}
```

### Layer Definition

| Field | Required | Description |
|-------|----------|-------------|
| `params` | no | Declared parameters: `name → {"type", "default", "description"}`. |
| `nodes` | yes | The layer's nodes. Any node type is allowed, including `Block` and other JSON layers. |
| `edges` | yes | Edges between the layer's nodes. |
| `output_node` | yes | ID of the node whose output is the layer's output. |
| `description` | no | Free text. |

The layer's **`Input` nodes are its input ports**, in declaration order — exactly like a top-level graph. At least one is required.

### Params

Each param declares a `type` — `int`, `float` (accepts ints), `bool`, `str`, `list`, `dict`, or `any` (the default). A param **without a `"default"` key is required**. A `"default": null` makes it optional with no value.

Using a layer binds params at **parse time** (`load`/`from_dict`), so these all raise `CustomLayerError` with the node's ID immediately:

- a required param is missing,
- a param is not declared by the layer (catches typos),
- a value has the wrong type (`true` is not an `int`; `16.0` is not an `int`).

Param names must be valid identifiers, since expressions refer to them by name.

### Param References

Inside the layer's node `params`, strings starting with `$` are references:

| Syntax | Result |
|--------|--------|
| `"$name"` | The param's value, of any JSON type: `"$kernel_size"` → `3`, `"$sizes"` → `[1, 2]` |
| `"${expr}"` | Arithmetic over params: `"${kernel_size // 2}"`, `"${2 * dim}"`, `"${max(1, channels // reduction)}"` |
| `"$$text"` | The literal string `"$text"` (escape) |

Expressions support numbers, param names, `+ - * / // % **`, unary `-`, parentheses, and `min`, `max`, `abs`, `int`, `float`, `round`. They are evaluated by walking a whitelisted syntax tree — **never `eval()`**. Anything else (attribute access, calls to other functions, strings, conditionals) is rejected with a clear error. References are substituted in node `params` only; node IDs, types, and edges are fixed.

### Multiple Inputs

A layer with several `Input` nodes takes several inputs. Bind them with `target_port` set to the Input node's ID, or leave all edges unlabeled to bind in declaration order — the same rules as [multi-input core layers](json-schema.md#multi-input-layers-and-target_port):

```json
"GatedFusion": {
  "params": {"dim": {"type": "int"}},
  "nodes": [
    {"id": "a", "type": "Input"},
    {"id": "b", "type": "Input"},
    {"id": "cat",   "type": "Concat", "params": {"dim": -1}},
    {"id": "proj",  "type": "Linear", "params": {"in_features": "${2 * dim}", "out_features": "$dim"}},
    {"id": "gate",  "type": "Sigmoid"},
    {"id": "gated", "type": "Multiply"},
    {"id": "out",   "type": "Add"}
  ],
  "edges": [
    {"source": "a", "target": "cat"}, {"source": "b", "target": "cat"},
    {"source": "cat", "target": "proj"}, {"source": "proj", "target": "gate"},
    {"source": "gate", "target": "gated"}, {"source": "a", "target": "gated"},
    {"source": "gated", "target": "out"}, {"source": "b", "target": "out"}
  ],
  "output_node": "out"
}
```

```json
{"source": "encoder", "target": "fuse", "target_port": "a"},
{"source": "decoder", "target": "fuse", "target_port": "b"}
```

A single-input layer that receives several edges needs the node-level `reduction` fallback (`"add"`, `"concat"`, `"stack"`), as core layers do.

### Layer Libraries and `LayerRegistry`

To share layers across models, keep them out of the model file and register them in a `LayerRegistry`:

```python
from mlnode import LayerRegistry, MLnodeModel

registry = LayerRegistry()
registry.register_from_file("ConvBNAct", "layers/conv_bn_act.json")   # one definition
registry.register_library("layers/vision.json")                       # {"custom_layers": {name: definition, ...}}

model = MLnodeModel.load("model.mlnode.json", layer_registry=registry)
```

Lookup order for a node's `layer` name: the model file's own `custom_layers`, then the registry passed to `load`/`from_dict`, then the process-wide default registry (`mlnode.layers.get_default_layer_registry()`). A model file's definition therefore shadows a registry entry of the same name.

### Validation and Building

- **Shapes:** `validate()` runs the layer's sub-graph with the actual incoming shapes. Inner shapes and errors are reported under namespaced IDs — `res1/conv1/conv` for a layer nested in a layer — and error messages start with `In custom layer '<node id>' (<layer name>)`.
- **Modules:** each JSON layer node becomes its own nested `nn.Module`, so `state_dict()` keys follow the structure: `_layers.stem._layers.conv.weight`.
- **Weight sharing:** `shared_id` works on JSON layer nodes; shared nodes must use the same `layer` and identical `params`.
- **Nesting:** layers may use other layers (and Blocks) up to a depth of 10; deeper chains — e.g. a layer that uses itself — raise `CustomLayerError`.
- **Saving:** `save()` writes `custom_layers` back unchanged.

### JSON Layers vs Blocks

A [Block](blocks.md) is pasted into the parent graph (flattened, no params, one input, one output). A JSON layer is a reusable *type*: it takes params, can have several inputs, and keeps its own module boundary. Prefer JSON layers for anything you instantiate more than once with different settings.

## Python Layers

Any Python class that subclasses `nn.Module` can be a node. MLnode imports it with `importlib` — never `exec()`/`eval()` — and can probe its output shape with a dummy forward pass. Implementation: `src/mlnode/layers/custom.py`.

### Usage

```json
{
  "id": "proj",
  "type": "custom",
  "path": "my_project.layers.BottleneckProjection",
  "params": {"in_features": 512, "bottleneck": 64}
}
```

```python
# my_project/layers.py
import torch.nn as nn
import torch.nn.functional as F

class BottleneckProjection(nn.Module):
    def __init__(self, in_features: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(in_features, bottleneck)
        self.up = nn.Linear(bottleneck, in_features)

    def forward(self, x):
        return self.up(F.relu(self.down(x)))
```

Requirements:

- `path` is a dotted import path ending in the class name, importable from the current environment (`sys.path`).
- The class must subclass `nn.Module`.
- `params` are passed as keyword arguments to `__init__`, so they must be JSON-representable.
- For full validation, `forward` must accept one tensor and return one tensor.

### What Happens When

**At parse time** (`load`/`from_dict`): the schema only checks that `path` is present.

**At shape-validation time** (`model.validate()`): the validator has no rule for `type: "custom"`, so a single-input custom node is treated as **shape-preserving** (output shape = input shape). If your layer changes the shape, downstream checks will be computed from the wrong shape — see below for the fix.

**At build time** (`model.build()`): the class is imported, verified to be an `nn.Module` subclass, and instantiated with `params`. Import failures, missing classes, non-`nn.Module` classes, and constructor errors all raise `CustomLayerError` carrying the node ID and the original exception.

### Full Validation with a Dummy Forward

The spec §4 pipeline — import, verify, instantiate, probe, cache — is available as a function:

```python
from mlnode.layers.custom import validate_custom_layer

result = validate_custom_layer(
    "my_project.layers.BottleneckProjection",
    {"in_features": 512, "bottleneck": 64},
    input_shape=("B", 512),
    node_id="proj",
)
result.output_shape   # ('B', 512)
result.instance       # the validated nn.Module
```

Steps:

1. Import the class (`importlib` only) and verify it subclasses `nn.Module`.
2. Instantiate with `params`.
3. Run a dummy forward in `eval()` mode under `torch.no_grad()`. Symbolic dims are concretized with defaults `B=2, Seq=16, C=H=W=D=8, *=4`; e.g. `("B", 512)` → `torch.zeros(2, 512)`.
4. Wrap any exception in `CustomLayerError` with the node reference.
5. Cache the result, keyed by `(path, params)`.

The concrete output dims are mapped back to symbols where they match a symbolic input dim — a `(2, 512) → (2, 512)` probe with input `("B", 512)` reports `("B", 512)`, keeping the batch symbolic.

To make `model.validate()` use real shapes for custom nodes, register a shape rule (applies to **all** `custom` nodes in the process):

```python
from mlnode.validator.shapes import register_custom_shape
from mlnode.layers.custom import validate_custom_layer

def probe(inputs, params, *, _path="my_project.layers.BottleneckProjection"):
    return validate_custom_layer(_path, params, inputs[0], "custom").output_shape

register_custom_shape("custom", probe)
```

### Caching

`validate_custom_layer` caches by `(import_path, frozen_params)`; a cache hit with the same `input_shape` skips the import and forward entirely. Clear it (e.g. between test runs or after editing the layer source) with:

```python
from mlnode.layers.custom import clear_cache
clear_cache()
```

### Failure Modes

| Error message contains | Cause |
|------------------------|-------|
| `invalid custom path` | `path` has no dot — must be `module.path.ClassName` |
| `could not import module` | Module not on `sys.path`, or it raised on import |
| `class ... not found` | Typo in class name |
| `not a subclass of nn.Module` | Plain class/function at that path |
| `failed to instantiate` | `params` don't match `__init__` |
| `dummy forward failed` | `forward` rejected the probe tensor — check the concretized shape in the message |
| `returned <type>, expected torch.Tensor` | `forward` returned a tuple/dict — return a single tensor |

All are `CustomLayerError` (a subclass of `MLnodeError`) with `.node_id` and `.original_error`.
