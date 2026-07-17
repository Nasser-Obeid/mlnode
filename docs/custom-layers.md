# Custom Layers

Any Python class that subclasses `nn.Module` can be a node. MLnode imports it with `importlib` — never `exec()`/`eval()` — and can probe its output shape with a dummy forward pass. Implementation: `src/mlnode/layers/custom.py`.

## Usage

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

## What Happens When

**At parse time** (`load`/`from_dict`): the schema only checks that `path` is present.

**At shape-validation time** (`model.validate()`): the validator has no rule for `type: "custom"`, so a single-input custom node is treated as **shape-preserving** (output shape = input shape). If your layer changes the shape, downstream checks will be computed from the wrong shape — see below for the fix.

**At build time** (`model.build()`): the class is imported, verified to be an `nn.Module` subclass, and instantiated with `params`. Import failures, missing classes, non-`nn.Module` classes, and constructor errors all raise `CustomLayerError` carrying the node ID and the original exception.

## Full Validation with a Dummy Forward

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

## Caching

`validate_custom_layer` caches by `(import_path, frozen_params)`; a cache hit with the same `input_shape` skips the import and forward entirely. Clear it (e.g. between test runs or after editing the layer source) with:

```python
from mlnode.layers.custom import clear_cache
clear_cache()
```

## Failure Modes

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
