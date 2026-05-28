# MLnode

**JSON-described DAG → executable PyTorch `nn.Module`.**

MLnode lets you define deep learning architectures as JSON graphs. It parses, validates, and compiles them into standard PyTorch modules you can train and deploy. No `exec()`, no `eval()`, no magic. The purpose of MLnode is to speed up the developement of ML and DL nueral networks, by eliminating the need to focus and more on the model architecture.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.11, PyTorch ≥ 2.0, Transformers ≥ 4.30, NetworkX ≥ 3.0, Pydantic ≥ 2.0.

## Quick Start

```python
from mlnode import MLnodeModel
import torch

model = MLnodeModel.from_dict({
    "mlnode_schema_version": "1.0",
    "input_shape": ["B", 10],
    "output_node": "fc",
    "nodes": [
        {"id": "input", "type": "Input", "params": {}},
        {"id": "fc", "type": "Linear", "params": {"in_features": 10, "out_features": 5}},
    ],
    "edges": [{"source": "input", "target": "fc"}]
})

result = model.validate()
print(result.valid)        # True
print(result.node_shapes)  # {'input': ('B', 10), 'fc': ('B', 5)}

module = model.build()
x = torch.randn(4, 10)
out = module(x)            # shape: (4, 5)
```

## Architecture

Every model goes through three distinct phases:

```
JSON → Parser → Validator → Executor → nn.Module
```

**Parser** takes JSON, validates it against the Pydantic schema, builds a NetworkX directed graph, runs topological sort, and detects cycles. Block nodes are expanded here by flattening their internal sub-graphs with namespaced IDs.

**Validator** propagates tensor shapes through the graph statically, without allocating GPU memory. Each layer type has a registered shape transform rule. Errors are collected (not fail-fast) so you can fix multiple issues in one pass.

**Executor** constructs a live `nn.Module` with proper weight sharing, reduction ops, and multi-output ports. The resulting module works like any PyTorch module — you can train it, save state dicts, export to ONNX, etc.

## API Reference

### MLnodeModel

The top-level class tying everything together.

```python
class MLnodeModel:
    @classmethod
    def load(cls, path: str, block_registry=None) -> MLnodeModel
    @classmethod
    def from_dict(cls, data: dict, block_registry=None) -> MLnodeModel
    def validate(self) -> ValidationResult
    def build(self) -> nn.Module
    def save(self, path: str, *, expanded: bool = False) -> None
```

**`load(path)`** — Load from a `.mlnode.json` file.

**`from_dict(data)`** — Create from a Python dictionary.

**`validate()`** — Run static shape validation. Returns a `ValidationResult` with `valid: bool`, `errors: list[ShapeError]`, and `node_shapes: dict[str, Shape]` mapping each node ID to its output shape.

**`build()`** — Compile to an executable `nn.Module`. Calls `validate()` first; raises `ValidationError` if invalid.

**`save(path)`** — Write to `.mlnode.json`. By default preserves block structure. Pass `expanded=True` to save the flattened graph.

### BlockRegistry

Stores reusable block definitions that can be referenced by `block_ref`.

```python
from mlnode import BlockRegistry, BlockDefinition

registry = BlockRegistry()
registry.register("resnet_block", BlockDefinition.model_validate({
    "input_node": "in", "output_node": "out",
    "nodes": [...], "edges": [...]
}))
registry.register_from_file("transformer_block", "blocks/transformer.json")

model = MLnodeModel.load("model.mlnode.json", block_registry=registry)
```

### Error Hierarchy

```
MLnodeError
├── ValidationError
│   ├── CycleDetectedError
│   └── ShapeMismatchError
├── CustomLayerError
└── SchemaVersionError
```

All errors carry context: `node_id`, `expected_shape`, `actual_shape`, `source_node_id` where applicable.

## JSON Schema Reference

### Top-Level Structure

```json
{
  "mlnode_schema_version": "1.0",
  "metadata": {"name": "My Model", "description": "...", "author": "..."},
  "input_shape": ["B", 3, 224, 224],
  "output_node": "classifier",
  "nodes": [...],
  "edges": [...]
}
```

**`mlnode_schema_version`** — Must be `"1.0"`. Unsupported versions raise `SchemaVersionError`.

**`metadata`** — Optional. Name, description, author.

**`input_shape`** — Shape fed to each Input node. Dimensions can be symbolic strings (`"B"`, `"Seq"`, `"D"`) or concrete integers. Symbolic dims let the validator check architecture correctness without fixing batch size or sequence length.

**`output_node`** — ID of the node whose output becomes the model's return value. Must exist in `nodes`.

### Node Schema

```json
{
  "id": "conv1",
  "type": "Conv2d",
  "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "padding": 1},
  "shared_id": "proj",
  "outputs": ["part1", "part2"],
  "reduction": "concat",
  "reduction_dim": -1,
  "hf_model": "bert-base-uncased",
  "hf_config": "GPT2Config",
  "path": "user.layers.MyLayer",
  "block": { ... },
  "block_ref": "resnet_block"
}
```

Only `id`, `type`, and `params` are required. All other fields are optional.

**`shared_id`** — Nodes with the same `shared_id` share a single `nn.Module` instance (weight sharing). They must have identical `type` and `params`.

**`outputs`** — Named output ports for multi-output nodes like `Split`. Referenced by edges via `source_port`.

**`reduction`** / **`reduction_dim`** — Fallback reduction strategy for non-reduction-op nodes that receive multiple inputs. Only allowed on nodes whose type is NOT a reduction op (`Add`, `Concat`, etc.).

**`hf_model`** — Load a pretrained HuggingFace model via `AutoModel.from_pretrained()`. Top-level field, not inside `params`.

**`hf_config`** — Initialize from a HuggingFace config class (random weights, no download). Top-level field, not inside `params`.

**`path`** — Import path for custom layers (e.g. `"user.layers.MyLayer"`). Only valid when `type` is `"custom"`. The class must inherit `nn.Module`.

**`block`** — Inline sub-graph definition for `Block` type nodes. Contains `input_node`, `output_node`, `nodes`, `edges`.

**`block_ref`** — Name of a block in the `BlockRegistry` for `Block` type nodes.

### Edge Schema

```json
{
  "source": "conv1",
  "source_port": "part1",
  "target": "fc1",
  "target_port": null
}
```

**`source_port`** — Selects a named output port when the source has multiple outputs. `null` for default single output.

**`target_port`** — Selects a named input port (e.g. `"query"`, `"key"`, `"value"` for attention). `null` for default.

### Supported Node Types

| Category | Types |
|----------|-------|
| I/O | `Input` |
| Linear & Conv | `Linear`, `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d` |
| Activation | `ReLU`, `GELU`, `Sigmoid`, `Tanh` |
| Normalization | `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`, `LayerNorm`, `GroupNorm` |
| Regularization | `Dropout` |
| Pooling | `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, `AdaptiveAvgPool1d/2d/3d`, `AdaptiveMaxPool1d/2d/3d` |
| Sequence | `Embedding`, `RNN`, `LSTM`, `GRU` |
| Attention | `MultiheadAttention`, `TransformerEncoderLayer`, `TransformerDecoderLayer` |
| Reduction Ops | `Add`, `Subtract`, `Multiply`, `Divide`, `Concat`, `Stack` |
| Tensor Ops | `Flatten`, `Reshape`, `Permute`, `Transpose`, `View`, `Unsqueeze`, `Squeeze`, `Split` |
| Composable | `Block` |
| Custom | `custom` |

Reduction ops have implicit reduction semantics — they accept multiple inputs without needing a `reduction` field. Setting `reduction` on a reduction op is an error.

### Symbolic Dimensions

The shape validator understands these symbolic dimension names:

| Symbol | Meaning | Typical use |
|--------|---------|-------------|
| `B` | Batch size | First dimension of any tensor |
| `Seq` | Sequence length | NLP/sequence models |
| `C` | Channels | CNN feature maps |
| `H`, `W` | Height, Width | Spatial dimensions |
| `D` | Generic dimension | Embedding size, hidden size |
| `*` | Unknown/inferred | When shape can't be determined |

Symbolic dims propagate through the graph. `Conv2d` with input `(B, 3, 224, 224)` outputs `(B, 64, 112, 112)` — the `B` passes through unchanged.

## Custom Layers

Custom layers are Python classes that inherit `nn.Module`. MLnode validates them by:

1. Importing the class via `importlib` (no `exec`/`eval`)
2. Verifying it inherits `nn.Module`
3. Instantiating with the provided `params`
4. Running a dummy forward pass with `torch.zeros(2, *shape)` to probe the output shape
5. Caching the result

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
class BottleneckProjection(nn.Module):
    def __init__(self, in_features: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(in_features, bottleneck)
        self.up = nn.Linear(bottleneck, in_features)

    def forward(self, x):
        return self.up(F.relu(self.down(x)))
```

## Blocks

Blocks are reusable sub-graphs. They're expanded at parse time — the rest of the system doesn't know they exist.

```json
{
  "id": "res1",
  "type": "Block",
  "block": {
    "input_node": "block_in",
    "output_node": "block_relu",
    "nodes": [
      {"id": "block_in", "type": "Input", "params": {}},
      {"id": "block_conv", "type": "Conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
      {"id": "block_bn", "type": "BatchNorm2d", "params": {"num_features": 64}},
      {"id": "block_skip", "type": "Add", "params": {}},
      {"id": "block_relu", "type": "ReLU", "params": {}}
    ],
    "edges": [
      {"source": "block_in", "target": "block_conv"},
      {"source": "block_conv", "target": "block_bn"},
      {"source": "block_in", "target": "block_skip"},
      {"source": "block_bn", "target": "block_skip"},
      {"source": "block_skip", "target": "block_relu"}
    ]
  }
}
```

After expansion, internal nodes get namespaced IDs: `res1/block_conv`, `res1/block_bn`, etc. The block's input node is removed and incoming edges are rewired to the first internal nodes.

Blocks can be chained (`res1 → res2`), nested (blocks containing blocks, up to depth 10), and referenced from a registry via `block_ref` instead of inline `block`.

## Examples

### ResNet Skip Connection

```json
{
  "mlnode_schema_version": "1.0",
  "metadata": {"name": "ResNet Block"},
  "input_shape": ["B", 64, 56, 56],
  "output_node": "relu",
  "nodes": [
    {"id": "input", "type": "Input", "params": {}},
    {"id": "conv1", "type": "Conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
    {"id": "conv2", "type": "Conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
    {"id": "skip", "type": "Add", "params": {}},
    {"id": "relu", "type": "ReLU", "params": {}}
  ],
  "edges": [
    {"source": "input", "target": "conv1"},
    {"source": "conv1", "target": "conv2"},
    {"source": "input", "target": "skip"},
    {"source": "conv2", "target": "skip"},
    {"source": "skip", "target": "relu"}
  ]
}
```

### Split and Concat

```json
{
  "nodes": [
    {"id": "input", "type": "Input", "params": {}},
    {"id": "split", "type": "Split", "params": {"split_size_or_sections": [32, 32], "dim": -1}, "outputs": ["left", "right"]},
    {"id": "fc_left", "type": "Linear", "params": {"in_features": 32, "out_features": 16}},
    {"id": "fc_right", "type": "Linear", "params": {"in_features": 32, "out_features": 16}},
    {"id": "cat", "type": "Concat", "params": {"dim": -1}},
    {"id": "out", "type": "Linear", "params": {"in_features": 32, "out_features": 10}}
  ],
  "edges": [
    {"source": "input", "target": "split"},
    {"source": "split", "source_port": "left", "target": "fc_left"},
    {"source": "split", "source_port": "right", "target": "fc_right"},
    {"source": "fc_left", "target": "cat"},
    {"source": "fc_right", "target": "cat"},
    {"source": "cat", "target": "out"}
  ]
}
```

### Multi-Input Graph

```python
model = MLnodeModel.from_dict({
    "mlnode_schema_version": "1.0",
    "input_shape": ["B", 32],
    "output_node": "add",
    "nodes": [
        {"id": "enc_input", "type": "Input", "params": {}},
        {"id": "dec_input", "type": "Input", "params": {}},
        {"id": "fc1", "type": "Linear", "params": {"in_features": 32, "out_features": 16}},
        {"id": "fc2", "type": "Linear", "params": {"in_features": 32, "out_features": 16}},
        {"id": "add", "type": "Add", "params": {}}
    ],
    "edges": [
        {"source": "enc_input", "target": "fc1"},
        {"source": "dec_input", "target": "fc2"},
        {"source": "fc1", "target": "add"},
        {"source": "fc2", "target": "add"}
    ]
})
module = model.build()
out = module(torch.randn(4, 32), torch.randn(4, 32))  # two inputs
```

Multiple Input nodes map to `forward(*args)` by declaration order in the `nodes` array.

## Running Tests

```bash
pytest tests/ -v              # all 111 tests
pytest tests/ --cov=mlnode    # with coverage (87%)
pytest tests/unit/            # unit tests only
pytest tests/integration/     # full pipeline tests
```

## Project Structure

```
src/mlnode/
├── __init__.py          # Public API exports
├── model.py             # MLnodeModel — top-level API
├── errors.py            # Error hierarchy
├── schema/
│   ├── models.py        # Pydantic models (GraphSchema, NodeSchema, EdgeSchema, BlockDefinition)
│   └── types.py         # Dim, Shape, ValidationResult, constants
├── parser/
│   └── dag.py           # DAGParser → ParsedGraph (topo sort, cycle detection)
├── validator/
│   └── shapes.py        # ShapeValidator with 45 shape transform rules
├── executor/
│   └── builder.py       # ModuleBuilder → MLnode(nn.Module)
├── layers/
│   ├── registry.py      # Maps type strings to nn.Module constructors
│   └── custom.py        # Custom layer validation (spec §4, steps 1-5)
├── blocks/
│   ├── registry.py      # BlockRegistry for reusable block definitions
│   └── expansion.py     # Flattens Block nodes into parent graph
└── huggingface/
    └── loader.py        # Pretrained and config-based HF model loading
```

## Design Principles

1. **One schema, one parser** — JSON is the single source of truth.
2. **DAG only** — cycles rejected at parse time.
3. **Explicit over implicit** — all tensor flows are edges.
4. **Fail fast** — validation before instantiation.
5. **Versioned contract** — every JSON carries a schema version.
6. **No runtime codegen** — no `exec()`, `eval()`, `compile()`.
7. **Parser → Validator → Executor** — distinct phases with clear boundaries.
