<p align="center">
  <img src="https://raw.githubusercontent.com/Nasser-Obeid/MLnode/main/docs/assets/logo.png" alt="mlnode" width="340">
</p>

<p align="center"><b>JSON-described DAG → executable PyTorch <code>nn.Module</code></b></p>

mlnode lets you define deep learning architectures as JSON graphs. It parses, validates, and compiles them into standard PyTorch modules you can train and deploy. No `exec()`, no `eval()`, no magic. mlnode speeds up ML/DL development by letting you focus on the model architecture instead of the wiring code — and by catching shape errors statically, before a single tensor is allocated.

Graphs can be written by hand, generated programmatically, or drawn visually in **[mlnode-editor](https://nasser-obeid.github.io/mlnode-editor/)** ([source](https://github.com/Nasser-Obeid/mlnode-editor)), the companion web editor that exports the same `.mlnode.json` format. It runs entirely in your browser — nothing to install.

<p align="center">
  <img src="https://raw.githubusercontent.com/Nasser-Obeid/MLnode/main/docs/assets/attention_is_all_you_need.png" alt="Figure 1 from 'Attention Is All You Need'" width="380">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Nasser-Obeid/MLnode/main/docs/assets/attention_graph.png" alt="The same Transformer drawn as a DAG in mlnode-editor" width="100%">
</p>

<p align="center"><sub>Figure 1 of <a href="https://arxiv.org/abs/1706.03762"><i>Attention Is All You Need</i></a> — and the same architecture drawn in mlnode-editor. The graph is <a href="https://github.com/Nasser-Obeid/mlnode/blob/main/examples/attention_is_all_you_need.mlnode.json"><code>examples/attention_is_all_you_need.mlnode.json</code></a>: it validates, builds, and runs as a 45M-parameter <code>nn.Module</code>, with the paper's tied source/target embeddings (<code>shared_id</code>), Q/K/V-labeled attention edges, and encoder→decoder cross-attention.</sub></p>

## Installation

```bash
pip install mlnode            # core library
pip install "mlnode[hf]"      # + HuggingFace nodes (transformers)
```

Supports Python 3.11–3.15 on Linux and Windows. PyTorch ≥ 2.4, NetworkX ≥ 3.0, and Pydantic ≥ 2.0 are installed automatically (Python 3.14+ raises the floors to the first releases with wheels for it).

**Python 3.15:** PyPI does not carry PyTorch wheels for 3.15 yet; PyTorch publishes them on its own index. Install torch from there first, then mlnode:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA index, e.g. .../whl/cu128
pip install mlnode
```

For development, clone [the repo](https://github.com/Nasser-Obeid/mlnode) and `pip install -e ".[dev]"` — or run the setup script, which checks the Python version, creates `.venv`, installs everything (including torch from the PyTorch index on 3.15), and runs the tests:

```bash
./setup.sh                                          # Linux / macOS / Git Bash
powershell -ExecutionPolicy Bypass -File .\setup.ps1 # Windows
```

## Documentation

| Doc | Contents |
|-----|----------|
| [QUICKSTART.md](https://github.com/Nasser-Obeid/mlnode/blob/main/QUICKSTART.md) | Zero to trained model in five minutes |
| [docs/getting-started.md](https://github.com/Nasser-Obeid/mlnode/blob/main/docs/getting-started.md) | Install, first model, validate/build/save, training |
| [docs/json-schema.md](https://github.com/Nasser-Obeid/mlnode/blob/main/docs/json-schema.md) | Complete `.mlnode.json` format reference |
| [docs/api.md](https://github.com/Nasser-Obeid/mlnode/blob/main/docs/api.md) | Python API reference for every public class |
| [docs/validation.md](https://github.com/Nasser-Obeid/mlnode/blob/main/docs/validation.md) | How static shape validation works |
| [docs/blocks.md](https://github.com/Nasser-Obeid/mlnode/blob/main/docs/blocks.md) | Reusable sub-graphs: inline blocks, `block_ref`, `BlockRegistry` |
| [docs/custom-layers.md](https://github.com/Nasser-Obeid/mlnode/blob/main/docs/custom-layers.md) | Custom layers defined in JSON, or as your own `nn.Module` classes |
| [docs/huggingface.md](https://github.com/Nasser-Obeid/mlnode/blob/main/docs/huggingface.md) | Pretrained and config-initialized HuggingFace nodes |
| [docs/MLNODE_SPEC_v1_0.md](https://github.com/Nasser-Obeid/mlnode/blob/main/docs/MLNODE_SPEC_v1_0.md) | The frozen v1.0 specification |

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

**`hf_model`** — Load a pretrained HuggingFace model via `AutoModel.from_pretrained()`. Top-level field, not inside `params`. Requires the `[hf]` extra: `pip install "mlnode[hf]"`.

**`hf_config`** — Initialize from a HuggingFace config class (random weights, no download). Top-level field, not inside `params`. Requires the `[hf]` extra.

**`path`** — Import path for Python custom layers (e.g. `"user.layers.MyLayer"`). Only valid when `type` is `"custom"`. The class must inherit `nn.Module`.

**`layer`** — Name of a JSON custom layer (schema `1.1`), defined in the file's `custom_layers` or a `LayerRegistry`. Only valid when `type` is `"custom"`; a custom node takes `path` or `layer`.

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
| Linear | `Linear`, `Bilinear`, `LazyLinear`, `Identity` |
| Convolution | `Conv1d/2d/3d`, `ConvTranspose1d/2d/3d`, `LazyConv1d/2d/3d`*, `LazyConvTranspose1d/2d/3d`*, `Unfold`*, `Fold`* |
| Pooling | `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, `AdaptiveAvgPool1d/2d/3d`, `AdaptiveMaxPool1d/2d/3d`, `MaxUnpool1d/2d/3d`*, `FractionalMaxPool2d/3d`*, `LPPool1d/2d/3d`* |
| Activation | `ReLU`, `ReLU6`, `GELU`, `SiLU`, `SELU`, `CELU`, `ELU`, `LeakyReLU`, `PReLU`, `RReLU`, `Sigmoid`, `Hardsigmoid`, `LogSigmoid`, `Tanh`, `Tanhshrink`, `Hardtanh`, `Hardswish`, `Hardshrink`, `Softplus`, `Softshrink`, `Softsign`, `Mish`, `Threshold`, `GLU`*, `Softmax`, `Softmin`, `Softmax2d`, `LogSoftmax`, `AdaptiveLogSoftmaxWithLoss`* |
| Normalization | `BatchNorm1d/2d/3d`, `LazyBatchNorm1d/2d/3d`, `LayerNorm`, `GroupNorm`, `SyncBatchNorm`, `InstanceNorm1d/2d/3d`, `LazyInstanceNorm1d/2d/3d`, `LocalResponseNorm`, `RMSNorm` |
| Dropout | `Dropout`, `Dropout1d/2d/3d`, `AlphaDropout`, `FeatureAlphaDropout` |
| Sparse | `Embedding`, `EmbeddingBag` |
| Recurrent | `RNN`, `LSTM`, `GRU`, `RNNCell`, `GRUCell`, `LSTMCell`* |
| Attention | `MultiheadAttention`, `Transformer`, `TransformerEncoder`, `TransformerDecoder`, `TransformerEncoderLayer`, `TransformerDecoderLayer` |
| Vision Ops | `PixelShuffle`, `PixelUnshuffle` |
| Reduction Ops | `Add`, `Subtract`, `Multiply`, `Divide`, `Concat`, `Stack` |
| Tensor Ops | `Flatten`, `Reshape`, `Permute`, `Transpose`, `View`, `Unsqueeze`, `Squeeze`, `Split` |
| Composable | `Block` |
| Custom | `custom` |

\* Accepted by the schema and buildable, but the static shape rule is not implemented yet — graphs containing these types currently fail `validate()` (and therefore `build()`). See [docs/validation.md](https://github.com/Nasser-Obeid/mlnode/blob/main/docs/validation.md) for the full status table.

Reduction ops have implicit reduction semantics — they accept multiple inputs without needing a `reduction` field. Setting `reduction` on a reduction op is an error.

**Multi-input layers.** Some layers take several positional tensor arguments instead of being reduced. Label their incoming edges with `target_port`:

| Type | Ports (in order) |
|------|------------------|
| `Transformer` | `src`, `tgt` |
| `TransformerDecoder`, `TransformerDecoderLayer` | `tgt`, `memory` |
| `MultiheadAttention` | `query`, `key`, `value` — one input means self-attention (`q=k=v`); two means `query` + shared `key`/`value` |
| `Bilinear` | `input1`, `input2` |

Unlabeled edges are assigned to ports in declaration order. `TransformerEncoder`/`TransformerDecoder` are built from flat layer params plus `num_layers` (default 6) and `norm` (bool — appends a final `LayerNorm`).

All transformer/attention/RNN shape rules assume `batch_first: true` — set it in params (the editor does this by default).

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

### JSON Layers

Define a layer once in JSON — a parameterized sub-graph — and use it like any built-in type. Requires `"mlnode_schema_version": "1.1"`.

```json
{
  "mlnode_schema_version": "1.1",
  "input_shape": ["B", 16],
  "output_node": "ffn",
  "custom_layers": {
    "MLP": {
      "params": {"dim": {"type": "int"}, "expansion": {"type": "int", "default": 4}},
      "nodes": [
        {"id": "x",   "type": "Input"},
        {"id": "fc1", "type": "Linear", "params": {"in_features": "$dim", "out_features": "${dim * expansion}"}},
        {"id": "act", "type": "GELU"},
        {"id": "fc2", "type": "Linear", "params": {"in_features": "${dim * expansion}", "out_features": "$dim"}}
      ],
      "edges": [
        {"source": "x", "target": "fc1"},
        {"source": "fc1", "target": "act"},
        {"source": "act", "target": "fc2"}
      ],
      "output_node": "fc2"
    }
  },
  "nodes": [
    {"id": "input", "type": "Input"},
    {"id": "ffn", "type": "custom", "layer": "MLP", "params": {"dim": 16}}
  ],
  "edges": [{"source": "input", "target": "ffn"}]
}
```

Params are declared with types and defaults and checked when the file loads. `"$dim"` substitutes a param and `"${dim * expansion}"` evaluates safe arithmetic (no `eval()`). Layers can have several inputs, nest other layers, and live in a shared `LayerRegistry`. Each use is validated through its sub-graph and built as a nested `nn.Module`. See [docs/custom-layers.md](https://github.com/Nasser-Obeid/mlnode/blob/main/docs/custom-layers.md) and [examples/models/custom_layers.mlnode.json](https://github.com/Nasser-Obeid/mlnode/blob/main/examples/models/custom_layers.mlnode.json).

### Python Layers

Python custom layers are classes that inherit `nn.Module`. mlnode validates them by:

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

Runnable training notebooks live in `examples/notebooks/`:

- `mlnode_house_prices_training.ipynb` — MLP regressor on California Housing, the simplest end-to-end run (`house_prices_mlp.mlnode.json`)
- `mlnode_mnist_training.ipynb` — CNN classifier on MNIST (`mnist_cnn.mlnode.json`)
- `mlnode_transformer_training.ipynb` — TransformerEncoder classifier over MNIST rows-as-sequence (`mnist_transformer.mlnode.json`)
- `mlnode_gan_training.ipynb` — DCGAN-style GAN where generator and discriminator are both mlnode graphs (`gan_generator.mlnode.json`, `gan_discriminator.mlnode.json`)

Standalone graphs live in `examples/`: `attention_is_all_you_need.mlnode.json` (the images at the top of this README), `resnet_block.mlnode.json`, and `stacked_blocks.mlnode.json`.

All `.mlnode.json` files open directly in [mlnode-editor](https://nasser-obeid.github.io/mlnode-editor/) (Import → auto-layout).

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
pytest tests/ -v              # full suite (237 tests)
pytest tests/ --cov=mlnode    # with coverage (~90%)
pytest tests/unit/            # unit tests only
pytest tests/integration/     # full pipeline tests
pytest -m "not slow"          # skip tests that download HuggingFace models
```

Markers: `slow` (downloads HuggingFace models), `gpu` (requires CUDA).

## Project Structure

```
src/mlnode/
├── __init__.py          # Public API exports
├── model.py             # MLnodeModel — top-level API
├── errors.py            # Error hierarchy
├── schema/
│   ├── models.py        # Pydantic models (GraphSchema, NodeSchema, EdgeSchema, BlockDefinition, LayerDefinition)
│   └── types.py         # Dim, Shape, ValidationResult, constants
├── parser/
│   └── dag.py           # DAGParser → ParsedGraph (topo sort, cycle detection)
├── validator/
│   └── shapes.py        # ShapeValidator with per-type shape transform rules
├── executor/
│   └── builder.py       # ModuleBuilder → MLnodeModule(nn.Module)
├── layers/
│   ├── registry.py      # Maps type strings to nn.Module constructors
│   ├── custom.py        # Python custom layer validation (spec §4, steps 1-5)
│   └── json_layers.py   # JSON custom layers: LayerRegistry, params, $-references
├── blocks/
│   ├── registry.py      # BlockRegistry for reusable block definitions
│   └── expansion.py     # Flattens Block nodes into parent graph
├── huggingface/
│   └── loader.py        # Pretrained and config-based HF model loading
└── utils/
    └── __init__.py      # resolve_multi_inputs() — port binding shared by
                         # the executor and the shape validator
```

## Design Principles

1. **One schema, one parser** — JSON is the single source of truth.
2. **DAG only** — cycles rejected at parse time.
3. **Explicit over implicit** — all tensor flows are edges.
4. **Fail fast** — validation before instantiation.
5. **Versioned contract** — every JSON carries a schema version.
6. **No runtime codegen** — no `exec()`, `eval()`, `compile()`.
7. **Parser → Validator → Executor** — distinct phases with clear boundaries.
