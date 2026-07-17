# Getting Started

MLnode turns a JSON description of a neural network into an executable PyTorch `nn.Module`. This guide takes you from installation to a trained model.

## Installation

```bash
pip install mlnode            # core library
pip install "mlnode[hf]"      # + HuggingFace nodes (transformers)
```

For development, clone the repo and install editable — or use the bundled script, which checks your Python version, creates `.venv`, and installs the package with dev dependencies:

```bash
git clone https://github.com/Nasser-Obeid/MLnode.git && cd MLnode
pip install -e ".[dev]"       # or: ./setup.sh
```

Requirements: Python ≥ 3.11, PyTorch ≥ 2.4, NetworkX ≥ 3.0, Pydantic ≥ 2.0 (Transformers ≥ 4.30 only with the `[hf]` extra).

## Your First Model

A model is a DAG: `nodes` are layers, `edges` are tensor flows. Every graph needs at least one `Input` node, an `input_shape`, and an `output_node`.

```python
from mlnode import MLnodeModel
import torch

model = MLnodeModel.from_dict({
    "mlnode_schema_version": "1.0",
    "metadata": {"name": "Tiny MLP"},
    "input_shape": ["B", 784],
    "output_node": "out",
    "nodes": [
        {"id": "input", "type": "Input", "params": {}},
        {"id": "fc1", "type": "Linear", "params": {"in_features": 784, "out_features": 128}},
        {"id": "act", "type": "ReLU", "params": {}},
        {"id": "out", "type": "Linear", "params": {"in_features": 128, "out_features": 10}},
    ],
    "edges": [
        {"source": "input", "target": "fc1"},
        {"source": "fc1", "target": "act"},
        {"source": "act", "target": "out"},
    ],
})
```

`"B"` is a **symbolic dimension** — the validator checks the architecture without fixing a batch size. See [validation.md](validation.md) for the full list of symbols.

## Validate

```python
result = model.validate()
print(result.valid)        # True
print(result.node_shapes)  # {'input': ('B', 784), 'fc1': ('B', 128), 'act': ('B', 128), 'out': ('B', 10)}
```

Validation is **static** — no tensors are allocated, no GPU is touched. All errors are collected in `result.errors` (a list of `ShapeError`) rather than failing on the first one, so you can fix several issues in one pass.

## Build and Run

```python
module = model.build()          # a regular nn.Module
x = torch.randn(32, 784)
logits = module(x)              # shape: (32, 10)
```

`build()` calls `validate()` first and raises `ValidationError` if the graph is invalid. The returned module is ordinary PyTorch: it works with optimizers, `state_dict()`, `torch.save`, ONNX export, `.to(device)`, and so on.

## Train

Nothing MLnode-specific here — it is a standard training loop:

```python
import torch.nn.functional as F

optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

for x, y in dataloader:
    optimizer.zero_grad()
    loss = F.cross_entropy(module(x), y)
    loss.backward()
    optimizer.step()
```

Complete, runnable examples live in [`examples/notebooks/`](../examples/notebooks/):

- `mlnode_mnist_training.ipynb` — CNN classifier on MNIST
- `mlnode_transformer_training.ipynb` — TransformerEncoder classifier
- `mlnode_gan_training.ipynb` — GAN with generator and discriminator both defined as MLnode graphs

## Load and Save Files

```python
model = MLnodeModel.load("model.mlnode.json")
model.save("copy.mlnode.json")                  # preserves Block structure
model.save("flat.mlnode.json", expanded=True)   # saves the flattened graph
```

The `.mlnode.json` files are the same format the visual editor ([mlnode-editor](../../mlnode-editor/)) imports and exports, so you can draw a model in the editor, export it, and train it here — or hand-write JSON and open it in the editor.

## Multiple Inputs

Declare several `Input` nodes; `forward()` then takes one tensor per input, mapped **by declaration order** in the `nodes` array:

```python
model = MLnodeModel.from_dict({
    "mlnode_schema_version": "1.0",
    "input_shape": ["B", 32],
    "output_node": "add",
    "nodes": [
        {"id": "a", "type": "Input", "params": {}},
        {"id": "b", "type": "Input", "params": {}},
        {"id": "fc_a", "type": "Linear", "params": {"in_features": 32, "out_features": 16}},
        {"id": "fc_b", "type": "Linear", "params": {"in_features": 32, "out_features": 16}},
        {"id": "add", "type": "Add", "params": {}},
    ],
    "edges": [
        {"source": "a", "target": "fc_a"},
        {"source": "b", "target": "fc_b"},
        {"source": "fc_a", "target": "add"},
        {"source": "fc_b", "target": "add"},
    ],
})
module = model.build()
out = module(torch.randn(4, 32), torch.randn(4, 32))
```

Note that `input_shape` describes the shape fed to **every** `Input` node — per-input shapes are not supported in schema v1.0.

## Common First Errors

| Symptom | Cause |
|---------|-------|
| `SchemaVersionError` | `mlnode_schema_version` is not `"1.0"` |
| `ValueError: unknown type '...'` | Node `type` is not in the supported list (check spelling/case) |
| `CycleDetectedError` | The graph has a cycle or a self-loop — only DAGs are allowed |
| `ValidationError: Output node ... is not reachable` | No path of edges from any `Input` to `output_node` |
| `Cannot build: N validation error(s)` | Shape mismatch somewhere — read `model.validate().errors` for details |
| Shape errors on transformer/RNN nodes | Set `batch_first: true` in params; all sequence shape rules assume it |

## Next Steps

- [json-schema.md](json-schema.md) — every field of the format
- [validation.md](validation.md) — how shapes propagate, symbolic dims, rule status per type
- [blocks.md](blocks.md) — reusable sub-graphs
- [custom-layers.md](custom-layers.md) — plug in your own `nn.Module`
- [huggingface.md](huggingface.md) — pretrained models as nodes
- [api.md](api.md) — the full Python API
