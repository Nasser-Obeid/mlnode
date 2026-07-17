# MLnode Quickstart

From zero to a trained PyTorch model in about five minutes. Every snippet here is copy-paste runnable.

## 1. Install

```bash
pip install mlnode              # requires Python ≥ 3.11
```

Or, for a development setup:

```bash
git clone https://github.com/Nasser-Obeid/MLnode.git && cd MLnode
./setup.sh                      # checks Python ≥ 3.11, creates .venv, installs everything
source .venv/bin/activate
```

Verify:

```bash
python -c "import mlnode; print('ok')"
```

## 2. Define a Model as JSON

A model is a DAG: nodes are layers, edges are tensor flows. Save this as `quickstart.py`:

```python
import torch
import torch.nn.functional as F
from mlnode import MLnodeModel

model = MLnodeModel.from_dict({
    "mlnode_schema_version": "1.0",
    "metadata": {"name": "Quickstart MLP"},
    "input_shape": ["B", 784],            # "B" = symbolic batch dimension
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

## 3. Validate — Before Any Tensor Exists

```python
result = model.validate()
print("valid:", result.valid)      # True
print("shapes:", result.node_shapes)
# {'input': ('B', 784), 'fc1': ('B', 128), 'act': ('B', 128), 'out': ('B', 10)}
```

Shape checking is static — no GPU, no allocation. If something doesn't fit, `result.errors` lists **every** problem at once.

## 4. Build and Train

```python
module = model.build()             # a regular nn.Module

x = torch.randn(64, 784)           # stand-in for your real data
y = torch.randint(0, 10, (64,))
optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

for step in range(5):
    optimizer.zero_grad()
    loss = F.cross_entropy(module(x), y)
    loss.backward()
    optimizer.step()
    print(f"step {step}: loss {loss.item():.4f}")
```

The module is plain PyTorch — `state_dict()`, `.to("cuda")`, ONNX export, everything works.

## 5. Save / Reload

```python
model.save("quickstart_mlp.mlnode.json")
reloaded = MLnodeModel.load("quickstart_mlp.mlnode.json")
```

Run it:

```bash
python quickstart.py
```

## 6. Or Draw It Instead

The same `.mlnode.json` files can be drawn visually in [mlnode-editor](https://github.com/Nasser-Obeid/MLnode-editor) — see its [QUICKSTART](https://github.com/Nasser-Obeid/MLnode-editor/blob/main/QUICKSTART.md). Export from the editor, `MLnodeModel.load()` here, train. Files round-trip in both directions.

## Where to Go Next

- Real training runs on MNIST (CNN, Transformer, GAN): [`examples/notebooks/`](examples/notebooks/)
- Skip connections, `Split`/`Concat`, multiple inputs: [docs/getting-started.md](docs/getting-started.md)
- The full JSON format: [docs/json-schema.md](docs/json-schema.md)
- Reusable sub-graphs, custom layers, HuggingFace nodes: [docs/](docs/README.md)
