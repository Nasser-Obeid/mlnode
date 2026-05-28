# MLnode v1.0 — Spec

**JSON-described DAG → executable PyTorch `nn.Module`.**

---

## 1. JSON Model Format

```json
{
  "mlnode._schema_version": "1.0",
  "metadata": {"name": "ResNet-50", "description": "Image classifier", "author": "alice"},
  "input_shape": ["B", 3, 224, 224],
  "output_node": "classifier",
  "nodes": [
    {"id": "input", "type": "Input", "params": {}},
    {"id": "conv1", "type": "Conv2d", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 7}},
    {"id": "classifier", "type": "Linear", "params": {"in_features": 25088, "out_features": 1000}}
  ],
  "edges": [
    {"source": "input", "source_port": null, "target": "conv1", "target_port": null},
    {"source": "conv1", "source_port": null, "target": "classifier", "target_port": null}
  ]
}
```

### Node Types

| Category | Types |
|----------|-------|
| I/O | `Input` — marks graph entry points (see §1.1). |
| Core | `Linear`, `Conv1d/2d/3d`, `ConvTranspose*`, `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `BatchNorm*`, `LayerNorm`, `GroupNorm`, `Dropout`, `MaxPool`, `AvgPool`, `AdaptivePool`, `Embedding`, `RNN`, `LSTM`, `GRU` |
| Attention | `MultiheadAttention`, `TransformerEncoderLayer`, `TransformerDecoderLayer` |
| Reduction Ops | `Add`, `Concat`, `Stack`, `Subtract`, `Multiply`, `Divide` — reduction semantics are **implicit by type** (see §1.2). |
| Tensor Ops | `Flatten`, `Reshape`, `Permute`, `Transpose`, `View`, `Unsqueeze`, `Squeeze`, `Split` |
| HuggingFace | `hf_model` — loads pretrained weights. `hf_config` — instantiates randomly initialized model from config class (see §5). |
| Custom | `{"type": "custom", "path": "user.layers.MyLayer", "params": {...}}` |

### 1.1 Input Node(s) & Multi-Input Forward

Every graph has at least one `Input` node. The `input_shape` field describes the shape fed into **each** `Input` node.

- **Single Input:** `forward(self, x)` → `store = {"input_0": x}`
- **Multiple Inputs:** `forward(self, *args)` → mapped **by node declaration order** in `nodes[]`. First `Input` node gets `args[0]`, second gets `args[1]`, etc.

```json
{
  "nodes": [
    {"id": "encoder_input", "type": "Input", "params": {}},
    {"id": "decoder_input", "type": "Input", "params": {}},
    {"id": "cross_attn", "type": "MultiheadAttention", "params": {"embed_dim": 512, "num_heads": 8}}
  ],
  "edges": [
    {"source": "encoder_input", "target": "cross_attn", "target_port": "key"},
    {"source": "decoder_input", "target": "cross_attn", "target_port": "query"}
  ]
}
# forward(self, encoder_tokens, decoder_tokens) maps encoder_input→args[0], decoder_input→args[1]
```

### 1.2 Multi-Input Reduction Rules

Reduction behavior is **determined by node type**, not a field:

| Node Type | Behavior |
|-----------|----------|
| `Add`, `Subtract`, `Multiply`, `Divide` | Element-wise operation. Validator enforces all input shapes are identical. |
| `Concat` | `torch.cat(inputs, dim=node.params.get("dim", -1))` |
| `Stack` | `torch.stack(inputs, dim=node.params.get("dim", 0))` |
| All other types receiving >1 input | Error unless `reduction` field is present (see fallback below). |

**Fallback `reduction` field:** For non-reduction-op nodes (e.g. `Linear`) that unexpectedly receive multiple inputs, allow `"reduction": "concat"|"add"|"stack"` with optional `"reduction_dim"`. If absent and >1 input → validation error.

```json
{"id": "dense", "type": "Linear", "params": {"in_features": 512, "out_features": 256}, "reduction": "concat", "reduction_dim": -1}
```

### 1.3 Edge Schema (Uniform)

Every edge uses the same 4-field schema. Ports omitted or `null` mean "default output/input":

```json
{"source": "node_id", "source_port": "part1", "target": "node_id", "target_port": null}
```

- `source_port`: Selects a named output port when the source node has multiple outputs (e.g. `Split`). `null` = default single output.
- `target_port`: Selects a named input port when the target node has multiple input slots (e.g. `MultiheadAttention` query/key/value). `null` = default single input.

### 1.4 Multi-Output Nodes

Named output ports declared on the node:

```json
{"id": "split", "type": "Split", "params": {"split_size_or_sections": [384, 384], "dim": -1}, "outputs": ["key", "value"]}
```

Referenced by edges:
```json
{"source": "split", "source_port": "key", "target": "attn", "target_port": "key"}
```

### 1.5 Weight Sharing

Nodes with identical `"shared_id"` reference the same `nn.Module` instance. Shared nodes must have **identical `type` and `params`**. Validator checks that input shapes to all shared instances are compatible — mismatches raise `ShapeMismatchError` with shared_id reference.

---

## 2. DAG Execution

```python
def forward(self, *args):
    input_nodes = [n for n in self.nodes if n.type == "Input"]
    store = {}
    for i, inp in enumerate(input_nodes):
        store[inp.id] = args[i] if i < len(args) else _raise_missing_input(inp.id)
    for node in self.execution_order:  # topo-sorted
        incoming = [store[e.source] for e in node.incoming_edges if e.source_port is None]
        incoming += [store[f"{e.source}:{e.source_port}"] for e in node.incoming_edges if e.source_port]
        if node.type in ("Add","Subtract","Multiply","Divide"):
            assert all_equal_shapes(incoming); result = elementwise_op(node.type, incoming)
        elif node.type == "Concat": result = torch.cat(incoming, dim=node.params.get("dim", -1))
        elif node.type == "Stack":  result = torch.stack(incoming, dim=node.params.get("dim", 0))
        elif node.reduction == "concat": result = torch.cat(incoming, dim=node.reduction_dim)
        elif node.reduction == "add":    result = sum(incoming)
        elif node.reduction == "stack":  result = torch.stack(incoming, dim=node.reduction_dim)
        elif len(incoming) > 1: raise RuntimeError
        else: result = node.layer(*incoming)
        if node.outputs:
            for j, port_name in enumerate(node.outputs):
                store[f"{node.id}:{port_name}"] = result[j] if isinstance(result, (tuple, list)) else result
        else:
            store[node.id] = result
    return store[self.output_node_id]
```

- Topological sort determines execution order.
- Cycles and self-loops rejected at parse time.

---

## 3. Shape Validation (Static)

- Simulates tensor shapes **without GPU memory** before `nn.Module` instantiation.
- Each layer type registers a shape transform rule: `Conv2d`: `(B,C,H,W) → (B,outC,⌊(H+2P-K)/S⌋+1, ⌊(W+2P-K)/S⌋+1)`.
- Symbolic dims: `B`, `C`, `H`, `W`, `Seq`, `D`, `*`. Concrete dims in params override symbols.
- Shared nodes: validator checks all instances receive compatible input shapes.
- Error format: `Node 'conv1' expects (B,3,224,224), got (B,1,224,224) from 'input'`.

---

## 4. Custom Layer Validation

1. Verify class inherits `nn.Module`.
2. Instantiate with provided params.
3. Run dummy forward with `torch.zeros(2, *expected_shape)`.
4. Catch exceptions → wrap in `CustomLayerError` with JSON node reference.
5. Cache result. No `exec()` / `eval()` ever.

---

## 5. HuggingFace Contract

| Field | Behavior | Shape Contract |
|-------|----------|----------------|
| `hf_model` | `AutoModel.from_pretrained(name)` — loads pretrained weights. | Input shape depends on model type (e.g. BERT expects `(B, Seq)` int64 token IDs). Shape validator uses `model.config` to infer output shape `(B, Seq, hidden_size)` for base models. |
| `hf_config` | `AutoConfig.from_name(cfg)` → `ModelClass(config)` — random init. | Same as above, but no weight download. |

Both variants expose the model's `forward()` directly. The shape validator inspects `config.hidden_size`, `config.num_attention_heads`, and model type tag to infer I/O shapes.

---

## 6. Schema Versioning

| Version | Meaning |
|---------|---------|
| Minor (`1.0`) | Backward-compatible additions |
| Major (`2.0.0`) | Breaking changes, explicit migration guide |

No auto-migration. Unsupported versions raise `SchemaVersionError`.

---

## 7. API

```python
class MLnodeModel:
    @classmethod
    def load(cls, path: str) -> MLnodeModel          # from .mlnode.json
    @classmethod
    def from_dict(cls, data: dict) -> MLnodeModel    # from dict
    def validate(self) -> ValidationResult        # shape check, no GPU
    def build(self) -> nn.Module                  # compile to executable PyTorch
    def save(self, path: str) -> None             # to .mlnode.json
```

**Error hierarchy:** `MLnodeError` → `ValidationError` (`CycleDetectedError`, `ShapeMismatchError`) / `CustomLayerError` / `SchemaVersionError`

---

## 8. Non-Goals

| Out of Scope | Reason |
|--------------|--------|
| Training loops | User responsibility |
| Datasets / dataloaders | Separate concern |
| Conditional execution | Static DAG only |
| Dynamic layers | PyTorch doesn't support |
| Auto schema migration | Clarity > convenience |
| CUDA kernels | Too low-level |
| Inference server | Future project |
| PyTorch → JSON | One-way only |
| Distributed training | v2+ scope |

---

## 9. Tech Stack

- Python ≥3.11, PyTorch ≥2.0, Transformers ≥4.30, NetworkX ≥3.0, Pydantic ≥2.0
- ReactFlow ≥11 (separate repo, shares only JSON schema)

---

## 10. Roadmap

| Week | Deliverable |
|------|-------------|
| 0 | Project scaffolding — repo, CI, test harness, Pydantic schema models |
| 1–2 | DAG parser, topo sort, execution loop, tensor store |
| 3–4 | Shape validator, symbolic dims, layer transform registry |
| 5 | Custom layer injection + sandboxed validation |
| 6 | Multi-output ports, Split/Concat/Stack ops |
| 7–8 | Schema v1.0 freeze, docs, examples (ResNet, Transformer, ViT) |

---

## 11. Example: ResNet Skip Connection

```json
{
  "mlnode_schema_version": "1.0",
  "metadata": {"name": "ResNet Block", "description": "Basic residual block with skip connection"},
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
    {"source": "input", "source_port": null, "target": "conv1", "target_port": null},
    {"source": "conv1", "source_port": null, "target": "conv2", "target_port": null},
    {"source": "input", "source_port": null, "target": "skip", "target_port": null},
    {"source": "conv2", "source_port": null, "target": "skip", "target_port": null},
    {"source": "skip", "source_port": null, "target": "relu", "target_port": null}
  ]
}
```
