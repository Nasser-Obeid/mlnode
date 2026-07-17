# Static Shape Validation

`MLnodeModel.validate()` simulates tensor shapes through the DAG **without allocating any tensors**. It answers "will this architecture fit together?" before you spend GPU time. Implementation: `src/mlnode/validator/shapes.py`.

## How It Works

1. Every `Input` node is seeded with the graph's `input_shape`.
2. Nodes are visited in topological order. For each node the validator:
   - gathers its input shapes from the incoming edges (respecting `source_port` for multi-output sources);
   - for multi-input layers (`Transformer`, `MultiheadAttention`, …), reorders the shapes by `target_port` using the same `resolve_multi_inputs()` logic the executor uses — so static shapes match runtime semantics;
   - for elementwise ops (`Add`, `Subtract`, `Multiply`, `Divide`), checks all input shapes are identical;
   - looks up the node type's **shape transform rule** and computes the output shape.
3. Errors are **collected, not raised** — a failed node simply doesn't produce a shape, and downstream nodes that depend on it report a missing input. You fix everything in one pass.

```python
result = model.validate()
result.valid        # bool
result.errors       # list[ShapeError] — node_id, message, expected/actual shapes
result.node_shapes  # dict[str, Shape] — every successfully computed shape,
                    # including per-port entries like "split:part1"
```

`build()` runs `validate()` first and raises `ValidationError` listing every message if the graph is invalid.

## Symbolic Dimensions

Shapes are tuples of `int | str`. Recognized symbols: `B`, `Seq`, `C`, `H`, `W`, `D`, `*` (unknown). Rules do arithmetic on concrete ints and pass symbols through:

- `Conv2d` with input `(B, 3, 224, 224)`, `kernel_size=3, padding=1` → `(B, 64, 224, 224)` — `B` survives, spatial dims are computed.
- `Flatten` over symbolic dims yields `*` (product unknown).
- `Concat` along a dim with any symbolic contribution yields `*` for that dim.

## Consistency Checks

Beyond pure propagation, some rules verify params against incoming shapes:

| Rule | Check |
|------|-------|
| `MultiheadAttention` | query's last dim == `embed_dim` |
| `Transformer`, `TransformerEncoder(-Layer)`, `TransformerDecoder(-Layer)` | each input's last dim == `d_model` |
| `Bilinear` | last dims == `in1_features` / `in2_features` |
| Elementwise ops | all input shapes identical |
| Weight sharing | all nodes in a `shared_id` group must have identical type/params (schema-level) |

Decoder output follows `tgt`; `Transformer` output follows `tgt`; `MultiheadAttention` output follows `query`.

## Rule Status by Type

Three kinds of rules exist:

**Computed** — real output-shape arithmetic:
`Conv1d/2d/3d`, `ConvTranspose1d/2d/3d`, `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, `AdaptiveAvgPool1d/2d/3d`, `AdaptiveMaxPool1d/2d/3d`, `Linear`, `LazyLinear`, `Bilinear`, `Embedding`, `EmbeddingBag`, `RNN`, `GRU`, `LSTM`, `RNNCell`, `GRUCell`, `MultiheadAttention`, `Transformer`, `TransformerEncoder(-Layer)`, `TransformerDecoder(-Layer)`, `PixelShuffle`, `PixelUnshuffle`, `Add/Subtract/Multiply/Divide`, `Concat`, `Stack`, `Flatten`, `Reshape`, `View`, `Permute`, `Transpose`, `Squeeze`, `Unsqueeze`, `Split`.

**Passthrough** — output shape equals input shape:
all element-wise activations (`ReLU`, `GELU`, `Sigmoid`, `Tanh`, `ELU`, `SiLU`, `Softmax`, `LogSoftmax`, …), all norms (`BatchNorm*`, `LayerNorm`, `GroupNorm`, `InstanceNorm*`, lazy variants, `RMSNorm`, `SyncBatchNorm`, `LocalResponseNorm`), all dropouts, `Identity`.

**Pending (`NotImplementedError`)** — accepted by the schema but currently **fail validation** if used:
`LazyConv1d/2d/3d`, `LazyConvTranspose1d/2d/3d`, `Unfold`, `Fold`, `MaxUnpool1d/2d/3d`, `FractionalMaxPool2d/3d`, `LPPool1d/2d/3d`, `GLU`, `AdaptiveLogSoftmaxWithLoss`, `LSTMCell`.

A pending rule surfaces as a `Node 'x': shape computation failed: …` error. Since `build()` requires a clean validation, graphs using these types cannot currently be built through `MLnodeModel`.

## Known Limitations

- **Fallback `reduction` is not simulated.** For a non-reduction-op node with multiple inputs and a `reduction` field, the shape rule receives the raw input list and typically uses only the first shape. A `Linear` fed by two concatenated inputs may validate against the wrong width; runtime is authoritative there.
- **Custom layers are passthrough by default.** The validator has no rule for `type: "custom"`, so a single-input custom node is assumed shape-preserving. To validate real shapes, either call `mlnode.layers.custom.validate_custom_layer()` (dummy-forward probe — see [custom-layers.md](custom-layers.md)) or register a rule:

  ```python
  from mlnode.validator.shapes import register_custom_shape
  register_custom_shape("custom", lambda inputs, params: (*inputs[0][:-1], params["out_features"]))
  ```

  Note that this registers per *type string*, so it applies to every `custom` node in the graph.
- **Unknown types with one input** also pass through silently (this is what makes HuggingFace nodes parseable); with multiple inputs they error.
- **One `input_shape` for all inputs.** Multi-input graphs whose inputs need different shapes will produce spurious errors or spurious passes; schema v1.0 cannot express per-input shapes.

## Extending

Add a rule for a new type with the decorator:

```python
from mlnode.validator.shapes import register_shape

@register_shape("Upsample")
def _upsample_shape(inputs, params):
    inp = inputs[0]
    factor = params.get("scale_factor", 2)
    return (*inp[:2], *(d * factor if isinstance(d, int) else d for d in inp[2:]))
```

The function receives `(input_shapes: list[Shape], params: dict)` and returns a `Shape`. Raise any exception to report a validation error for the node. Remember to also register the constructor in `layers/registry.py` and add the type to `CORE_LAYER_TYPES`.
