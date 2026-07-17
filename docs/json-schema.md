# JSON Schema Reference

Complete reference for the `.mlnode.json` format (schema version `1.0`). The schema is enforced by the Pydantic models in `src/mlnode/schema/models.py`; structural DAG rules are enforced by the parser in `src/mlnode/parser/dag.py`.

## Top-Level Structure

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

| Field | Required | Description |
|-------|----------|-------------|
| `mlnode_schema_version` | yes | Must be `"1.0"`. Anything else raises `SchemaVersionError`. |
| `metadata` | no | `name`, `description`, `author` — all optional strings. |
| `input_shape` | yes | Shape fed to **each** `Input` node. Mix of symbolic strings and integers. |
| `output_node` | yes | ID of the node whose output is the model's return value. Must exist in `nodes`. |
| `nodes` | yes | List of node objects (below). At least one must have `type: "Input"`. |
| `edges` | yes | List of edge objects (below). |

## Node Schema

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

Only `id`, `type`, and `params` are required; everything else is optional.

| Field | Applies to | Description |
|-------|------------|-------------|
| `id` | all | Unique, non-empty string. Duplicates are rejected. |
| `type` | all | A core layer type, `"Block"`, or `"custom"`. Nodes with `hf_model`/`hf_config` set may use any type string. |
| `params` | all | Keyword arguments passed to the layer constructor (e.g. `nn.Conv2d(**params)`). |
| `shared_id` | any layer | Nodes with the same `shared_id` share one `nn.Module` instance (weight sharing). They must have **identical `type` and `params`**. |
| `outputs` | multi-output nodes (`Split`) | Named output ports, referenced by edges via `source_port`. |
| `reduction` | non-reduction-op nodes | Fallback for nodes receiving >1 input: `"concat"`, `"add"`, or `"stack"`. **Forbidden** on reduction ops (`Add`, `Concat`, …), which reduce implicitly. |
| `reduction_dim` | with `reduction` | Dimension for `concat`/`stack` (default `-1`). |
| `hf_model` | HuggingFace nodes | Model ID for `AutoModel.from_pretrained()`. Top-level field, **not** inside `params`. |
| `hf_config` | HuggingFace nodes | Config name for random-init construction (no weight download). Top-level field. |
| `path` | `type: "custom"` | Import path `"pkg.module.ClassName"`. Required for custom nodes. The class must subclass `nn.Module`. |
| `block` | `type: "Block"` | Inline sub-graph definition (see below). |
| `block_ref` | `type: "Block"` | Name of a block registered in a `BlockRegistry`. A Block node needs `block` **or** `block_ref`. |

## Edge Schema

Every edge uses the same uniform 4-field shape; omitted or `null` ports mean "default":

```json
{"source": "conv1", "source_port": "part1", "target": "fc1", "target_port": null}
```

| Field | Description |
|-------|-------------|
| `source` | ID of the producing node. Must exist. |
| `source_port` | Selects a named output port declared in the source node's `outputs`. Referencing an undeclared port is a parse error. |
| `target` | ID of the consuming node. Must exist. |
| `target_port` | Selects a named input port on multi-input layers (see below). |

Self-loops (`source == target`) and cycles are rejected at parse time with `CycleDetectedError`.

## Multi-Input Layers and `target_port`

Some layers take several positional tensor arguments instead of being reduced. Their ports, in positional order:

| Type | Ports |
|------|-------|
| `Transformer` | `src`, `tgt` |
| `TransformerDecoder`, `TransformerDecoderLayer` | `tgt`, `memory` |
| `MultiheadAttention` | `query`, `key`, `value` |
| `Bilinear` | `input1`, `input2` |

Rules (implemented in `mlnode.utils.resolve_multi_inputs`, shared by the executor and the validator):

- Label incoming edges with `target_port`, **or** leave all of them unlabeled — a mix is an error.
- Unlabeled edges bind to ports in edge declaration order.
- Each port accepts at most one edge; unknown port names are errors.
- `MultiheadAttention` shorthands: one input → self-attention (`q = k = v`); two inputs → `query` plus shared `key`/`value`.
- All other multi-input layers require every port to be connected (a `TransformerDecoder` with no `memory` edge fails validation).

All transformer/attention/RNN shape rules assume `batch_first: true` — set it in `params` (mlnode-editor emits it by default).

## Multi-Output Nodes and `source_port`

Declare named ports on the node, then reference them from edges:

```json
{"id": "split", "type": "Split",
 "params": {"split_size_or_sections": [384, 384], "dim": -1},
 "outputs": ["key", "value"]}
```

```json
{"source": "split", "source_port": "key", "target": "attn", "target_port": "key"}
```

At runtime the node's result tuple is unpacked into the ports in order. Declaring more ports than the layer returns is an execution error.

## Multiple Input Nodes

A graph may have any number of `Input` nodes. The built module's `forward(*args)` maps arguments to `Input` nodes **by declaration order in the `nodes` array**: the first `Input` node gets `args[0]`, the second `args[1]`, and so on. `input_shape` applies to every input.

## Blocks

A `Block` node embeds (or references) a reusable single-input/single-output sub-graph:

```json
{
  "id": "res1",
  "type": "Block",
  "block": {
    "input_node": "in",
    "output_node": "relu",
    "nodes": [ ...  ],
    "edges": [ ... ]
  }
}
```

Blocks are expanded (flattened) at parse time; internal IDs become `res1/<internal_id>`. See [blocks.md](blocks.md) for the full expansion semantics, nesting, and the registry.

## Validation Order

A graph passes through these gates, in order:

1. **Pydantic schema validation** — field types, version check, `output_node` existence, at least one `Input`, unique IDs, edge endpoints exist, weight-sharing groups consistent, per-node type rules (`custom` needs `path`, `Block` needs `block`/`block_ref`, no `reduction` on reduction ops).
2. **Block expansion** — `Block` nodes flattened (recursively, max depth 10).
3. **DAG construction** — self-loop and cycle detection, topological sort, `output_node` reachable from at least one `Input`, `source_port` references declared.
4. **Shape validation** (`model.validate()`) — static shape propagation; see [validation.md](validation.md).

Steps 1–3 run eagerly in `MLnodeModel.load()` / `from_dict()` and raise on failure. Step 4 collects all errors into a `ValidationResult`.

## Symbolic Dimensions

Dimensions in `input_shape` may be symbolic strings; the validator propagates them without fixing their value:

| Symbol | Meaning |
|--------|---------|
| `B` | Batch size |
| `Seq` | Sequence length |
| `C` | Channels |
| `H`, `W` | Height, width |
| `D` | Generic dimension (embedding/hidden size) |
| `*` | Unknown / not statically inferable |

Concrete integers participate in arithmetic (a `Conv2d` computes real output sizes); symbols pass through untouched where the rule allows it.

## Supported Node Types

The authoritative list is `CORE_LAYER_TYPES` in `src/mlnode/schema/types.py` — the README's [Supported Node Types](../README.md#supported-node-types) table mirrors it. Some accepted types do not have a static shape rule yet and therefore fail `validate()`; [validation.md](validation.md) tracks per-type status.
