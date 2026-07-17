# Blocks — Reusable Sub-Graphs

A `Block` node embeds a whole sub-graph behind a single node. Blocks are **expanded at parse time** — flattened into the parent graph before validation and execution — so the rest of the system never sees them. Implementation: `src/mlnode/blocks/expansion.py`.

## Inline Blocks

```json
{
  "id": "res1",
  "type": "Block",
  "params": {},
  "block": {
    "input_node": "in",
    "output_node": "out",
    "nodes": [
      {"id": "in",   "type": "Input",       "params": {}},
      {"id": "conv", "type": "Conv2d",      "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
      {"id": "bn",   "type": "BatchNorm2d", "params": {"num_features": 64}},
      {"id": "skip", "type": "Add",         "params": {}},
      {"id": "out",  "type": "ReLU",        "params": {}}
    ],
    "edges": [
      {"source": "in",   "target": "conv"},
      {"source": "conv", "target": "bn"},
      {"source": "in",   "target": "skip"},
      {"source": "bn",   "target": "skip"},
      {"source": "skip", "target": "out"}
    ]
  }
}
```

A block definition has four required fields — `input_node`, `output_node`, `nodes`, `edges` — and is validated on parse: the named input/output nodes and every edge endpoint must exist inside the block.

## Registry Blocks (`block_ref`)

Define once, use many times:

```python
from mlnode import BlockRegistry, BlockDefinition, MLnodeModel

registry = BlockRegistry()
registry.register("resnet_block", BlockDefinition.model_validate({
    "input_node": "in", "output_node": "out",
    "nodes": [...], "edges": [...],
}))
registry.register_from_file("transformer_block", "blocks/transformer.json")

model = MLnodeModel.load("model.mlnode.json", block_registry=registry)
```

```json
{"id": "res1", "type": "Block", "params": {}, "block_ref": "resnet_block"},
{"id": "res2", "type": "Block", "params": {}, "block_ref": "resnet_block"}
```

If no registry is passed to `load`/`from_dict`, the process-wide default registry (`mlnode.blocks.registry.get_default_registry()`) is consulted. Registering the same name twice raises `MLnodeError`.

> Note: two `block_ref` nodes referencing the same block produce **independent copies** of the layers (separate weights). For weight sharing use `shared_id` on individual nodes.

## Expansion Semantics

For a Block node `res1` wrapping the definition above:

1. Internal nodes are copied into the parent graph with **namespaced IDs**: `res1/conv`, `res1/bn`, `res1/skip`, `res1/out`.
2. The block's `input_node` is **removed** — it is pure wiring. Every edge that pointed *at* `res1` is rewired to the internal nodes that consumed the block input (`res1/conv` and `res1/skip` here), preserving any `target_port` the internal edges carried.
3. Edges *leaving* `res1` are rewired to originate from `res1/out` (the namespaced `output_node`).
4. Block-to-block edges (`res1 → res2`) connect `res1/<out>` directly to `res2`'s first internal consumers.
5. If the graph's `output_node` was the block itself, it is remapped to `res1/<out>`.

Expansion recurses for nested blocks (blocks containing blocks) up to a depth of **10**, after which `MLnodeError` is raised — this also catches circular `block_ref` chains.

You can see the expanded form of any model:

```python
model = MLnodeModel.load("model.mlnode.json")
print([n.id for n in model.schema.nodes])         # namespaced IDs — schema is post-expansion
model.save("flat.mlnode.json", expanded=True)     # write the flattened graph
model.save("orig.mlnode.json")                    # default: original JSON with blocks intact
```

## Constraints

- A block has exactly **one input node and one output node**. Multi-input/multi-output blocks are not expressible in schema v1.0.
- Since expansion happens before shape validation, `validate()` reports errors against the namespaced IDs (`res1/conv`), which tells you both which block instance and which internal node failed.
- `params` on the Block node itself is ignored — configuration lives on the internal nodes.
