# HuggingFace Nodes

A node can wrap a HuggingFace Transformers model — either pretrained weights or a randomly initialized model built from a config. Implementation: `src/mlnode/huggingface/loader.py`.

## The Two Fields

Both are **top-level node fields, not `params` entries**:

```json
{"id": "encoder", "type": "Linear", "hf_model": "bert-base-uncased", "params": {}}
```

```json
{"id": "gpt", "type": "Linear", "hf_config": "gpt2", "params": {}}
```

| Field | Behavior | Weights |
|-------|----------|---------|
| `hf_model` | `AutoModel.from_pretrained(name)` | Downloaded, pretrained |
| `hf_config` | `AutoConfig.from_pretrained(name)` → `AutoModel.from_config(config)` | Random init, no download |

When either field is present, the schema accepts the node regardless of its `type` string, and the executor builds the HF model instead of a core layer. The model's own `forward()` is called directly with the incoming tensor.

## Shape Validation

There is no dedicated shape rule for HF nodes: with a single input they are treated as **shape-preserving** during `validate()`. That is usually wrong for a transformer (`(B, Seq)` token IDs → `(B, Seq, hidden_size)`), so treat static shapes downstream of an HF node as advisory, or register a rule for your model:

```python
from mlnode.validator.shapes import register_custom_shape
# BERT-base: (B, Seq) → (B, Seq, 768). Register under the node's `type` string.
register_custom_shape("Linear", lambda inputs, params: (*inputs[0], 768))
```

(Note the rule keys off the node's `type` string, so give HF nodes a dedicated type name if you do this.)

## Practical Notes

- **Inputs are passed positionally.** The incoming tensor becomes the model's first `forward` argument (`input_ids` for BERT-style models) — so feed integer token-ID tensors, not floats.
- **Outputs are model-specific.** Most `AutoModel` classes return a `ModelOutput` object, not a raw tensor. The generic tuple-extraction in the executor only handles `RNN/LSTM/GRU/MultiheadAttention`, so a downstream core layer may receive a `ModelOutput` and fail. Today HF nodes are most useful as the **final** node of a graph, or feeding a custom layer that unwraps `output.last_hidden_state`.
- **Downloads and caching** follow the standard HuggingFace environment (`HF_HOME`, offline mode, etc.).
- Tests touching pretrained downloads are marked `slow` — run `pytest -m "not slow"` to skip them.

## In the Editor

mlnode-editor currently ships its HuggingFace palette entries commented out, but it round-trips `hf_model` / `hf_config` fields on import/export and shows an editable 🤗 panel for nodes that carry them. Hand-written JSON with HF nodes opens fine in the editor.
