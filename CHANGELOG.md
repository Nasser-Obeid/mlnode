# Changelog

All notable changes to **mlnode** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.0.1] — 2026-08-02

Documentation and packaging metadata only — no library code changed.

### Changed

- The `Editor` project URL now points at the **hosted** editor,
  <https://nasser-obeid.github.io/mlnode-editor/>, which runs entirely in the
  browser; the repository moved to a separate `Editor source` URL.
- README links the live editor, and this release refreshes the PyPI project
  page with the logo and Transformer showcase added after 1.0.0 shipped.

## [1.0.0] — 2026-07-18

First public release on PyPI: `pip install mlnode`.

### Packaging

- Distribution metadata for PyPI: author, `[project.urls]`, keywords,
  classifiers, MIT `LICENSE` file (PEP 639), and a `py.typed` marker (PEP 561).
- Version is single-sourced from `mlnode.__version__` via hatchling.
- `transformers` is now an **optional dependency** — install with
  `pip install "mlnode[hf]"` to use HuggingFace nodes. Without it, building a
  graph with `hf_model`/`hf_config` raises an `MLnodeError` explaining the
  install command.
- Minimum `torch` raised to **2.4** (the registry maps `RMSNorm` to
  `nn.RMSNorm`, added in torch 2.4).

### Added

- **`TransformerEncoder` / `TransformerDecoder` are now buildable from JSON.**
  PyTorch's classes require a layer *instance* plus `num_layers`, which cannot
  come straight from JSON params. Dedicated builders in `layers/registry.py`
  accept the flat `TransformerEncoderLayer`/`DecoderLayer` params plus:
  - `num_layers` (int, default `6`) — depth of the stack
  - `norm` (bool, default `false`) — append a final `LayerNorm(d_model)`
- **Multi-input layer support in the executor** (`executor/builder.py`).
  Layers whose `forward()` takes several positional tensors are no longer
  forced through the reduction fallback. The port table lives in
  `schema/types.py` (`MULTI_INPUT_LAYERS`):

  | Type | Ports (in order) |
  |------|------------------|
  | `Transformer` | `src`, `tgt` |
  | `TransformerDecoder`, `TransformerDecoderLayer` | `tgt`, `memory` |
  | `MultiheadAttention` | `query`, `key`, `value` |
  | `Bilinear` | `input1`, `input2` |

  Edges labeled with `target_port` bind by name; unlabeled edges bind in
  declaration order. `MultiheadAttention` supports shorthands: one input →
  self-attention (`q=k=v`), two inputs → `query` + shared `key`/`value`.
  Shared resolution logic: `mlnode.utils.resolve_multi_inputs()` — used by
  both the executor and the shape validator so static shapes match runtime
  semantics.
- **Static consistency checks** in the shape validator: transformer-family
  rules verify the input's last dim against `d_model` (and `embed_dim` for
  `MultiheadAttention`); `Bilinear` checks `in1_features`/`in2_features`.
  Decoder output follows `tgt`; `Transformer` output follows `tgt`.
- `LazyBatchNorm1d/2d/3d` registered in the layer registry (they were listed
  in `CORE_LAYER_TYPES` but not buildable).
- Proper shape rules for `LazyLinear` (out_features known statically) and the
  lazy norm layers (passthrough).
- **New example notebooks** in `examples/notebooks/`:
  - `mlnode_transformer_training.ipynb` + `mnist_transformer.mlnode.json` —
    `TransformerEncoder` classifier treating MNIST images as 28-row sequences.
  - `mlnode_gan_training.ipynb` + `gan_generator.mlnode.json` /
    `gan_discriminator.mlnode.json` — DCGAN-style GAN where generator and
    discriminator are both MLnode graphs.
- Test coverage: `tests/unit/test_multi_input_layers.py` (transformer
  build/forward, port binding and error cases, vision ops, registry fixes).

### Fixed

- `RReLU` was mapped to `nn.PReLU` in the layer registry.
- `PixelUnshuffle` was constructed with `upscale_factor` instead of
  `downscale_factor` (always a `TypeError` at build).
- `PixelShuffle`/`PixelUnshuffle` were listed in `ELEMENTWISE_OPS`, which made
  the executor return their **input unchanged** (silent identity op). They now
  execute as real layers.
- `register_shape("LazyLinear")` was missing the `@`, so the rule was never
  registered; the same dead pattern existed for the Upsample family (removed —
  those types are not in `CORE_LAYER_TYPES` yet).
- `LazyInstanceNorm1d` was registered three times instead of 1d/2d/3d.
- Duplicate `AdaptiveAvgPool1d/2d/3d` entries in `CORE_LAYER_TYPES` and the
  registry map.

### Changed (behavior)

- **`TransformerDecoder`/`TransformerDecoderLayer` graphs now require both
  `tgt` and `memory` inputs.** Previously a single-input decoder graph passed
  static validation but could never execute; validation now reports the
  missing input instead.
- **`PixelShuffle`/`PixelUnshuffle` outputs change** for any graph that used
  them: they previously ran as identity (see Fixed above) and now actually
  shuffle pixels.
- All transformer/attention/RNN shape rules assume `batch_first: true`; set it
  in node params (mlnode-editor now emits it by default).

### Removed

- **`RNNBase`** from `CORE_LAYER_TYPES` and the registry. It is an abstract
  base class without `forward()` — it could be constructed but never executed.
  Use `RNN`, `LSTM`, or `GRU`.

### Docs

- **`QUICKSTART.md`** at the repo root — zero to trained model in five
  minutes; the code snippet is verified runnable end-to-end. Linked from the
  README and `docs/README.md` indexes.
- **New guide set in `docs/`**: `getting-started.md`, `json-schema.md`,
  `api.md`, `validation.md` (including per-type shape-rule status and known
  limitations), `blocks.md`, `custom-layers.md`, `huggingface.md`, plus a
  `docs/README.md` index.
- README: supported-types table now mirrors `CORE_LAYER_TYPES` (full
  activation/norm/pooling/dropout/lazy families) with a footnote marking
  types whose shape rule is pending; documentation index; `setup.sh`
  mention; test counts refreshed (128 tests, ~88% coverage); `utils/` added
  to the project tree; intro typos fixed.
- `docs/MLNODE_SPEC_v1_0.md`: fixed `mlnode._schema_version` →
  `mlnode_schema_version` typo in the example; §7 API signatures now include
  `block_registry` and `save(expanded=)`; core-types row points at
  `CORE_LAYER_TYPES` as the authoritative list.
- README: multi-input port table, `batch_first` convention note, notebook
  index, updated supported-types table (Vision Ops, full transformer family).
- `docs/MLNODE_SPEC_v1_0.md`: attention row documents port-labeled edges.
