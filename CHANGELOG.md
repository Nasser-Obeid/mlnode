# Changelog

All notable changes to **mlnode** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.1.0] — 2026-10-19

### Added

- **JSON custom layers (schema `1.1`).** Define a layer as a named,
  parameterized sub-graph under a top-level `custom_layers` map and use it with
  `{"type": "custom", "layer": "<name>", "params": {...}}`. See
  `docs/custom-layers.md` and `examples/models/custom_layers.mlnode.json`.
  - Declared params with types (`int`, `float`, `bool`, `str`, `list`, `dict`,
    `any`) and defaults; missing, unknown and mistyped params raise
    `CustomLayerError` at load time.
  - `"$name"` param references and `"${expr}"` arithmetic (`+ - * / // % **`,
    `min`/`max`/`abs`/`int`/`float`/`round`), evaluated by an AST whitelist —
    no `eval()`.
  - Multi-input layers: the definition's `Input` nodes are its ports, bound by
    `target_port` or declaration order.
  - Each use builds a nested `nn.Module`; `validate()` checks shapes through
    the sub-graph and reports inner errors as `<node>/<inner node>`.
  - Layers can nest other layers and Blocks (depth ≤ 10), share weights via
    `shared_id`, and work inside Blocks.
  - `LayerRegistry` (`register`, `register_from_dict`, `register_from_file`,
    `register_library`) plus a process-wide default registry;
    `MLnodeModel.load`/`from_dict` accept `layer_registry=`.
- **Python 3.14 and 3.15 support**, verified on 3.15.0 (all tests pass with
  `DeprecationWarning` as error). Dependency floors rise on new interpreters to
  the first releases with wheels for them: torch ≥ 2.9 / pydantic ≥ 2.12 on
  3.14, torch ≥ 2.14 / pydantic ≥ 2.14 on 3.15. PyPI has no torch 3.15 wheels
  yet; install from `https://download.pytorch.org/whl/<cpu|cuXXX>` (the setup
  scripts do this automatically).
- **Windows support:** `setup.ps1` (PowerShell counterpart of `setup.sh`);
  `setup.sh` also works under Git Bash. GitHub Actions CI runs the test suite
  on Ubuntu and Windows × Python 3.11–3.15, plus both setup scripts.
- `.gitattributes` pinning LF for `.sh`/`.json` and CRLF for `.ps1`.
- **Input-width checks in `validate()`**: `Linear` (`in_features`), convolutions
  (`in_channels`), `BatchNorm*`/`SyncBatchNorm` (`num_features`), `GroupNorm`
  (`num_channels`), `LayerNorm`/`RMSNorm` (`normalized_shape`) and the
  recurrent layers (`input_size`) are checked against the incoming shape.
  Errors follow the spec §3 format — `Node 'fc' (Linear, in_features=512)
  expects (B, 512), got (B, 256) from 'flatten'` — and set `expected_shape`,
  `actual_shape` and `source_node_id`. Every case is a hard runtime error in
  PyTorch, so no previously working model is rejected; symbolic dims pass.
- **Fallback `reduction` is now simulated** by the validator for every
  single-input layer fed by several edges (previously the rule saw only the
  first input). A node with several inputs and no `reduction` now fails
  validation, as spec §1.2 requires, instead of failing at runtime.
- mlnode-editor 1.1.0 supports JSON custom layers (separate repository).

### Changed

- Schema version `1.1` is now accepted alongside `1.0`; files declaring
  `custom_layers` must use `1.1`.
- `custom` nodes take `path` **or** `layer` (exactly one).
- Weight-sharing groups also require identical `path`/`layer`.
- All JSON files are read and written as UTF-8 regardless of platform locale
  (previously Windows used the ANSI code page, so non-ASCII metadata failed to
  load). `save()` now writes LF line endings, non-ASCII characters unescaped,
  and a trailing newline.
- `resolve_multi_inputs()` takes an optional `ports=` override;
  `ShapeValidator` takes optional per-Input `input_shapes=`.
- Ruff enforces `PLW1514` (`open()` without `encoding=`).

### Fixed

- `Stack` shape rule placed the new dimension one position too early for
  negative `dim` (e.g. `dim=-1`).

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
