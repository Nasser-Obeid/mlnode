"""Tests for JSON-defined custom layers (schema 1.1)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch

from mlnode import LayerDefinition, LayerRegistry, MLnodeModel
from mlnode.errors import CustomLayerError, MLnodeError, SchemaVersionError
from mlnode.layers.json_layers import (
    MAX_LAYER_DEPTH,
    evaluate_expression,
    get_default_layer_registry,
    substitute,
)
from mlnode.schema.models import GraphSchema, NodeSchema

EXAMPLE = Path(__file__).parents[2] / "examples" / "models" / "custom_layers.mlnode.json"

MLP_LAYER: dict[str, Any] = {
    "description": "Linear → ReLU → Linear with a hidden width multiplier",
    "params": {
        "dim": {"type": "int"},
        "expansion": {"type": "int", "default": 4},
    },
    "nodes": [
        {"id": "x", "type": "Input"},
        {"id": "fc1", "type": "Linear",
         "params": {"in_features": "$dim", "out_features": "${dim * expansion}"}},
        {"id": "act", "type": "ReLU"},
        {"id": "fc2", "type": "Linear",
         "params": {"in_features": "${dim * expansion}", "out_features": "$dim"}},
    ],
    "edges": [
        {"source": "x", "target": "fc1"},
        {"source": "fc1", "target": "act"},
        {"source": "act", "target": "fc2"},
    ],
    "output_node": "fc2",
}

GATED_FUSION_LAYER: dict[str, Any] = {
    "description": "out = sigmoid(W[a; b]) * a + b",
    "params": {"dim": {"type": "int"}},
    "nodes": [
        {"id": "a", "type": "Input"},
        {"id": "b", "type": "Input"},
        {"id": "cat", "type": "Concat", "params": {"dim": -1}},
        {"id": "proj", "type": "Linear",
         "params": {"in_features": "${2 * dim}", "out_features": "$dim"}},
        {"id": "gate", "type": "Sigmoid"},
        {"id": "gated", "type": "Multiply"},
        {"id": "out", "type": "Add"},
    ],
    "edges": [
        {"source": "a", "target": "cat"},
        {"source": "b", "target": "cat"},
        {"source": "cat", "target": "proj"},
        {"source": "proj", "target": "gate"},
        {"source": "gate", "target": "gated"},
        {"source": "a", "target": "gated"},
        {"source": "gated", "target": "out"},
        {"source": "b", "target": "out"},
    ],
    "output_node": "out",
}


def _mlp_graph(**node_overrides: Any) -> dict[str, Any]:
    node = {"id": "mlp", "type": "custom", "layer": "MLP", "params": {"dim": 16}}
    node.update(node_overrides)
    return {
        "mlnode_schema_version": "1.1",
        "input_shape": ["B", 16],
        "output_node": "head",
        "custom_layers": {"MLP": copy.deepcopy(MLP_LAYER)},
        "nodes": [
            {"id": "input", "type": "Input"},
            node,
            {"id": "head", "type": "Linear", "params": {"in_features": 16, "out_features": 3}},
        ],
        "edges": [
            {"source": "input", "target": "mlp"},
            {"source": "mlp", "target": "head"},
        ],
    }


def _fusion_graph(edges: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mlnode_schema_version": "1.1",
        "input_shape": ["B", 8],
        "output_node": "fuse",
        "custom_layers": {"GatedFusion": copy.deepcopy(GATED_FUSION_LAYER)},
        "nodes": [
            {"id": "left", "type": "Input"},
            {"id": "right", "type": "Input"},
            {"id": "fuse", "type": "custom", "layer": "GatedFusion", "params": {"dim": 8}},
        ],
        "edges": edges,
    }


# ──────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────


class TestSchema:
    def test_custom_requires_path_or_layer(self) -> None:
        with pytest.raises(ValueError, match=r"requires 'path'.*or 'layer'"):
            NodeSchema(id="c", type="custom")

    def test_custom_rejects_path_and_layer(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            NodeSchema(id="c", type="custom", path="a.B", layer="MLP")

    def test_layer_definition_valid(self) -> None:
        layer = LayerDefinition.model_validate(GATED_FUSION_LAYER)
        assert layer.input_ports == ["a", "b"]
        assert layer.params["dim"].required

    def test_param_with_default_is_optional(self) -> None:
        layer = LayerDefinition.model_validate(MLP_LAYER)
        assert not layer.params["expansion"].required
        assert layer.params["expansion"].default == 4

    def test_null_default_is_optional(self) -> None:
        layer = LayerDefinition.model_validate(
            {**MLP_LAYER, "params": {"dim": {"type": "int", "default": None}}}
        )
        assert not layer.params["dim"].required

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            ({"nodes": [{"id": "fc", "type": "Linear", "params": {}}], "edges": [],
              "output_node": "fc"}, "at least one Input"),
            ({"output_node": "missing"}, "output_node 'missing' not found"),
            ({"edges": [{"source": "x", "target": "ghost"}]}, "edge target 'ghost'"),
            ({"params": {"bad-name": {}}}, "valid identifier"),
            ({"params": {"lambda": {}}}, "valid identifier"),
            ({"nodes": [{"id": "x", "type": "Input"}, {"id": "x", "type": "ReLU"}],
              "edges": [], "output_node": "x"}, "Duplicate node IDs"),
        ],
    )
    def test_invalid_layer_definition(self, change: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            LayerDefinition.model_validate({**MLP_LAYER, **change})

    def test_custom_layers_need_schema_1_1(self) -> None:
        data = _mlp_graph()
        data["mlnode_schema_version"] = "1.0"
        with pytest.raises(ValueError, match=r"requires mlnode_schema_version '1\.1'"):
            GraphSchema.model_validate(data)

    def test_schema_1_0_still_supported(self) -> None:
        data = _mlp_graph()
        data["mlnode_schema_version"] = "1.0"
        del data["custom_layers"]
        data["nodes"][1] = {"id": "mlp", "type": "Linear",
                            "params": {"in_features": 16, "out_features": 16}}
        assert MLnodeModel.from_dict(data).validate().valid

    def test_unknown_version_still_rejected(self) -> None:
        data = _mlp_graph()
        data["mlnode_schema_version"] = "2.0"
        with pytest.raises(SchemaVersionError):
            GraphSchema.model_validate(data)


# ──────────────────────────────────────────────
# Param substitution and expressions
# ──────────────────────────────────────────────


class TestSubstitution:
    PARAMS: ClassVar[dict[str, Any]] = {
        "dim": 64, "heads": 8, "sizes": [1, 2], "flag": True, "name": "x",
    }

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("$dim", 64),
            ("$sizes", [1, 2]),
            ("$flag", True),
            ("${dim // heads}", 8),
            ("${dim * 2 + 1}", 129),
            ("${-dim}", -64),
            ("${(dim + 2) % 5}", 1),
            ("${dim / heads}", 8.0),
            ("${2 ** 3}", 8),
            ("${max(1, dim // 128)}", 1),
            ("${int(dim * 0.5)}", 32),
            ("${round(dim / 3)}", 21),
            ("$$dim", "$dim"),
            ("plain", "plain"),
            (5, 5),
            ([["$dim"], {"k": "${heads}"}], [[64], {"k": 8}]),
        ],
    )
    def test_substitute(self, value: Any, expected: Any) -> None:
        assert substitute(value, self.PARAMS) == expected

    def test_unknown_reference(self) -> None:
        with pytest.raises(ValueError, match="unknown param reference '\\$nope'"):
            substitute("$nope", self.PARAMS)

    @pytest.mark.parametrize(
        "expr",
        [
            "__import__('os')",
            "dim.__class__",
            "(lambda: 1)()",
            "[1, 2]",
            "dim if flag else 0",
            "open('x')",
            "'a' + 'b'",
            "max(dim, key=abs)",
        ],
    )
    def test_disallowed_syntax(self, expr: str) -> None:
        with pytest.raises(ValueError, match="unsupported syntax"):
            evaluate_expression(expr, self.PARAMS)

    def test_unknown_name_in_expression(self) -> None:
        with pytest.raises(ValueError, match="unknown param 'nope'"):
            evaluate_expression("nope * 2", self.PARAMS)

    def test_syntax_error(self) -> None:
        with pytest.raises(ValueError, match="invalid expression"):
            evaluate_expression("dim *", self.PARAMS)

    def test_division_by_zero(self) -> None:
        with pytest.raises(ValueError, match="failed"):
            evaluate_expression("dim // 0", self.PARAMS)

    def test_huge_exponent_rejected(self) -> None:
        with pytest.raises(ValueError, match="exponent"):
            evaluate_expression("10 ** 10 ** 10", self.PARAMS)


# ──────────────────────────────────────────────
# Param binding (at parse time)
# ──────────────────────────────────────────────


class TestParamBinding:
    def test_default_applied(self) -> None:
        model = MLnodeModel.from_dict(_mlp_graph())
        fc1 = model.graph.custom_layers["mlp"].nodes["fc1"].schema
        assert fc1.params == {"in_features": 16, "out_features": 64}

    def test_override_default(self) -> None:
        model = MLnodeModel.from_dict(
            _mlp_graph(params={"dim": 16, "expansion": 2})
        )
        fc1 = model.graph.custom_layers["mlp"].nodes["fc1"].schema
        assert fc1.params["out_features"] == 32

    def test_missing_required(self) -> None:
        with pytest.raises(CustomLayerError, match="missing required param 'dim'") as exc:
            MLnodeModel.from_dict(_mlp_graph(params={}))
        assert exc.value.node_id == "mlp"

    def test_unknown_param(self) -> None:
        with pytest.raises(CustomLayerError, match="unknown param\\(s\\) \\['width'\\]"):
            MLnodeModel.from_dict(_mlp_graph(params={"dim": 16, "width": 3}))

    @pytest.mark.parametrize("bad", [True, 16.0, "16", [16]])
    def test_int_type_check(self, bad: Any) -> None:
        with pytest.raises(CustomLayerError, match="param 'dim' expects int"):
            MLnodeModel.from_dict(_mlp_graph(params={"dim": bad}))

    def test_float_accepts_int(self) -> None:
        data = _mlp_graph()
        layer = data["custom_layers"]["MLP"]
        layer["params"]["p"] = {"type": "float", "default": 0.1}
        layer["nodes"].insert(2, {"id": "drop", "type": "Dropout", "params": {"p": "$p"}})
        layer["edges"] = [
            {"source": "x", "target": "fc1"},
            {"source": "fc1", "target": "drop"},
            {"source": "drop", "target": "act"},
            {"source": "act", "target": "fc2"},
        ]
        data["nodes"][1]["params"]["p"] = 0
        model = MLnodeModel.from_dict(data)
        assert model.graph.custom_layers["mlp"].nodes["drop"].schema.params == {"p": 0}

    def test_bad_expression_reports_inner_node(self) -> None:
        data = _mlp_graph()
        data["custom_layers"]["MLP"]["nodes"][1]["params"]["out_features"] = "${dim *}"
        with pytest.raises(CustomLayerError, match=r"inner node 'fc1'.*invalid expression"):
            MLnodeModel.from_dict(data)


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────


class TestLayerRegistry:
    def test_register_and_get(self) -> None:
        registry = LayerRegistry()
        layer = LayerDefinition.model_validate(MLP_LAYER)
        registry.register("MLP", layer)
        assert registry.has("MLP")
        assert registry.get("MLP") is layer
        assert registry.names == ["MLP"]
        assert len(registry) == 1

    def test_duplicate_register_raises(self) -> None:
        registry = LayerRegistry()
        registry.register_from_dict("MLP", MLP_LAYER)
        with pytest.raises(MLnodeError, match="already registered"):
            registry.register_from_dict("MLP", MLP_LAYER)

    def test_get_missing_raises(self) -> None:
        with pytest.raises(MLnodeError, match="not found"):
            LayerRegistry().get("nope")

    def test_register_from_file(self, tmp_path: Path) -> None:
        path = tmp_path / "mlp.json"
        path.write_text(json.dumps(MLP_LAYER, ensure_ascii=False), encoding="utf-8")
        registry = LayerRegistry()
        registry.register_from_file("MLP", path)
        assert registry.get("MLP").description == MLP_LAYER["description"]

    def test_register_library(self, tmp_path: Path) -> None:
        path = tmp_path / "lib.json"
        path.write_text(
            json.dumps({"custom_layers": {"MLP": MLP_LAYER, "GatedFusion": GATED_FUSION_LAYER}}),
            encoding="utf-8",
        )
        registry = LayerRegistry()
        assert registry.register_library(path) == ["GatedFusion", "MLP"]

    def test_register_library_requires_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "lib.json"
        path.write_text(json.dumps(MLP_LAYER), encoding="utf-8")
        with pytest.raises(MLnodeError, match="'custom_layers' mapping"):
            LayerRegistry().register_library(path)

    def test_model_resolves_from_registry(self) -> None:
        registry = LayerRegistry()
        registry.register_from_dict("MLP", MLP_LAYER)
        data = _mlp_graph()
        del data["custom_layers"]
        model = MLnodeModel.from_dict(data, layer_registry=registry)
        assert model.validate().valid

    def test_model_resolves_from_default_registry(self) -> None:
        default = get_default_layer_registry()
        default.register_from_dict("_TestDefaultMLP", MLP_LAYER)
        try:
            data = _mlp_graph(layer="_TestDefaultMLP")
            del data["custom_layers"]
            assert MLnodeModel.from_dict(data).validate().valid
        finally:
            default._layers.pop("_TestDefaultMLP")

    def test_inline_definition_wins_over_registry(self) -> None:
        registry = LayerRegistry()
        registry.register_from_dict("MLP", {**MLP_LAYER, "params": {"dim": {"type": "str"}}})
        model = MLnodeModel.from_dict(_mlp_graph(), layer_registry=registry)
        assert model.validate().valid

    def test_undefined_layer(self) -> None:
        data = _mlp_graph(layer="Nope")
        with pytest.raises(CustomLayerError, match="custom layer 'Nope' is not defined"):
            MLnodeModel.from_dict(data, layer_registry=LayerRegistry())


# ──────────────────────────────────────────────
# Validation and execution
# ──────────────────────────────────────────────


class TestEndToEnd:
    def test_example_file(self) -> None:
        model = MLnodeModel.load(EXAMPLE)
        result = model.validate()
        assert result.valid, result.errors
        assert result.node_shapes["down"] == ("B", 64, 16, 16)
        assert result.node_shapes["res1/conv1/conv"] == ("B", 32, 32, 32)
        module = model.build()
        assert module(torch.randn(2, 3, 32, 32)).shape == (2, 10)

    def test_single_input_layer_matches_manual_mlp(self) -> None:
        model = MLnodeModel.from_dict(_mlp_graph())
        result = model.validate()
        assert result.valid, result.errors
        assert result.node_shapes["mlp"] == ("B", 16)
        assert result.node_shapes["mlp/fc1"] == ("B", 64)

        module = model.build()
        inner = module._layers["mlp"]
        assert isinstance(inner, torch.nn.Module)
        x = torch.randn(4, 16)
        fc1, fc2 = inner._layers["fc1"], inner._layers["fc2"]
        head = module._layers["head"]
        expected = head(fc2(torch.relu(fc1(x))))
        torch.testing.assert_close(module(x), expected)

    def test_state_dict_is_nested(self) -> None:
        module = MLnodeModel.from_dict(_mlp_graph()).build()
        keys = set(module.state_dict())
        assert "_layers.mlp._layers.fc1.weight" in keys
        assert module.state_dict()["_layers.mlp._layers.fc1.weight"].shape == (64, 16)

    @staticmethod
    def _mismatched_fusion() -> dict[str, Any]:
        # dim=4 but inputs are (B, 8): proj expects 2*4=8 features, gets 8+8=16.
        data = _fusion_graph([
            {"source": "left", "target": "fuse"},
            {"source": "right", "target": "fuse"},
        ])
        data["nodes"][2]["params"]["dim"] = 4
        return data

    def test_shape_error_inside_layer_is_namespaced(self) -> None:
        result = MLnodeModel.from_dict(self._mismatched_fusion()).validate()
        assert not result.valid
        error = result.errors[0]
        assert error.node_id == "fuse/proj"
        assert "In custom layer 'fuse' (GatedFusion)" in error.message
        assert "in_features=8" in error.message
        assert error.expected_shape == ("B", 8)
        assert error.actual_shape == ("B", 16)
        assert error.source_node_id == "fuse/cat"
        assert "fuse" not in result.node_shapes

    def test_wrong_param_value_caught_statically(self) -> None:
        data = _mlp_graph()
        data["input_shape"] = ["B", 12]   # layer built for dim=16
        result = MLnodeModel.from_dict(data).validate()
        assert result.errors[0].node_id == "mlp/fc1"
        assert "in_features=16" in result.errors[0].message

    def test_build_raises_on_inner_shape_error(self) -> None:
        with pytest.raises(MLnodeError, match="Cannot build"):
            MLnodeModel.from_dict(self._mismatched_fusion()).build()

    def test_multi_input_labeled_ports(self) -> None:
        data = _fusion_graph([
            {"source": "right", "target": "fuse", "target_port": "b"},
            {"source": "left", "target": "fuse", "target_port": "a"},
        ])
        model = MLnodeModel.from_dict(data)
        assert model.validate().valid
        module = model.build()
        a, b = torch.randn(3, 8), torch.randn(3, 8)
        inner = module._layers["fuse"]
        gate = torch.sigmoid(inner._layers["proj"](torch.cat([a, b], dim=-1)))
        torch.testing.assert_close(module(a, b), gate * a + b)

    def test_multi_input_declaration_order(self) -> None:
        data = _fusion_graph([
            {"source": "left", "target": "fuse"},
            {"source": "right", "target": "fuse"},
        ])
        module = MLnodeModel.from_dict(data).build()
        a, b = torch.randn(3, 8), torch.randn(3, 8)
        inner = module._layers["fuse"]
        gate = torch.sigmoid(inner._layers["proj"](torch.cat([a, b], dim=-1)))
        torch.testing.assert_close(module(a, b), gate * a + b)

    def test_multi_input_missing_port(self) -> None:
        data = _fusion_graph([{"source": "left", "target": "fuse", "target_port": "a"}])
        data["edges"].append({"source": "right", "target": "fuse", "target_port": "a"})
        result = MLnodeModel.from_dict(data).validate()
        assert not result.valid
        assert "received more than one edge" in result.errors[0].message

    def test_multi_input_unknown_port(self) -> None:
        data = _fusion_graph([
            {"source": "left", "target": "fuse", "target_port": "a"},
            {"source": "right", "target": "fuse", "target_port": "c"},
        ])
        result = MLnodeModel.from_dict(data).validate()
        assert "unknown target_port 'c'" in result.errors[0].message

    def test_single_input_unknown_port(self) -> None:
        data = _mlp_graph()
        data["edges"][0]["target_port"] = "y"
        result = MLnodeModel.from_dict(data).validate()
        assert "unknown target_port ['y']" in result.errors[0].message

    def test_reduction_fallback_for_single_port_layer(self) -> None:
        data = _mlp_graph(reduction="add")
        data["edges"].append({"source": "input", "target": "mlp"})
        model = MLnodeModel.from_dict(data)
        assert model.validate().valid
        module = model.build()
        assert module(torch.randn(2, 16)).shape == (2, 3)

    def test_reduction_concat_shape(self) -> None:
        data = _mlp_graph(reduction="concat", params={"dim": 32})
        data["edges"].append({"source": "input", "target": "mlp"})
        data["nodes"][2]["params"]["in_features"] = 32
        model = MLnodeModel.from_dict(data)
        result = model.validate()
        assert result.valid, result.errors
        assert result.node_shapes["mlp/x"] == ("B", 32)
        assert model.build()(torch.randn(2, 16)).shape == (2, 3)

    def test_multiple_inputs_without_reduction(self) -> None:
        data = _mlp_graph()
        data["edges"].append({"source": "input", "target": "mlp"})
        result = MLnodeModel.from_dict(data).validate()
        assert "no reduction strategy" in result.errors[0].message

    def test_weight_sharing(self) -> None:
        data = _mlp_graph(shared_id="tied")
        data["nodes"].insert(2, {"id": "mlp2", "type": "custom", "layer": "MLP",
                                 "params": {"dim": 16}, "shared_id": "tied"})
        data["edges"] = [
            {"source": "input", "target": "mlp"},
            {"source": "mlp", "target": "mlp2"},
            {"source": "mlp2", "target": "head"},
        ]
        module = MLnodeModel.from_dict(data).build()
        assert module._layers["mlp"] is module._layers["mlp2"]

    def test_weight_sharing_requires_same_layer(self) -> None:
        data = _mlp_graph(shared_id="tied")
        data["custom_layers"]["MLP2"] = copy.deepcopy(MLP_LAYER)
        data["nodes"].insert(2, {"id": "mlp2", "type": "custom", "layer": "MLP2",
                                 "params": {"dim": 16}, "shared_id": "tied"})
        with pytest.raises(ValueError, match="identical type and params"):
            GraphSchema.model_validate(data)

    def test_layer_inside_block(self) -> None:
        data = _mlp_graph()
        mlp_node = data["nodes"][1]
        data["nodes"][1] = {
            "id": "blk", "type": "Block",
            "block": {
                "input_node": "in", "output_node": "mlp",
                "nodes": [{"id": "in", "type": "Input"}, mlp_node],
                "edges": [{"source": "in", "target": "mlp"}],
            },
        }
        data["edges"] = [
            {"source": "input", "target": "blk"},
            {"source": "blk", "target": "head"},
        ]
        model = MLnodeModel.from_dict(data)
        assert "blk/mlp" in model.graph.custom_layers
        assert model.validate().valid
        assert model.build()(torch.randn(2, 16)).shape == (2, 3)

    def test_self_reference_hits_depth_limit(self) -> None:
        data = _mlp_graph()
        layer = data["custom_layers"]["MLP"]
        layer["nodes"][1] = {"id": "fc1", "type": "custom", "layer": "MLP",
                             "params": {"dim": "$dim"}}
        with pytest.raises(CustomLayerError, match=f"nesting depth exceeded {MAX_LAYER_DEPTH}"):
            MLnodeModel.from_dict(data)

    def test_save_preserves_custom_layers(self, tmp_path: Path) -> None:
        data = _mlp_graph()
        data["metadata"] = {"name": "ünïcode → model"}
        path = tmp_path / "m.mlnode.json"
        MLnodeModel.from_dict(data).save(path)
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        reloaded = MLnodeModel.load(path)
        assert reloaded.schema.metadata.name == "ünïcode → model"
        assert "MLP" in reloaded.schema.custom_layers
        assert reloaded.validate().valid

    def test_python_path_custom_layers_unaffected(self) -> None:
        data = _mlp_graph(layer=None, path="tests.fixtures.custom_layers.ScaleLayer",
                          params={"scale": 2.0})
        model = MLnodeModel.from_dict(data)
        assert model.graph.custom_layers == {}
        assert model.build()(torch.randn(2, 16)).shape == (2, 3)
