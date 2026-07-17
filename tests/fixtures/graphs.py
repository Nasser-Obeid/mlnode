"""Shared test fixtures for MLnode tests."""

from __future__ import annotations

import pytest

# ──────────────────────────────────────────────
# Minimal valid graphs
# ──────────────────────────────────────────────

MINIMAL_LINEAR = {
    "mlnode_schema_version": "1.0",
    "metadata": {"name": "Minimal Linear"},
    "input_shape": ["B", 10],
    "output_node": "fc",
    "nodes": [
        {"id": "input", "type": "Input", "params": {}},
        {"id": "fc", "type": "Linear", "params": {"in_features": 10, "out_features": 5}},
    ],
    "edges": [
        {"source": "input", "source_port": None, "target": "fc", "target_port": None},
    ],
}

RESNET_SKIP_BLOCK = {
    "mlnode_schema_version": "1.0",
    "metadata": {"name": "ResNet Block", "description": "Basic residual block"},
    "input_shape": ["B", 64, 56, 56],
    "output_node": "relu",
    "nodes": [
        {"id": "input", "type": "Input", "params": {}},
        {"id": "conv1", "type": "Conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
        {"id": "conv2", "type": "Conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
        {"id": "skip", "type": "Add", "params": {}},
        {"id": "relu", "type": "ReLU", "params": {}},
    ],
    "edges": [
        {"source": "input", "source_port": None, "target": "conv1", "target_port": None},
        {"source": "conv1", "source_port": None, "target": "conv2", "target_port": None},
        {"source": "input", "source_port": None, "target": "skip", "target_port": None},
        {"source": "conv2", "source_port": None, "target": "skip", "target_port": None},
        {"source": "skip", "source_port": None, "target": "relu", "target_port": None},
    ],
}

MULTI_INPUT_GRAPH = {
    "mlnode_schema_version": "1.0",
    "metadata": {"name": "Multi-Input"},
    "input_shape": ["B", 32],
    "output_node": "add",
    "nodes": [
        {"id": "enc_input", "type": "Input", "params": {}},
        {"id": "dec_input", "type": "Input", "params": {}},
        {"id": "fc1", "type": "Linear", "params": {"in_features": 32, "out_features": 16}},
        {"id": "fc2", "type": "Linear", "params": {"in_features": 32, "out_features": 16}},
        {"id": "add", "type": "Add", "params": {}},
    ],
    "edges": [
        {"source": "enc_input", "source_port": None, "target": "fc1", "target_port": None},
        {"source": "dec_input", "source_port": None, "target": "fc2", "target_port": None},
        {"source": "fc1", "source_port": None, "target": "add", "target_port": None},
        {"source": "fc2", "source_port": None, "target": "add", "target_port": None},
    ],
}

# ──────────────────────────────────────────────
# Invalid graphs for error testing
# ──────────────────────────────────────────────

CYCLE_GRAPH = {
    "mlnode_schema_version": "1.0",
    "input_shape": ["B", 10],
    "output_node": "fc2",
    "nodes": [
        {"id": "input", "type": "Input", "params": {}},
        {"id": "fc1", "type": "Linear", "params": {"in_features": 10, "out_features": 10}},
        {"id": "fc2", "type": "Linear", "params": {"in_features": 10, "out_features": 10}},
    ],
    "edges": [
        {"source": "input", "source_port": None, "target": "fc1", "target_port": None},
        {"source": "fc1", "source_port": None, "target": "fc2", "target_port": None},
        {"source": "fc2", "source_port": None, "target": "fc1", "target_port": None},
    ],
}

SELF_LOOP_GRAPH = {
    "mlnode_schema_version": "1.0",
    "input_shape": ["B", 10],
    "output_node": "fc1",
    "nodes": [
        {"id": "input", "type": "Input", "params": {}},
        {"id": "fc1", "type": "Linear", "params": {"in_features": 10, "out_features": 10}},
    ],
    "edges": [
        {"source": "input", "source_port": None, "target": "fc1", "target_port": None},
        {"source": "fc1", "source_port": None, "target": "fc1", "target_port": None},
    ],
}

SHAPE_MISMATCH_GRAPH = {
    "mlnode_schema_version": "1.0",
    "input_shape": ["B", 10],
    "output_node": "add",
    "nodes": [
        {"id": "input", "type": "Input", "params": {}},
        {"id": "fc1", "type": "Linear", "params": {"in_features": 10, "out_features": 16}},
        {"id": "fc2", "type": "Linear", "params": {"in_features": 10, "out_features": 32}},
        {"id": "add", "type": "Add", "params": {}},
    ],
    "edges": [
        {"source": "input", "source_port": None, "target": "fc1", "target_port": None},
        {"source": "input", "source_port": None, "target": "fc2", "target_port": None},
        {"source": "fc1", "source_port": None, "target": "add", "target_port": None},
        {"source": "fc2", "source_port": None, "target": "add", "target_port": None},
    ],
}

BAD_VERSION_GRAPH = {
    "mlnode_schema_version": "1.0.0",
    "input_shape": ["B", 10],
    "output_node": "fc",
    "nodes": [
        {"id": "input", "type": "Input", "params": {}},
        {"id": "fc", "type": "Linear", "params": {"in_features": 10, "out_features": 5}},
    ],
    "edges": [
        {"source": "input", "source_port": None, "target": "fc", "target_port": None},
    ],
}

SPLIT_GRAPH = {
    "mlnode_schema_version": "1.0",
    "metadata": {"name": "Split Test"},
    "input_shape": ["B", 64],
    "output_node": "fc_out",
    "nodes": [
        {"id": "input", "type": "Input", "params": {}},
        {"id": "split", "type": "Split", "params": {"split_size_or_sections": [32, 32], "dim": -1}, "outputs": ["left", "right"]},
        {"id": "fc_left", "type": "Linear", "params": {"in_features": 32, "out_features": 16}},
        {"id": "fc_right", "type": "Linear", "params": {"in_features": 32, "out_features": 16}},
        {"id": "cat", "type": "Concat", "params": {"dim": -1}},
        {"id": "fc_out", "type": "Linear", "params": {"in_features": 32, "out_features": 10}},
    ],
    "edges": [
        {"source": "input", "source_port": None, "target": "split", "target_port": None},
        {"source": "split", "source_port": "left", "target": "fc_left", "target_port": None},
        {"source": "split", "source_port": "right", "target": "fc_right", "target_port": None},
        {"source": "fc_left", "source_port": None, "target": "cat", "target_port": None},
        {"source": "fc_right", "source_port": None, "target": "cat", "target_port": None},
        {"source": "cat", "source_port": None, "target": "fc_out", "target_port": None},
    ],
}

WEIGHT_SHARING_GRAPH = {
    "mlnode_schema_version": "1.0",
    "metadata": {"name": "Weight Sharing"},
    "input_shape": ["B", 10],
    "output_node": "add",
    "nodes": [
        {"id": "input", "type": "Input", "params": {}},
        {"id": "fc1", "type": "Linear", "params": {"in_features": 10, "out_features": 10}, "shared_id": "shared_fc"},
        {"id": "fc2", "type": "Linear", "params": {"in_features": 10, "out_features": 10}, "shared_id": "shared_fc"},
        {"id": "add", "type": "Add", "params": {}},
    ],
    "edges": [
        {"source": "input", "source_port": None, "target": "fc1", "target_port": None},
        {"source": "input", "source_port": None, "target": "fc2", "target_port": None},
        {"source": "fc1", "source_port": None, "target": "add", "target_port": None},
        {"source": "fc2", "source_port": None, "target": "add", "target_port": None},
    ],
}

# ──────────────────────────────────────────────
# Block fixtures
# ──────────────────────────────────────────────

# A reusable residual block definition
RESNET_BLOCK_DEF = {
    "input_node": "block_in",
    "output_node": "block_relu",
    "nodes": [
        {"id": "block_in", "type": "Input", "params": {}},
        {"id": "block_conv1", "type": "Conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
        {"id": "block_bn1", "type": "BatchNorm2d", "params": {"num_features": 64}},
        {"id": "block_relu1", "type": "ReLU", "params": {}},
        {"id": "block_conv2", "type": "Conv2d", "params": {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1}},
        {"id": "block_bn2", "type": "BatchNorm2d", "params": {"num_features": 64}},
        {"id": "block_skip", "type": "Add", "params": {}},
        {"id": "block_relu", "type": "ReLU", "params": {}},
    ],
    "edges": [
        {"source": "block_in", "target": "block_conv1"},
        {"source": "block_conv1", "target": "block_bn1"},
        {"source": "block_bn1", "target": "block_relu1"},
        {"source": "block_relu1", "target": "block_conv2"},
        {"source": "block_conv2", "target": "block_bn2"},
        {"source": "block_in", "target": "block_skip"},
        {"source": "block_bn2", "target": "block_skip"},
        {"source": "block_skip", "target": "block_relu"},
    ],
}

# Graph that uses an inline block
INLINE_BLOCK_GRAPH = {
    "mlnode_schema_version": "1.0",
    "metadata": {"name": "Inline Block Test"},
    "input_shape": ["B", 64, 56, 56],
    "output_node": "res1",
    "nodes": [
        {"id": "input", "type": "Input", "params": {}},
        {"id": "res1", "type": "Block", "block": RESNET_BLOCK_DEF},
    ],
    "edges": [
        {"source": "input", "target": "res1"},
    ],
}

# Graph that chains two blocks
CHAINED_BLOCKS_GRAPH = {
    "mlnode_schema_version": "1.0",
    "metadata": {"name": "Chained Blocks"},
    "input_shape": ["B", 64, 56, 56],
    "output_node": "res2",
    "nodes": [
        {"id": "input", "type": "Input", "params": {}},
        {"id": "res1", "type": "Block", "block": RESNET_BLOCK_DEF},
        {"id": "res2", "type": "Block", "block": RESNET_BLOCK_DEF},
    ],
    "edges": [
        {"source": "input", "target": "res1"},
        {"source": "res1", "target": "res2"},
    ],
}

# Graph using block_ref (requires registry)
BLOCK_REF_GRAPH = {
    "mlnode_schema_version": "1.0",
    "metadata": {"name": "Block Ref Test"},
    "input_shape": ["B", 64, 56, 56],
    "output_node": "res1",
    "nodes": [
        {"id": "input", "type": "Input", "params": {}},
        {"id": "res1", "type": "Block", "block_ref": "resnet_block"},
    ],
    "edges": [
        {"source": "input", "target": "res1"},
    ],
}

# Simple MLP block definition for testing
MLP_BLOCK_DEF = {
    "input_node": "mlp_in",
    "output_node": "mlp_out",
    "nodes": [
        {"id": "mlp_in", "type": "Input", "params": {}},
        {"id": "mlp_fc1", "type": "Linear", "params": {"in_features": 32, "out_features": 64}},
        {"id": "mlp_relu", "type": "ReLU", "params": {}},
        {"id": "mlp_out", "type": "Linear", "params": {"in_features": 64, "out_features": 32}},
    ],
    "edges": [
        {"source": "mlp_in", "target": "mlp_fc1"},
        {"source": "mlp_fc1", "target": "mlp_relu"},
        {"source": "mlp_relu", "target": "mlp_out"},
    ],
}

MLP_BLOCK_GRAPH = {
    "mlnode_schema_version": "1.0",
    "metadata": {"name": "MLP Block"},
    "input_shape": ["B", 32],
    "output_node": "block1",
    "nodes": [
        {"id": "input", "type": "Input", "params": {}},
        {"id": "block1", "type": "Block", "block": MLP_BLOCK_DEF},
    ],
    "edges": [
        {"source": "input", "target": "block1"},
    ],
}
