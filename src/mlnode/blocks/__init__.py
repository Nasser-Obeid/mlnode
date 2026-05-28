"""Blocks — reusable sub-graph composition.

A Block is a self-contained sub-graph that can be instantiated as a single node.
Blocks are expanded at parse time by flattening their internal nodes into the
parent graph with namespaced IDs: {block_node_id}/{internal_id}.
"""

from mlnode.blocks.expansion import expand_blocks
from mlnode.blocks.registry import BlockRegistry

__all__ = ["expand_blocks", "BlockRegistry"]
