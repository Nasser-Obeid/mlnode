"""Sample custom layers for testing."""

from __future__ import annotations

import torch
import torch.nn as nn


class ScaleLayer(nn.Module):
    """Simple custom layer that multiplies by a learnable scalar."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class BottleneckProjection(nn.Module):
    """Projects input through a bottleneck and back."""

    def __init__(self, in_features: int, bottleneck: int) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, bottleneck)
        self.up = nn.Linear(bottleneck, in_features)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class BadLayer:
    """Not an nn.Module — should be rejected by validation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class CrashyLayer(nn.Module):
    """Crashes during forward — should be caught by dummy probe."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("I always crash!")


class BadInitLayer(nn.Module):
    """Crashes during __init__ with bad params."""

    def __init__(self, required_param: str) -> None:
        super().__init__()
        if required_param != "magic":
            raise ValueError("required_param must be 'magic'")
