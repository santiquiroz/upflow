from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

CallFn = Callable[[nn.Module, torch.Tensor], Any]


def _plain_forward(model: nn.Module, image: torch.Tensor) -> Any:
    return model(image)


def descriptor_call_fn(descriptor: Any) -> CallFn:
    # Spandrel keeps call_fn private; spandrel is pinned <0.5 and the parity
    # check against descriptor(x) catches a rename, so a silent fallback here
    # can never ship a wrong graph.
    return getattr(descriptor, "_call_fn", None) or _plain_forward


class DescriptorGraph(nn.Module):
    def __init__(self, model: nn.Module, call_fn: CallFn) -> None:
        super().__init__()
        self.model = model
        self._call_fn = call_fn

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self._call_fn(self.model, image)
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Model call returned {type(output).__name__}, expected a single image tensor")
        return output.clamp(0, 1)


def graph_from_descriptor(descriptor: Any) -> DescriptorGraph:
    graph = DescriptorGraph(descriptor.model, descriptor_call_fn(descriptor))
    return graph.eval()
