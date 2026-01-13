"""
Author: Mahdi Chamseddine
"""

from typing import Literal

import torch
import torch.nn as nn


class BasicFusion(nn.Module):
    def __init__(
        self,
        fusion_type: Literal["concat", "add", "mult"],
        in_channels: int,
        n_modalities: int,
        in_size: int,
        out_size: int,
        depth: int = 4,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.fusion_type = fusion_type
        self.out_channels = (
            n_modalities * in_channels if fusion_type == "concat" else in_channels
        )

        self.fusion_blocks = nn.ModuleList()
        for _ in range(self.depth):
            block = BasicBlock(
                in_channels=in_channels,
                n_modalities=n_modalities,
                in_size=in_size,
                out_size=out_size,
            )
            self.fusion_blocks.append(block)

    def forward(
        self,
        inputs: list[torch.Tensor],
    ) -> torch.Tensor:
        fused_output = [
            block(encoder_branch, self.fusion_type)
            for encoder_branch, block in zip(inputs, self.fusion_blocks)
        ]
        x = torch.cat(fused_output, dim=1)
        return x


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_modalities: int,
        in_size: int,
        out_size: int,
    ) -> None:
        super().__init__()
        self.n_modalities = n_modalities

        self.norm = nn.LayerNorm([in_channels, in_size, in_size])
        self.upscale = nn.Upsample(scale_factor=out_size // in_size, mode="bilinear")

    def forward(
        self,
        input: torch.Tensor,
        fusion_type: Literal["concat", "add", "mult"],
    ) -> torch.Tensor:
        x = self.norm(input)
        match fusion_type:
            case "concat":
                x = self._concat_fusion(x)
            case "add":
                x = self._addition_fusion(x)
            case "mult":
                x = self._mult_fusion(x)
            case _:
                raise NotImplementedError
        return self.upscale(x)

    def _concat_fusion(self, input: torch.Tensor) -> torch.Tensor:
        N, C, H, W = input.shape
        return input.view(int(N / self.n_modalities), self.n_modalities * C, H, W)

    def _addition_fusion(self, input: torch.Tensor) -> torch.Tensor:
        N, C, H, W = input.shape
        x = input.view(int(N / self.n_modalities), self.n_modalities, C, H, W)
        return x.sum(dim=1)

    def _mult_fusion(self, input: torch.Tensor) -> torch.Tensor:
        N, C, H, W = input.shape
        x = input.view(int(N / self.n_modalities), self.n_modalities, C, H, W)
        return x.prod(dim=1)
