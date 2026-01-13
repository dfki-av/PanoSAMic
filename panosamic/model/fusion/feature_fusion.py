"""
Author: Mahdi Chamseddine
"""

import torch
import torch.nn as nn

from panosamic.model.attention import (
    AttentionBuilder,
    ChannelAttention,
    EfficientChannelAttention,
    MovingAttention,
    SpatialAttention,
)


class FeatureFusion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modalities: int,
        attention_builder: AttentionBuilder,
        in_size: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.out_channels = out_channels

        self.fusion_blocks = nn.ModuleList()
        for _ in range(self.depth):
            block = FusionBlock(
                in_channels=in_channels,
                out_channels=self.out_channels,
                n_modalities=n_modalities,
                channel_attention=attention_builder.build_channel_attention(
                    channels=in_channels * n_modalities
                ),
                spatial_attention=attention_builder.build_spatial_attention(),
                in_size=in_size,
            )
            self.fusion_blocks.append(block)

    def forward(
        self,
        inputs: list[torch.Tensor],
    ) -> torch.Tensor:
        fused_output = [
            block(encoder_branch, feed_forward=True)
            for encoder_branch, block in zip(inputs, self.fusion_blocks)
        ]
        x = torch.cat(fused_output, dim=1)
        return x


class FusionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modalities: int,
        channel_attention: (
            ChannelAttention | EfficientChannelAttention | MovingAttention | None
        ) = None,
        spatial_attention: SpatialAttention | MovingAttention | None = None,
        in_size: int = 64,
    ) -> None:
        super().__init__()
        self.n_modalities = n_modalities
        self.norm = nn.LayerNorm([in_channels, in_size, in_size])
        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels * n_modalities, in_channels, 1, bias=False),
            nn.LayerNorm([in_channels, in_size, in_size]),
            nn.GELU(),
            nn.Conv2d(in_channels, 2 * out_channels, 1, bias=False),
            nn.LayerNorm([2 * out_channels, in_size, in_size]),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),
            nn.LayerNorm([out_channels, 2 * in_size, 2 * in_size]),
            nn.Upsample(scale_factor=2, mode="bilinear"),
        )

    def forward(self, input: torch.Tensor, feed_forward: bool = False) -> torch.Tensor:
        N, C, H, W = input.shape
        # only feed forward if an attention is present
        ff: bool = True if self.channel_attention or self.spatial_attention else False
        x = self.norm(input)
        x = x.view(int(N / self.n_modalities), self.n_modalities * C, H, W)
        _x = self.channel_attention(x) if self.channel_attention else x
        _x = self.spatial_attention(_x) if self.spatial_attention else _x
        x = x + _x if (feed_forward and ff) else _x
        x = self.neck(x)
        return x
