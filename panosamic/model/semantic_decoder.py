"""
Author: Mahdi Chamseddine
"""

import torch
import torch.nn as nn


# The decoder is based on the Segformer semantic decoder with some modifications
class ConvDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,  # out_channels of FeatureFusion
        num_classes: int,
        out_size: int = 256,  # = 4 * in_size of FeatureFusion
        depth: int = 4,  # same as depth of FeatureFusion
        dual_view_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.decoder_head = nn.Sequential(
            nn.Conv2d(depth * in_channels, in_channels, 1),
            nn.LayerNorm([in_channels, out_size, out_size]),
            nn.GELU(),
            nn.Conv2d(in_channels, num_classes, 1),
        )

        self.decoder_attention = (
            nn.Sequential(
                SphericalConv2D(
                    2 * num_classes, num_classes, kernel_size=7
                ),  # (N, 2C, H, W) → (N, C, H, W)
                nn.GELU(),
                SphericalConv2D(
                    num_classes, 1, kernel_size=3
                ),  # (N, C, H, W) → (N, 1, H, W)
                nn.Sigmoid(),
            )
            if dual_view_fusion
            else None
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.decoder_head(input)
        if self.decoder_attention:
            x1 = x[0::2, ...]
            x2 = x[1::2, ...]
            x2 = torch.roll(x2, shifts=int(-x2.shape[-1] // 2), dims=-1)
            x = torch.cat([x1, x2], dim=1)  # (N, 2C, H, W)
            alpha = self.decoder_attention(x)  # (N, 1, H, W)
            x = alpha * x1 + (1 - alpha) * x2

        return x


class SphericalConv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        # Currently only works with a square kernel
        self.kernel_h, self.kernel_w = kernel_size, kernel_size
        self.padding = (
            self.kernel_w // 2,
            self.kernel_w // 2,
            self.kernel_h // 2,
            self.kernel_h // 2,
        )

        # Define padding layers
        self.circular_pad = nn.CircularPad2d((self.padding[0], self.padding[1], 0, 0))
        self.zero_pad = nn.ZeroPad2d((0, 0, self.padding[2], self.padding[3]))

        # Define Conv2d layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs 2D convolution with circular horizontal padding and zero vertical padding.
        """
        # Apply custom padding
        x = self.circular_pad(input)  # Apply circular padding to left and right
        x = self.zero_pad(x)  # Apply zero padding to top and bottom

        # Perform convolution using nn.Conv2d
        x = self.conv(x)

        return x


class BaselineDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,  # out_channels of SAM
        n_modalities: int,
        num_classes: int,
        out_size: int = 256,
    ) -> None:
        super().__init__()
        self.decoder_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(n_modalities * in_channels, in_channels, 1),
            nn.LayerNorm([in_channels, out_size // 2, out_size // 2]),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels, num_classes, 1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.decoder_head(input)
        return x
