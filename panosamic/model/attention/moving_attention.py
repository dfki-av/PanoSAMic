"""
Author: Mahdi Chamseddine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAttention(nn.Module):
    def __init__(
        self,
        att_layer: nn.Module,
        window_size: int,
        stride: int,
        aggregation: str = "none",
    ) -> None:
        super().__init__()
        assert aggregation in ("none", "mean", "max")

        self.att_layer = att_layer
        self.window_size = window_size
        self.stride = stride

        if aggregation == "none":
            self.aggregation_method = self.__fold_attention_aggregation
            self.with_sigmoid = False
        elif aggregation == "mean":
            self.aggregation_method = self.__mean_attention_aggregation
            self.with_sigmoid = True
        elif aggregation == "max":
            self.aggregation_method = self.__max_attention_aggregation
            self.with_sigmoid = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        N, C, H, W = input.shape
        k = self.window_size
        s = self.stride

        # Step 1: Extract sliding windows
        # Input: (N, C, H, W) -> Unfold: (N, C * k * k, L)
        windows = F.unfold(input, kernel_size=k, stride=s)
        L = windows.size(-1)  # Number of windows

        # Reshape to process each window:
        windows = windows.view(N, C, k, k, L).permute(0, 4, 1, 2, 3)  # (N, L, C, k, k)
        windows = windows.reshape(N * L, C, k, k)

        # Step 2: Apply Channel Attention to each window
        windows = self.att_layer(
            windows,
            only_attention=True,
            with_sigmoid=self.with_sigmoid,
        ).view(N, L, C, k, k)
        windows = windows.reshape(N, L, C * k * k).transpose(1, 2)  # (N, C * k * k, L)

        # Step 3: Aggregate Attention Maps
        attention = self.aggregation_method(windows, output_size=(H, W))

        return input * attention

    def __fold_attention_aggregation(  # Fold windows with sigmoid on top
        self,
        windows: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        attention = F.fold(
            windows,
            output_size=output_size,
            kernel_size=self.window_size,
            stride=self.stride,
        )
        attention = F.sigmoid(attention)
        return attention

    def __mean_attention_aggregation(
        self,
        windows: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        normalization_mask = F.fold(
            torch.ones_like(windows),
            output_size=output_size,
            kernel_size=self.window_size,
            stride=self.stride,
        )

        attention = F.fold(
            windows,
            output_size=output_size,
            kernel_size=self.window_size,
            stride=self.stride,
        )

        attention = attention / normalization_mask

        return attention

    def __max_attention_aggregation(
        self,
        windows: torch.Tensor,
        output_size: tuple,
    ) -> torch.Tensor:
        attention = F.fold(
            windows,
            output_size=output_size,
            kernel_size=self.window_size,
            stride=self.stride,
        )

        attention = F.sigmoid(15 * (attention - 0.5))

        return attention
