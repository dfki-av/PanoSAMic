"""
Author: Mahdi Chamseddine
"""

from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, bias: bool = False) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(2 * channels, channels, bias=bias),
            nn.GELU(),
            nn.Linear(channels, channels, bias=bias),
        )

    def forward(
        self,
        input: torch.Tensor,
        only_attention: bool = False,
        with_sigmoid: bool = True,
    ) -> torch.Tensor:
        x_avg = self.avg_pool(input)
        x_max = self.max_pool(input)
        x = torch.cat((x_avg, x_max), dim=1)
        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        x = self.fc(x)

        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.sigmoid(x) if with_sigmoid else x
        x = x.expand_as(input)

        return x if only_attention else input * x


class EfficientChannelAttention(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()

        t = int(abs(log2(channels) + b) / gamma)
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)

    def forward(
        self,
        input: torch.Tensor,
        only_attention: bool = False,
        with_sigmoid: bool = True,
    ) -> torch.Tensor:
        x = self.avg_pool(input)
        x = x.squeeze(-1).transpose(-1, -2)
        x = self.conv(x)

        x = x.transpose(-1, -2).unsqueeze(-1)
        x = F.sigmoid(x) if with_sigmoid else x
        x = x.expand_as(input)

        return x if only_attention else input * x
