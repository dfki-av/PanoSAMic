"""
Author: Mahdi Chamseddine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 5) -> None:
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(
        self,
        input: torch.Tensor,
        only_attention: bool = False,
        with_sigmoid: bool = True,
    ) -> torch.Tensor:
        x_avg = torch.mean(input, dim=1, keepdim=True)
        x_max, _ = torch.max(input, dim=1, keepdim=True)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.conv(x)

        x = F.sigmoid(x) if with_sigmoid else x
        x = x.expand_as(input)

        return x if only_attention else input * x
