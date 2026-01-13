"""
Author: Mahdi Chamseddine
"""

from panosamic.evaluation.utils.config import ModelConfig
from panosamic.model.attention.channel_attention import (
    ChannelAttention,
    EfficientChannelAttention,
)
from panosamic.model.attention.moving_attention import MovingAttention
from panosamic.model.attention.spatial_attention import SpatialAttention


class AttentionBuilder:
    def __init__(self, config: ModelConfig) -> None:
        self.channel_attention = config.channel_attention
        self.spatial_attention = config.spatial_attention

    def build_channel_attention(
        self, channels: int
    ) -> ChannelAttention | EfficientChannelAttention | MovingAttention | None:
        if self.channel_attention is None:
            return None

        if self.channel_attention.get("efficient_attention", False):
            attention = EfficientChannelAttention(
                channels=channels,
                gamma=self.channel_attention.get("gamma", 2),  # default value
                b=self.channel_attention.get("b", 1),  # default value
            )
        else:
            attention = ChannelAttention(
                channels=channels,
                bias=self.channel_attention.get("bias", False),  # default value
            )

        if self.channel_attention.get("moving_attention", False):
            return MovingAttention(
                att_layer=attention,
                window_size=self.channel_attention.get("window_size", 8),
                stride=self.channel_attention.get("stride", 4),
                aggregation=self.channel_attention.get("aggregation", "none"),
            )

        return attention

    def build_spatial_attention(self) -> SpatialAttention | MovingAttention | None:
        if self.spatial_attention is None:
            return None

        attention = SpatialAttention(
            kernel_size=self.spatial_attention.get("kernel_size", 5)
        )

        if self.spatial_attention.get("moving_attention", False):
            return MovingAttention(
                att_layer=attention,
                window_size=self.spatial_attention.get("window_size", 8),
                stride=self.spatial_attention.get("stride", 4),
                aggregation=self.spatial_attention.get("aggregation", "none"),
            )
        return attention
