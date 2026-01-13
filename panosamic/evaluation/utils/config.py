"""
Author: Mahdi Chamseddine
"""

import json
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(kw_only=True, slots=True)
class PlatformConfig:
    dataset_path: str
    config_path: str
    experiments_path: str
    sam_weights_path: str
    workers_per_gpu: int = 2
    num_gpus: int = 1


@dataclass(kw_only=True, slots=True)
class TrainingConfig:
    dataset_name: str
    fold: int
    batch_size: int
    epochs: int
    loss: dict[str, Any]
    optimizer: str
    start_lr: float
    max_lr: float
    intermediate_lr: float | None = None
    warm_up_ratio: float
    wind_down_ratio: float


# @dataclass
# class AttentionConfig:
#     moving_attention: bool = True
#     window_size: int = 8
#     stride: int = 4
#     aggregation: str = "none"


@dataclass(kw_only=True, slots=True)
class ModelConfig:
    vit_model: Literal["vit_h", "vit_l", "vit_b"] = "vit_h"
    modalities: tuple[str, ...] = ("image", "depth", "normals")
    semantic_only: bool = False
    channel_attention: dict[str, Any] | None = None
    spatial_attention: dict[str, Any] | None = None
    dual_view_fusion: bool = True
    basic_fusion: Literal["concat", "add", "mult"] | None = None


def generate_configs(
    args: Namespace,
) -> tuple[PlatformConfig, TrainingConfig, ModelConfig]:
    with open(args.config_path, "r") as fp:
        config_file = json.load(fp)

    platform_config = PlatformConfig(
        dataset_path=args.dataset_path,
        config_path=args.config_path,
        experiments_path=args.experiments_path,
        sam_weights_path=args.sam_weights_path,
        num_gpus=args.num_gpus,
        workers_per_gpu=args.workers_per_gpu,
    )

    training_config = TrainingConfig(
        dataset_name=args.dataset.lower(),
        fold=args.fold,
        batch_size=args.batch_size if args.batch_size > 0 else 1,
        epochs=args.epochs if args.epochs > 0 else 1,
        loss=config_file.get("loss", None),
        optimizer=config_file.get("optimizer", None),
        start_lr=config_file["lr_scheduler"].get("start_lr", 1e-4),
        max_lr=config_file["lr_scheduler"].get("max_lr", 1e-3),
        intermediate_lr=config_file["lr_scheduler"].get("intermediate_lr", None),
        warm_up_ratio=config_file["lr_scheduler"].get("warm_up_ratio", 0.1),
        wind_down_ratio=config_file["lr_scheduler"].get("wind_down_ratio", 0.7),
    )

    model_config = ModelConfig(
        vit_model=args.vit_model,
        modalities=parse_modalities(args.modalities),
        semantic_only=True,
        channel_attention=config_file.get("channel_attention", None),
        spatial_attention=config_file.get("spatial_attention", None),
        dual_view_fusion=config_file.get("dual_view_fusion", True),
        basic_fusion=config_file.get("basic_fusion", None),
    )

    return platform_config, training_config, model_config


def parse_modalities(arg: str) -> tuple[str, ...]:
    modalities = []
    if "image" in arg.lower():
        modalities.append("image")
    if "depth" in arg.lower():
        modalities.append("depth")
    if "normals" in arg.lower():
        modalities.append("normals")

    return tuple(modalities)
