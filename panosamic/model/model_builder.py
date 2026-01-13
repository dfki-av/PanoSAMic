"""
Author: Mahdi Chamseddine
"""

from functools import partial

import torch
import torch.nn as nn
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling.transformer import TwoWayTransformer

from panosamic.evaluation.utils.config import ModelConfig
from panosamic.model.attention import AttentionBuilder
from panosamic.model.fusion import BasicFusion, FeatureFusion
from panosamic.model.image_encoder import ImageEncoderViT
from panosamic.model.initialization import orthogonal_module_init
from panosamic.model.panosamic_net import PanoSAMic
from panosamic.model.semantic_decoder import BaselineDecoder, ConvDecoder


def freeze_parameters(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


@torch.no_grad()
def image_encoder_builder(
    image_size: int,
    vit_patch_size: int,
    prompt_embed_dim: int,
    vit_model: str = "",
) -> ImageEncoderViT:
    # sam_vit_h params (default)
    encoder_embed_dim = 1280
    encoder_depth = 32
    encoder_num_heads = 16
    encoder_global_attn_indexes = (7, 15, 23, 31)

    if "vit_l" in vit_model:
        encoder_embed_dim = 1024
        encoder_depth = 24
        encoder_num_heads = 16
        encoder_global_attn_indexes = (5, 11, 17, 23)
    elif "vit_b" in vit_model:
        encoder_embed_dim = 768
        encoder_depth = 12
        encoder_num_heads = 12
        encoder_global_attn_indexes = (2, 5, 8, 11)

    model = ImageEncoderViT(  # params based on model size
        img_size=image_size,
        patch_size=vit_patch_size,
        in_chans=3,  # default from SAM
        embed_dim=encoder_embed_dim,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=4,  # default from SAM
        out_chans=prompt_embed_dim,
        qkv_bias=True,  # default from SAM
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # default from SAM
        act_layer=nn.GELU,  # default from SAM
        use_abs_pos=True,  # default from SAM
        use_rel_pos=True,  # default from SAM
        rel_pos_zero_init=True,  # default from SAM
        window_size=14,  # default from SAM
        global_attn_indexes=encoder_global_attn_indexes,
    )
    model.eval()

    return model


@torch.no_grad()
def prompt_encoder_builder(
    embed_dim: int,
    image_embedding_size: int,
    input_image_size: int,
) -> PromptEncoder:
    model = PromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(input_image_size, input_image_size),
        mask_in_chans=16,  # default from SAM
    )
    model.eval()

    return model


@torch.no_grad()
def mask_decoder_builder(
    embedding_dim: int,
) -> MaskDecoder:
    model = MaskDecoder(
        num_multimask_outputs=3,  # default from SAM
        transformer=TwoWayTransformer(
            depth=2,  # default from SAM
            embedding_dim=embedding_dim,
            mlp_dim=2048,  # default from SAM
            num_heads=8,  # default from SAM
        ),
        transformer_dim=embedding_dim,
        iou_head_depth=3,  # default from SAM
        iou_head_hidden_dim=256,  # default from SAM
    )
    model.eval()

    return model


def feature_fusion_builder(
    in_channels: int,
    out_channels: int,
    n_modalities: int,
    config: ModelConfig,
    in_size: int,
    depth: int,
) -> BasicFusion | FeatureFusion:
    if not config.basic_fusion:
        model = FeatureFusion(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modalities=n_modalities,
            attention_builder=AttentionBuilder(config),
            in_size=in_size,
            depth=depth,
        )
    else:
        model = BasicFusion(
            fusion_type=config.basic_fusion,
            in_channels=in_channels,
            n_modalities=n_modalities,
            in_size=in_size,
            out_size=256,  # hardcoded
            depth=depth,
        )
    return model


def semantic_decoder_builder(
    in_channels: int,
    num_classes: int,
    out_size: int,
    depth: int,
    dual_view_fusion: bool,
) -> ConvDecoder:
    model = ConvDecoder(
        in_channels=in_channels,
        num_classes=num_classes,
        out_size=out_size,
        depth=depth,
        dual_view_fusion=dual_view_fusion,
    )
    return model


def panosamic_builder(
    config: ModelConfig,
    num_classes: int,
    freeze_encoder: bool = True,
) -> PanoSAMic:
    image_size: int = 1024  # default from SAM
    vit_patch_size: int = 16  # default from SAM
    image_embedding_size: int = image_size // vit_patch_size  # default from SAM
    prompt_embed_dim: int = 256  # default from SAM

    # ImageEncoderViT
    image_encoder = image_encoder_builder(
        image_size=image_size,
        vit_patch_size=vit_patch_size,
        prompt_embed_dim=prompt_embed_dim,
        vit_model=config.vit_model,
    )

    # PromptEncoder
    prompt_encoder = (
        None
        if config.semantic_only
        else prompt_encoder_builder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=image_size,
        )
    )
    # MaskDecoder
    mask_decoder = (
        None
        if config.semantic_only
        else mask_decoder_builder(
            embedding_dim=prompt_embed_dim,
        )
    )

    # FeatureFusion
    feature_fuser = feature_fusion_builder(
        in_channels=image_encoder.embed_dim,
        out_channels=prompt_embed_dim,
        n_modalities=len(config.modalities),
        config=config,
        in_size=image_embedding_size,
        depth=len(image_encoder.global_attn_indexes),
    )

    # ConvDecoder or BaselineDecoder
    # Use BaselineDecoder when basic_fusion is specified (for ablation studies)
    # Otherwise use ConvDecoder with FeatureFusion
    use_baseline_decoder = config.basic_fusion is not None
    semantic_decoder = (
        BaselineDecoder(
            in_channels=prompt_embed_dim,
            num_classes=num_classes,
            n_modalities=len(config.modalities),
            out_size=4 * image_embedding_size,
        )
        if use_baseline_decoder
        else semantic_decoder_builder(
            in_channels=feature_fuser.out_channels,
            num_classes=num_classes,
            out_size=4 * image_embedding_size,  # 4 times upscale
            depth=feature_fuser.depth,
            dual_view_fusion=config.dual_view_fusion,
        )
    )

    # PanoSAMic
    model = PanoSAMic(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        feature_fuser=feature_fuser,
        semantic_decoder=semantic_decoder,
        input_modalities=config.modalities,
        semantic_only=config.semantic_only,
        dual_view_fusion=config.dual_view_fusion,
        # pixel_mean=[123.675, 116.28, 103.53],  # default from SAM
        # pixel_std=[58.395, 57.12, 57.375],  # default from SAM
    )

    orthogonal_module_init(model)

    # Freeze the encoder weights
    freeze_parameters(model.image_encoder) if freeze_encoder else None
    freeze_parameters(model.prompt_encoder) if model.prompt_encoder else None
    freeze_parameters(model.mask_decoder) if model.mask_decoder else None

    return model
