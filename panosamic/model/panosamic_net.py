"""
Author: Mahdi Chamseddine
"""

from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.utils.amg import batch_iterator

from panosamic.model.cached_prompt_encoder import CachedPromptEncoder
from panosamic.model.fusion import BasicFusion, FeatureFusion
from panosamic.model.image_encoder import ImageEncoderViT
from panosamic.model.mask_fusion import fuse_dual_view_masks
from panosamic.model.mask_postprocessing import (
    merge_masks_greedy,
    postprocess_instances,
)
from panosamic.model.prompt_validator import prompt_validator
from panosamic.model.semantic_decoder import BaselineDecoder, ConvDecoder


class PanoSAMic(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder | None,
        mask_decoder: MaskDecoder | None,
        feature_fuser: BasicFusion | FeatureFusion,
        semantic_decoder: BaselineDecoder | ConvDecoder,
        input_modalities: tuple[str, ...] = ("image", "depth", "normals"),
        semantic_only: bool = False,
        dual_view_fusion: bool = True,
        low_memory_mode: bool = True,
        # Used for SAM (segment everything) prompt
        points_per_side: int = 32,
        points_per_batch: int = 64,
        # Post-processing parameters for instance segmentation
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        min_mask_region_area: int = 0,
        # Pixel mean and std values from original SAM
        pixel_mean: list[float] = [123.675, 116.28, 103.53],
        pixel_std: list[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()

        # Sanity checks
        assert semantic_only or (
            prompt_encoder is not None and mask_decoder is not None
        ), "prompt_encoder and mask_decoder are required for full panosamic."
        assert "image" in input_modalities, (
            "'image' modality is required for instance segmentation"
        )

        self.modalities = input_modalities
        self.n_modalities = len(self.modalities)
        self.semantic_only = semantic_only
        self.dual_view_fusion = dual_view_fusion
        self.use_all_modalities = True
        self.low_memory_mode = low_memory_mode

        # Modules
        # TODO assert dims
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        # Cached wrapper for prompt_encoder (used in instance_segmentation_block)
        # Keeps prompt_encoder as a proper submodule for state_dict while adding caching
        self._cached_prompt_encoder = prompt_encoder
        # self._cached_prompt_encoder = (  # Warning Experimental..
        #     CachedPromptEncoder(prompt_encoder) if prompt_encoder is not None else None
        # )
        self.feature_fuser = (
            None if isinstance(semantic_decoder, BaselineDecoder) else feature_fuser
        )
        self.semantic_decoder = semantic_decoder

        self.img_size = self.image_encoder.img_size
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch

        # Post-processing parameters
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.min_mask_region_area = min_mask_region_area

        self.pixel_mean: torch.Tensor
        self.pixel_std: torch.Tensor
        self.register_buffer(
            "pixel_mean",
            torch.Tensor(pixel_mean).view(-1, 1, 1),
            False,
        )
        self.register_buffer(
            "pixel_std",
            torch.Tensor(pixel_std).view(-1, 1, 1),
            False,
        )

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: list[dict[str, torch.Tensor]],
        batched_prompts: list[dict[str, Any]] = list(),
        multimask_output: bool = True,
    ) -> list[dict[str, torch.Tensor]]:
        input_images, image_shapes = self.data_preparation_block(batched_input)

        encoder_output_list, encoder_branch_lists = self.image_encoder_block(
            input_images
        )

        encoder_embeddings, encoder_branches = self.encoder_postprocessing_block(
            encoder_output_list,
            encoder_branch_lists,
        )

        # Use ALL modalities for instance segmentation (creates better over-segmentation)
        # Strategy: Segment EACH modality separately, then merge all masks together
        # This preserves boundaries from RGB, depth, AND normals for maximum over-segmentation

        # Encoder output structure with dual_view:
        # [rgb_v1, depth_v1, normals_v1, rgb_v2, depth_v2, normals_v2, ...]

        if self.dual_view_fusion:
            # Segment each modality+view separately, then merge
            instance_predictions = self.dual_view_instance_segmentation_block(
                encoder_embeddings=encoder_embeddings,
                image_shapes=image_shapes,
                batched_prompts=batched_prompts,
                multimask_output=multimask_output,
                use_all_modalities=self.use_all_modalities,  # Use all modalities for best over-segmentation
            )
        else:
            # Segment each modality separately, then merge
            instance_predictions = self.instance_segmentation_block(
                encoder_embeddings=encoder_embeddings,
                image_shapes=image_shapes,
                batched_prompts=batched_prompts,
                multimask_output=multimask_output,
                use_all_modalities=self.use_all_modalities,  # Use all modalities for best over-segmentation
            )

        fused_features = self.feature_fusion_block(
            encoder_embeddings=encoder_embeddings,
            encoder_branch_batched=encoder_branches,
        )

        # Free embeddings after use
        if self.low_memory_mode:
            del encoder_embeddings, encoder_branches

        semantic_predictions = self.semantic_segmentation_block(
            fused_features=fused_features,
            image_shapes=image_shapes,
        )

        if self.low_memory_mode:
            del fused_features

        segmentation_output = []
        for instances, semantics in zip(instance_predictions, semantic_predictions):
            combined = {
                "instance_masks": instances,
                "sem_preds": semantics,
            }
            segmentation_output.append(combined)

        return segmentation_output

    def data_preparation_block(
        self,
        batched_input: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
        prep_methods = (
            [
                partial(self._preprocess, rotate=False),
                partial(self._preprocess, rotate=True),
            ]
            if self.dual_view_fusion
            else [self._preprocess]
        )
        # Run encoder on all modalities before and after shifting
        # Example:
        # batch size is 1 with 3 modalities (image, depth, normals) ->
        # input_images batch size will be 6 (image, shifted_depth, shifted_normals)
        input_images = torch.stack(
            [
                f(batch[modality])
                for batch in batched_input
                for f in prep_methods
                for modality in self.modalities
            ],
            dim=0,
        )

        # Save the original size of the images for resizing later
        image_shapes = [
            # if image_shape is not provided the default value is assumed to be the input size
            # batch.get("image_shape", tuple(batch["image"].shape[-2:]))
            # NOTE removed "image_shape" from input dictionary
            # NOTE shape is added once per image: rotations and modalities do not count
            tuple(batch["image"].shape[-2:])
            for batch in batched_input
        ]

        return input_images, image_shapes

    @torch.no_grad()
    def image_encoder_block(
        self,
        input_images: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        N, _, _, _ = input_images.shape
        encoder_output_list = []
        encoder_branch_lists = []

        if self.low_memory_mode and self.dual_view_fusion:
            # Process one image's full set of views and modalities at a time to save memory.
            modalities_per_view = 2 * self.n_modalities  # num_views is always 2 here
            for i in range(0, N, modalities_per_view):
                # This inner loop processes each view's modalities separately within the image's chunk.
                for j in range(i, i + modalities_per_view, self.n_modalities):
                    output, branches = self.image_encoder(
                        input_images[j : j + self.n_modalities, :]
                    )
                    encoder_output_list.append(output)
                    encoder_branch_lists.append(branches)
        else:
            # Original path: process one view's modalities at a time across all images.
            for i in range(0, N, self.n_modalities):
                output, branches = self.image_encoder(
                    input_images[i : i + self.n_modalities, :]
                )
                encoder_output_list.append(output)
                encoder_branch_lists.append(branches)

        return encoder_output_list, encoder_branch_lists

    def encoder_postprocessing_block(
        self,
        encoder_output_list: list[torch.Tensor],
        encoder_branch_lists: list[list[torch.Tensor]],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.feature_fuser:
            # Reshape a list of batches (N) of a list of branches (depth) we want a list of
            # branches (depth) of batched tensors NxCxHxW
            encoder_branch_batched = [[] for _ in range(self.feature_fuser.depth)]
            for batch_list in encoder_branch_lists:
                for b in range(self.feature_fuser.depth):
                    encoder_branch_batched[b].append(batch_list[b])

            encoder_branch_batched = [
                torch.cat(temp_list, dim=0) for temp_list in encoder_branch_batched
            ]
        else:  # Baseline: branches are unused
            encoder_branch_batched = [torch.Tensor([])]

        return (
            torch.cat(encoder_output_list, dim=0),
            encoder_branch_batched,
        )

    @torch.no_grad()
    def instance_segmentation_block(
        self,
        encoder_embeddings: torch.Tensor,
        image_shapes: list[tuple[int, ...]],
        batched_prompts: list[dict[str, Any]] = list(),
        multimask_output: bool = True,
        use_all_modalities: bool = True,
    ) -> list[list[dict[str, Any]]]:
        """
        Perform instance segmentation with configurable modality handling.

        Args:
            encoder_embeddings: Embeddings [rgb, depth, normals, ...] or just [rgb, ...]
            image_shapes: Original image shapes
            batched_prompts: Optional prompts for guided segmentation
            multimask_output: Whether to output multiple masks per point
            use_all_modalities: If True, segment each modality separately and merge.
                              If False, use only first modality (RGB only mode).

        Returns:
            List of merged mask lists, one per batch item
        """
        if self.prompt_encoder is None or self.mask_decoder is None:
            return [[] for _ in image_shapes]

        if not use_all_modalities:
            # RGB-only mode: segment only the first modality (RGB)
            rgb_embeddings = encoder_embeddings[:: self.n_modalities, :]
            return self.modality_instance_segmentation(
                image_embeddings=rgb_embeddings,
                image_shapes=image_shapes,
                batched_prompts=batched_prompts,
                multimask_output=multimask_output,
            )

        # Multi-modality mode: segment each modality separately, then merge
        # Structure: [rgb_img1, depth_img1, normals_img1, rgb_img2, depth_img2, ...]
        modality_masks_per_image = [[] for _ in image_shapes]

        # Segment each modality
        for mod_idx in range(self.n_modalities):
            # Extract embeddings for this modality across all images
            modality_embeddings = encoder_embeddings[mod_idx :: self.n_modalities, :]

            # Segment this modality
            modality_predictions = self.modality_instance_segmentation(
                image_embeddings=modality_embeddings,
                image_shapes=image_shapes,
                batched_prompts=batched_prompts,
                multimask_output=multimask_output,
            )

            # Collect masks for each image
            for img_idx, masks in enumerate(modality_predictions):
                modality_masks_per_image[img_idx].extend(masks)

        # Post-process: remove duplicates within each image using NMS
        final_predictions = []
        for masks in modality_masks_per_image:
            if len(masks) == 0:
                final_predictions.append([])
                continue

            # Apply NMS to remove duplicate masks from different modalities
            merged_masks = merge_masks_greedy(masks, iou_threshold=self.box_nms_thresh)
            final_predictions.append(merged_masks)

        return final_predictions

    @torch.no_grad()
    def dual_view_instance_segmentation_block(
        self,
        encoder_embeddings: torch.Tensor,
        image_shapes: list[tuple[int, ...]],
        batched_prompts: list[dict[str, Any]] = list(),
        multimask_output: bool = True,
        use_all_modalities: bool = True,
    ) -> list[list[dict[str, Any]]]:
        """
        Perform instance segmentation with dual-view fusion.

        Strategy:
        1. Separate encoder_embeddings into unshifted and shifted views
        2. Run instance_segmentation_block on unshifted view
        3. Run instance_segmentation_block on shifted view
        4. Fuse masks from both views using fuse_dual_view_masks

        Args:
            encoder_embeddings: All modality+view embeddings
                               [rgb_v1, depth_v1, normals_v1, rgb_v2, depth_v2, normals_v2, ...]
            image_shapes: Original image shapes
            batched_prompts: Optional prompts for guided segmentation
            multimask_output: Whether to output multiple masks per point
            use_all_modalities: If True, use all modalities. If False, RGB only.

        Returns:
            List of merged mask lists, one per batch item
        """
        if self.prompt_encoder is None or self.mask_decoder is None:
            return [[] for _ in image_shapes]

        # Structure: [rgb_v1, depth_v1, normals_v1, rgb_v2, depth_v2, normals_v2, ...]
        # Extract embeddings per view: collect all modalities for view 0, then view 1
        N = encoder_embeddings.shape[0]
        num_embeddings_per_image = self.n_modalities * 2

        # Separate into unshifted (view 0) and shifted (view 1)
        unshifted_indices = []
        shifted_indices = []

        for img_idx in range(N // num_embeddings_per_image):
            base = img_idx * num_embeddings_per_image
            # View 0: rgb, depth, normals (first n_modalities)
            unshifted_indices.extend(range(base, base + self.n_modalities))
            # View 1: rgb, depth, normals (next n_modalities)
            shifted_indices.extend(
                range(base + self.n_modalities, base + num_embeddings_per_image)
            )

        unshifted_embeddings = encoder_embeddings[unshifted_indices, :]
        shifted_embeddings = encoder_embeddings[shifted_indices, :]

        # Run instance segmentation on each view
        unshifted_masks = self.instance_segmentation_block(
            encoder_embeddings=unshifted_embeddings,
            image_shapes=image_shapes,
            batched_prompts=batched_prompts,
            multimask_output=multimask_output,
            use_all_modalities=use_all_modalities,
        )

        shifted_masks = self.instance_segmentation_block(
            encoder_embeddings=shifted_embeddings,
            image_shapes=image_shapes,
            batched_prompts=batched_prompts,
            multimask_output=multimask_output,
            use_all_modalities=use_all_modalities,
        )

        # Fuse masks from both views using existing dual-view fusion
        fused_predictions = []
        for unshifted_list, shifted_list in zip(unshifted_masks, shifted_masks):
            fused_masks = fuse_dual_view_masks(
                unshifted_masks=unshifted_list,
                shifted_masks=shifted_list,
                iou_threshold=self.box_nms_thresh,
            )
            fused_predictions.append(fused_masks)

        return fused_predictions

    @torch.no_grad()
    def modality_instance_segmentation(
        self,
        image_embeddings: torch.Tensor,
        image_shapes: list[tuple[int, ...]],
        batched_prompts: list[dict[str, Any]] = list(),
        multimask_output: bool = True,
    ) -> list[list[dict[str, Any]]]:
        """
        Perform instance segmentation on a single modality's embeddings.

        This is the core SAM segmentation: prompts → mask decoder → post-processing.

        Args:
            image_embeddings: Embeddings for one modality across all images
            image_shapes: Original image shapes
            batched_prompts: Optional prompts for guided segmentation
            multimask_output: Whether to output multiple masks per point

        Returns:
            List of mask lists, one per batch item
        """
        prompt_encoder = (
            self.prompt_encoder if self.low_memory_mode else self._cached_prompt_encoder
        )
        if prompt_encoder is None or self.mask_decoder is None:
            return [[] for _ in image_shapes]

        if len(batched_prompts) != 0:
            prompt_encoder.invalidate_cache() if isinstance(
                prompt_encoder, CachedPromptEncoder
            ) else None

        instance_predictions: list[list[dict[str, Any]]] = []

        # Create prompts for each image if not provided
        if len(batched_prompts) == 0:
            batched_prompts = [{} for _ in range(len(image_shapes))]

        # For loop over all the batches
        for prompt, embeddings, shape in zip(
            batched_prompts, image_embeddings, image_shapes
        ):
            prompt, point_coords, point_labels = prompt_validator(
                prompt, self.points_per_side, self.device, self.img_size
            )

            # Process masks in streaming fashion to minimize peak memory
            # We'll collect filtered candidates, then apply global NMS at the end
            filtered_masks: list[torch.Tensor] = []
            filtered_ious: list[torch.Tensor] = []

            for points in batch_iterator(
                self.points_per_batch, point_coords, point_labels
            ):
                sparse_embeddings, dense_embeddings = prompt_encoder(
                    points=points,
                    boxes=prompt.get("boxes", None),
                    masks=prompt.get("mask_inputs", None),
                )

                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=embeddings.unsqueeze(0),
                    image_pe=prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                high_res_masks = self._postprocess(
                    low_res_masks,
                    input_size=shape,
                    original_size=shape,
                )
                if self.low_memory_mode:
                    # Early filtering to reduce memory: filter by IoU threshold immediately
                    # This reduces the number of masks we need to keep in memory
                    N, num_pred, H, W = high_res_masks.shape
                    masks_flat = high_res_masks.reshape(N * num_pred, H, W)
                    ious_flat = iou_predictions.reshape(N * num_pred)

                    # Keep only masks above IoU threshold
                    iou_mask = ious_flat >= self.pred_iou_thresh
                    if iou_mask.any():
                        filtered_masks.append(masks_flat[iou_mask])
                        filtered_ious.append(ious_flat[iou_mask])

                    # Free memory immediately
                    del (
                        high_res_masks,
                        low_res_masks,
                        iou_predictions,
                        masks_flat,
                        ious_flat,
                    )
                    if iou_mask.any():
                        del iou_mask
                else:
                    # Original behavior: keep everything
                    filtered_masks.append(high_res_masks)
                    filtered_ious.append(iou_predictions)

            if len(filtered_masks) == 0:
                instance_predictions.append([])
                continue

            # Concatenate filtered candidates
            all_masks = torch.cat(filtered_masks, dim=0)
            all_ious = torch.cat(filtered_ious, dim=0)

            # Free intermediate lists
            del filtered_masks, filtered_ious

            if not self.low_memory_mode:
                # Flatten multi-mask predictions (only needed if we didn't already flatten)
                N, num_pred, H, W = all_masks.shape
                all_masks = all_masks.reshape(N * num_pred, H, W)
                all_ious = all_ious.reshape(N * num_pred)

            # Post-process with internal batching for memory efficiency
            # NMS is always applied globally inside postprocess_instances
            processed_masks = postprocess_instances(
                masks=all_masks,
                iou_predictions=all_ious,
                pred_iou_thresh=self.pred_iou_thresh
                if not self.low_memory_mode
                else 0.0,  # Already filtered in low_memory_mode
                stability_score_thresh=self.stability_score_thresh,
                stability_score_offset=self.stability_score_offset,
                box_nms_thresh=self.box_nms_thresh,
                min_mask_region_area=self.min_mask_region_area,
                mask_threshold=self.mask_threshold,
                batch_size=self.points_per_batch if self.low_memory_mode else None,
            )

            instance_predictions.append(processed_masks)

        return instance_predictions

    def feature_fusion_block(
        self,
        encoder_embeddings: torch.Tensor,
        encoder_branch_batched: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.feature_fuser:
            fused_features: torch.Tensor = self.feature_fuser(encoder_branch_batched)
            _, C, _, W = fused_features.shape
            if self.dual_view_fusion:
                pe, rotated_pe = horizontal_positional_encoding(
                    int(W), int(C), self.device
                )
                fused_features[0::2, ...] += pe
                fused_features[1::2, ...] += rotated_pe
        else:  # Baseline: just concatenate the modalities along channel dimension
            N, C, H, W = encoder_embeddings.shape
            fused_features = encoder_embeddings.reshape(
                N // self.n_modalities, C * self.n_modalities, H, W
            )
        return fused_features

    def semantic_segmentation_block(
        self,
        fused_features: torch.Tensor,
        image_shapes: list[tuple[int, ...]],
    ) -> list[torch.Tensor]:
        semantic_predictions = self.semantic_decoder(fused_features)

        upscaled_predictions = [
            self._postprocess(prediction[None, :], shape, shape)
            for prediction, shape in zip(semantic_predictions, image_shapes)
        ]

        return upscaled_predictions

    def _preprocess(self, x: torch.Tensor, rotate: bool = False) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if rotate:  # rotate by 180 degrees
            x = torch.roll(x, shifts=int(x.shape[-1] // 2), dims=-1)

        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    # NB. not needed when training since shape is already enforced in preprocessing/preparation
    # def get_preprocess_shape(self, oldh: int, oldw: int) -> tuple[int, int]:
    #     """
    #     Compute the output size given input size and target long side length.
    #     """
    #     scale = self.img_size * 1.0 / max(oldh, oldw)
    #     newh, neww = oldh * scale, oldw * scale
    #     neww = int(neww + 0.5)
    #     newh = int(newh + 0.5)
    #     return (newh, neww)

    def _postprocess(
        self,
        masks: torch.Tensor,
        input_size: tuple[int, ...],
        original_size: tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def freeze_module(self):
        # Potential replacement to "with torch.no_grad()" for more flexibility
        # To be called after loading weights
        raise NotImplementedError


def horizontal_positional_encoding(width: int, d_model: int, device):
    """
    Compute sinusoidal positional encoding for the horizontal dimension.

    Args:
        width (int): Width of the panoramic image (number of columns).
        d_model (int): Dimension of the feature map or encoding.

    Returns:
        torch.Tensor: Positional encoding matrix of shape (1, width, d_model).
    """
    # Initialize positional encoding
    pe = np.zeros((width, d_model))
    position = np.arange(0, width)[:, np.newaxis]  # Shape: (width, 1)
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )  # Shape: (d_model/2,)

    # Compute sine and cosine components
    pe[:, 0::2] = np.sin(position * div_term)  # Even indices: sine
    pe[:, 1::2] = np.cos(position * div_term)  # Odd indices: cosine

    # Add batch and height dimensions for compatibility
    pe = pe[np.newaxis, :, :]  # Shape: (1, width, d_model)

    pe = torch.tensor(pe, dtype=torch.float32).permute(0, 2, 1).unsqueeze(2)
    rotated_pe = torch.roll(pe, shifts=width // 2, dims=-1)
    return pe.to(device), rotated_pe.to(device)
