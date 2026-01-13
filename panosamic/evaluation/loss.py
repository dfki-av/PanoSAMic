"""
Author: Mahdi Chamseddine
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_optimizer import DiceLoss, JaccardLoss

from panosamic.evaluation.utils.data import dict_list_to_tensors


class PanoSAMicLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,  # CrossEntropyLoss
        ignore_index: int = -1,
        reduction: str = "mean",
        gamma: int = 2,  # FocalLoss
        config: dict[str, Any] = {},
        total_steps: int = 0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.config = config

        self.scheduled_loss = self.config.get("ScheduledLoss", {})
        self.transition_start = total_steps
        self.transition_finish = total_steps
        self.w1, self.w2 = 1, 1
        self.eps = eps

        self.loss_weights = (
            self.config.get("CrossEntropyLoss", None),
            self.config.get("FocalLoss", None),
            self.config.get("DiceLoss", None),
            self.config.get("JaccardLoss", None),
        )

        if self.scheduled_loss:
            assert (
                self.loss_weights[0]
                and not self.loss_weights[1]
                and (  # XOR operation since only one is allowed
                    (self.loss_weights[2] is not None)
                    ^ (self.loss_weights[3] is not None)
                )
            ), (
                "ScheduledLoss is only supported for CrossEntropyLoss with DiceLoss or JaccardLoss"
            )

            self.transition_start *= self.scheduled_loss["transition_start_ratio"]
            self.transition_finish *= self.scheduled_loss["transition_finish_ratio"]
            self.factor = (1.0 - self.eps) / (
                self.transition_finish - self.transition_start
            )

        label_smoothing = self.config.get("label_smoothing", 0)
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self.focal_loss = FocalLoss(
            alpha=class_weights,
            gamma=gamma,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        self.dice_loss = DiceLoss(
            mode="multiclass",
            log_loss=False,
            from_logits=True,
            label_smooth=label_smoothing,
            ignore_index=ignore_index,
        )
        self._jaccard_loss = JaccardLoss(
            # Must use "multilabel" to handle the ignore_index outside the class
            mode="multilabel",
            log_loss=False,
            from_logits=True,
            label_smooth=label_smoothing,
        )

    def forward(
        self,
        prediction_list: list[dict[str, torch.Tensor]],
        label_list: list[dict[str, torch.Tensor]],
        steps: int | None = None,
    ) -> torch.Tensor:
        inputs = dict_list_to_tensors(prediction_list, key="sem_preds")
        targets = dict_list_to_tensors(label_list, key="semantics")

        losses = []
        losses.append(
            self.loss_weights[0] * self.cross_entropy_loss(inputs, targets)
            if self.loss_weights[0]
            else None
        )

        losses.append(
            self.loss_weights[1] * self.focal_loss(inputs, targets)
            if self.loss_weights[1]
            else None
        )

        losses.append(
            self.loss_weights[2] * self.dice_loss(inputs, targets)
            if self.loss_weights[2]
            else None
        )

        losses.append(
            self.loss_weights[3] * self.jaccard_loss(inputs, targets)
            if self.loss_weights[3]
            else None
        )

        if self.scheduled_loss and steps:  # step loss
            self.w1, self.w2 = self.get_scheduled_weights(steps=steps)

        overall_loss = None
        for loss in filter(None, losses):
            overall_loss = (
                overall_loss + (loss * self.w2) if overall_loss else (loss * self.w1)
            )

        assert overall_loss, "Loss cannot be None, loss function is misconfigured."
        return overall_loss

    def jaccard_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        N, C, _, _ = inputs.size()
        inputs = inputs.view(N, C, -1)
        targets = targets.view(N, -1)

        mask = targets != self.ignore_index
        inputs = inputs * mask[:, None, :]
        targets = F.one_hot((targets * mask).to(torch.long), C)
        targets = targets.permute(0, 2, 1) * mask[:, None, :]

        return self._jaccard_loss(inputs, targets)

    def get_scheduled_weights(self, steps: int) -> tuple[float, float]:
        # _t1 = self.eps - self.transition_start * self.factor + self.factor * steps
        # w = max(self.eps, min(1, _t1))

        _t1 = min(1, max(self.eps, 1 - self.factor * steps))
        _t2 = max(
            0,
            (math.pi * (steps - self.transition_start))
            / (2 * (self.transition_finish - self.transition_start)),
        )
        _t2 = 1 - math.cos(min(math.pi / 2, _t2))
        w = min(1, _t1 + _t2)  # ensure w doesn't go higher than 1

        return (w, 1 - w)


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: int = 2,
        ignore_index: int = -1,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            alpha (Tensor or None): Class weights (C,) (same as used in CrossEntropyLoss)
            gamma (float): Focusing parameter.
            ignore_index (int): Label to ignore in loss computation.
            reduction (str): "mean", "sum", or "none".
        """
        super().__init__()
        assert reduction in (
            "mean",
            "sum",
            "none",
        ), f"{reduction} is not a valud value for reduction ('mean', 'sum', 'none')"
        self.alpha = alpha  # Class weights (Tensor of shape (C,))
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (Tensor): Predicted logits, shape (N, C, H, W).
            target (Tensor): Ground truth labels, shape (N, H, W).
        Returns:
            focal_loss (Tensor): The computed focal loss.
        """
        # Compute standard CrossEntropyLoss without reduction
        ce_loss = F.cross_entropy(
            input,
            target,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)  # Compute probability of the correct class

        # Apply Focal Loss scaling
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            target_weights = self.alpha[target]  # Correct indexing for (N, H, W)
            focal_loss *= target_weights  # Element-wise multiplication

        # Handle reduction method
        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()

        return focal_loss
