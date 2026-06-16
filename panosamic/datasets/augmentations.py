import enum

import torch


def rotate_horizontal_tensor(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Cyclically shift tensor columns by ``shift`` pixels (wraps panoramic boundary)."""
    return torch.roll(x, shifts=shift, dims=-1)


def flip_horizontal(
    sample_data: dict[str, torch.Tensor],
    sample_labels: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Flip all modality and label tensors along the horizontal axis."""
    for key, value in sample_data.items():
        sample_data[key] = value.flip(-1)

    for key, value in sample_labels.items():
        sample_labels[key] = value.flip(-1)

    return sample_data, sample_labels


def rotate_horizontal(
    sample_data: dict[str, torch.Tensor],
    sample_labels: dict[str, torch.Tensor],
    shift: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Cyclically shift all tensors horizontally by ``shift`` pixels (random if ``None``)."""
    if shift is None:
        shift = int(torch.rand(1) * sample_data["image"].shape[-1])

    for key, value in sample_data.items():
        sample_data[key] = rotate_horizontal_tensor(value, shift)

    for key, value in sample_labels.items():
        if value is None:
            continue
        sample_labels[key] = rotate_horizontal_tensor(value, shift)

    return sample_data, sample_labels


def permute_colors(
    sample_data: dict[str, torch.Tensor],
    sample_labels: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Randomly permute the three RGB channels (depth and normals are unchanged)."""
    # Only modifies RGB
    idx = torch.randperm(3)
    sample_data["image"] = sample_data["image"][idx, :, :]

    return sample_data, sample_labels


class Augmentation(enum.Enum):
    """Enum of callable augmentation functions for use with ``augment_image``."""

    FLIP = enum.member(flip_horizontal)
    ROTATE = enum.member(rotate_horizontal)
    PERMUTE = enum.member(permute_colors)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


def augment_image(
    sample_data: dict[str, torch.Tensor],
    sample_labels: dict[str, torch.Tensor],
    augmentations: tuple[Augmentation, ...],
    enabled: bool = True,
    probabilities: tuple[float, ...] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Apply augmentations stochastically; each is applied independently with its own probability.

    When ``probabilities`` is ``None``, defaults to uniform ``1 / (n+1)`` per augmentation.
    """
    if not enabled:
        return sample_data, sample_labels
    n_aug = len(augmentations)
    if probabilities is None:
        probabilities = tuple([1 / (n_aug + 1) for _ in range(n_aug)])

    if n_aug != len(probabilities):
        raise ValueError(
            "Augmentation probabilities must be the same length as the augmentations."
        )

    for p, fn in zip(probabilities, augmentations, strict=False):
        if torch.rand(1) > p:
            continue
        sample_data, sample_labels = fn(sample_data, sample_labels)

    return sample_data, sample_labels
