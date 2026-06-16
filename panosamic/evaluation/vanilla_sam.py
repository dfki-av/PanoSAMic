from argparse import ArgumentParser
from typing import Any

import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def show_anns(anns: list[dict[str, Any]]) -> np.ndarray | None:
    """Render SAM mask annotations as a colour overlay; returns ``None`` if the list is empty."""
    if len(anns) == 0:
        return None
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)

    img = np.zeros(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            3,
        )
    )
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.random.random(3)
        img[m] = color_mask

    return img


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        required=True,
        help="SAM checkpoint path",
    )

    parser.add_argument(
        "--input_img",
        type=str,
        required=True,
        help="Path to input image to be used",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    sam_checkpoint: str = args.sam_checkpoint
    input_img: str = args.input_img

    idx = sam_checkpoint.index("vit_")
    model_type = sam_checkpoint[idx : idx + len("vit_x")]

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    image = np.array(Image.open(input_img))
    masks = mask_generator.generate(image)

    img = show_anns(masks)
    if img is None:
        print("No annotations found.")
        return
    img = img * 255 if np.max(img) <= 1 else img
    Image.fromarray(img.astype(np.uint8)).save("output.png")


if __name__ == "__main__":
    main()
