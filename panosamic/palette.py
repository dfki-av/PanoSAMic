"""
Class-color palettes, shared by the standalone demo script and dataset
preprocessing scripts that don't ship their own colors.npy.

Author: Mahdi Chamseddine
"""

import colorsys

import numpy as np

# Stanford2D3DS's own published colors.npy, in Stanford2d3dsDataset.CLASS_NAMES
# order (colors[c + 1] indexing, index 0 = void/background). ToF360Dataset
# reuses this exact 13-class taxonomy, so this is the correct palette for it
# too, not just a generic approximation.
STANFORD2D3DS_COLORS = np.array(
    [
        [0, 0, 0],  # void/background
        [254, 158, 137],  # beam
        [85, 116, 127],  # board
        [255, 31, 33],  # bookcase
        [241, 255, 82],  # ceiling
        [0, 18, 141],  # chair
        [234, 234, 234],  # clutter
        [89, 173, 163],  # column
        [113, 143, 65],  # door
        [102, 168, 226],  # floor
        [100, 22, 116],  # sofa
        [84, 84, 84],  # table
        [190, 123, 75],  # wall
        [0, 244, 1],  # window
    ],
    dtype=np.uint8,
)


def build_palette(num_classes: int) -> np.ndarray:
    """Deterministic HSV-spaced (num_classes + 1, 3) uint8 palette.

    Generic fallback for class lists that aren't Stanford2D3DS's -- prefer
    ``STANFORD2D3DS_COLORS`` when the classes match. Index 0 is reserved for
    the ignore/void class; matches the ``colors[c + 1]`` indexing convention
    used throughout the evaluation code (e.g. ``PanoSAMicEvaluator``,
    ``eval_sam3_panosamic.py``).
    """
    colors = np.zeros((num_classes + 1, 3), dtype=np.uint8)
    for c in range(num_classes):
        hue = c / max(num_classes, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        colors[c + 1] = (round(r * 255), round(g * 255), round(b * 255))
    return colors
