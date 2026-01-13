"""
Author: Mahdi Chamseddine
"""

import torch
import torch.nn as nn


def count_params(model: nn.Module, verbose: bool = True) -> dict[str, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: PyTorch model to analyze
        verbose: If True, print the results

    Returns:
        Dictionary with 'total' and 'trainable' parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    if verbose:
        print(f"Total parameters:      {total_params:>15,}")
        print(f"Trainable parameters:  {trainable_params:>15,}")
        print(f"Frozen parameters:     {frozen_params:>15,}")
        print(f"Trainable ratio:       {trainable_params/total_params*100:>14.2f}%")

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
    }


def count_flops(
    model: nn.Module,
    n_modalities: int,
    verbose: bool = True,
    device: str = "cuda",
) -> dict[str, float] | None:
    """Count FLOPs for a model using fvcore.

    Args:
        model: PyTorch model to analyze
        n_modalities: Number of modalities (1=RGB, 2=RGBD, 3=RGBDN)
        verbose: If True, print the results
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Dictionary with FLOP counts or None if fvcore not available
    """
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
    except ImportError:
        if verbose:
            print("Can't compute model FLOPs. fvcore is not installed.")
            print("Install with: pip install fvcore")
        return None

    # Create dummy input matching expected dimensions (512x1024 panorama)
    input_dict = {"image": torch.zeros(3, 512, 1024, device=device)}
    if n_modalities > 1:
        input_dict["depth"] = torch.zeros(3, 512, 1024, device=device)
    if n_modalities > 2:
        input_dict["normals"] = torch.zeros(3, 512, 1024, device=device)

    flops = FlopCountAnalysis(model.eval(), [input_dict])  # type: ignore
    total_flops = flops.total()
    gflops = total_flops / 1e9

    if verbose:
        print(f"Total FLOPs:           {total_flops:>15,}")
        print(f"Total GFLOPs:          {gflops:>15.2f}")

    return {
        "flops": total_flops,
        "gflops": gflops,
    }
