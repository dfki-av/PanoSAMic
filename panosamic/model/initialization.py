"""
Author: Mahdi Chamseddine
"""

import torch


def delta_orthogonal_(tensor: torch.Tensor, gain: float = 1) -> torch.Tensor:
    """
    Delta-orthogonal initialization of convolution filter weights.

    @param tensor: The tensor that shall be initialized.
    @param gain: The target gain value for initialization.
    @returns: The initialized tensor.
    """
    with torch.no_grad():
        torch.nn.init.zeros_(tensor)
        ortho = torch.zeros(tensor.shape[0], tensor.shape[1])
        torch.nn.init.orthogonal_(ortho, gain=gain)  # type:ignore
        tensor[:, :, tensor.shape[2] // 2, tensor.shape[3] // 2] = ortho
    return tensor


def orthogonal_module_init(module: torch.nn.Module) -> None:
    """
    Orthogonal initialization of an entire module.

    Convolution filters with a kernel size larger than one are initialized with the delta-orthogonal method. Fully
    connected layers (nn.Linear) and 1x1 convolution are initialized with the standard orthogonal initialization.

    @param module: The pyTorch neural network module that shall be initialized.
    """
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        if module.weight.shape[2] == 1 and module.weight.shape[3] == 1:
            torch.nn.init.orthogonal_(module.weight)
        else:
            delta_orthogonal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
