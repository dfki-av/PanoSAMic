"""
Author: Mahdi Chamseddine
"""

import torch


def collate_as_lists(
    batch_list: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    input_list = []
    label_list = []
    for input_dict, label_dict in batch_list:
        input_list.append(input_dict)
        label_list.append(label_dict)
    return input_list, label_list


def dict_list_to_tensors(
    dict_list: list[dict[str, torch.Tensor]], key: str
) -> torch.Tensor:
    return torch.cat([item[key] for item in dict_list], dim=0)
