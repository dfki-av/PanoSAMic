"""
Author: Mahdi Chamseddine
"""

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from panosamic.datasets import BaseDataset
from panosamic.evaluation.metrics import compute_metrics
from panosamic.evaluation.utils.data import collate_as_lists
from panosamic.evaluation.utils.distributed_handler import DistributedHandler
from panosamic.model import PanoSAMic


class PanoSAMicEvaluator:
    PBAR_FORMAT: str = (
        ""
        + "{l_bar}{bar}| "
        + "{n_fmt: >4}/{total_fmt: >4} "
        + "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
    MAX_PRECISION: int = 5

    def __init__(
        self,
        distributed_helper: DistributedHandler,
        model: PanoSAMic,
        model_path: Path,
        dataset: BaseDataset,
        batch_size: int,
        num_gpus: int,
        workers_per_gpu: int,
    ) -> None:
        self.dh = distributed_helper
        self.dataset = dataset

        self.validation_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=num_gpus == 1,
            sampler=(
                DistributedSampler(dataset, shuffle=True, drop_last=True)
                if num_gpus > 1
                else None
            ),
            num_workers=workers_per_gpu,
            collate_fn=collate_as_lists,
            drop_last=False if num_gpus > 1 else True,
        )

        if self.dh.has_cuda:
            model = model.to(self.dh.local_rank)

        self.model = self.dh.prepare_model(model)
        self.model_path = model_path

        self.colors = None
        if (color_file := self.dataset.dataset_path / "assets" / "colors.npy").exists():
            self.colors = np.load(color_file)

    def load_checkpoint(self):
        self.dh.print(f"Loading weights from: {self.model_path}")

        checkpoint = self.dh.load_state(self.model_path)
        model_weights = checkpoint["model"]
        if isinstance(self.model, DistributedDataParallel):
            self.model.module.load_state_dict(model_weights, strict=True)
        else:
            self.model.load_state_dict(model_weights, strict=True)
        self.epoch = checkpoint.get("epoch", -1)

    def eval_one_epoch(self, epoch: int) -> tuple[float, float, float]:
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        self.model.eval()

        running_loss = None

        length = len(self.validation_loader)

        # Disable gradient computation and reduce memory consumption.
        area_intersection = torch.zeros(self.dataset.NUM_CLASSES)
        area_union = torch.zeros(self.dataset.NUM_CLASSES)
        area_target = torch.zeros(self.dataset.NUM_CLASSES)
        area_intersection = self.dh.move_to_gpu(area_intersection)
        area_union = self.dh.move_to_gpu(area_union)
        area_target = self.dh.move_to_gpu(area_target)
        with (
            torch.no_grad(),
            tqdm(
                total=length,
                desc=f"[Val: Ep {epoch: >2}]{'': >15}",
                # only show the progress bar on the main process
                disable=not self.dh.is_master,
                dynamic_ncols=True,
                bar_format=self.PBAR_FORMAT,
            ) as pbar,
        ):
            for vdata in self.validation_loader:
                vinputs, vlabels = vdata
                vinputs = self.dh.move_to_gpu(vinputs)
                vlabels = self.dh.move_to_gpu(vlabels)
                voutputs = self.model(vinputs)

                metrics = compute_metrics(
                    pred_list=voutputs,
                    label_list=vlabels,
                    num_classes=self.dataset.NUM_CLASSES,
                    ignore_index=self.dataset.ignore_index,
                    device=self.model.device,
                )

                area_intersection += metrics[0]
                area_union += metrics[1]
                area_target += metrics[2]
                pbar.update() if self.dh.is_master else None

            avg_loss = (
                running_loss / length
                if running_loss
                else self.dh.move_to_gpu(torch.tensor([0]))
            )

            self.dh.merge_distributed_values(avg_loss)
            # Sum up the values over all GPUs to get the equivalent of single GPU validation
            self.dh.merge_distributed_values(area_intersection, dist.ReduceOp.SUM)
            self.dh.merge_distributed_values(area_union, dist.ReduceOp.SUM)
            self.dh.merge_distributed_values(area_target, dist.ReduceOp.SUM)

            if self.dh.is_master:
                iou_per_class = area_intersection / area_union
                acc_per_class = area_intersection / area_target

                miou = torch.nanmean(iou_per_class)
                macc = torch.nanmean(acc_per_class)

                class_miou = {
                    name: torch.tensor([iou], device=iou_per_class.device)
                    for name, iou in zip(self.dataset.CLASS_NAMES, iou_per_class)
                }

                avg_loss = self.dh.move_to_cpu(avg_loss)
                miou = self.dh.move_to_cpu(miou)
                macc = self.dh.move_to_cpu(macc)
                class_miou = self.dh.move_to_cpu(class_miou)
                avg_loss = avg_loss.item()
                miou = miou.item()
                macc = macc.item()

                _p = self._get_precision(avg_loss)
                pbar.set_description(
                    f"[Val: Ep {epoch: >2}] - loss {avg_loss:.{_p}f}",
                    refresh=True,
                )
                return miou, macc, avg_loss

        return 0, 0, 0

    # TODO check
    # def save_results(
    #     self,
    #     sample_name: str,
    #     metrics: tuple[np.ndarray, ...],
    #     vinputs: list[dict[str, torch.Tensor]],
    #     vlabels: list[dict[str, torch.Tensor]],
    #     voutputs: list[dict[str, torch.Tensor]],
    #     format: str = "png",
    # ) -> None:
    #     assert (
    #         self.colors is not None
    #     ), "Dataset colors are undefined, can't save results."

    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", category=RuntimeWarning)
    #         sample_iou = float(np.nanmean(metrics[0] / metrics[1]))

    #     image = vinputs[0]["image"].squeeze(0).numpy(force=True).transpose((1, 2, 0))
    #     prediction = (
    #         torch.argmax(voutputs[0]["sem_preds"], dim=1).squeeze(0).numpy(force=True)
    #     )
    #     ground_truth = vlabels[0]["semantics"].squeeze(0).numpy(force=True)
    #     h, w = ground_truth.shape

    #     pred_img = np.zeros((h, w, 3))
    #     gt_img = np.zeros((h, w, 3))
    #     for c in range(self.dataset.NUM_CLASSES):
    #         pred_img[prediction == c] = self.colors[c + 1]
    #         gt_img[ground_truth == c] = self.colors[c + 1]

    #     file_name = f"{sample_iou*100:.2f}_{sample_name.split("/")[-1]}"
    #     image = Image.fromarray(image.astype(np.uint8))
    #     image.save(
    #         f"visualizations/colored/{file_name}.{format}",
    #         format=format,
    #         lossless=True,
    #     )

    #     pred_img = Image.fromarray(pred_img.astype(np.uint8))
    #     pred_img.save(
    #         f"visualizations/prediction/{file_name}.{format}",
    #         format=format,
    #         lossless=True,
    #     )

    #     gt_img = Image.fromarray(gt_img.astype(np.uint8))
    #     gt_img.save(
    #         f"visualizations/ground_truth/{file_name}.{format}",
    #         format=format,
    #         lossless=True,
    #     )

    def _get_precision(self, num: float) -> int:
        if num < 10:
            return self.MAX_PRECISION
        elif num < 100:
            return self.MAX_PRECISION - 1
        elif num < 1000:
            return self.MAX_PRECISION - 2
        else:  # loss is greater than 1000, aesthetics don't matter...
            return self.MAX_PRECISION - 3
