"""
Author: Mahdi Chamseddine
"""

# from math import sqrt
from pathlib import Path

import pytorch_optimizer
import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from panosamic.datasets import BaseDataset
from panosamic.evaluation.loss import PanoSAMicLoss
from panosamic.evaluation.metrics import compute_metrics
from panosamic.evaluation.scheduler import PanoSAMicLRScheduler
from panosamic.evaluation.utils.config import TrainingConfig
from panosamic.evaluation.utils.data import collate_as_lists
from panosamic.evaluation.utils.distributed_handler import DistributedHandler
from panosamic.model import PanoSAMic


class PanoSAMicTrainer:
    PBAR_FORMAT: str = (
        ""
        + "{l_bar}{bar}| "
        + "{n_fmt: >4}/{total_fmt: >4} "
        + "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
    MAX_PRECISION: int = 5
    LOG_INTERVAL: int = 5
    NOMINAL_BATCH_SIZE: int = 4

    def __init__(
        self,
        distributed_helper: DistributedHandler,
        model: PanoSAMic,
        model_path: Path,
        training_set: BaseDataset,
        validation_set: BaseDataset,
        config: TrainingConfig,
        num_gpus: int,
        workers_per_gpu: int,
        checkpoint_path: Path,
    ) -> None:
        self.dh = distributed_helper
        self.dataset = training_set
        self.training_loader = DataLoader(
            training_set,
            batch_size=config.batch_size,
            shuffle=num_gpus == 1,
            sampler=(
                DistributedSampler(training_set, shuffle=True, drop_last=True)
                if num_gpus > 1
                else None
            ),
            num_workers=workers_per_gpu,
            collate_fn=collate_as_lists,
            pin_memory=True,
            drop_last=False if num_gpus > 1 else True,
        )

        self.validation_loader = DataLoader(
            validation_set,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=(
                DistributedSampler(validation_set, shuffle=False, drop_last=True)
                if num_gpus > 1
                else None
            ),
            num_workers=workers_per_gpu,
            collate_fn=collate_as_lists,
            drop_last=False if num_gpus > 1 else True,
        )

        # Loss Function
        loss_function = PanoSAMicLoss(
            class_weights=training_set.class_weights,
            ignore_index=training_set.ignore_index,
            config=config.loss,
            total_steps=config.epochs * len(self.training_loader),
        )

        # Optimizer and scheduler
        warm_up_iter = int(config.warm_up_ratio * len(self.training_loader))
        warm_down_iter = int(config.wind_down_ratio * len(self.training_loader))

        # Scaling the learning rate according to Granziol et al.
        # Learning Rates as a Function of Batch Size: A Random Matrix Theory Approach to Neural Network Training
        # https://www.jmlr.org/papers/v23/20-1258.html
        batch_size_ratio = num_gpus * config.batch_size / self.NOMINAL_BATCH_SIZE
        # Testing showed that linear scaling worked better for Stanford2D3DS dataset
        # max_lr = config.max_lr * sqrt(batch_size_ratio)
        max_lr = config.max_lr * batch_size_ratio

        if config.optimizer == "Ranger21":
            self.optimizer = pytorch_optimizer.Ranger21(
                params=model.parameters(),
                num_iterations=len(self.training_loader) * config.epochs,
                lr=max_lr,
                num_warm_up_iterations=warm_up_iter,
                num_warm_down_iterations=warm_down_iter,
                warm_down_min_lr=max_lr * 1e-4,
            )
            self.lr_scheduler = None
        else:
            self.optimizer = torch.optim.RAdam(
                params=model.parameters(),
                lr=1,  # Must use initial LR as 1 when using LambdaLR
                decoupled_weight_decay=True,
            )
            self.lr_scheduler = PanoSAMicLRScheduler(
                optimizer=self.optimizer,
                start_lr=config.start_lr,
                max_lr=max_lr,
                warm_up_steps=warm_up_iter,
                wind_down_step=warm_down_iter,
                total_steps=int(config.epochs * len(self.training_loader)),
                # Optional: Set to None for constant max_lr before wind-down
                intermediate_lr=config.intermediate_lr,
            )

        if self.dh.has_cuda:
            model = model.to(self.dh.local_rank)
            self.loss_function = loss_function.to(self.dh.local_rank)

        self.model = self.dh.prepare_model(model)
        self.model_path = model_path

        # Metrics
        self.best_miou = 0
        self.miou = 0
        self.macc = 0

        self.tb_writer = SummaryWriter(checkpoint_path) if self.dh.is_master else None

    def load_checkpoint(self, resume: bool = False):
        self.dh.print(f"Loading weights from: {self.model_path}")

        checkpoint = self.dh.load_state(self.model_path)
        model_weights = checkpoint if not resume else checkpoint["model"]
        if isinstance(self.model, DistributedDataParallel):
            self.model.module.load_state_dict(model_weights, strict=resume)
        else:
            self.model.load_state_dict(model_weights, strict=resume)
        self.epoch = 0 if not resume else checkpoint["epoch"]
        self.best_miou = 0 if not resume else checkpoint["best_miou"]
        self.miou = 0 if not resume else checkpoint["miou"]
        self.macc = 0 if not resume else checkpoint["macc"]

        if not resume:
            return

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def save_checkpoint(self, path: Path, full_checkpoint: bool = True) -> None:
        if self.dh.is_master:
            if isinstance(self.model, (DataParallel, DistributedDataParallel)):
                # this is a little hack to make loading weights work with
                # code that does not know about DDP training
                model_state = self.model.module.state_dict()
            else:
                model_state = self.model.state_dict()
            data = {
                "epoch": self.epoch,
                "model": model_state,
                "best_miou": self.best_miou,
                "miou": self.miou,
                "macc": self.macc,
            }
            if full_checkpoint:
                data.update(
                    {
                        "optimizer": self.optimizer.state_dict(),
                        "lr_scheduler": (
                            self.lr_scheduler.state_dict()
                            if self.lr_scheduler is not None
                            else None
                        ),
                    }
                )
            torch.save(data, path)

    def train_one_epoch(self, epoch: int) -> float:
        # Make sure gradient tracking is on
        self.epoch = epoch
        self.model.train()
        if isinstance(self.training_loader.sampler, DistributedSampler):
            self.training_loader.sampler.set_epoch(epoch)

        smooth_loss = None
        running_loss = None

        length = len(self.training_loader)

        with tqdm(
            total=length,
            desc=f"[Trg: Ep {epoch: >2}] - loss {'': >7}",
            # only show the progress bar on the main process
            disable=not self.dh.is_master,
            dynamic_ncols=True,
            bar_format=self.PBAR_FORMAT,
        ) as pbar:
            for step, batch_data in enumerate(self.training_loader):
                input_list, label_list = batch_data
                input_list = self.dh.move_to_gpu(input_list)
                label_list = self.dh.move_to_gpu(label_list)

                # Zero gradients for every iteration
                self.optimizer.zero_grad()

                # Generate predictions for this batch
                prediction_list = self.model(input_list)

                # Compute the loss and its gradients
                loss = self.loss_function(
                    prediction_list,
                    label_list,
                    steps=(epoch - 1) * length + step + 1,
                )
                loss.backward()

                # for name, param in self.model.named_parameters():
                #     if param.grad is None:
                #         print(param.requires_grad)
                # exit()

                # Adjust learning weights and learning rate
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # Gather data and report
                last_loss = loss.clone()
                smooth_loss = smooth_loss + loss if smooth_loss else loss.clone()
                running_loss = running_loss + loss if running_loss else loss.clone()

                self.dh.merge_distributed_values(smooth_loss)
                self.dh.merge_distributed_values(last_loss)
                if self.dh.is_master:
                    last_loss = self.dh.move_to_cpu(last_loss)
                    last_loss = last_loss.item()
                    _p = self._get_precision(last_loss)
                    pbar.set_description(
                        f"[Trg: Ep {epoch: >2}] - loss {last_loss:.{_p}f}"
                    )
                    pbar.update()

                if self.dh.is_master and ((step + 1) % self.LOG_INTERVAL == 0):
                    if self.lr_scheduler:
                        last_lr = self.lr_scheduler.get_last_lr()[-1]
                    elif isinstance(self.optimizer, pytorch_optimizer.Ranger21):
                        last_lr = self.optimizer.current_lr
                    else:
                        last_lr = self.optimizer.param_groups[-1]["lr"]

                    tb_x = (epoch - 1) * length + step + 1
                    smooth_loss = self.dh.move_to_cpu(smooth_loss)
                    smooth_loss = smooth_loss.item()
                    self.tb_writer.add_scalar(  # type: ignore (dh.is_master)
                        "Loss/train", smooth_loss / self.LOG_INTERVAL, tb_x
                    )
                    self.tb_writer.add_scalar(  # type: ignore (dh.is_master)
                        "Learning Rate", last_lr, tb_x
                    )

                smooth_loss = (
                    None if ((step + 1) % self.LOG_INTERVAL == 0) else smooth_loss
                )

            avg_loss = (
                running_loss / length
                if running_loss
                else self.dh.move_to_gpu(torch.tensor([0]))
            )
            self.dh.merge_distributed_values(avg_loss)
            if self.dh.is_master:
                avg_loss = self.dh.move_to_cpu(avg_loss)
                avg_loss = avg_loss.item()
                _p = self._get_precision(avg_loss)
                pbar.set_description(
                    f"[Trg: Ep {epoch: >2}] - loss {avg_loss:.{_p}f}",
                    refresh=True,
                )
                return avg_loss
        return 0

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
                vloss = self.loss_function(voutputs, vlabels)
                running_loss = running_loss + vloss if running_loss else vloss.clone()

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

                self.tb_writer.add_scalar(  # type: ignore (dh.is_master)
                    "mIoU", miou, epoch
                )
                self.tb_writer.add_scalar(  # type: ignore (dh.is_master)
                    "mAcc", macc, epoch
                )
                self.tb_writer.add_scalars(  # type: ignore (dh.is_master)
                    "Per class mIoU", class_miou, epoch
                )

                _p = self._get_precision(avg_loss)
                pbar.set_description(
                    f"[Val: Ep {epoch: >2}] - loss {avg_loss:.{_p}f}",
                    refresh=True,
                )
                return miou, macc, avg_loss

        return 0, 0, 0

    def _get_precision(self, num: float) -> int:
        if num < 10:
            return self.MAX_PRECISION
        elif num < 100:
            return self.MAX_PRECISION - 1
        elif num < 1000:
            return self.MAX_PRECISION - 2
        else:  # loss is greater than 1000, aesthetics don't matter...
            return self.MAX_PRECISION - 3
