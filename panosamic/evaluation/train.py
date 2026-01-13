"""
Author: Mahdi Chamseddine
"""

import json
import os
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import torch
import torch.nn as nn

from panosamic.datasets import build_dataset
from panosamic.evaluation.trainer import PanoSAMicTrainer
from panosamic.evaluation.utils.config import generate_configs
from panosamic.evaluation.utils.distributed_handler import DistributedHandler
from panosamic.evaluation.utils.efficiency import count_flops, count_params
from panosamic.evaluation.utils.parser import create_parser
from panosamic.evaluation.utils.slurm_utils import job_time_left
from panosamic.model import panosamic_builder

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_checkpoint_path(path: Path, id: str) -> Path | None:
    # Check if a specific experiment is provided
    if id in str(path.name):
        return path
    # Find latest experiment that matches the id in the provided path
    all_exp = [dir for dir in path.glob(f"*_{id}") if dir.is_dir()]
    all_exp.sort()
    if all_exp:
        return all_exp[-1]
    return None


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    platform_config, training_config, model_config = generate_configs(args=args)

    training_set = build_dataset(
        dataset_path=platform_config.dataset_path,
        config=training_config,
        n_modalities=len(model_config.modalities),
        test_mode=False,
    )
    validation_set = build_dataset(
        dataset_path=platform_config.dataset_path,
        config=training_config,
        n_modalities=len(model_config.modalities),
        test_mode=True,
    )

    model: nn.Module = panosamic_builder(
        config=model_config,
        num_classes=training_set.NUM_CLASSES,
        freeze_encoder=True,
    )

    CONFIG_PATH = Path(platform_config.config_path)
    EXPERIMENTS_PATH = Path(platform_config.experiments_path)
    if EXPERIMENTS_PATH.name != training_config.dataset_name:
        EXPERIMENTS_PATH = EXPERIMENTS_PATH / training_config.dataset_name
    SAM_PATH = Path(platform_config.sam_weights_path)

    EXPERIMENT_ID = (
        ""
        + f"{CONFIG_PATH.stem}"
        + f"_F{training_config.fold}"
        + f"_V{model_config.vit_model[-1]}"
        + f"_M{len(model_config.modalities)}"
    )

    CHECKPOINT_PATH = (
        get_checkpoint_path(path=EXPERIMENTS_PATH, id=EXPERIMENT_ID)
        if args.resume
        else None
    )
    args.resume = args.resume if CHECKPOINT_PATH else None

    MODEL_PATH = (
        CHECKPOINT_PATH / f"model_{args.resume}.pth"
        if CHECKPOINT_PATH and args.resume
        else None
    )

    # Can't resume unless the 3 conditions are true
    if not (CHECKPOINT_PATH and MODEL_PATH and MODEL_PATH.exists()):
        args.resume = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        CHECKPOINT_PATH = EXPERIMENTS_PATH / f"{timestamp}_{EXPERIMENT_ID}"

        MODEL_PATH = next(SAM_PATH.glob(f"*_{model_config.vit_model}_*.pth"), None)
        assert MODEL_PATH, (
            ""
            + "No SAM weights for the chosen ViT size were found in: "
            + f"{platform_config.sam_weights_path}"
        )

    temp_file = (
        Path(os.environ["HOME"])
        / ".tmp"
        / f"{training_config.dataset_name}_{EXPERIMENT_ID}".lower()
    )

    # Create the DistributedHelper as late as possible to avoid handling process group closing
    dh = DistributedHandler(platform_config.num_gpus)

    try:
        trainer = PanoSAMicTrainer(
            distributed_helper=dh,
            model=model,
            model_path=MODEL_PATH,
            training_set=training_set,
            validation_set=validation_set,
            config=training_config,
            num_gpus=platform_config.num_gpus,
            workers_per_gpu=platform_config.workers_per_gpu,
            checkpoint_path=CHECKPOINT_PATH,
        )
        trainer.load_checkpoint(args.resume is not None)

        if args.compute_efficiency:
            if dh.is_distributed:
                dh.print(
                    "Efficiency computation is not implemented for distributed setup."
                )
                exit(1)
            count_params(model=model)
            count_flops(model=model, n_modalities=len(model_config.modalities))
            exit(0)

        start_epoch = trainer.epoch

        CONFIG = json.dumps(
            {
                "platform": asdict(platform_config),
                "training": asdict(training_config),
                "model": asdict(model_config),
            },
            indent=4,
        )

        if trainer.dh.is_master:
            trainer.tb_writer.add_text(  # type: ignore (dh.is_master)
                "Experiment configuration", CONFIG, start_epoch
            )

        # Stores if the job running has enough time to complete the next epoch
        time_sufficient = torch.tensor([True], dtype=torch.bool)
        time_sufficient = trainer.dh.move_to_gpu(time_sufficient)
        # Stores the runtime in seconds of each epoch
        epoch_runtime: list[float] = []

        for epoch in range(start_epoch + 1, training_config.epochs + 1):
            if not time_sufficient.all():
                # If there's not enough time left in a job then exit the training
                trainer.dh.print("Stopping job due to insufficient time remaining")
                exit(0)

            start_time = perf_counter()

            tloss = trainer.train_one_epoch(epoch=epoch)
            miou, macc, vloss = trainer.eval_one_epoch(epoch=epoch)

            print_miou = f"[IoU: Ep {epoch: >2}] {miou * 100:.2f}%"
            print_macc = f"[Acc: Ep {epoch: >2}] {macc * 100:.2f}%"

            if trainer.dh.is_master:
                trainer.miou = miou
                trainer.macc = macc

                trainer.tb_writer.add_scalars(  # type: ignore (dh.is_master)
                    "Training vs. Validation Loss",
                    {"Training": tloss, "Validation": vloss},
                    epoch,
                )
                trainer.tb_writer.flush()  # type: ignore (dh.is_master)

            # Track best performance, and save the model's state
            if miou > trainer.best_miou:
                trainer.best_miou = miou
                print_miou += " new best mIoU"
                trainer.save_checkpoint(CHECKPOINT_PATH / "model_best.pth")

            trainer.save_checkpoint(CHECKPOINT_PATH / "model_last.pth")
            trainer.dh.print(print_miou + "\n" + print_macc)

            epoch_runtime.append(perf_counter() - start_time)
            mean_runtime = mean(epoch_runtime)
            stdev_runtime = stdev(epoch_runtime) if len(epoch_runtime) > 1 else 0

            if trainer.dh.is_master:
                time_left = job_time_left()
                time_sufficient[0] = time_left > (mean_runtime + stdev_runtime)

            # Synchronization of all processes to exit at the same time if needed
            trainer.dh.broadcast_value(time_sufficient)

        trainer.dh.print(
            "\n"
            + f"Training Done - best score: {trainer.best_miou * 100:.2f}%\n"
            + "Deleting last checkpoint"
        )

        # Cleanup
        checkpoint = CHECKPOINT_PATH / "model_last.pth"
        os.remove(checkpoint) if checkpoint.exists() and trainer.dh.is_master else None
        os.remove(temp_file) if temp_file.exists() and trainer.dh.is_master else None

    except Exception as e:
        log_file = CHECKPOINT_PATH / f"rank_{dh.rank}_err.log"
        error_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        if dh.is_distributed:
            # If distributed training, write errors to log files
            dh.print(
                ""
                + "Logging exceptions to disk in:\n"
                + f"{CHECKPOINT_PATH.absolute()}"
            )
            with open(log_file, "w") as f:
                f.write(error_str)
        else:
            # Else print errors to console
            dh.print(error_str)

        exit(1)  # Exit with an error

    finally:
        # Always close process group
        dh.close()


if __name__ == "__main__":
    main()
