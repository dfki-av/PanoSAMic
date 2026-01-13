"""
Author: Mahdi Chamseddine
"""

import os
import traceback
from pathlib import Path

import torch

from panosamic.datasets import build_dataset
from panosamic.evaluation.evaluator import PanoSAMicEvaluator
from panosamic.evaluation.utils.config import generate_configs
from panosamic.evaluation.utils.distributed_handler import DistributedHandler
from panosamic.evaluation.utils.parser import create_parser
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

    dataset = build_dataset(
        dataset_path=platform_config.dataset_path,
        config=training_config,
        n_modalities=len(model_config.modalities),
        test_mode=True,
    )

    model: torch.nn.Module = panosamic_builder(
        config=model_config,
        num_classes=dataset.NUM_CLASSES,
        freeze_encoder=True,
    )

    CONFIG_PATH = Path(platform_config.config_path)
    EXPERIMENTS_PATH = Path(platform_config.experiments_path)
    if EXPERIMENTS_PATH.name != training_config.dataset_name:
        EXPERIMENTS_PATH = EXPERIMENTS_PATH / training_config.dataset_name

    EXPERIMENT_ID = (
        ""
        + f"{CONFIG_PATH.stem}"
        + f"_F{training_config.fold}"
        + f"_V{model_config.vit_model[-1]}"
        + f"_M{len(model_config.modalities)}"
    )

    CHECKPOINT_PATH = get_checkpoint_path(path=EXPERIMENTS_PATH, id=EXPERIMENT_ID)
    if not CHECKPOINT_PATH:
        return

    MODEL_PATH = CHECKPOINT_PATH / "model_best.pth"
    if not (MODEL_PATH and MODEL_PATH.exists()):
        return

    # Create the DistributedHelper as late as possible to avoid handling process group closing for earlier exceptions
    dh = DistributedHandler(platform_config.num_gpus)

    try:
        evaluator = PanoSAMicEvaluator(
            distributed_helper=dh,
            model=model,
            model_path=MODEL_PATH,
            dataset=dataset,
            batch_size=training_config.batch_size,
            num_gpus=platform_config.num_gpus,
            workers_per_gpu=platform_config.workers_per_gpu,
        )
        evaluator.load_checkpoint()

        epoch = evaluator.epoch
        miou, macc, _ = evaluator.eval_one_epoch(epoch=epoch)

        print_miou = f"[IoU: Ep {epoch: >2}] {miou * 100:.2f}%"
        print_macc = f"[Acc: Ep {epoch: >2}] {macc * 100:.2f}%"

        evaluator.dh.print(print_miou + "\n" + print_macc)

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

    finally:
        # Always close process group
        dh.close()


if __name__ == "__main__":
    main()
