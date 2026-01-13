"""
Author: Mahdi Chamseddine
"""

from argparse import ArgumentParser


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # Platform configuration
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the processed panoramic dataset",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config file",
    )

    parser.add_argument(
        "--experiments_path",
        type=str,
        required=True,
        help="Path to the experiments path to store checkpoints",
    )

    parser.add_argument(
        "--sam_weights_path",
        default=None,
        type=str,
        required=False,
        help="Path to the model weights",
    )

    parser.add_argument(
        "--num_gpus",
        default=1,
        type=int,
        required=False,
        help="Number of GPUs",
    )

    parser.add_argument(
        "--workers_per_gpu",
        default=2,
        type=int,
        required=False,
        help="Number of workers per GPU",
    )

    # Training configuration
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["stanford2d3ds", "structured3d", "matterport3d"],
        required=True,
        help="Panoramic dataset name",
    )

    parser.add_argument(
        "--fold",
        default=1,
        type=int,
        required=False,
        help="The dataset fold number to be used",
    )

    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        required=False,
        help="Batch size",
    )

    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        required=False,
        help="Number of training epochs",
    )

    # Model configuration
    parser.add_argument(
        "--vit_model",
        default="vit_h",
        type=str,
        choices=["vit_h", "vit_l", "vit_b"],
        required=False,
        help="ViT encoder model to be used",
    )

    parser.add_argument(
        "--modalities",
        default="image,depth,normals",
        type=str,
        required=False,
        help="The modalities to use for training ('image', 'depth', 'normals')",
    )

    # Continue training
    parser.add_argument(
        "--resume",
        default=None,
        type=lambda arg: None if arg == "None" else arg,
        choices=[None, "last", "best"],
        required=False,
        help="Continue training from last or best epoch",
    )

    # Efficiency computation
    parser.add_argument(
        "--compute_efficiency",
        default=False,
        type=bool,
        required=False,
        help="Compute model parameters and efficiency. Exists after printing.",
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    main()
