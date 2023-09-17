import torch
import torch.nn as nn
import argparse
import os
import wandb

from data_loader import DataLoader
from models import MLPModel
from losses import MSE_L1L2Loss, BCELoss
from trainer import Trainer

# DATA_DIR = os.path.join(os.curdir, "..", "data", "final_dataset")
DATA_DIR = r"D:\harik\Desktop\Book_Recommendation\data\final_dataset"


def create_model_config(model_name, m, n, **kwargs):
    """Creates the model configuration.

    Parameters
    ----------
    model_name: str
        Name of the model.
    m: int
        Number of users.
    n: int
        Number of books.
    kwargs: dict
        Dictionary containing the model configuration.
        Expected keys:
            embedding_dim: int
                Hidden dimension for the hidden layer.
            cf_layer_neurons: list
                List of integers specifying the number of neurons in each layer of the collaborative filtering part of the model.
            use_sigmoid: bool
                Whether to use sigmoid activation function for the last layer of the model.
            init_weights: bool
                Whether to initialize the weights of the model.

    Returns
    -------
    model_config: dict
        Dictionary containing the model configuration.
    """
    model_config = {
        "name": model_name,
        "m": m,
        "n": n,
        "embedding_dim": 32,
        "cf_layer_neurons": [32, 16],
        "use_sigmoid": True,
        "init_weights": False,
    }
    model_config.update(kwargs)
    return model_config


def create_data_config(data_dir, batch_size, **kwargs):
    """Creates the data configuration.

    Parameters
    ----------
    data_dir: str
        Path to the data directory.
    batch_size: int
        Batch size to use for the data loaders.
    kwargs: dict
        Dictionary containing the data configuration.
        Expected keys:
            negative_samples_ratio: float
                Ratio of negative samples to use for training.
            split_ratio: float
                Ratio to use for splitting the data into train, validation and test sets.
            features: bool
                Whether to use features or not.
            normalize: bool
                Whether to normalize the ratings or not.
            normalize_by: str
                If std, then the ratings are normalized by subtracting the mean and dividing by the standard deviation.
                If max, then the ratings are normalized by dividing by the maximum rating.
    Returns
    -------
    data_config: dict
        Dictionary containing the data configuration.
    """
    data_config = {
        "data_dir": data_dir,
        "batch_size": batch_size,
        "negative_samples_ratio": 0.4,
        "split_ratio": 0.1,
        "features": False,
        "normalize": True,
        "normalize_by": "max",
    }
    data_config.update(kwargs)
    return data_config


def create_loss_config(loss_function, **kwargs):
    """Creates the loss configuration.

    Parameters
    ----------
    loss_function: str
        Name of the loss function.
        Can be one of: mse_l1l2, bce
    kwargs: dict
        Dictionary containing the loss configuration.
        Expected keys:
            l1_weight: float
                Weight for the L1 regularization term.
            l2_weight: float
                Weight for the L2 regularization term.
    Returns
    -------
    loss_config: dict
        Dictionary containing the loss configuration.
    """
    loss_config = {"loss_function": loss_function, "l1_weight": 0.0, "l2_weight": 0.0}
    loss_config.update(kwargs)
    return loss_config


def main(args):
    if args["wandb"]:
        # Initialize wandb
        wandb.init(project=args["project_name"], name=args["run_name"])
        wandb.run.name = args["run_name"]
        wandb.log_interval = args["log_interval"]

    # data config
    data_config = create_data_config(
        data_dir=args["data_dir"],
        batch_size=args["batch_size"],
        negative_samples_ratio=args["negative_samples_ratio"],
        split_ratio=args["split_ratio"],
        features=args["features"],
        normalize=args["normalize"],
        normalize_by=args["normalize_by"],
    )
    data_loader = DataLoader(
        data_dir=data_config["data_dir"],
        config=data_config,
    )
    m = data_loader.m
    n = data_loader.n
    train_loader, test_loader = data_loader.get_data_loaders()

    # Create the model configuration
    model_config = create_model_config(
        model_name=args["model_name"],
        m=m,
        n=n,
        embedding_dim=args["embedding_dim"],
        cf_layer_neurons=args["cf_layer_neurons"],
        use_sigmoid=args["use_sigmoid"],
        init_weights=args["init_weights"],
    )
    # Initialize the model
    model = MLPModel(
        m=model_config["m"],
        n=model_config["n"],
        embedding_dim=model_config["embedding_dim"],
        cf_layer_neurons=model_config["cf_layer_neurons"],
        use_sigmoid=model_config["use_sigmoid"],
        init_weights_=model_config["init_weights"],
    )

    # Create the loss configuration
    loss_config = create_loss_config(
        loss_function=args["loss_function"],
        l1_weight=args["l1_weight"],
        l2_weight=args["l2_weight"],
    )

    # Initialize the loss function
    if loss_config["loss_function"] == "mse_l1l2":
        loss_fn = MSE_L1L2Loss(
            model,
            l1_weight=loss_config["l1_weight"],
            l2_weight=loss_config["l2_weight"],
        )
    elif loss_config["loss_function"] == "bce":
        loss_fn = BCELoss(model)

    # Initialize the optimizer
    if args["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    elif args["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"])

    # Initialize the scheduler
    if args["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, verbose=True, min_lr=1e-7
        )

    all_config = {
        "data_config": data_config,
        "model_config": model_config,
    }
    # flatten the config for wandb
    all_config = {k: v for d in all_config.values() for k, v in d.items()}
    all_config.update({"optimizer": args["optimizer"], "scheduler": args["scheduler"]})

    if args["wandb"]:
        wandb.config.update(all_config)

    if args["wandb"]:
        # Log the model
        wandb.watch(model)
        wandb_ = wandb

    if not args["wandb"]:
        wandb_ = None
    # Initialize the trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        test_loader=test_loader,
        data_loader=data_loader,
        wandb=wandb_,
    )

    # Train the model
    trainer.train(
        epochs=args["epochs"],
        verbose=args["verbose"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model config
    parser.add_argument("--model_name", type=str, default="mlp")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--cf_layer_neurons", nargs="+", type=int, default=[32, 16])
    parser.add_argument("--use_sigmoid", type=bool, default=True)
    parser.add_argument("--init_weights", type=bool, default=False)

    # Data config
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--negative_samples_ratio", type=float, default=0.4)
    parser.add_argument("--split_ratio", type=float, default=0.1)
    parser.add_argument("--features", type=bool, default=False)
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--normalize_by", type=str, default="max")

    # Loss config
    parser.add_argument("--loss_function", type=str, default="mse_l1l2")
    parser.add_argument("--l1_weight", type=float, default=0.0)
    parser.add_argument("--l2_weight", type=float, default=0.0)

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.001)

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="step")

    # Trainer config
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--verbose", type=int, default=2)

    # Wandb config
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--project_name", type=str, default="Book_Recommendation")
    parser.add_argument("--run_name", type=str, default="model_test_1")
    parser.add_argument("--log_interval", type=int, default=20)

    args = parser.parse_args()
    args = vars(args)
    main(args)
