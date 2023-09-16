import torch
import os
from data_loader import DataLoader


class Trainer:
    """A class to train models. It is responsible for taking a model, dataset, optimizers, loss functions, and other and training the model."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        data_loader: DataLoader,
        wandb=None,
    ):
        """Initializes the trainer.

        Parameters
        ----------
        model: torch.nn.Module
            The model to train.
        loss_fn: torch.nn.Module
            The loss function to use.
        optimizer: torch.optim.Optimizer
            The optimizer to use.
        scheduler: torch.optim.lr_scheduler
            The learning rate scheduler to use.
        train_loader: torch.utils.data.DataLoader
            The training data loader.
        test_loader: torch.utils.data.DataLoader
            The test data loader.
        data_loader: DataLoader
            The data loader object. To be used for decoding ratings.
        epochs: int
            The number of epochs to train for.
        device: torch.device
            The device to use for training.
        wandb: wandb
            The wandb object to use for logging.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.data_loader = data_loader
        self.wandb = wandb
        self.scheduler = scheduler

    def train_step(self):
        """A single training step."""
        self.model.train()
        train_loss = 0.0
        for user, item, rating in self.train_loader:
            # x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self.model(user, item)
            loss = self.loss_fn(y_hat, rating)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            if self.wandb:
                self.wandb.log({"train_loss": loss.item()})
        train_loss /= len(self.train_loader)
        if self.scheduler is not None:
            self.scheduler.step()
        if self.wandb:
            self.wandb.log({"train_loss": train_loss})
        return train_loss

    def test_step(self):
        """A single test step."""
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for user, item, rating in self.test_loader:
                # x, y = x.to(self.device), y.to(self.device)
                # y_hat = self.model(x)
                # loss = self.loss_fn(y_hat, y)
                y_hat = self.model(user, item)
                loss = self.loss_fn(y_hat, rating)
                test_loss += loss.item()
                if self.wandb:
                    self.wandb.log({"test_loss": loss.item()})
                    sample_rating_pred = self.data_loader.decode_rating(y_hat[0].item())
                    sample_rating_true = self.data_loader.decode_rating(
                        rating[0].item()
                    )
                    self.wandb.log(
                        {
                            "sample_rating_pred": sample_rating_pred,
                            "sample_rating_true": sample_rating_true,
                        }
                    )
        test_loss /= len(self.test_loader)

        return test_loss

    def train(self, epochs, verbose=1):
        """Trains the model."""
        for epoch in range(epochs):
            train_loss = self.train_step()
            test_loss = self.test_step()
            if self.wandb:
                self.wandb.log(
                    {"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss}
                )
            if verbose > 1:
                print(
                    f"Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}"
                )
            elif verbose == 1:
                if epoch % 10 == 0:
                    print(
                        f"Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}"
                    )
            elif verbose == 0:
                pass

    def save_model(self, model_dir):
        """Saves the model."""
        path = os.path.join(model_dir, "model.pth")
        torch.save(self.model.state_dict(), path)
        if self.wandb:
            self.wandb.save(path)
