import torch
import os


class Trainer:
    """A class to train models. It is responsible for taking a model, dataset, optimizers, loss functions, and other and training the model."""

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
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


# from trainer import Trainer


# def main(args):
# Load the data
# data_loader = DataLoader(args.data_dir, args.batch_size)
# train_loader, val_loader, test_loader = data_loader.get_loaders()

# # Initialize the model
# model = MLPModel(
#     m=data_loader.m,
#     n=data_loader.n,
#     embedding_dim=args.embedding_dim,
#     cf_layer_neurons=args.cf_layer_neurons,
#     use_sigmoid=args.use_sigmoid,
#     init_weights=args.init_weights,
# )

# # Initialize the loss function
# if args.use_sigmoid:
#     loss_fn = BCELoss(model)
# else:
#     loss_fn = MSE_L1L2Loss(
#         model, l1_weight=args.l1_weight, l2_weight=args.l2_weight
#     )

# # Initialize the optimizer
# optimizer = torch.optim.Adam(
#     model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
# )

# # Initialize the trainer
# trainer = Trainer(
#     model=model,
#     loss_fn=loss_fn,
#     optimizer=optimizer,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     test_loader=test_loader,
#     epochs=args.epochs,
#     device=args.device,
# )

# # Train the model
# trainer.train()

# # Save the model
# trainer.save_model(args.model_dir)
