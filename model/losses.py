import torch
import torch.nn as nn


class MSE_L1L2Loss(nn.Module):
    def __init__(self, model, l1_weight, l2_weight):
        """Initializes the loss function.

        Parameters
        ----------
        model: torch.nn.Module
            The model to use for the loss function.
        l1_weight: float
            Weight for the L1 regularization term.
        l2_weight: float
            Weight for the L2 regularization term.
        """
        super().__init__()
        self.model = model
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, y_hat, y):
        """The forward pass of the loss function."""
        mse_loss = torch.nn.functional.mse_loss(y_hat, y)
        l2_regularization = torch.tensor(0.0)
        l1_regularization = torch.tensor(0.0)
        for param in self.model.parameters():
            l2_regularization += torch.norm(param, 2)
            l1_regularization += torch.norm(param, 1)
        l1_regularization *= self.l1_weight
        l2_regularization *= self.l2_weight
        loss = mse_loss + l1_regularization + l2_regularization
        return loss


class BCELoss(nn.Module):
    def __init__(self, model):
        """Initializes the loss function.

        Parameters
        ----------
        model: torch.nn.Module
            The model to use for the loss function.
        """
        super().__init__()
        self.model = model

    def forward(self, y_hat, y):
        """The forward pass of the loss function."""
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        return loss
