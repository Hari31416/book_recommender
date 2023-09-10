import torch
import torch.nn as nn
import numpy as np


class MLPModel(nn.Module):
    """Model class for the BookNet model."""

    def __init__(
        self,
        m,
        n,
        embedding_dim,
        cf_layer_neurons,
        use_sigmoid=False,
        init_weights=True,
    ):
        """Initializes the BookNet model.

        Parameters
        ----------
        m: int
            Number of users.
        n: int
            Number of books.
        embedding_dim: int
            Hidden dimension for the hidden layer.
        cf_layer_neurons: list
            List of integers specifying the number of neurons in each layer of the collaborative filtering part of the model.
        use_sigmoid: bool
            Whether to use sigmoid activation function for the last layer of the model.
        init_weights: bool
            Whether to initialize the weights of the model.
        """
        super(MLPModel, self).__init__()
        self.m = m
        self.n = n
        self.embedding_dim = embedding_dim
        self.cf_layer_neurons = cf_layer_neurons
        self.use_sigmoid = use_sigmoid
        self.init_weights = init_weights

        self.user_embedding, self.book_embedding = self.create_embedding_layer()
        self.cf_layer = self.create_CF_layer()
        self.affine_output = torch.nn.Linear(
            in_features=self.cf_layer_neurons[-1], out_features=1
        )
        if self.use_sigmoid:
            self.logistic = torch.nn.Sigmoid()
        else:
            self.logistic = torch.nn.ReLU()

        if self.init_weights:
            self.init_weights()

    def create_embedding_layer(self):
        """Creates the embedding layer"""
        user_in_shape = self.m
        book_in_shape = self.n
        out_shape = self.embedding_dim
        user_embedding = torch.nn.Embedding(
            num_embeddings=user_in_shape, embedding_dim=out_shape
        )
        book_embedding = torch.nn.Embedding(
            num_embeddings=book_in_shape, embedding_dim=out_shape
        )
        return user_embedding, book_embedding

    def _init_embedding_weights(self, embedding_layer):
        """Initializes the embedding layer weights with a uniform distribution."""
        embedding_layer.weight.data.uniform_(0, 1)

    def init_weights(self):
        """Initializes the weights of the model."""
        self._init_embedding_weights(self.user_embedding)
        self._init_embedding_weights(self.book_embedding)
        for layer in self.cf_layer:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def create_CF_layer(self):
        """Creates the collaborative filtering layers. Uses the number of neurons specified in `cf_layer_neurons`."""
        num_layers = len(self.cf_layer_neurons)
        activation = torch.nn.ReLU()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(
                    torch.nn.Linear(self.embedding_dim * 2, self.cf_layer_neurons[i])
                )
            else:
                layers.append(
                    torch.nn.Linear(
                        self.cf_layer_neurons[i - 1], self.cf_layer_neurons[i]
                    )
                )
            layers.append(activation)
        return torch.nn.Sequential(*layers)

    def forward(self, user_id, book_id):
        """Forward pass of the model.

        Parameters
        ----------
        user_id: torch.Tensor
            Tensor containing the user id.
        book_id: torch.Tensor
            Tensor containing the book id.
        """
        user_index = user_id.long()  # convert to long to avoid errors
        user_embedded = self.user_embedding(user_index)
        book_index = book_id.long()
        book_embedded = self.book_embedding(book_index)

        # Concatenate the user and book embeddings to form one vector.
        x = torch.cat([user_embedded, book_embedded], dim=-1)
        x = self.cf_layer(x)
        x = torch.squeeze(x)
        return x
