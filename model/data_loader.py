import os
import torch
import numpy as np
import pandas as pd

DATA_DIR = os.path.join("..", "data", "final_dataset")


class BookDataset(torch.utils.data.Dataset):
    """Dataset class for the model.""" ""

    def __init__(
        self,
        users,
        books,
        interactions,
        negative_samples_ratio=0.5,
        features=True,
        normalize=True,
        normalize_by="max",
    ):
        """Initializes the BookDataset.

        Parameters
        ----------
        users: pandas.DataFrame
            DataFrame containing user features.
        books: pandas.DataFrame
            DataFrame containing book features.
        interactions: pandas.DataFrame
            DataFrame containing user-book interactions.
        negative_samples_ratio: float
            Ratio of negative samples to positive samples.
        features: bool
            Whether to use features or not.
        normalize: bool
            Whether to normalize the ratings or not.
        normalize_by: str
            If std, then the ratings are normalized by subtracting the mean and dividing by the standard deviation.
            If max, then the ratings are normalized by dividing by the maximum rating.
        """
        self.negative_samples_ratio = negative_samples_ratio
        self.features = features
        self.normalize = normalize
        self.normalize_by = normalize_by
        # self.negative_samples_ratio = config["negative_samples_ratio"]
        # self.features = config["features"]
        # self.normalize = config["normalize"]
        # try:
        #     self.normalize_by = config["normalize_by"]
        # except KeyError:
        #     self.normalize_by = "max"
        self.users = users
        self.books = books
        self.interactions = interactions.copy()
        self.m = len(self.users)
        self.n = len(self.books)
        self.m_f = self.users.shape[1]
        self.n_f = self.books.shape[1]
        if self.normalize:
            self.normalize_ratings()
        if self.negative_samples_ratio < 0 or self.negative_samples_ratio > 1:
            raise ValueError("negative_samples_ratio must be between 0 and 1.")
        self.negative_samples_ratio = self.negative_samples_ratio

    def normalize_ratings(self):
        """Normalizes the ratings by dividing by 10."""
        if self.normalize_by == "max":
            self.interactions["provided_rating"] = (
                self.interactions["provided_rating"] / 10
            ).round(2)
        elif self.normalize_by == "std":
            mean = self.interactions["provided_rating"].mean()
            std = self.interactions["provided_rating"].std()
            self.mean = mean
            self.std = std
            self.interactions["provided_rating"] = (
                self.interactions["provided_rating"]
                - self.interactions["provided_rating"].mean()
            ) / self.interactions["provided_rating"].std()
        else:
            raise ValueError(
                f"normalize_by must be either max or std. Got {self.normalize_by}"
            )

    def decode_rating(self, rating):
        """Decodes a rating."""
        if not self.normalize:
            return rating
        if self.normalize_by == "max":
            return rating * 10
        elif self.normalize_by == "std":
            return rating * self.std + self.mean
        else:
            raise ValueError(
                f"normalize_by must be either max or std. Got {self.normalize_by}"
            )

    def get_user_features(self, user_id):
        # Assumes that the user_id column is the index in the users DataFrame.
        user = self.users.iloc[user_id].values
        return user.reshape((self.m_f,))

    def get_book_features(self, book_id):
        # Assumes that the book_id column is the index in the books DataFrame.
        book = self.books.iloc[book_id].values
        return book.reshape((self.n_f,))

    def __len__(self):
        # tried this but this will give error as the interaction dataframe will become out of index
        # num_times = 1+self.negative_samples_ratio
        num_times = 1
        return int(len(self.interactions) * num_times)

    def get_positive_sample(self, idx):
        """Gets a positive sample from the interactions dataframe."""
        row = self.interactions.iloc[idx]
        user_id = row["user_id"]
        book_id = row["book_id"]
        rating = row["provided_rating"]
        user_id = np.array([user_id])
        book_id = np.array([book_id])
        if self.features:
            user_features = self.get_user_features(user_id)
            book_features = self.get_book_features(book_id)
            user_input = np.concatenate([user_id, user_features], axis=-1)
            book_input = np.concatenate([book_id, book_features], axis=-1)
        else:
            user_input = user_id
            book_input = book_id

        # make sure the length of the input is correct
        if self.features:
            assert len(user_input) == 1 + self.m_f
            assert len(book_input) == 1 + self.n_f
        else:
            assert len(user_input) == 1
            assert len(book_input) == 1
        return user_input, book_input, rating

    def get_negative_sample(self, idx):
        """Gets a negative sample from the interactions dataframe.""" ""
        row = self.interactions.iloc[idx]
        user_id = row["user_id"]
        negative_book_id = np.random.choice(self.books.index.values)
        while (
            negative_book_id
            in self.interactions[self.interactions["user_id"] == user_id][
                "book_id"
            ].values
        ):
            negative_book_id = np.random.choice(self.books.index.values)

        user_id = np.array([user_id])
        book_id = np.array([negative_book_id])
        if self.features:
            user_features = self.get_user_features(user_id)
            book_features = self.get_book_features(book_id)
            user_input = np.concatenate([user_id, user_features], axis=-1)
            book_input = np.concatenate([book_id, book_features], axis=-1)
        else:
            user_input = user_id
            book_input = book_id

        # make sure the length of the input is correct
        if self.features:
            assert len(user_input) == 1 + self.m_f
            assert len(book_input) == 1 + self.n_f
        else:
            assert len(user_input) == 1
            assert len(book_input) == 1

        rating = 0  # negative sample
        return user_input, book_input, rating

    def get_one_sample(self, idx):
        """Gets one sample from the dataset. Uses negative sampling with probability `negative_samples_ratio`."""
        if np.random.random() < self.negative_samples_ratio:
            return self.get_negative_sample(idx)
        else:
            return self.get_positive_sample(idx)

    def __getitem__(self, idx):
        """Gets one sample from the dataset."""
        # A workaround to make interaction dataframe circular when using negative sampling
        # with `num_times` > 1. Leaving it as this should not be necessary here.
        # if idx >= len(self.interactions):
        #     idx = idx%len(self.interactions)
        user_input, book_input, rating = self.get_one_sample(idx)
        if self.features:
            dtype = torch.float32
        else:
            dtype = torch.long
        user_input = torch.tensor(user_input, dtype=dtype)
        book_input = torch.tensor(book_input, dtype=dtype)
        targets = torch.tensor(rating, dtype=torch.float32)
        return user_input, book_input, targets


class DataLoader:
    def __init__(self, data_dir, config):
        """Initializes the DataLoader.

        Parameters
        ----------
        data_dir: str
            Path to the directory containing the data.
        config: dict
            Config object containing configuration parameters.
            Expected parameters:
                - negative_samples_ratio: float
                - features: bool
                - normalize: bool
                - normalize_by: str
                - split_ratio: float
                - batch_size: int

        """
        self.data_dir = data_dir
        self.config = config
        self.interactions = pd.read_parquet(
            os.path.join(self.data_dir, "ratings_final.parquet")
        )
        self.users = pd.read_parquet(os.path.join(self.data_dir, "users_final.parquet"))
        self.books = pd.read_parquet(os.path.join(self.data_dir, "books_final.parquet"))
        self.m = len(self.users)
        self.n = len(self.books)

    def split_dataframe(self, df, holdout_fraction=0.1):
        """Splits a DataFrame into training and test sets.
        Parameters:
        -----------
        df: pandas.DataFrame
            DataFrame to split.
        holdout_fraction: float
            Fraction of the data to use for the test set.

        Returns:
        --------
        train: pandas.DataFrame
            Training set.
        test: pandas.DataFrame
            Test set.
        """
        test = df.sample(frac=holdout_fraction, replace=False)
        train = df[~df.index.isin(test.index)]
        return train, test

    def get_data_loaders(self, split_ratio=0.1):
        """Returns training and test data loaders."""
        train, test = self.split_dataframe(self.interactions, split_ratio)
        train_dataset = BookDataset(
            self.users,
            self.books,
            train,
            negative_samples_ratio=self.config["negative_samples_ratio"],
            features=self.config["features"],
            normalize=self.config["normalize"],
            normalize_by=self.config["normalize_by"],
        )
        test_dataset = BookDataset(
            self.users,
            self.books,
            test,
            negative_samples_ratio=self.config["negative_samples_ratio"],
            features=self.config["features"],
            normalize=self.config["normalize"],
            normalize_by=self.config["normalize_by"],
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["batch_size"], shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config["batch_size"], shuffle=True
        )
        return train_loader, test_loader
