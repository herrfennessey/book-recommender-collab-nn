import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryF1Score


class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)

        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            user_ratings (pd.DataFrame): Dataframe containing the movie user_ratings for training
            all_book_ids (list): List containing all movieIds (train + test)
    """

    def __init__(self, training_dataset, num_users, num_books, learning_rate=0.01):
        super().__init__()
        self.user_id_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=16)
        self.book_id_embedding = nn.Embedding(num_embeddings=num_books, embedding_dim=14)

        self.fc1 = nn.Linear(in_features=74, out_features=140)
        self.fc2 = nn.Linear(in_features=140, out_features=70)
        self.output = nn.Linear(in_features=70, out_features=1)
        self.train_dataset = training_dataset
        self.learning_rate = learning_rate
        self.f1 = BinaryF1Score()

    def forward(self, user_input, item_input, item_details, item_meta):
        # Pass through embedding layers
        user_embedded = self.user_id_embedding(user_input)
        item_embedded = self.book_id_embedding(item_input)
        book_details = item_details
        book_meta = item_meta

        # Concat the embeddings with the other tensors
        vector = torch.cat([user_embedded, item_embedded, book_details, book_meta], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred

    # Truncated t
