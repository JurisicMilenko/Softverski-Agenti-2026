import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import DeadlockMatchDataset
from model import MatchPredictionModel


class FederatedClient:

    def __init__(
        self,
        name,
        dataset,
        batch_size=64,
        learning_rate=0.001
    ):
        self.name = name
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = MatchPredictionModel()

        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

    def set_model_weights(self, global_weights):
        """
        Replace the client's current model weights
        with the weights of the global model.
        """

        self.model.load_state_dict(
            copy.deepcopy(global_weights)
        )

    def get_model_weights(self):
        """
        Return a copy of the client's model weights.
        """

        return copy.deepcopy(
            self.model.state_dict()
        )

    def train(self, epochs=1):
        """
        Train the local model using only this client's data.
        """

        self.model.train()

        criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        print()
        print(f"Training client: {self.name}")
        print(f"Samples: {len(self.dataset)}")

        for epoch in range(epochs):

            total_loss = 0.0

            for features, labels in self.data_loader:

                optimizer.zero_grad()

                predictions = self.model(features)

                loss = criterion(
                    predictions,
                    labels
                )

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            average_loss = (
                total_loss /
                len(self.data_loader)
            )

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss: {average_loss:.4f}"
            )

        return {
            "weights": self.get_model_weights(),
            "sample_count": len(self.dataset),
            "loss": average_loss
        }


if __name__ == "__main__":

    # Load all 20,000 matches
    full_dataset = DeadlockMatchDataset()

    # For this first test, use only the first 5,000 matches.
    test_indices = range(5000)

    client_dataset = Subset(
        full_dataset,
        test_indices
    )

    client = FederatedClient(
        name="TestClient",
        dataset=client_dataset
    )

    result = client.train(
        epochs=3
    )

    print()
    print("=" * 80)
    print("TRAINING FINISHED")
    print("=" * 80)

    print(
        f"Client: {client.name}"
    )

    print(
        f"Samples used: "
        f"{result['sample_count']}"
    )

    print(
        f"Final loss: "
        f"{result['loss']:.4f}"
    )