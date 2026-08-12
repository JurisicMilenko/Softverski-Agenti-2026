import torch
from torch import nn


INPUT_SIZE = 514


class MatchPredictionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


if __name__ == "__main__":
    model = MatchPredictionModel()

    print(model)

    # Create 4 fake matches just to test the model
    test_input = torch.randn(4, INPUT_SIZE)

    output = model(test_input)

    print()
    print("Input shape:")
    print(test_input.shape)

    print()
    print("Output shape:")
    print(output.shape)

    print()
    print("Raw model output:")
    print(output)

    probabilities = torch.sigmoid(output)

    print()
    print("Win probabilities:")
    print(probabilities)