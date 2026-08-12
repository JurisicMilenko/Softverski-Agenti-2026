from pathlib import Path

import duckdb
import torch
from torch.utils.data import Dataset


DATA_FILE = Path("data/matches.parquet")

# hero_id in the Deadlock dump is UTINYINT,
# so every hero ID fits into the range 0-255.
NUM_HERO_IDS = 256


class DeadlockMatchDataset(Dataset):

    def __init__(self, file_path=DATA_FILE):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        print(f"Loading dataset from {self.file_path}...")

        with duckdb.connect() as con:
            rows = con.execute("""
                SELECT
                    badge_team0,
                    badge_team1,
                    team0_heroes,
                    team1_heroes,
                    team0_won
                FROM read_parquet(?)
            """, [str(self.file_path)]).fetchall()

        features = []
        labels = []

        for (
            badge_team0,
            badge_team1,
            team0_heroes,
            team1_heroes,
            team0_won
        ) in rows:

            feature_vector = self._create_feature_vector(
                badge_team0,
                badge_team1,
                team0_heroes,
                team1_heroes
            )

            features.append(feature_vector)
            labels.append(float(team0_won))

        self.features = torch.stack(features)

        self.labels = torch.tensor(
            labels,
            dtype=torch.float32
        )

        print(f"Loaded {len(self.features)} matches.")

    def _create_feature_vector(
        self,
        badge_team0,
        badge_team1,
        team0_heroes,
        team1_heroes
    ):
        """
        Creates one input vector for one match.

        Structure:

        [
            badge_team0,
            badge_team1,

            256 values representing heroes in Team0,

            256 values representing heroes in Team1
        ]
        """

        feature_vector = torch.zeros(
            2 + NUM_HERO_IDS * 2,
            dtype=torch.float32
        )

        # Normalize badge values so they are not much larger
        # than the hero one-hot values.
        feature_vector[0] = badge_team0 / 100.0
        feature_vector[1] = badge_team1 / 100.0

        # Team0 hero one-hot encoding
        team0_offset = 2

        for hero_id in team0_heroes:
            feature_vector[
                team0_offset + int(hero_id)
            ] = 1.0

        # Team1 hero one-hot encoding
        team1_offset = 2 + NUM_HERO_IDS

        for hero_id in team1_heroes:
            feature_vector[
                team1_offset + int(hero_id)
            ] = 1.0

        return feature_vector

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.features[index],
            self.labels[index]
        )


if __name__ == "__main__":
    dataset = DeadlockMatchDataset()

    print()
    print("=" * 80)
    print("DATASET INFO")
    print("=" * 80)

    print(f"Number of matches: {len(dataset)}")
    print(
        f"Number of input features: "
        f"{dataset.features.shape[1]}"
    )

    print(
        f"Feature tensor shape: "
        f"{dataset.features.shape}"
    )

    print(
        f"Label tensor shape: "
        f"{dataset.labels.shape}"
    )

    print()
    print("First match:")

    features, label = dataset[0]

    print(f"Team0 badge: {features[0].item() * 100:.0f}")
    print(f"Team1 badge: {features[1].item() * 100:.0f}")

    team0_heroes = []

    for hero_id in range(NUM_HERO_IDS):
        if features[2 + hero_id] == 1:
            team0_heroes.append(hero_id)

    team1_heroes = []

    team1_offset = 2 + NUM_HERO_IDS

    for hero_id in range(NUM_HERO_IDS):
        if features[team1_offset + hero_id] == 1:
            team1_heroes.append(hero_id)

    print(f"Team0 heroes: {team0_heroes}")
    print(f"Team1 heroes: {team1_heroes}")
    print(f"Team0 won: {int(label.item())}")