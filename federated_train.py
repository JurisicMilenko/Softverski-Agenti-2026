from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import random_split, Subset

from aggregator import FederatedAggregator
from client import FederatedClient
from dataset import DeadlockMatchDataset


def create_clients(dataset, client_names: list[str], seed: int) -> list[FederatedClient]:
    if len(client_names) < 2:
        raise ValueError("At least two clients are required for federated learning")

    generator = torch.Generator().manual_seed(seed)
    lengths = _balanced_lengths(len(dataset), len(client_names))
    subsets = random_split(dataset, lengths, generator=generator)

    clients = []
    for name, subset in zip(client_names, subsets):
        clients.append(
            FederatedClient(
                name=name,
                dataset=subset,
            )
        )

    return clients


def _balanced_lengths(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def run_federated_training(
    rounds: int,
    local_epochs: int,
    client_names: list[str],
    seed: int,
    output_path: Path,
) -> None:
    dataset = DeadlockMatchDataset()
    clients = create_clients(dataset, client_names, seed=seed)
    aggregator = FederatedAggregator()

    print()
    print("=" * 80)
    print("FEDERATED LEARNING")
    print("=" * 80)
    print(f"Clients: {', '.join(client_names)}")
    print(f"Rounds: {rounds}")
    print(f"Local epochs per round: {local_epochs}")

    for round_number in range(1, rounds + 1):
        print()
        print("-" * 80)
        print(f"ROUND {round_number}/{rounds}")
        print("-" * 80)

        global_weights = aggregator.get_global_weights()

        for client in clients:
            client.set_model_weights(global_weights)
            result = client.train(epochs=local_epochs)

            aggregator.submit_update(
                client_name=client.name,
                weights=result["weights"],
                sample_count=result["sample_count"],
                loss=result["loss"],
            )

        new_global_weights = aggregator.aggregate()
        print(f"Global model updated from {len(clients)} clients.")
        for client in clients:
            client.set_model_weights(new_global_weights)

    aggregator.save_global_model(output_path)

    print()
    print("=" * 80)
    print("FEDERATED TRAINING FINISHED")
    print("=" * 80)
    print(f"Global model saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run federated training for the Deadlock model")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/global_model.pt"),
    )
    parser.add_argument(
        "--clients",
        nargs="+",
        default=["EU", "NA", "ASIA"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.rounds <= 0:
        raise SystemExit("--rounds must be greater than 0")
    if args.local_epochs <= 0:
        raise SystemExit("--local-epochs must be greater than 0")

    run_federated_training(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        client_names=args.clients,
        seed=args.seed,
        output_path=args.output,
    )
