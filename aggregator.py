from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn

from model import MatchPredictionModel


ModelState = Mapping[str, Tensor]


@dataclass(frozen=True)
class ClientUpdate:
    client_name: str
    weights: Dict[str, Tensor]
    sample_count: int
    loss: Optional[float] = None


class FederatedAggregator:
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model if model is not None else MatchPredictionModel()
        self._pending_updates: Dict[str, ClientUpdate] = {}

    @property
    def pending_update_count(self) -> int:
        return len(self._pending_updates)

    def get_global_weights(self) -> Dict[str, Tensor]:
        return copy.deepcopy(self.model.state_dict())

    def submit_update(
        self,
        client_name: str,
        weights: ModelState,
        sample_count: int,
        loss: Optional[float] = None,
    ) -> None:

        if not client_name or not client_name.strip():
            raise ValueError("client_name must not be empty")

        if sample_count <= 0:
            raise ValueError("sample_count must be greater than zero")

        normalized_weights = self._validate_and_copy_weights(weights)

        self._pending_updates[client_name] = ClientUpdate(
            client_name=client_name,
            weights=normalized_weights,
            sample_count=int(sample_count),
            loss=None if loss is None else float(loss),
        )

    def aggregate(self) -> Dict[str, Tensor]:
        if not self._pending_updates:
            raise ValueError("No client updates are available for aggregation")

        updates = list(self._pending_updates.values())
        aggregated = self.aggregate_updates(updates, reference_state=self.model.state_dict())

        self.model.load_state_dict(aggregated)
        self._pending_updates.clear()

        return self.get_global_weights()

    @staticmethod
    def aggregate_updates(
        updates: Iterable[ClientUpdate],
        reference_state: Optional[ModelState] = None,
    ) -> Dict[str, Tensor]:
        updates = list(updates)
        if not updates:
            raise ValueError("At least one client update is required")

        total_samples = sum(update.sample_count for update in updates)
        if total_samples <= 0:
            raise ValueError("Total sample count must be greater than zero")

        first_state = updates[0].weights
        expected_keys = tuple(first_state.keys())

        if reference_state is not None:
            if tuple(reference_state.keys()) != expected_keys:
                raise ValueError("Client model state does not match the global model state")

        for update in updates[1:]:
            if tuple(update.weights.keys()) != expected_keys:
                raise ValueError(
                    f"Client {update.client_name!r} has a different model state structure"
                )

        aggregated: Dict[str, Tensor] = {}

        for key in expected_keys:
            first_tensor = first_state[key]

            if not torch.is_tensor(first_tensor):
                raise TypeError(f"Model state entry {key!r} is not a tensor")

            if not torch.is_floating_point(first_tensor):
                largest = max(updates, key=lambda u: u.sample_count)
                aggregated[key] = largest.weights[key].clone()
                continue

            result = torch.zeros_like(first_tensor)

            for update in updates:
                tensor = update.weights[key]
                if tensor.shape != first_tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {key!r}: "
                        f"expected {tuple(first_tensor.shape)}, "
                        f"got {tuple(tensor.shape)} from {update.client_name!r}"
                    )
                weight = update.sample_count / total_samples
                result.add_(tensor, alpha=weight)

            aggregated[key] = result

        return aggregated

    def get_aggregated_loss(self) -> Optional[float]:
        if not self._pending_updates:
            return None

        total_samples = sum(update.sample_count for update in self._pending_updates.values())
        known_losses = [u for u in self._pending_updates.values() if u.loss is not None]

        if not known_losses:
            return None

        return sum(
            update.loss * update.sample_count
            for update in known_losses
        ) / sum(update.sample_count for update in known_losses)

    def save_global_model(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.get_global_weights(), path)

    def load_global_model(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Global model not found: {path}")

        weights = torch.load(path, map_location="cpu", weights_only=True)
        validated = self._validate_and_copy_weights(weights)
        self.model.load_state_dict(validated)

    def _validate_and_copy_weights(self, weights: ModelState) -> Dict[str, Tensor]:
        reference = self.model.state_dict()

        if tuple(weights.keys()) != tuple(reference.keys()):
            raise ValueError("Client weights do not match the global model")

        result: Dict[str, Tensor] = {}
        for key, reference_tensor in reference.items():
            tensor = weights[key]

            if not torch.is_tensor(tensor):
                raise TypeError(f"Weight {key!r} is not a torch.Tensor")

            if tensor.shape != reference_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {key!r}: "
                    f"expected {tuple(reference_tensor.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )

            result[key] = tensor.detach().cpu().clone()

        return result


if __name__ == "__main__":
    aggregator = FederatedAggregator()

    base = aggregator.get_global_weights()
    client_a = {key: value + 1.0 for key, value in base.items()}
    client_b = {key: value + 3.0 for key, value in base.items()}

    aggregator.submit_update("EU", client_a, sample_count=100)
    aggregator.submit_update("NA", client_b, sample_count=300)
    result = aggregator.aggregate()

    print("FedAvg smoke test completed.")
    print(f"Aggregated {len(result)} tensors from 2 clients.")
