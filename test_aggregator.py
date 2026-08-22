import torch

from aggregator import ClientUpdate, FederatedAggregator


def test_weighted_fedavg():
    aggregator = FederatedAggregator()
    base = aggregator.get_global_weights()

    weights_a = {key: torch.full_like(value, 1.0) for key, value in base.items()}
    weights_b = {key: torch.full_like(value, 3.0) for key, value in base.items()}

    result = FederatedAggregator.aggregate_updates([
        ClientUpdate("A", weights_a, 1),
        ClientUpdate("B", weights_b, 3),
    ])

    for tensor in result.values():
        if torch.is_floating_point(tensor):
            assert torch.allclose(tensor, torch.full_like(tensor, 2.5))


def test_round_updates_global_model():
    aggregator = FederatedAggregator()
    base = aggregator.get_global_weights()

    plus_one = {key: value + 1.0 for key, value in base.items()}
    plus_three = {key: value + 3.0 for key, value in base.items()}

    aggregator.submit_update("A", plus_one, 100)
    aggregator.submit_update("B", plus_three, 300)
    result = aggregator.aggregate()

    for key, tensor in result.items():
        if torch.is_floating_point(tensor):
            assert torch.allclose(tensor, base[key] + 2.5)

    assert aggregator.pending_update_count == 0


def test_rejects_invalid_sample_count():
    aggregator = FederatedAggregator()
    try:
        aggregator.submit_update("A", aggregator.get_global_weights(), 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for zero samples")


if __name__ == "__main__":
    test_weighted_fedavg()
    test_round_updates_global_model()
    test_rejects_invalid_sample_count()
    print("All aggregator tests passed.")
