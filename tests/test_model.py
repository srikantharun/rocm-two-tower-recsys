"""CPU-only smoke tests so CI stays green without a GPU."""

from __future__ import annotations

import numpy as np
import torch

from src.data import make_synthetic
from src.device import backend_name, get_device
from src.model import TwoTowerModel
from src.train import evaluate, train


def test_device_resolves() -> None:
    assert get_device().type in {"cuda", "cpu"}
    assert backend_name() in {"ROCm (HIP)", "CUDA", "CPU", "GPU (unknown runtime)"}


def test_model_shapes() -> None:
    model = TwoTowerModel(num_users=50, num_items=40, embedding_dim=16)
    users = torch.arange(8)
    items = torch.arange(8)
    scores = model(users, items)
    assert scores.shape == (8,)
    assert model.all_item_repr().shape == (40, 16)


def test_synthetic_split_is_leave_one_out() -> None:
    data = make_synthetic(num_users=100, num_items=60, seed=1)
    assert data.test_pairs.shape == (100, 2)  # exactly one held-out item per user
    assert data.train_pairs.shape[1] == 2
    assert len(data.user_train_items) == 100


def test_training_improves_ranking() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    data = make_synthetic(num_users=400, num_items=200, seed=0)
    device = get_device()
    model = TwoTowerModel(data.num_users, data.num_items, embedding_dim=32).to(device)
    before, _ = evaluate(model, data, device, k=10)
    model = train(data, device, embedding_dim=32, epochs=5, batch_size=256)
    after, _ = evaluate(model, data, device, k=10)
    # A trained model must beat an untrained one on recoverable low-rank signal.
    assert after > before
