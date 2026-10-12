"""Synthetic implicit-feedback dataset for the two-tower recommender.

Ready-to-run with zero downloads: user/item interactions are generated from a
hidden low-rank preference structure, so a well-trained two-tower model can
actually recover signal and post non-trivial Recall@K / NDCG@K.

Evaluation uses leave-one-out: one held-out positive per user, ranked against the
full item catalogue (with the user's training items masked out).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Interactions:
    num_users: int
    num_items: int
    train_pairs: np.ndarray           # (N, 2) int64 columns: [user, item]
    test_pairs: np.ndarray            # (M, 2) int64 -- one held-out positive per user
    user_train_items: list[np.ndarray]  # per-user training item ids (for masking)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def make_synthetic(
    num_users: int = 3000,
    num_items: int = 1500,
    latent_dim: int = 16,
    avg_interactions: int = 25,
    seed: int = 0,
) -> Interactions:
    """Generate a low-rank implicit-feedback dataset with a leave-one-out test split."""
    rng = np.random.default_rng(seed)
    user_factors = rng.normal(size=(num_users, latent_dim))
    item_factors = rng.normal(size=(num_items, latent_dim))

    train_rows: list[tuple[int, int]] = []
    test_rows: list[tuple[int, int]] = []
    user_train_items: list[np.ndarray] = []

    for user in range(num_users):
        probs = _softmax(user_factors[user] @ item_factors.T)
        k = min(num_items, max(2, int(rng.poisson(avg_interactions))))
        items = rng.choice(num_items, size=k, replace=False, p=probs)
        # Leave-one-out: hold out the last sampled item for evaluation.
        held_out = int(items[-1])
        train_items = items[:-1].astype(np.int64)
        user_train_items.append(train_items)
        for item in train_items:
            train_rows.append((user, int(item)))
        test_rows.append((user, held_out))

    return Interactions(
        num_users=num_users,
        num_items=num_items,
        train_pairs=np.array(train_rows, dtype=np.int64),
        test_pairs=np.array(test_rows, dtype=np.int64),
        user_train_items=user_train_items,
    )
