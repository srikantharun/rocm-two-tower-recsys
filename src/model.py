"""Two-tower retrieval model.

A user tower and an item tower each map an id to an embedding; relevance is the
dot product of the two. This is the standard architecture behind large-scale
recommender retrieval (candidate generation), and it maps cleanly onto GPU:
scoring is one matrix multiply, so it benchmarks well on ROCm and CUDA alike.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def user_repr(self, users: torch.Tensor) -> torch.Tensor:
        return self.user_tower(self.user_emb(users))

    def item_repr(self, items: torch.Tensor) -> torch.Tensor:
        return self.item_tower(self.item_emb(items))

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Elementwise relevance score for paired (user, item) tensors."""
        return (self.user_repr(users) * self.item_repr(items)).sum(dim=-1)

    def all_item_repr(self) -> torch.Tensor:
        """Encode the full item catalogue -- used for ranking at eval time."""
        all_ids = torch.arange(self.item_emb.num_embeddings, device=self.item_emb.weight.device)
        return self.item_repr(all_ids)
