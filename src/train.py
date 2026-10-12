"""Train the two-tower recommender with in-batch sampled-softmax negatives.

For a batch of (user, positive_item) pairs we score every user against every
positive item in the batch; the correct pairing is the diagonal, so the other
items act as negatives. This is the sampled-softmax trick used in production
two-tower retrieval -- efficient and GPU-friendly.

Runs on CPU out of the box; set no flags and it uses the AMD GPU automatically
on a ROCm PyTorch build.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from src.data import Interactions, make_synthetic
from src.device import get_device, print_device_info, synchronize
from src.model import TwoTowerModel


def train(
    data: Interactions,
    device: torch.device,
    embedding_dim: int = 64,
    epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 3e-3,
    temperature: float = 0.1,
    seed: int = 0,
) -> TwoTowerModel:
    torch.manual_seed(seed)
    model = TwoTowerModel(data.num_users, data.num_items, embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    users = torch.from_numpy(data.train_pairs[:, 0]).to(device)
    items = torch.from_numpy(data.train_pairs[:, 1]).to(device)
    num_pairs = users.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_pairs, device=device)
        total_loss = 0.0
        num_batches = 0
        for start in range(0, num_pairs, batch_size):
            idx = perm[start : start + batch_size]
            if idx.shape[0] < 2:
                continue  # in-batch negatives need at least two rows
            u = model.user_repr(users[idx])           # (B, d)
            i = model.item_repr(items[idx])           # (B, d)
            logits = (u @ i.t()) / temperature        # (B, B)
            labels = torch.arange(idx.shape[0], device=device)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        recall, ndcg = evaluate(model, data, device, k=10)
        print(
            f"epoch {epoch:3d} | loss {total_loss / max(num_batches, 1):.4f} "
            f"| Recall@10 {recall:.4f} | NDCG@10 {ndcg:.4f}"
        )
    return model


@torch.no_grad()
def evaluate(model: TwoTowerModel, data: Interactions, device: torch.device, k: int = 10) -> tuple[float, float]:
    """Leave-one-out Recall@K and NDCG@K against the full catalogue, training items masked."""
    model.eval()
    item_matrix = model.all_item_repr()               # (num_items, d)

    test_users = torch.from_numpy(data.test_pairs[:, 0]).to(device)
    test_items = data.test_pairs[:, 1]

    hits = 0
    ndcg_sum = 0.0
    eval_batch = 512
    for start in range(0, test_users.shape[0], eval_batch):
        batch_users = test_users[start : start + eval_batch]
        user_vecs = model.user_repr(batch_users)       # (b, d)
        scores = user_vecs @ item_matrix.t()           # (b, num_items)

        # Mask out items the user already interacted with in training.
        for row, user_id in enumerate(batch_users.tolist()):
            train_items = data.user_train_items[user_id]
            if train_items.size:
                scores[row, train_items] = float("-inf")

        topk = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()  # (b, k)
        targets = test_items[start : start + eval_batch]
        for row, target in enumerate(targets):
            ranked = topk[row]
            pos = np.where(ranked == target)[0]
            if pos.size:
                hits += 1
                ndcg_sum += 1.0 / np.log2(pos[0] + 2)

    n = test_users.shape[0]
    return hits / n, ndcg_sum / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a two-tower recommender on ROCm/CUDA/CPU.")
    parser.add_argument("--users", type=int, default=3000)
    parser.add_argument("--items", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="", help="optional path to save model weights")
    args = parser.parse_args()

    print_device_info()
    device = get_device()
    data = make_synthetic(num_users=args.users, num_items=args.items, seed=args.seed)
    print(f"dataset: {data.train_pairs.shape[0]} train / {data.test_pairs.shape[0]} test interactions")

    model = train(
        data,
        device,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    synchronize()
    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"saved weights to {args.save}")


if __name__ == "__main__":
    main()
