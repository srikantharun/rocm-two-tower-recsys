"""Benchmark the two-tower model on whatever backend is present (ROCm / CUDA / CPU).

Reports two numbers that matter for a recommender in production:
  * Training throughput  -- forward+backward pairs/second
  * Retrieval latency    -- time to score the full item catalogue for a batch of
                            users (p50 / p95 / p99), i.e. candidate generation cost

Run it on an AMD GPU and again on an NVIDIA GPU (or CPU) and paste both rows into
the README table -- same code, different ``backend`` header.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.device import backend_name, get_device, print_device_info, synchronize
from src.model import TwoTowerModel


def benchmark_training(model: TwoTowerModel, device: torch.device, batch_size: int, steps: int) -> float:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_users = model.user_emb.num_embeddings
    num_items = model.item_emb.num_embeddings

    def one_step() -> None:
        users = torch.randint(0, num_users, (batch_size,), device=device)
        items = torch.randint(0, num_items, (batch_size,), device=device)
        u = model.user_repr(users)
        i = model.item_repr(items)
        logits = (u @ i.t()) / 0.1
        labels = torch.arange(batch_size, device=device)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for _ in range(5):  # warmup
        one_step()
    synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        one_step()
    synchronize()
    elapsed = time.perf_counter() - start
    return (steps * batch_size) / elapsed  # pairs/second


def benchmark_retrieval(model: TwoTowerModel, device: torch.device, user_batch: int, trials: int) -> dict[str, float]:
    model.eval()
    num_users = model.user_emb.num_embeddings
    with torch.no_grad():
        item_matrix = model.all_item_repr()  # (num_items, d)

        def one_query() -> None:
            users = torch.randint(0, num_users, (user_batch,), device=device)
            user_vecs = model.user_repr(users)
            _ = user_vecs @ item_matrix.t()  # (user_batch, num_items) scores

        for _ in range(5):  # warmup
            one_query()
        synchronize()

        latencies = []
        for _ in range(trials):
            start = time.perf_counter()
            one_query()
            synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)

    latencies.sort()
    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(np.mean(latencies)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark two-tower RecSys on ROCm/CUDA/CPU.")
    parser.add_argument("--users", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=50_000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--user-batch", type=int, default=1024)
    parser.add_argument("--retrieval-trials", type=int, default=100)
    args = parser.parse_args()

    print_device_info()
    device = get_device()
    model = TwoTowerModel(args.users, args.items, embedding_dim=args.embedding_dim).to(device)

    throughput = benchmark_training(model, device, args.batch_size, args.train_steps)
    latency = benchmark_retrieval(model, device, args.user_batch, args.retrieval_trials)

    print("\n" + "-" * 62)
    print(f" Results on: {backend_name()}")
    print("-" * 62)
    print(f"  catalogue            : {args.users:,} users x {args.items:,} items (dim {args.embedding_dim})")
    print(f"  training throughput  : {throughput:,.0f} pairs/s (batch {args.batch_size})")
    print(f"  retrieval p50        : {latency['p50_ms']:.2f} ms  (score {args.items:,} items x {args.user_batch} users)")
    print(f"  retrieval p95        : {latency['p95_ms']:.2f} ms")
    print(f"  retrieval p99        : {latency['p99_ms']:.2f} ms")
    print("-" * 62)
    print("\nMarkdown row for the README table:")
    print(
        f"| {backend_name()} | {throughput:,.0f} | "
        f"{latency['p50_ms']:.2f} | {latency['p95_ms']:.2f} | {latency['p99_ms']:.2f} |"
    )


if __name__ == "__main__":
    main()
