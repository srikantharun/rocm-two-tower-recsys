# rocm-two-tower-recsys

A **two-tower recommender** that trains and benchmarks on **AMD GPUs via ROCm** — written
device-agnostically so the *same* PyTorch code runs on ROCm, CUDA, or CPU with no changes.

Built to demonstrate two things at once:

1. **Recommender-systems engineering** — embedding-based candidate retrieval (the two-tower
   architecture behind large-scale RecSys), trained with in-batch sampled-softmax negatives and
   evaluated with leave-one-out **Recall@K / NDCG@K**.
2. **AMD ROCm proficiency** — training and serving a real model on AMD hardware (HIP), plus a
   portable CUDA-vs-ROCm benchmark harness.

> **Why this matters:** CUDA is NVIDIA-only and proprietary; **ROCm is AMD's open-source stack**.
> A PyTorch ROCm build exposes AMD GPUs through the same `torch.cuda` API and the same `"cuda"`
> device string — so production PyTorch code ports to AMD with *zero* source changes. This repo is
> that claim, made runnable.

## What's inside

| File | Purpose |
|------|---------|
| `src/device.py` | Backend detection (ROCm / CUDA / CPU) and honest GPU timing (`synchronize`) |
| `src/data.py` | Synthetic low-rank implicit-feedback dataset, leave-one-out split (zero downloads) |
| `src/model.py` | `TwoTowerModel` — user tower + item tower, dot-product relevance |
| `src/train.py` | Sampled-softmax training loop + Recall@K / NDCG@K evaluation |
| `src/benchmark.py` | Training throughput (pairs/s) + retrieval latency (p50/p95/p99) with a Markdown row |
| `Dockerfile` | Runs on an AMD GPU using AMD's official `rocm/pytorch` image |

## Quick start

Install PyTorch for your backend (pick one), then the light deps:

```bash
# AMD ROCm (Linux):
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
# NVIDIA CUDA:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
# CPU only (works everywhere, for a quick smoke test):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Train, then benchmark (the backend is auto-detected and printed at the top):

```bash
python -m src.train --epochs 10
python -m src.benchmark
pytest            # CPU-only smoke tests
```

The model is device-agnostic — on a ROCm PyTorch build it uses the AMD GPU automatically; the
header will read `backend: ROCm (HIP)` with your MI300X / Radeon name.

## Run it on an AMD Instinct MI300X (free credits)

The fastest way to run this on real AMD silicon:

1. **Claim credits** — join the [AMD AI Developer Program](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)
   for **$100 in credits (~50 hrs on an MI300X)**.
2. **Create a GPU Droplet** on **AMD Developer Cloud** (powered by DigitalOcean): region ATL1,
   plan *MI300X (1 GPU)* — 192 GB HBM3 VRAM, $1.99/hr. Pick the **PyTorch** or **ROCm** image.
3. **SSH in and run:**
   ```bash
   git clone https://github.com/srikantharun/rocm-two-tower-recsys
   cd rocm-two-tower-recsys
   pip install -r requirements.txt     # torch ships in the ROCm image
   python -m src.train --epochs 10
   python -m src.benchmark
   rocm-smi                            # AMD's nvidia-smi: VRAM / util / temp
   ```
4. **Destroy the droplet** when done — powering off still bills; only *Destroy* stops charges.

> The same cloud box also hosts LLMs via **vLLM** (OpenAI-compatible endpoint) if you want to
> extend this into a RAG / agent demo on AMD hardware.

## Benchmark results

Run `python -m src.benchmark` on each backend and paste the printed Markdown row here.
_(100,000 users × 50,000 items, embedding dim 64, batch 4096.)_

| Backend | Train throughput (pairs/s) | Retrieval p50 (ms) | p95 (ms) | p99 (ms) |
|---------|---------------------------:|-------------------:|---------:|---------:|
| ROCm (HIP) — MI300X | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| CUDA — _(optional)_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| CPU | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Notes

- Dataset is synthetic but has recoverable low-rank structure, so a trained model posts
  non-trivial Recall@10 / NDCG@10 — swap in MovieLens by replacing `src/data.py`.
- Timing uses `torch.cuda.synchronize()` (works on ROCm too) so GPU numbers are honest.
