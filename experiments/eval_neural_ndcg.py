"""Evaluate Avito official neural_baseline checkpoint on chronological val feeds.

Competition NDCG: DCG across ALL ranked candidates / IDCG@30,
then macro-average over feeds. Ties put negative targets before positives.

Usage (from ~/avito/official_baseline):
    python ../experiments/eval_neural_ndcg.py \
      --model ../experiments/runs/neural_smoke_cpu_200f/model.pt \
      --split ../competition/val --max-feeds 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

# This script is stored in ~/avito/experiments/ on the user's server.
BASELINE_DIR = Path(__file__).resolve().parent.parent / "official_baseline"
sys.path.insert(0, str(BASELINE_DIR))
import neural_baseline as nb  # noqa: E402


def competition_ndcg(target: np.ndarray, score: np.ndarray) -> float:
    """Exact documented ranking convention: full DCG, ideal DCG@30."""
    target = np.asarray(target, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if not np.isfinite(score).all():
        raise ValueError("Non-finite scores")
    positives = int(target.sum())
    if positives == 0:
        return 1.0
    # lexsort: primary key = descending score; secondary = target ascending
    # (on exact score ties, negatives go first).
    order = np.lexsort((target, -score))
    discounts = 1.0 / np.log2(np.arange(2, len(target) + 2))
    dcg = float(np.dot(target[order], discounts))
    top = min(30, positives)
    ideal_discounts = 1.0 / np.log2(np.arange(2, top + 2))
    return dcg / float(ideal_discounts.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--max-feeds", type=int, default=100,
                        help="Limit feeds for smoke evaluation; 0 means full split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"Checkpoint not found: {args.model}")
    if not (args.split / "impressions").is_dir():
        parser.error(f"Split not found: {args.split}")

    torch.set_num_threads(8)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    mapping = checkpoint["categories_mapping"]
    ds = nb.CandidateData(args.split, args.max_feeds, categories_mapping=mapping)
    model = nb.Ranker(len(mapping), **checkpoint["model_params"]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=nb.collate_feeds)

    ndcgs = []
    total_bce = 0.0
    total_candidates = 0
    with torch.inference_mode():
        for features, categories, users, padding, targets in loader:
            logits = model(features.to(args.device), categories.to(args.device),
                           users.to(args.device), padding.to(args.device))
            targets_gpu = targets.to(args.device)
            total_bce += F.binary_cross_entropy_with_logits(
                logits[~padding.to(args.device)], targets_gpu[~padding.to(args.device)],
                reduction="sum",
            ).item()
            for i in range(len(features)):
                n = int((~padding[i]).sum().item())
                ndcgs.append(competition_ndcg(
                    targets[i, :n].numpy(), logits[i, :n].cpu().numpy()
                ))
                total_candidates += n

    print(f"feeds:          {len(ndcgs):,}")
    print(f"candidates:     {total_candidates:,}")
    print(f"val_logloss:    {total_bce / total_candidates:.10f}")
    print(f"val_ndcg_full:  {float(np.mean(ndcgs)):.10f}")
    print("Note: smoke NDCG is NOT comparable to full-val CatBoost NDCG.")


if __name__ == "__main__":
    main()
