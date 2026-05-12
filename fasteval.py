#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class FunctionPairDataset(Dataset):
    def __init__(self, anchors: List[torch.Tensor], positives: List[torch.Tensor]):
        if len(anchors) != len(positives):
            raise ValueError("Anchor and positive embedding lists must have the same length.")
        self.anchors = anchors
        self.positives = positives

    def __getitem__(self, idx):
        return self.anchors[idx].squeeze(0), self.positives[idx].squeeze(0)

    def __len__(self):
        return len(self.anchors)


def _normalize_embedding(value):
    if isinstance(value, torch.Tensor):
        emb = value.float()
    else:
        emb = torch.as_tensor(value, dtype=torch.float32)
    return emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def build_pairs(records, opt_a: str, opt_b: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    anchors, positives = [], []
    for item in records:
        if not isinstance(item, dict):
            continue
        emb_a = item.get(opt_a)
        emb_b = item.get(opt_b)
        if emb_a is None or emb_b is None:
            continue
        if isinstance(emb_a, int) or isinstance(emb_b, int):
            continue
        anchors.append(_normalize_embedding(emb_a))
        positives.append(_normalize_embedding(emb_b))
    return anchors, positives


def evaluate_pair(records, opt_a: str, opt_b: str, poolsize: int, batch_workers: int) -> Tuple[float, float, int]:
    anchors, positives = build_pairs(records, opt_a, opt_b)
    dataset = FunctionPairDataset(anchors, positives)
    loader = DataLoader(dataset, batch_size=poolsize, num_workers=batch_workers, shuffle=True, drop_last=True)

    reciprocal_ranks = []
    recall_at_1 = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for anchor, positive in tqdm(loader, desc=f"{opt_a}-{opt_b}", dynamic_ncols=True):
        anchor = anchor.to(device)
        positive = positive.to(device)
        sim = torch.mm(anchor, positive.t()).detach().cpu().numpy()
        ranked = np.argsort(-sim, axis=1)

        for i in range(ranked.shape[0]):
            rank = int(np.where(ranked[i] == i)[0][0]) + 1
            reciprocal_ranks.append(1.0 / rank)
            recall_at_1.append(1.0 if rank == 1 else 0.0)

    if not reciprocal_ranks:
        return float("nan"), float("nan"), 0
    return float(np.mean(reciprocal_ranks)), float(np.mean(recall_at_1)), len(reciprocal_ranks)


def parse_pairs(raw_pairs: str) -> Iterable[Tuple[str, str]]:
    for item in raw_pairs.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Invalid pair {item!r}. Expected format such as O0-O3.")
        left, right = item.split("-", 1)
        yield left.strip(), right.strip()


def main():
    parser = argparse.ArgumentParser("Evaluate exported SemFormer embeddings")
    parser.add_argument("--experiment-path", "--experiment_path", dest="experiment_path", default="./outputs/eval_embeddings.pkl")
    parser.add_argument("--poolsize", type=int, default=32)
    parser.add_argument(
        "--pairs",
        default="O0-O3,O0-Os,O1-O3,O1-Os,O2-O3,O2-Os",
        help="comma-separated optimization pairs",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    with open(args.experiment_path, "rb") as f:
        records = pickle.load(f)

    print(f"poolsize={args.poolsize}")
    for opt_a, opt_b in parse_pairs(args.pairs):
        mrr, recall1, count = evaluate_pair(records, opt_a, opt_b, args.poolsize, args.num_workers)
        print(f"{opt_a}-{opt_b}\tcount={count}\tMRR={mrr:.6f}\tRecall@1={recall1:.6f}")


if __name__ == "__main__":
    main()
