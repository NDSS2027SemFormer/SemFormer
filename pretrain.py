#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import glob
import time
import argparse
import random
import pickle
import json
import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Set

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from transformers import (
    BertConfig,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)

import readidadata

from data import gen_funcstr_with_rel_no_dup as gen_funcstr_with_rel
from model import BinBertForMaskedLMRDP, MODEL_MODES, set_model_mode


# =========================
# timeout helper (Linux only)
# =========================
try:
    import signal
    _HAS_SIGNAL = True
except Exception:
    _HAS_SIGNAL = False


class _TimeoutExc(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    """
    Raise _TimeoutExc if the block doesn't finish within `seconds`.
    Works on Linux (signal). If unsupported or seconds<=0, it's a no-op.
    """
    if (not _HAS_SIGNAL) or seconds is None or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise _TimeoutExc(f"timeout>{seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def _get_worker_tag():
    rank = int(os.environ.get("RANK", "0"))
    w = torch.utils.data.get_worker_info()
    wid = w.id if w is not None else 0
    return rank, wid


def append_error_line(
    error_dir: str,
    pkl_path: str,
    func_key: str,
    reason: str,
    T: int,
    n_nodes: int,
    n_edges: int,
    max_rel_dist: int,
    item_timeout_sec: float,
):
    """
    Each dataloader worker writes to its own file to avoid file locks.
    TSV:
      tag  pkl  func  reason  T  nodes  edges  max_rel_dist  item_timeout_sec  ts
    """
    if not error_dir:
        return
    os.makedirs(error_dir, exist_ok=True)
    rank, wid = _get_worker_tag()
    path = os.path.join(error_dir, f"errors_rank{rank}_worker{wid}.log")
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = "\t".join([
        "ERR",
        str(pkl_path),
        str(func_key),
        str(reason),
        str(T),
        str(n_nodes),
        str(n_edges),
        str(max_rel_dist),
        str(item_timeout_sec),
        ts,
    ])
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# =========================
# DDP utils
# =========================
def ddp_setup():
    """
    torchrun will set:
      RANK, LOCAL_RANK, WORLD_SIZE
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def ddp_cleanup():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def ddp_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def ddp_all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
        x = x / torch.distributed.get_world_size()
    return x


def ddp_all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x


# =========================
# misc utils
# =========================
def seed_all(seed: int, rank: int = 0):
    s = int(seed) + 1000 * int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_ce_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    valid = (labels != ignore_index)
    if valid.any():
        B, L, V = logits.shape
        return nn.functional.cross_entropy(
            logits.view(-1, V),
            labels.view(-1),
            ignore_index=ignore_index,
            reduction="mean",
        )
    else:
        return logits.sum() * 0.0


def ce_loss_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    B, P, R = logits.shape
    return nn.functional.cross_entropy(
        logits.reshape(B * P, R),
        labels.reshape(B * P),
        reduction="mean",
    )


@torch.no_grad()
def acc1_flat(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    hit = (pred == labels).float().mean().item()
    return float(hit)


@torch.no_grad()
def acc1_masked(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    if mask.any():
        hit = (pred[mask] == labels[mask]).float().mean().item()
        return float(hit)
    return 0.0


def bucket_id(label: int, no_edge_id: int) -> int:
    # 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-127, 128-255, 256-512, no_edge, other
    if label == no_edge_id:
        return 10
    if label == 0:
        return 0
    if label == 1:
        return 1
    if 2 <= label <= 3:
        return 2
    if 4 <= label <= 7:
        return 3
    if 8 <= label <= 15:
        return 4
    if 16 <= label <= 31:
        return 5
    if 32 <= label <= 63:
        return 6
    if 64 <= label <= 127:
        return 7
    if 128 <= label <= 255:
        return 8
    if 256 <= label <= 512:
        return 9
    return 11


# =========================
# CFG -> mutex bb pairs (unchanged)
# =========================
def _add_super_exit_if_needed(cfg: nx.DiGraph) -> Tuple[nx.DiGraph, object]:
    if cfg is None or cfg.number_of_nodes() == 0:
        return cfg, None

    exits = [n for n in cfg.nodes if cfg.out_degree(n) == 0]
    if len(exits) <= 1:
        if len(exits) == 1:
            return cfg, exits[0]
        any_node = next(iter(cfg.nodes))
        return cfg, any_node

    g = cfg.copy()
    super_exit = ("__SUPER_EXIT__", id(cfg))
    g.add_node(super_exit)
    for x in exits:
        g.add_edge(x, super_exit)
    return g, super_exit


def _immediate_postdominators(cfg: nx.DiGraph) -> Dict[object, object]:
    if cfg is None or cfg.number_of_nodes() == 0:
        return {}

    g, exit_node = _add_super_exit_if_needed(cfg)
    if exit_node is None:
        return {}

    rg = g.reverse(copy=True)
    try:
        idom = nx.immediate_dominators(rg, exit_node)
    except Exception:
        return {}
    return idom


def _arm_nodes_between(branch: object, succ: object, stop: object, cfg: nx.DiGraph) -> Set[object]:
    arm: Set[object] = set()
    if succ == stop:
        return arm

    stack = [succ]
    visited = set([succ])
    while stack:
        u = stack.pop()
        if u == stop:
            continue
        arm.add(u)
        for v in cfg.successors(u):
            if v == stop:
                continue
            if v not in visited:
                visited.add(v)
                stack.append(v)
    return arm


def _compute_mutex_bb_pairs(cfg: nx.DiGraph) -> Set[Tuple[object, object]]:
    if cfg is None or cfg.number_of_nodes() == 0:
        return set()

    ipdom = _immediate_postdominators(cfg)
    if not ipdom:
        return set()

    mutex_pairs: Set[Tuple[object, object]] = set()

    for b in cfg.nodes:
        succs = list(cfg.successors(b))
        if len(succs) < 2:
            continue

        stop = ipdom.get(b, None)
        if stop is None or stop == b:
            continue

        arms: List[Set[object]] = []
        for s in succs:
            arm = _arm_nodes_between(b, s, stop, cfg)
            if arm:
                arms.append(arm)

        if len(arms) < 2:
            continue

        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                Ai, Aj = arms[i], arms[j]
                for u in Ai:
                    for v in Aj:
                        mutex_pairs.add((u, v))
                        mutex_pairs.add((v, u))
    return mutex_pairs


# =========================
# Dataset: infinite stream, yields (tokens, rel_small)
#   - per-function timeout + per-worker error logs
#   - unreachable id is ALWAYS (max_rel_dist + 1)
# =========================
class BinaryCorpIterableDatasetWithRel(IterableDataset):
    def __init__(
        self,
        root_dir: str,
        max_raw_tokens: int,
        max_rel_dist: int,
        use_longest_path: bool = False,
        use_operand_anchor: bool = True,
        item_timeout_sec: float = 0.0,
        error_dir: str = "./error_logs",
    ):
        super().__init__()
        self.root_dir = root_dir
        self.max_raw_tokens = max_raw_tokens
        self.max_rel_dist = int(max_rel_dist)
        self.use_longest_path = bool(use_longest_path)
        self.use_operand_anchor = bool(use_operand_anchor)

        self.no_edge_id = int(self.max_rel_dist + 1)   # e.g. 512 -> 513

        self.item_timeout_sec = float(item_timeout_sec) if item_timeout_sec is not None else 0.0
        self.error_dir = error_dir

        self.pkl_files = sorted(glob.glob(os.path.join(root_dir, "**", "*.pkl"), recursive=True))
        if not self.pkl_files:
            raise ValueError(f"No .pkl found under: {root_dir}")

        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print(f"[DATA] found {len(self.pkl_files)} pkls under {root_dir}")
            print(f"[DATA] example pkl: {self.pkl_files[0]}")
            print(f"[REL] max_rel_dist={self.max_rel_dist} -> no_edge_id(no_relation_id)={self.no_edge_id}")

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        files = self.pkl_files

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        files = files[rank :: world_size]

        if worker is not None:
            files = files[worker.id :: worker.num_workers]

        while True:
            random.shuffle(files)
            for pkl_path in files:
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                except Exception as e:
                    append_error_line(
                        self.error_dir, pkl_path, "-", f"PKL_LOAD_ERR:{type(e).__name__}:{e}",
                        T=-1, n_nodes=-1, n_edges=-1, max_rel_dist=self.max_rel_dist, item_timeout_sec=self.item_timeout_sec
                    )
                    continue

                if not isinstance(data, dict):
                    append_error_line(
                        self.error_dir, pkl_path, "-", "PKL_FMT_ERR:not_dict",
                        T=-1, n_nodes=-1, n_edges=-1, max_rel_dist=self.max_rel_dist, item_timeout_sec=self.item_timeout_sec
                    )
                    continue

                for func_key, item in data.items():
                    if not (isinstance(item, (list, tuple)) and len(item) >= 5):
                        append_error_line(
                            self.error_dir, pkl_path, str(func_key), "ITEM_FMT_ERR:bad_item",
                            T=-1, n_nodes=-1, n_edges=-1, max_rel_dist=self.max_rel_dist, item_timeout_sec=self.item_timeout_sec
                        )
                        continue

                    n_nodes = -1
                    n_edges = -1
                    T = -1

                    try:
                        with time_limit(self.item_timeout_sec):
                            ftuple = (item[0], item[1], item[2], item[3], item[4])

                            _func_str, rel_small, _mask_unused, aux = gen_funcstr_with_rel(
                                ftuple,
                                max_rel_dist=self.max_rel_dist,
                                max_raw_tokens=self.max_raw_tokens,
                                use_operand_anchor=self.use_operand_anchor,
                                use_longest_path=self.use_longest_path,
                            )
                            toks = aux.get("tokens", None)
                            if not toks:
                                continue

                            T = len(toks)

                            data_no_rel = aux.get("no_relation_id", None)
                            if data_no_rel is not None and int(data_no_rel) != int(self.no_edge_id):
                                append_error_line(
                                    self.error_dir, pkl_path, str(func_key),
                                    f"NO_EDGE_MISMATCH:data_no_rel={data_no_rel} expected={self.no_edge_id}",
                                    T=T, n_nodes=n_nodes, n_edges=n_edges,
                                    max_rel_dist=self.max_rel_dist, item_timeout_sec=self.item_timeout_sec
                                )
                                continue

                            cfg = ftuple[3]
                            if cfg is not None and hasattr(cfg, "number_of_nodes"):
                                try:
                                    n_nodes = int(cfg.number_of_nodes())
                                    n_edges = int(cfg.number_of_edges())
                                except Exception:
                                    pass
                    except _TimeoutExc as e:
                        append_error_line(
                            self.error_dir, pkl_path, str(func_key), f"TIMEOUT:{e}",
                            T=T, n_nodes=n_nodes, n_edges=n_edges, max_rel_dist=self.max_rel_dist, item_timeout_sec=self.item_timeout_sec
                        )
                        continue
                    except Exception as e:
                        append_error_line(
                            self.error_dir, pkl_path, str(func_key), f"ITEM_ERR:{type(e).__name__}:{e}",
                            T=T, n_nodes=n_nodes, n_edges=n_edges, max_rel_dist=self.max_rel_dist, item_timeout_sec=self.item_timeout_sec
                        )
                        continue

                    yield toks, rel_small


# =========================
# Collator: MLM + RDP + rel_ids
#   - unreachable id is ALWAYS (max_rel_dist + 1)
# =========================
@dataclass
class BatchStats:
    Lmax: int
    mlm_n: int
    pairs: int
    ne_ratio: float
    lab_hist: Optional[np.ndarray] = None


class PretrainCollatorRDPNoMask:
    def __init__(
        self,
        vocab: Dict[str, int],
        pad_id: int,
        cls_id: int,
        sep_id: int,
        mask_id: int,
        vocab_size: int,
        max_len: int,
        max_rel_dist: int,
        mlm_prob: float,
        pairs_per_seq: int,
        pair_mode: str = "uniform",
        nontrivial_frac: float = 0.8,
        max_resample: int = 80,
        # bucket-balanced
        nontriv_bucket_min: int = 2,
        nontriv_bucket_max: int = 9,
        unk_fallback_id: Optional[int] = None,
    ):
        self.vocab = vocab
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self.max_len = int(max_len)

        self.max_rel_dist = int(max_rel_dist)

        self.no_edge_id = int(self.max_rel_dist + 1)  # 512 -> 513
        self.no_rel_id = int(self.no_edge_id)         # compat
        self.rel_num_types = int(self.max_rel_dist + 2)  # 514

        self.mlm_prob = float(mlm_prob)
        self.pairs_per_seq = int(pairs_per_seq)

        self.pair_mode = pair_mode
        self.nontrivial_frac = float(nontrivial_frac)
        self.max_resample = int(max_resample)

        self.nontriv_bucket_ids = list(range(int(nontriv_bucket_min), int(nontriv_bucket_max) + 1))
        self.unk_fallback_id = unk_fallback_id if unk_fallback_id is not None else mask_id

    def tok2id(self, tok: str) -> int:
        return self.vocab.get(tok, self.unk_fallback_id)

    def _sample_pair_uniform(self, Lvalid: int) -> Tuple[int, int]:
        if Lvalid <= 3:
            return 0, 0
        q = random.randint(1, Lvalid - 2)
        k = random.randint(1, Lvalid - 2)
        return q, k

    def _sample_pair_bucket(self, rel_t: torch.Tensor, Lvalid: int, target_bucket: int) -> Tuple[int, int]:
        for _ in range(self.max_resample):
            q, k = self._sample_pair_uniform(Lvalid)
            lab = int(rel_t[q, k].item())
            if lab == self.no_edge_id:
                continue
            bid = bucket_id(lab, no_edge_id=self.no_edge_id)
            if bid == target_bucket:
                return q, k
        return self._sample_pair_uniform(Lvalid)

    def __call__(self, batch: List[Tuple[List[str], np.ndarray]]) -> Dict[str, Any]:
        input_ids, attn, lm_labels, rel_ids = [], [], [], []
        pair_idx, pair_lab = [], []

        mlm_n = 0
        Lmax = 0
        ne_cnt = 0
        tot_pairs = 0

        lab_hist = np.zeros((self.rel_num_types,), dtype=np.int64)

        for toks, rel_small in batch:
            T = min(len(toks), self.max_len - 2)
            toks = toks[:T]
            rel_small = rel_small[:T, :T]

            toks2 = ["[CLS]"] + toks + ["[SEP]"]
            ids = [self.tok2id(t) for t in toks2]

            # padding region = no_edge_id (=max_rel_dist+1)
            rel_full = np.full((self.max_len, self.max_len), fill_value=self.no_edge_id, dtype=np.int16)
            rel_full[1:1 + T, 1:1 + T] = rel_small.astype(np.int16)

            Lvalid = T + 2
            # allow CLS/SEP connections
            rel_full[0, :Lvalid] = 0
            rel_full[:Lvalid, 0] = 0
            rel_full[Lvalid - 1, :Lvalid] = 0
            rel_full[:Lvalid, Lvalid - 1] = 0
            for i in range(min(Lvalid, self.max_len)):
                rel_full[i, i] = 0

            pad = self.max_len - len(ids)
            if pad < 0:
                ids = ids[:self.max_len]
                pad = 0
                Lvalid = min(Lvalid, self.max_len)
            ids = ids + [self.pad_id] * pad
            mask = [1] * (self.max_len - pad) + [0] * pad

            ids_t = torch.tensor(ids, dtype=torch.long)
            mask_t = torch.tensor(mask, dtype=torch.long)
            lm_t = torch.full_like(ids_t, -100)
            rel_t = torch.tensor(rel_full.astype(np.int64), dtype=torch.long)

            Lmax = max(Lmax, int(mask_t.sum().item()))

            # MLM
            for i in range(1, self.max_len - 1):
                tid = int(ids_t[i].item())
                if tid in (self.cls_id, self.sep_id, self.pad_id):
                    continue
                if random.random() < self.mlm_prob:
                    lm_t[i] = tid
                    mlm_n += 1
                    r = random.random()
                    if r < 0.8:
                        ids_t[i] = self.mask_id
                    elif r < 0.9:
                        ids_t[i] = random.randint(0, self.vocab_size - 1)

            # RDP pairs
            P = self.pairs_per_seq
            want_nontrivial = int(round(P * self.nontrivial_frac)) if self.pair_mode == "mix" else 0

            buckets = self.nontriv_bucket_ids[:]
            random.shuffle(buckets)

            pidx = []
            plab = []
            for pi in range(P):
                if self.pair_mode == "mix" and pi < want_nontrivial and buckets:
                    target_bucket = buckets[pi % len(buckets)]
                    q, k = self._sample_pair_bucket(rel_t, Lvalid, target_bucket)
                else:
                    q, k = self._sample_pair_uniform(Lvalid)

                lab = int(rel_t[q, k].item())
                pidx.append((q, k))
                plab.append(lab)

                tot_pairs += 1
                if 0 <= lab < self.rel_num_types:
                    lab_hist[lab] += 1
                if lab == self.no_edge_id:
                    ne_cnt += 1

            input_ids.append(ids_t)
            attn.append(mask_t)
            lm_labels.append(lm_t)
            rel_ids.append(rel_t)
            pair_idx.append(torch.tensor(pidx, dtype=torch.long))
            pair_lab.append(torch.tensor(plab, dtype=torch.long))

        ne_ratio = (ne_cnt / max(tot_pairs, 1))

        stats = BatchStats(
            Lmax=Lmax,
            mlm_n=mlm_n,
            pairs=tot_pairs,
            ne_ratio=float(ne_ratio),
            lab_hist=lab_hist,
        )

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels_mlm": torch.stack(lm_labels),
            "rel_ids": torch.stack(rel_ids),
            "pair_idx": torch.stack(pair_idx),
            "pair_lab": torch.stack(pair_lab),
            "stats": stats,
        }


# =========================
# build model
# =========================
def build_model(
    vocab_size: int,
    max_len: int,
    pe_equals_we: bool,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    intermediate_size: int,
    max_rel_dist: int,
    rel_stat_every: int,
    rel_probe_every: int,
    rel_probe_id: int,
    model_mode: str,
) -> BinBertForMaskedLMRDP:
    max_rel_dist = int(max_rel_dist)

    no_edge_id = int(max_rel_dist + 1)   # 513
    rel_num_types = int(max_rel_dist + 2)  # 514

    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_len,

        rel_num_types=rel_num_types,
        no_edge_id=no_edge_id,
        no_rel_id=no_edge_id,  # compat

        rel_stat_every=rel_stat_every,
        rel_probe_every=rel_probe_every,
        rel_probe_id=rel_probe_id,
    )
    set_model_mode(cfg, model_mode)
    model = BinBertForMaskedLMRDP(cfg)

    if pe_equals_we:
        model.bert.embeddings.position_embeddings = model.bert.embeddings.word_embeddings
        print("[CHECK] position_embeddings = word_embeddings")

    return model


# =========================
# save ckpt
# =========================
def save_ckpt(out_dir, model, tokenizer, optimizer, scheduler, scaler, global_step, rank: int):
    if rank != 0:
        return

    os.makedirs(out_dir, exist_ok=True)
    to_save = model.module if hasattr(model, "module") else model
    to_save.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(
        {
            "global_step": int(global_step),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        os.path.join(out_dir, "trainer.pt"),
    )
    with open(os.path.join(out_dir, "train_state.json"), "w", encoding="utf-8") as f:
        json.dump({"global_step": int(global_step)}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] step={global_step} -> {out_dir}")


# =========================
# train
# =========================
def train(args):
    ddp_on, rank, local_rank, world_size = ddp_setup()

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    seed_all(args.seed, rank=rank)
    if is_main_process(rank):
        ensure_dir(args.out)
        if args.error_dir:
            ensure_dir(args.error_dir)

    ddp_barrier()

    tok = BertTokenizerFast.from_pretrained(args.tokenizer_dir)
    vocab = tok.get_vocab()
    vocab_size = len(tok)

    pad_id = tok.pad_token_id
    cls_id = tok.cls_token_id
    sep_id = tok.sep_token_id
    mask_id = tok.mask_token_id
    unk_id = tok.unk_token_id if tok.unk_token_id is not None else mask_id

    no_edge_id = int(args.max_rel_dist + 1)   # 513
    rel_num_types = int(args.max_rel_dist + 2)  # 514
    if is_main_process(rank):
        print(f"[REL] max_rel_dist={args.max_rel_dist} -> no_edge_id={no_edge_id}, rel_num_types={rel_num_types}")

    dataset = BinaryCorpIterableDatasetWithRel(
        root_dir=args.pkl_dir,
        max_raw_tokens=args.max_raw_tokens,
        max_rel_dist=args.max_rel_dist,
        use_longest_path=args.use_longest_path,
        use_operand_anchor=args.use_operand_anchor,
        item_timeout_sec=args.item_timeout_sec,
        error_dir=args.error_dir,
    )

    collator = PretrainCollatorRDPNoMask(
        vocab=vocab,
        pad_id=pad_id,
        cls_id=cls_id,
        sep_id=sep_id,
        mask_id=mask_id,
        vocab_size=vocab_size,
        max_len=args.max_len,
        max_rel_dist=args.max_rel_dist,
        mlm_prob=args.mlm_prob,
        pairs_per_seq=args.pairs_per_seq,
        pair_mode=args.pair_mode,
        nontrivial_frac=args.nontrivial_frac,
        max_resample=args.max_resample,
        nontriv_bucket_min=args.nontriv_bucket_min,
        nontriv_bucket_max=args.nontriv_bucket_max,
        unk_fallback_id=unk_id,
    )

    def _build_loader():
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            timeout=(args.loader_timeout_sec if args.num_workers > 0 else 0),
        )

    loader = _build_loader()
    it = iter(loader)

    # warmup
    if is_main_process(rank):
        t0 = time.time()
        first = next(it)
        print(
            f"[WARMUP] first batch ready in {time.time()-t0:.2f}s "
            f"(Lmax={first['stats'].Lmax} mlm_n={first['stats'].mlm_n} pairs={first['stats'].pairs})"
        )
    else:
        first = next(it)

    model = build_model(
        vocab_size=vocab_size,
        max_len=args.max_len,
        pe_equals_we=args.pe_equals_we,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_rel_dist=args.max_rel_dist,
        rel_stat_every=args.rel_stat_every,
        rel_probe_every=args.rel_probe_every,
        rel_probe_id=args.rel_probe_id,
        model_mode=args.model_mode,
    )

    if args.resize_token_embeddings:
        try:
            model.resize_token_embeddings(vocab_size)
        except NotImplementedError:
            if is_main_process(rank):
                print("[WARN] resize_token_embeddings NotImplementedError -> skip")
        except Exception as e:
            if is_main_process(rank):
                print(f"[WARN] resize_token_embeddings failed -> skip ({type(e).__name__}: {e})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    if ddp_on:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup_steps = int(args.max_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, args.max_steps)

    use_fp16 = bool(args.fp16) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    global_step = 0
    micro_step = 0
    last_step_time = time.time()
    t_start = time.time()

    optimizer.zero_grad(set_to_none=True)

    pending = first
    run_hist = np.zeros((rel_num_types,), dtype=np.int64)

    loader_resets = 0

    while global_step < args.max_steps:
        try:
            batch = pending if pending is not None else next(it)
            pending = None
        except Exception as e:
            loader_resets += 1
            if is_main_process(rank):
                print(f"[LOADER-RESET {loader_resets}] {type(e).__name__}: {e}")
            append_error_line(
                args.error_dir,
                pkl_path="-",
                func_key="-",
                reason=f"LOADER_TIMEOUT_OR_ERR:{type(e).__name__}:{e}",
                T=-1,
                n_nodes=-1,
                n_edges=-1,
                max_rel_dist=args.max_rel_dist,
                item_timeout_sec=args.item_timeout_sec,
            )
            if args.loader_max_resets > 0 and loader_resets >= args.loader_max_resets:
                raise RuntimeError(f"Too many loader resets: {loader_resets}") from e
            loader = _build_loader()
            it = iter(loader)
            pending = None
            continue

        stats = batch.pop("stats")
        if stats.lab_hist is not None:
            if stats.lab_hist.shape[0] == run_hist.shape[0]:
                run_hist += stats.lab_hist
            else:
                m = min(run_hist.shape[0], stats.lab_hist.shape[0])
                run_hist[:m] += stats.lab_hist[:m]

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=use_fp16):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                rel_ids=batch["rel_ids"],
                pair_idx=batch["pair_idx"],
            )
            logits = out.logits
            rdp_logits = out.rdp_logits  # (B,P,R)

            loss_mlm = safe_ce_loss(logits, batch["labels_mlm"])
            loss_rdp = ce_loss_flat(rdp_logits, batch["pair_lab"])
            loss = (loss_mlm + loss_rdp) / args.grad_accum_steps

        scaler.scale(loss).backward()
        micro_step += 1

        if (micro_step % args.grad_accum_steps) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            now = time.time()
            step_time = now - last_step_time
            last_step_time = now

            global_batch = args.batch_size * args.grad_accum_steps * world_size
            sps = global_batch / max(step_time, 1e-6)

            if global_step % args.logging_steps == 0:
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_start

                rdp_acc1 = acc1_flat(rdp_logits.detach(), batch["pair_lab"].detach())
                labels = batch["pair_lab"].detach()

                mask_nontriv = (labels != 0) & (labels != no_edge_id)
                rdp_acc1_nontriv = acc1_masked(rdp_logits.detach(), labels, mask_nontriv)

                pred = rdp_logits.detach().argmax(dim=-1).cpu().numpy()
                lab = labels.cpu().numpy()
                buck_hit = np.zeros((12,), dtype=np.int64)
                buck_cnt = np.zeros((12,), dtype=np.int64)
                Bn, Pn = lab.shape
                for b in range(Bn):
                    for p in range(Pn):
                        bid = bucket_id(int(lab[b, p]), no_edge_id=no_edge_id)
                        buck_cnt[bid] += 1
                        if int(pred[b, p]) == int(lab[b, p]):
                            buck_hit[bid] += 1

                # DDP aggregate
                t_loss = torch.tensor(loss.item(), device=device)
                t_mlm = torch.tensor(loss_mlm.item(), device=device)
                t_rdp = torch.tensor(loss_rdp.item(), device=device)
                t_acc = torch.tensor(rdp_acc1, device=device)
                t_acc_nt = torch.tensor(rdp_acc1_nontriv, device=device)
                t_ne = torch.tensor(stats.ne_ratio, device=device)

                t_loss = ddp_all_reduce_mean(t_loss)
                t_mlm = ddp_all_reduce_mean(t_mlm)
                t_rdp = ddp_all_reduce_mean(t_rdp)
                t_acc = ddp_all_reduce_mean(t_acc)
                t_acc_nt = ddp_all_reduce_mean(t_acc_nt)
                t_ne = ddp_all_reduce_mean(t_ne)

                buck_hit_t = torch.tensor(buck_hit, device=device, dtype=torch.long)
                buck_cnt_t = torch.tensor(buck_cnt, device=device, dtype=torch.long)
                buck_hit_t = ddp_all_reduce_sum(buck_hit_t)
                buck_cnt_t = ddp_all_reduce_sum(buck_cnt_t)

                run_hist_t = torch.tensor(run_hist, device=device, dtype=torch.long)
                run_hist_t = ddp_all_reduce_sum(run_hist_t)

                if is_main_process(rank):
                    tot = int(run_hist_t.sum().item())
                    r0 = float(run_hist_t[0].item()) / max(1, tot)

                    rne = 0.0
                    if 0 <= no_edge_id < run_hist_t.numel():
                        rne = float(run_hist_t[no_edge_id].item()) / max(1, tot)

                    buck_acc = []
                    for i in range(11):
                        c = int(buck_cnt_t[i].item())
                        h = int(buck_hit_t[i].item())
                        buck_acc.append(h / max(1, c))

                    print(
                        f"[step {global_step:06d}] "
                        f"loss={t_loss.item():.4f} mlm={t_mlm.item():.4f} rdp={t_rdp.item():.4f} "
                        f"rdp@1={t_acc.item():.3f} nontriv@1={t_acc_nt.item():.3f} "
                        f"lr={lr:.2e} | "
                        f"ne_ratio={t_ne.item():.3f} | "
                        f"label_ratio:0={r0:.3f} no_edge={rne:.3f} | "
                        f"bucket_acc[1]={buck_acc[1]:.3f} [2-3]={buck_acc[2]:.3f} [4-7]={buck_acc[3]:.3f} "
                        f"[8-15]={buck_acc[4]:.3f} [16-31]={buck_acc[5]:.3f} [32-63]={buck_acc[6]:.3f} "
                        f"[64-127]={buck_acc[7]:.3f} [128-255]={buck_acc[8]:.3f} [256-512]={buck_acc[9]:.3f} "
                        f"[no_edge]={buck_acc[10]:.3f} | "
                        f"{step_time:.2f}s/step, {sps:.1f} samples/s, elapsed={elapsed/3600:.2f}h"
                    )

            if args.ckpt_steps > 0 and (global_step % args.ckpt_steps == 0):
                ckpt_dir = os.path.join(args.out, f"step{global_step}")
                save_ckpt(ckpt_dir, model, tok, optimizer, scheduler, scaler, global_step, rank=rank)

    if is_main_process(rank):
        to_save = model.module if hasattr(model, "module") else model
        to_save.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        torch.save(
            {
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            },
            os.path.join(args.out, "trainer.pt"),
        )
        print("[DONE]")

    ddp_cleanup()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pkl-dir", default="./data/BinaryCorp/train")
    ap.add_argument("--tokenizer-dir", default="./tokenizer")
    ap.add_argument("--out", default="./outputs/pretrain_rdp_t5")

    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-raw-tokens", type=int, default=400)

    ap.add_argument("--max-rel-dist", type=int, default=512)
    ap.add_argument("--model-mode", "--model_mode", dest="model_mode",
                    type=str, default="t5", choices=MODEL_MODES,
                    help="semantic-link attention mode: t5, qr, or qkr")

    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--pairs-per-seq", type=int, default=8)

    ap.add_argument("--pair-mode", type=str, default="mix", choices=["uniform", "mix"])
    ap.add_argument("--nontrivial-frac", type=float, default=0.9)
    ap.add_argument("--max-resample", type=int, default=200)

    ap.add_argument("--nontriv-bucket-min", type=int, default=2)
    ap.add_argument("--nontriv-bucket-max", type=int, default=9)

    ap.add_argument("--epochs", type=int, default=5,
                    help="recorded for reproducibility; training is controlled by --max-steps")
    ap.add_argument("--max-steps", type=int, default=680_000)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--ckpt-steps", "--save-steps", dest="ckpt_steps", type=int, default=10_000)

    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--grad-accum-steps", type=int, default=4)

    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=2)

    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)

    ap.add_argument("--pe-equals-we", dest="pe_equals_we", action="store_true", default=True)
    ap.add_argument("--no-pe-equals-we", dest="pe_equals_we", action="store_false")
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")

    ap.add_argument("--resize-token-embeddings", action="store_true", default=True)

    ap.add_argument("--rel-stat-every", type=int, default=200)
    ap.add_argument("--rel-probe-every", type=int, default=0)
    ap.add_argument("--rel-probe-id", type=int, default=0)

    ap.add_argument("--hidden-size", type=int, default=768)
    ap.add_argument("--num-layers", type=int, default=12)
    ap.add_argument("--num-heads", type=int, default=12)
    ap.add_argument("--intermediate-size", type=int, default=3072)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-longest-path", action="store_true", default=False)
    ap.add_argument("--use-operand-anchor", dest="use_operand_anchor", action="store_true", default=True)
    ap.add_argument("--no-use-operand-anchor", dest="use_operand_anchor", action="store_false")

    ap.add_argument("--item-timeout-sec", type=float, default=300.0,
                    help="timeout for processing ONE function sample (seconds); 0 disables")
    ap.add_argument("--error-dir", type=str, default="./outputs/pretrain_rdp_t5/error_logs",
                    help="directory to write per-worker error logs")

    ap.add_argument("--loader-timeout-sec", type=float, default=0.0,
                    help="timeout for DataLoader to yield ONE batch; 0 disables DataLoader timeout")
    ap.add_argument("--loader-max-resets", type=int, default=0,
                    help="0 means unlimited resets; otherwise stop after N resets")

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
