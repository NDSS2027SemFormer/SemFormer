# -*- coding: utf-8 -*-


import os
import sys
import glob
import time
import math
import pickle
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from model import BinBertModelcfg, MODEL_MODES, set_model_mode

def get_logger(log_path: str):
    logger = logging.getLogger("finetune")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

OPT_ORDER = {"O0": 0, "O1": 1, "O2": 2, "O3": 3, "Os": 3}

def compute_opt_weight(opt_a: str, opt_b: str, base: float = 1.0, hard: float = 2.0) -> float:
    a = OPT_ORDER.get(opt_a, 1)
    b = OPT_ORDER.get(opt_b, 1)
    diff = abs(a - b)
    if diff == 0:
        return base
    w = base + (hard - base) * (diff / 3.0)
    return float(w)

class WeightedInfoNCELoss(nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        a: torch.Tensor,
        p: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        B = a.size(0)
        sim = torch.matmul(a, p.t()) / self.temperature
        labels = torch.arange(B, device=a.device)
        loss_ap = F.cross_entropy(sim, labels, reduction='none')
        loss_pa = F.cross_entropy(sim.t(), labels, reduction='none')
        loss_per_sample = (loss_ap + loss_pa) / 2.0
        weighted_loss = (loss_per_sample * weights).mean()

        return weighted_loss

class ShardPairWeightedIterableDataset(IterableDataset):

    ALL_OPTS = ["O0", "O1", "O2", "O3", "Os"]

    def __init__(
        self,
        shard_glob_pattern: str,
        max_seq_len: int,
        seed: int,
        shuffle_shards: bool = True,
        weight_base: float = 1.0,
        weight_hard: float = 2.0,
    ):
        super().__init__()
        self.shard_glob_pattern = shard_glob_pattern
        self.max_seq_len = int(max_seq_len)
        self.seed = int(seed)
        self.shuffle_shards = bool(shuffle_shards)
        self.weight_base = float(weight_base)
        self.weight_hard = float(weight_hard)
        self._epoch = 0

        self.help_tokenize = None
        self.pad_rel_and_mask = None
        self.pad_rel_ids_only = None

    def set_epoch(self, e: int):
        self._epoch = int(e)

    def _init_funcs(self):
        if self.help_tokenize is None:
            from data import help_tokenize, pad_rel_and_mask, pad_rel_ids_only
            self.help_tokenize = help_tokenize
            self.pad_rel_and_mask = pad_rel_and_mask
            self.pad_rel_ids_only = pad_rel_ids_only

    def _pack_to_full_rel(self, packed):
        if isinstance(packed, np.ndarray):
            arr = packed
            if arr.shape != (self.max_seq_len, self.max_seq_len):
                full = np.full((self.max_seq_len, self.max_seq_len), 0, dtype=np.int64)
                h = min(self.max_seq_len, arr.shape[0])
                w = min(self.max_seq_len, arr.shape[1])
                full[:h, :w] = arr[:h, :w]
                arr = full
            return torch.from_numpy(arr).long()

        if isinstance(packed, (tuple, list)):
            if len(packed) == 3:
                rel_small, mask_small, no_rel_id = packed
                full_rel, _ = self.pad_rel_and_mask(
                    np.asarray(rel_small), np.asarray(mask_small),
                    self.max_seq_len, int(no_rel_id),
                )
                return torch.from_numpy(full_rel).long()
            if len(packed) == 2:
                rel_small, no_rel_id = packed
                full_rel = self.pad_rel_ids_only(
                    np.asarray(rel_small), self.max_seq_len, int(no_rel_id),
                )
                return torch.from_numpy(full_rel).long()

        raise ValueError(f"Unsupported packed_rel: type={type(packed)}")

    def _prepare_one(self, func_str: str, packed_rel):
        tok = self.help_tokenize(func_str)
        rel_ids = self._pack_to_full_rel(packed_rel)
        return tok["input_ids"], tok["attention_mask"], rel_ids

    def __iter__(self):
        self._init_funcs()

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        rng = random.Random(self.seed + 1000 * self._epoch + worker_id)

        shard_files = sorted(glob.glob(self.shard_glob_pattern))
        if self.shuffle_shards:
            rng.shuffle(shard_files)
        shard_files = shard_files[worker_id::num_workers]

        for sp in shard_files:
            with open(sp, "rb") as f:
                functions_chunk, rel_chunk, ebds_chunk = pickle.load(f)

            n_pairs = len(functions_chunk)
            for i in range(n_pairs):
                pairs = functions_chunk[i]
                rel_pairs = rel_chunk[i]

                if len(pairs) < 2 or len(rel_pairs) < 2:
                    continue
                ebd = ebds_chunk[i] if (ebds_chunk is not None and i < len(ebds_chunk)) else None
                available_opts = []
                if ebd is not None:
                    for opt in self.ALL_OPTS:
                        idx = ebd.get(opt, None)
                        if idx is not None and isinstance(idx, int) and idx < len(pairs):
                            available_opts.append((opt, idx))

                if len(available_opts) >= 2:
                    rng.shuffle(available_opts)
                    opt_a, idx_a = available_opts[0]
                    opt_b, idx_b = available_opts[1]
                    hard_opts = [(o, i) for o, i in available_opts if o == "O0"]
                    easy_opts = [(o, i) for o, i in available_opts if o != "O0"]
                    if hard_opts and easy_opts and rng.random() < 0.5:
                        opt_a, idx_a = hard_opts[0]
                        hard_pos = [(o, i) for o, i in available_opts if o in ("O3", "Os")]
                        if hard_pos:
                            opt_b, idx_b = rng.choice(hard_pos)
                        else:
                            opt_b, idx_b = rng.choice(easy_opts)

                    w = compute_opt_weight(opt_a, opt_b, self.weight_base, self.weight_hard)
                    func_a, rel_a = pairs[idx_a], rel_pairs[idx_a]
                    func_b, rel_b = pairs[idx_b], rel_pairs[idx_b]
                else:
                    pos = rng.randint(0, len(pairs) - 1)
                    pos2 = rng.randint(0, len(pairs) - 1)
                    while pos2 == pos and len(pairs) > 1:
                        pos2 = rng.randint(0, len(pairs) - 1)
                    w = 1.0
                    func_a, rel_a = pairs[pos], rel_pairs[pos]
                    func_b, rel_b = pairs[pos2], rel_pairs[pos2]

                seq1, mask1, rel1 = self._prepare_one(func_a, rel_a)
                seq2, mask2, rel2 = self._prepare_one(func_b, rel_b)

                yield seq1, seq2, mask1, mask2, rel1, rel2, torch.tensor(w, dtype=torch.float32)

class ShardPairIterableDataset(IterableDataset):
    def __init__(self, shard_glob_pattern, max_seq_len, seed, shuffle_shards=False):
        super().__init__()
        self.shard_glob_pattern = shard_glob_pattern
        self.max_seq_len = int(max_seq_len)
        self.seed = int(seed)
        self.shuffle_shards = bool(shuffle_shards)
        self._epoch = 0
        self.help_tokenize = None
        self.pad_rel_and_mask = None
        self.pad_rel_ids_only = None

    def set_epoch(self, e):
        self._epoch = int(e)

    def _init_funcs(self):
        if self.help_tokenize is None:
            from data import help_tokenize, pad_rel_and_mask, pad_rel_ids_only
            self.help_tokenize = help_tokenize
            self.pad_rel_and_mask = pad_rel_and_mask
            self.pad_rel_ids_only = pad_rel_ids_only

    def _pack_to_full_rel(self, packed):
        if isinstance(packed, np.ndarray):
            arr = packed
            if arr.shape != (self.max_seq_len, self.max_seq_len):
                full = np.full((self.max_seq_len, self.max_seq_len), 0, dtype=np.int64)
                h = min(self.max_seq_len, arr.shape[0])
                w = min(self.max_seq_len, arr.shape[1])
                full[:h, :w] = arr[:h, :w]
                arr = full
            return torch.from_numpy(arr).long()
        if isinstance(packed, (tuple, list)):
            if len(packed) == 3:
                rel_small, mask_small, no_rel_id = packed
                full_rel, _ = self.pad_rel_and_mask(
                    np.asarray(rel_small), np.asarray(mask_small),
                    self.max_seq_len, int(no_rel_id),
                )
                return torch.from_numpy(full_rel).long()
            if len(packed) == 2:
                rel_small, no_rel_id = packed
                full_rel = self.pad_rel_ids_only(
                    np.asarray(rel_small), self.max_seq_len, int(no_rel_id),
                )
                return torch.from_numpy(full_rel).long()
        raise ValueError(f"Unsupported packed_rel: type={type(packed)}")

    def _prepare_one(self, func_str, packed_rel):
        tok = self.help_tokenize(func_str)
        rel_ids = self._pack_to_full_rel(packed_rel)
        return tok["input_ids"], tok["attention_mask"], rel_ids

    def __iter__(self):
        self._init_funcs()
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        rng = random.Random(self.seed + 1000 * self._epoch + worker_id)
        shard_files = sorted(glob.glob(self.shard_glob_pattern))
        if self.shuffle_shards:
            rng.shuffle(shard_files)
        shard_files = shard_files[worker_id::num_workers]

        for sp in shard_files:
            with open(sp, "rb") as f:
                functions_chunk, rel_chunk, _ = pickle.load(f)
            for i in range(len(functions_chunk)):
                pairs = functions_chunk[i]
                rel_pairs = rel_chunk[i]
                if len(pairs) < 2 or len(rel_pairs) < 2:
                    continue
                a = rng.randint(0, len(pairs) - 1)
                b = rng.randint(0, len(pairs) - 1)
                while b == a:
                    b = rng.randint(0, len(pairs) - 1)
                seq1, mask1, rel1 = self._prepare_one(pairs[a], rel_pairs[a])
                seq2, mask2, rel2 = self._prepare_one(pairs[b], rel_pairs[b])
                yield seq1, seq2, mask1, mask2, rel1, rel2

def _load_pair_count_from_one_shard(shard_path):
    with open(shard_path, "rb") as f:
        functions_chunk, _, _ = pickle.load(f)
    return int(len(functions_chunk))

def estimate_total_pairs(shard_glob, logger=None):
    files = sorted(glob.glob(shard_glob))
    if len(files) == 0:
        return 0
    if len(files) == 1:
        return _load_pair_count_from_one_shard(files[0])
    t0 = time.time()
    per = _load_pair_count_from_one_shard(files[0])
    last = _load_pair_count_from_one_shard(files[-1])
    total = per * (len(files) - 1) + last
    if logger:
        logger.info(
            f"[len/estimate] shards={len(files)} per={per} last={last} "
            f"total={total} time={time.time()-t0:.2f}s"
        )
    return int(total)

def strict_total_pairs(shard_glob, logger=None):
    files = sorted(glob.glob(shard_glob))
    if len(files) == 0:
        return 0
    t0 = time.time()
    total = sum(_load_pair_count_from_one_shard(sp) for sp in files)
    if logger:
        logger.info(f"[len/strict] shards={len(files)} total={total} time={time.time()-t0:.2f}s")
    return int(total)

def load_state_dict_any(path_or_dir):
    if os.path.isdir(path_or_dir):
        bin_path = os.path.join(path_or_dir, "pytorch_model.bin")
        st_path = os.path.join(path_or_dir, "model.safetensors")
        if os.path.exists(st_path):
            try:
                from safetensors.torch import load_file
                return load_file(st_path)
            except Exception:
                pass
        if os.path.exists(bin_path):
            return torch.load(bin_path, map_location="cpu")
        cands = glob.glob(os.path.join(path_or_dir, "*.bin"))
        if cands:
            return torch.load(cands[0], map_location="cpu")
        raise FileNotFoundError(f"No checkpoint found under {path_or_dir}")
    else:
        if path_or_dir.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path_or_dir)
        return torch.load(path_or_dir, map_location="cpu")

def clean_pretrain_state_dict_for_bert(state_dict):
    clean = {}
    for k, v in state_dict.items():
        if k.startswith("bert."):
            clean[k[len("bert."):]] = v
        elif k.startswith("cls."):
            continue
        else:
            clean[k] = v
    keys_to_del = [k for k in list(clean.keys()) if "position_embeddings" in k or "position_ids" in k]
    for k in keys_to_del:
        del clean[k]
    return clean

def freeze_layers_and_unfreeze_rel(model, freeze_cnt, logger):
    if freeze_cnt <= 0:
        return
    logger.info(f"[freeze] Freezing first {freeze_cnt} layers...")
    for layer in model.encoder.layer[:freeze_cnt]:
        for p in layer.parameters():
            p.requires_grad = False
    unfrozen = 0
    for i in range(freeze_cnt):
        lyr = model.encoder.layer[i]
        if hasattr(lyr.attention.self, "rel_embeddings") and lyr.attention.self.rel_embeddings is not None:
            lyr.attention.self.rel_embeddings.weight.requires_grad = True
            unfrozen += 1
    logger.info(f"[freeze] Unfroze rel_embeddings in {unfrozen}/{freeze_cnt} frozen layers.")

@torch.no_grad()
def finetune_eval(model, data_loader, device, total_steps=None, max_batches=-1):
    model.eval()
    mrr_list = []
    bar_fmt = "{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    it = tqdm(data_loader, desc="Evaluating", ncols=120, total=total_steps, bar_format=bar_fmt)

    for step, batch in enumerate(it):
        if max_batches > 0 and step >= max_batches:
            break
        if len(batch) == 6:
            seq1, seq2, mask1, mask2, rel1, rel2 = batch
        else:
            raise ValueError(f"Unexpected eval batch length: {len(batch)}")

        out1 = model(input_ids=seq1.to(device), attention_mask=mask1.to(device), rel_ids=rel1.to(device))
        out2 = model(input_ids=seq2.to(device), attention_mask=mask2.to(device), rel_ids=rel2.to(device))

        a = F.normalize(out1.last_hidden_state[:, 0, :], p=2, dim=1)
        p = F.normalize(out2.last_hidden_state[:, 0, :], p=2, dim=1)

        sim = torch.matmul(a, p.t()).detach().cpu().numpy()
        B = sim.shape[0]
        for i in range(B):
            order = np.argsort(-sim[i])
            rank = int(np.where(order == i)[0][0]) + 1
            mrr_list.append(1.0 / rank)

    return float(np.mean(np.asarray(mrr_list))) if mrr_list else 0.0
def train_loop(model, args, train_loader, valid_loader, logger, device,
               train_total_steps=None, eval_total_steps=None):
    infonce_loss_fn = WeightedInfoNCELoss(temperature=args.temperature)

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(params, lr=args.lr)

    use_amp = bool(args.fp16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum_steps = max(1, int(args.gradient_accumulation_steps))
    effective_batch = args.batch_size * accum_steps
    logger.info(f"[train] batch_size={args.batch_size}, accum_steps={accum_steps}, effective_batch={effective_batch}")
    logger.info(f"[train] temperature={args.temperature}, weight_base={args.weight_base}, weight_hard={args.weight_hard}")

    global_step = 0
    micro_step = 0

    bar_fmt = "{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"

    for epoch in range(args.epoch):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_w_mean = 0.0
        epoch_total_steps = (train_total_steps // accum_steps) if train_total_steps else None
        if args.max_train_batches > 0:
            epoch_total_steps = min(epoch_total_steps or 999999, args.max_train_batches // accum_steps)

        it = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epoch}",
            ncols=120,
            total=train_total_steps,
            bar_format=bar_fmt,
        )

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(it):
            if args.max_train_batches > 0 and step >= args.max_train_batches:
                break
            seq1, seq2, mask1, mask2, rel1, rel2, opt_weights = batch

            seq1 = seq1.to(device, non_blocking=True)
            seq2 = seq2.to(device, non_blocking=True)
            mask1 = mask1.to(device, non_blocking=True)
            mask2 = mask2.to(device, non_blocking=True)
            rel1 = rel1.to(device, non_blocking=True)
            rel2 = rel2.to(device, non_blocking=True)
            opt_weights = opt_weights.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out1 = model(input_ids=seq1, attention_mask=mask1, rel_ids=rel1)
                out2 = model(input_ids=seq2, attention_mask=mask2, rel_ids=rel2)
                a = F.normalize(out1.last_hidden_state[:, 0, :], p=2, dim=1)
                p = F.normalize(out2.last_hidden_state[:, 0, :], p=2, dim=1)
                loss = infonce_loss_fn(a, p, opt_weights) / accum_steps

            scaler.scale(loss).backward()
            micro_step += 1
            running_loss += loss.item() * accum_steps
            running_w_mean += opt_weights.mean().item()
            if micro_step % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = running_loss / (args.log_every * accum_steps)
                    avg_w = running_w_mean / (args.log_every * accum_steps)
                    lr = optimizer.param_groups[0]["lr"]
                    it.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "w_mean": f"{avg_w:.2f}",
                        "lr": f"{lr:.2e}",
                        "step": global_step,
                    })
                    running_loss = 0.0
                    running_w_mean = 0.0
        if micro_step % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        logger.info(f"[train] Epoch {epoch+1} finished. global_step={global_step} time={(time.time()-t0)/60:.1f} min")
        if valid_loader is not None and (epoch + 1) % args.eval_every == 0:
            mrr = finetune_eval(
                model, valid_loader, device,
                total_steps=eval_total_steps,
                max_batches=args.eval_max_batches,
            )
            logger.info(f"[eval] Epoch {epoch+1}/{args.epoch}: MRR={mrr:.4f}")
        if (epoch + 1) % args.save_every == 0:
            save_dir = os.path.join(args.output_path, f"finetune_epoch_{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(save_dir)
            logger.info(f"[save] Saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser("finetune - InfoNCE with hard positive weighting")
    parser.add_argument("--model_path", type=str, default="../checkpoints/semformer-rdp-pretrain")
    parser.add_argument("--output_path", type=str, default="../outputs/finetune_rdp_shortest_bs64")
    parser.add_argument("--tokenizer_dir", type=str, default="./tokenizer")
    parser.add_argument("--train_shard_glob", type=str, default="../data/rel_shards_train_shortest/rel_shard_*.pkl")
    parser.add_argument("--eval_shard_glob", type=str, default="../data/rel_shards_test_shortest/rel_shard_*.pkl")
    parser.add_argument("--max_rel_dist", type=int, default=512)
    parser.add_argument("--model-mode", "--model_mode", dest="model_mode",
                        type=str, default="t5", choices=MODEL_MODES,
                        help="semantic-link attention mode: t5, qr, or qkr")
    parser.add_argument("--temperature", type=float, default=0.03,
                        help="InfoNCE 温度参数，越小对比越尖锐")
    parser.add_argument("--weight_base", type=float, default=1.0,
                        help="相邻优化级别（如 O0-O1）的基础权重")
    parser.add_argument("--weight_hard", type=float, default=3.0,
                        help="困难正样本对（如 O0-O3/Os）的权重上限")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="梯度累积步数，等效 batch = batch_size * gradient_accumulation_steps")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", "--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--eval_max_batches", type=int, default=-1)
    parser.add_argument("--freeze_cnt", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no_shuffle_shards", action="store_true")
    parser.add_argument("--use_dp", action="store_true")
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true", default=True,
                        help="开启后可降低显存占用，代价是训练速度下降")
    parser.add_argument("--no_gradient_checkpointing", "--no-gradient-checkpointing",
                        dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--len_mode", type=str, default="estimate", choices=["estimate", "strict"])

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    torch.multiprocessing.set_sharing_strategy("file_system")

    if os.path.isdir(args.tokenizer_dir):
        os.environ.setdefault("SEMFORMER_TOKENIZER_DIR", os.path.abspath(args.tokenizer_dir))

    from datetime import datetime
    log_path = os.path.join(args.output_path, f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = get_logger(log_path)
    logger.info("===== finetune start =====")
    logger.info(str(vars(args)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from transformers import BertConfig
    config = BertConfig.from_pretrained(args.model_path)
    config.max_rel_dist = int(args.max_rel_dist)
    config.rel_num_types = int(args.max_rel_dist) + 2
    config.no_edge_id = int(args.max_rel_dist) + 1
    config.no_rel_id = int(args.max_rel_dist) + 1
    config.use_cfg_rel = True
    set_model_mode(config, args.model_mode)

    model = BinBertModelcfg(config, add_pooling_layer=False)

    sd = load_state_dict_any(args.model_path)
    sd = clean_pretrain_state_dict_for_bert(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"[ckpt] loaded. missing={len(missing)} unexpected={len(unexpected)}")

    if hasattr(model, "embeddings") and hasattr(model.embeddings, "position_embeddings"):
        with torch.no_grad():
            model.embeddings.position_embeddings.weight.zero_()
        model.embeddings.position_embeddings.weight.requires_grad = False

    freeze_layers_and_unfreeze_rel(model, args.freeze_cnt, logger)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("[mem] gradient_checkpointing enabled: saves ~40-60% VRAM, ~30% slower")

    model.to(device)

    if args.use_dp and torch.cuda.device_count() > 1:
        logger.info(f"[DP] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    if args.len_mode == "strict":
        train_pairs = strict_total_pairs(args.train_shard_glob, logger)
    else:
        train_pairs = estimate_total_pairs(args.train_shard_glob, logger)

    train_total_steps = train_pairs // args.batch_size
    if args.max_train_batches > 0:
        train_total_steps = min(train_total_steps, args.max_train_batches)
    logger.info(
        f"[ETA/train] total_pairs={train_pairs}, total_micro_steps={train_total_steps}, "
        f"total_optim_steps={train_total_steps // args.gradient_accumulation_steps}"
    )

    train_ds = ShardPairWeightedIterableDataset(
        shard_glob_pattern=args.train_shard_glob,
        max_seq_len=512,
        seed=args.seed,
        shuffle_shards=(not args.no_shuffle_shards),
        weight_base=args.weight_base,
        weight_hard=args.weight_hard,
    )

    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        drop_last=True,
    )
    if args.num_workers and args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, **dl_kwargs)

    valid_loader = None
    eval_total_steps = None
    if args.eval_shard_glob:
        if args.len_mode == "strict":
            eval_pairs = strict_total_pairs(args.eval_shard_glob, logger)
        else:
            eval_pairs = estimate_total_pairs(args.eval_shard_glob, logger)

        eval_total_steps = math.ceil(eval_pairs / args.eval_batch_size)
        if args.eval_max_batches > 0:
            eval_total_steps = min(eval_total_steps, args.eval_max_batches)

        logger.info(f"[ETA/eval] total_pairs={eval_pairs}, total_steps={eval_total_steps}")

        valid_ds = ShardPairIterableDataset(
            shard_glob_pattern=args.eval_shard_glob,
            max_seq_len=512,
            seed=args.seed,
            shuffle_shards=False,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=bool(args.pin_memory),
            drop_last=False,
        )

    train_loop(
        model, args, train_loader, valid_loader, logger, device,
        train_total_steps=train_total_steps,
        eval_total_steps=eval_total_steps,
    )

    logger.info("===== finetune done =====")

if __name__ == "__main__":
    main()



