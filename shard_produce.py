#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
from multiprocessing import Pool

from tqdm import tqdm

from data import _iter_pairs, _make_datasetbase, _process_one_pair


def normalize_rel_entry(entry):
    if isinstance(entry, (tuple, list)) and len(entry) == 3:
        rel_small, _mask_small, no_rel_id = entry
        return rel_small, int(no_rel_id)
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        rel_small, no_rel_id = entry
        return rel_small, int(no_rel_id)

    size = len(entry) if hasattr(entry, "__len__") else "NA"
    raise ValueError(f"Unexpected rel entry format: type={type(entry)} len={size}")


def process_tasks(tasks, pool, chunksize, pbar=None):
    functions_chunk = []
    rel_chunk = []
    ebds_chunk = []
    total_functions = 0

    for _idx, local_funcs, local_rels, local_ebd, count in pool.imap_unordered(
        _process_one_pair,
        tasks,
        chunksize=chunksize,
    ):
        if pbar is not None:
            pbar.update(1)

        if count == 0:
            continue

        functions_chunk.append(local_funcs)
        rel_chunk.append([normalize_rel_entry(item) for item in local_rels])
        ebds_chunk.append(local_ebd)
        total_functions += count

    return functions_chunk, rel_chunk, ebds_chunk, total_functions


def build_tasks(pair_iter, opt, args):
    for idx, item in enumerate(pair_iter):
        proj, funcname, funcs_dict = item[0], item[1], item[2]
        yield (
            idx,
            proj,
            funcname,
            funcs_dict,
            opt,
            True,
            args.max_rel_dist,
            args.max_raw_tokens,
            args.no_call_cti,
            args.use_mutex,
            args.use_operand_anchor,
            args.use_longest_path,
        )


def write_shard(out_dir, shard_idx, functions_chunk, rel_chunk, ebds_chunk):
    shard_path = os.path.join(out_dir, f"rel_shard_{shard_idx:03d}.pkl")
    with open(shard_path, "wb") as f:
        pickle.dump((functions_chunk, rel_chunk, ebds_chunk), f, protocol=pickle.HIGHEST_PROTOCOL)
    return shard_path


def build_rel_shards(args):
    os.makedirs(args.out_dir, exist_ok=True)
    dataset = _make_datasetbase(args.data_path, filt=None, alldata=True, convert_jump=False, opt=args.opt)
    tasks_iter = build_tasks(_iter_pairs(dataset), args.opt, args)

    shard_idx = 0
    total_pairs = 0
    total_functions = 0
    pending = []

    with Pool(processes=args.num_workers) as pool:
        pbar = tqdm(desc="Build shards", dynamic_ncols=True)

        for task in tasks_iter:
            pending.append(task)
            if len(pending) < args.max_pairs_per_shard:
                continue

            funcs, rels, ebds, count = process_tasks(pending, pool, args.chunksize, pbar)
            path = write_shard(args.out_dir, shard_idx, funcs, rels, ebds)
            print(f"[shard] {path} pairs={len(funcs)} functions={count}")

            total_pairs += len(funcs)
            total_functions += count
            shard_idx += 1
            pending = []

        if pending:
            funcs, rels, ebds, count = process_tasks(pending, pool, args.chunksize, pbar)
            path = write_shard(args.out_dir, shard_idx, funcs, rels, ebds)
            print(f"[shard] {path} pairs={len(funcs)} functions={count}")

            total_pairs += len(funcs)
            total_functions += count
            shard_idx += 1

        pbar.close()

    print(f"[done] shards={shard_idx} pairs={total_pairs} functions={total_functions}")


def build_arg_parser():
    parser = argparse.ArgumentParser("Build fine-tuning shards for SemFormer")
    parser.add_argument("--data-path", required=True, help="BinaryCorp training data directory")
    parser.add_argument("--out-dir", required=True, help="output directory for shard pkl files")
    parser.add_argument("--opt", nargs="+", default=["O0", "O1", "O2", "O3", "Os"])
    parser.add_argument("--max-rel-dist", type=int, default=512)
    parser.add_argument("--max-raw-tokens", type=int, default=509)
    parser.add_argument("--max-pairs-per-shard", type=int, default=6000)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--no-call-cti", action="store_true", default=True)
    parser.add_argument("--use-mutex", action="store_true", default=True)
    parser.add_argument("--use-operand-anchor", action="store_true", default=True)
    parser.add_argument("--use-longest-path", action="store_true")
    return parser


if __name__ == "__main__":
    build_rel_shards(build_arg_parser().parse_args())
