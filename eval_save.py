#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertConfig

from data import help_tokenize, load_paired_data_with_rel, pad_rel_ids_only
from model import BinBertModelcfg, MODEL_MODES, set_model_mode


def get_logger(log_path: str):
    logger = logging.getLogger("eval_save")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def load_model_state(path_or_dir):
    ckpt_path = os.path.join(path_or_dir, "pytorch_model.bin")
    if os.path.exists(ckpt_path):
        raw_state = torch.load(ckpt_path, map_location="cpu")
    else:
        raw_state = torch.load(path_or_dir, map_location="cpu")

    state = {}
    for key, value in raw_state.items():
        if key.startswith("bert."):
            state[key[len("bert."):]] = value
        elif key.startswith("cls."):
            continue
        else:
            state[key] = value

    for key in [k for k in list(state.keys()) if "position_embeddings" in k or "position_ids" in k]:
        del state[key]
    return state


def infer_no_rel_id(rel_masks, fallback: int) -> int:
    for per_pair in rel_masks:
        if not per_pair:
            continue
        for packed in per_pair:
            if packed is None or len(packed) < 3:
                continue
            no_rel_id = packed[2]
            if no_rel_id is not None:
                return int(no_rel_id)
    return int(fallback)


def build_arg_parser():
    parser = argparse.ArgumentParser("Export SemFormer embeddings for retrieval evaluation")
    parser.add_argument("--model-dir", required=True, help="checkpoint directory or pytorch_model.bin path")
    parser.add_argument("--dataset-dir", required=True, help="BinaryCorp evaluation data directory")
    parser.add_argument("--out-pkl", required=True, help="output pickle path for exported embeddings")
    parser.add_argument("--cache-file", default="", help="optional processed dataset cache")
    parser.add_argument("--model-mode", "--model_mode", dest="model_mode", default="t5", choices=MODEL_MODES)
    parser.add_argument("--max-rel-dist", type=int, default=512)
    parser.add_argument("--max-raw-tokens", type=int, default=509)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--use-longest-path", action="store_true")
    parser.add_argument("--log-file", default="")
    return parser


def main():
    args = build_arg_parser().parse_args()

    out_dir = os.path.dirname(args.out_pkl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    log_file = args.log_file or f"eval_save_{datetime.now().strftime('%Y%m%d%H%M')}.log"
    logger = get_logger(log_file)

    config = BertConfig.from_pretrained(args.model_dir)
    expected_no_edge = int(args.max_rel_dist + 1)
    config.no_edge_id = int(getattr(config, "no_edge_id", expected_no_edge))
    config.no_rel_id = int(config.no_edge_id)
    config.rel_num_types = int(getattr(config, "rel_num_types", args.max_rel_dist + 2))
    config.use_cfg_rel = True
    set_model_mode(config, args.model_mode)

    if int(config.rel_num_types) <= int(config.no_edge_id):
        raise ValueError(
            f"Bad config: rel_num_types={int(config.rel_num_types)} "
            f"must be greater than no_edge_id={int(config.no_edge_id)}"
        )

    cache_file = args.cache_file or os.path.join(args.dataset_dir, "cached_rel_data_processed.pkl")
    opt_list = ["O0", "O1", "O2", "O3", "Os"]

    if os.path.exists(cache_file):
        logger.info(f"Loading processed data from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            functions, rel_masks, ebds = pickle.load(f)
    else:
        logger.info(f"Building processed data from: {args.dataset_dir}")
        functions, rel_masks, ebds = load_paired_data_with_rel(
            datapath=args.dataset_dir,
            filt=None,
            alldata=True,
            convert_jump=False,
            opt=opt_list,
            add_ebd=True,
            max_rel_dist=args.max_rel_dist,
            max_raw_tokens=args.max_raw_tokens,
            num_workers=args.num_workers,
            chunksize=args.chunksize,
            no_call_cti=True,
            use_mutex=True,
            use_operand_anchor=True,
            use_longest_path=args.use_longest_path,
        )
        try:
            with open(cache_file, "wb") as f:
                pickle.dump((functions, rel_masks, ebds), f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.warning(f"Failed to save cache {cache_file}: {exc}")

    data_no_rel = infer_no_rel_id(rel_masks, fallback=args.max_rel_dist + 1)
    ckpt_no_edge = int(config.no_edge_id)

    model = BinBertModelcfg(config, add_pooling_layer=False)
    state_dict = load_model_state(args.model_dir)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

    if hasattr(model, "embeddings") and hasattr(model.embeddings, "position_embeddings"):
        model.embeddings.position_embeddings.weight.data.zero_()
        model.embeddings.position_embeddings.weight.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    torch.set_grad_enabled(False)
    for i in tqdm(range(len(functions)), desc="Embedding Generation", dynamic_ncols=True):
        pairs = functions[i]
        rel_pair = rel_masks[i]
        meta = ebds[i]

        for opt in opt_list:
            if meta.get(opt) is None:
                continue

            idx = meta[opt]
            if isinstance(idx, torch.Tensor):
                continue

            func_str = pairs[idx]
            rel_packed, _mask_packed, _no_rel_id = rel_pair[idx]
            rel_arr = rel_packed.astype(np.int64)

            if int(data_no_rel) != ckpt_no_edge:
                rel_arr[rel_arr == int(data_no_rel)] = ckpt_no_edge

            full_rel = pad_rel_ids_only(rel_arr, max_seq_len=512, no_relation_id=ckpt_no_edge)
            ret = help_tokenize(func_str)

            out = model(
                input_ids=ret["input_ids"].unsqueeze(0).to(device),
                attention_mask=ret["attention_mask"].unsqueeze(0).to(device),
                rel_ids=torch.from_numpy(full_rel).long().unsqueeze(0).to(device),
                output_attentions=False,
            )

            meta[opt] = out.last_hidden_state[:, 0, :].detach().cpu()

    logger.info(f"Writing embeddings to: {args.out_pkl}")
    with open(args.out_pkl, "wb") as f:
        pickle.dump(ebds, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
