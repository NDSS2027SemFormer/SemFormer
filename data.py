
import os
import random
import pickle
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional, Iterable

import numpy as np
import torch
import networkx as nx
from tqdm import tqdm

from datautils.playdata import DatasetBase as DatasetBase
import readidadata

MAXLEN = 512

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_TOKENIZER_DIR = os.environ.get("SEMFORMER_TOKENIZER_DIR", "")
_vocab_candidates = [
    os.path.join(_TOKENIZER_DIR, "vocab.txt") if _TOKENIZER_DIR else "",
    os.path.join(_CODE_DIR, "tokenizer", "vocab.txt"),
    os.path.join(os.getcwd(), "tokenizer", "vocab.txt"),
]
_vocab_path = next((p for p in _vocab_candidates if p and os.path.exists(p)), None)
if _vocab_path is None:
    raise FileNotFoundError(
        "Cannot find vocab.txt. Set SEMFORMER_TOKENIZER_DIR or place vocab.txt "
        "under code/tokenizer/."
    )
vocab_data = open(_vocab_path).read().strip().split("\n") + ["[SEP]", "[PAD]", "[CLS]", "[MASK]"]
my_vocab = defaultdict(lambda: 512, {vocab_data[i]: i for i in range(len(vocab_data))})


def help_tokenize(line: str):
    split_line = line.strip().split(" ") if line else []
    if len(split_line) <= 509:
        split_line = ["[CLS]"] + split_line + ["[SEP]"]
        attention_mask = [1] * len(split_line) + [0] * (512 - len(split_line))
        split_line = split_line + (512 - len(split_line)) * ["[PAD]"]
    else:
        split_line = ["[CLS]"] + split_line[:510] + ["[SEP]"]
        attention_mask = [1] * 512

    input_ids = [my_vocab[e] for e in split_line]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def pad_rel_and_mask(rel_small: np.ndarray,
                     mask_small: np.ndarray,
                     max_seq_len: int = 512,
                     no_relation_id: int = 0):
    L = int(rel_small.shape[0])

    if L > max_seq_len - 2:
        L = max_seq_len - 2
        rel_small = rel_small[:L, :L]
        mask_small = mask_small[:L, :L]

    full_rel = np.full((max_seq_len, max_seq_len), no_relation_id, dtype=np.int64)
    full_mask = np.zeros((max_seq_len, max_seq_len), dtype=np.int64)

    full_rel[1:L+1, 1:L+1] = rel_small
    full_mask[1:L+1, 1:L+1] = mask_small

    full_mask[0, :L+2] = 1
    full_mask[:L+2, 0] = 1
    full_rel[0, :L+2] = 0
    full_rel[:L+2, 0] = 0

    full_rel[L+1, :L+2] = 0
    full_rel[:L+2, L+1] = 0

    return full_rel, full_mask


def _extract_cfg_and_asm_source(f) -> Tuple[nx.DiGraph, Optional[Dict[Any, List[Tuple[str, str]]]]]:
    if not isinstance(f, (tuple, list)) or len(f) < 4:
        raise ValueError(
            f"Unexpected function object type/len: type={type(f)} len={len(f) if hasattr(f,'__len__') else 'NA'}"
        )

    cfg = f[3]
    if not hasattr(cfg, "nodes") or not hasattr(cfg, "edges"):
        raise TypeError(f"f[3] is not a networkx graph: type(f[3])={type(cfg)}")

    asm_dict = None
    if isinstance(f[2], dict):
        asm_dict = f[2]

    return cfg, asm_dict


def _is_cti_op(op: str, no_call_cti: bool = True) -> bool:
    if not op:
        return False
    op = op.lower()
    if op == "ret":
        return True
    if op.startswith("j"):
        return True
    if op == "call":
        return (not no_call_cti)
    return False


def _parse_target_addr(tok: Any) -> Optional[int]:
    if tok is None:
        return None
    if isinstance(tok, int):
        return tok
    if not isinstance(tok, str):
        return None
    s = tok.strip()
    if s.startswith("hex_"):
        try:
            return int(s[4:], 16)
        except Exception:
            return None
    if s.startswith("0x"):
        try:
            return int(s, 16)
        except Exception:
            return None
    return None


def _find_operand_anchor(u, v, bb2idxs, tokens=None):
    if tokens is None:
        return None
    if isinstance(v, int):
        targets = {hex(v), "hex_" + hex(v)[2:]}
    else:
        targets = {str(v)}
    for idx in reversed(sorted(bb2idxs.get(u, []))):
        t = tokens[idx]
        if isinstance(t, str) and t.strip() in targets:
            return idx
    return None


def linearize_cfg_no_dup(f, no_call_cti: bool = True):
    cfg, asm_dict = _extract_cfg_and_asm_source(f)

    tokens: List[str] = []
    bb_ids: List[Any] = []
    pos_in_bb: List[int] = []

    bb_last_cti_tok: Dict[Any, int] = {}
    bb_last_cti_tgt: Dict[Any, Any] = {}

    cti_operand_edges: List[Tuple[int, Any]] = []

    for bb in sorted(cfg.nodes):
        local_pos = 0
        last_op_tok_idx = None
        last_tgt = None

        if asm_dict is not None:
            insts = asm_dict.get(bb, [])
            for (op, operand) in insts:
                op = op if op is not None else ""
                operand = operand if operand is not None else ""

                tokens.append(op)
                bb_ids.append(bb)
                pos_in_bb.append(local_pos)
                op_tok_idx = len(tokens) - 1
                last_op_tok_idx = op_tok_idx
                local_pos += 1

                operand_tok_idx = None
                if operand != "":
                    tokens.append(operand)
                    bb_ids.append(bb)
                    pos_in_bb.append(local_pos)
                    operand_tok_idx = len(tokens) - 1
                    local_pos += 1

                if _is_cti_op(op, no_call_cti=no_call_cti) and operand_tok_idx is not None:
                    tgt = _parse_target_addr(operand)
                    if tgt is not None and tgt in cfg.nodes:
                        cti_operand_edges.append((operand_tok_idx, tgt))
                        last_tgt = tgt

        else:
            asm_list = cfg.nodes[bb].get("asm", [])
            for code in asm_list:
                op, o1, o2, o3, ann = readidadata.parse_asm(code)

                tokens.append(op)
                bb_ids.append(bb)
                pos_in_bb.append(local_pos)
                op_tok_idx = len(tokens) - 1
                last_op_tok_idx = op_tok_idx
                local_pos += 1

                operands = []
                operand_tok_indices = []
                for o in (o1, o2, o3):
                    if o is not None:
                        operands.append(o)
                        tokens.append(o)
                        bb_ids.append(bb)
                        pos_in_bb.append(local_pos)
                        operand_tok_indices.append(len(tokens) - 1)
                        local_pos += 1

                if _is_cti_op(op, no_call_cti=no_call_cti) and operand_tok_indices:
                    tgt = None
                    tgt_operand_tok_idx = None
                    for tok_idx, o in zip(operand_tok_indices, operands):
                        t = _parse_target_addr(o)
                        if t is not None:
                            tgt = t
                            tgt_operand_tok_idx = tok_idx
                            break
                    if tgt is not None and tgt_operand_tok_idx is not None and tgt in cfg.nodes:
                        cti_operand_edges.append((tgt_operand_tok_idx, tgt))
                        last_tgt = tgt

        if last_op_tok_idx is not None:
            bb_last_cti_tok[bb] = last_op_tok_idx
            if last_tgt is not None:
                bb_last_cti_tgt[bb] = last_tgt

    return tokens, bb_ids, pos_in_bb, bb_last_cti_tok, bb_last_cti_tgt, cti_operand_edges


def _compute_branch_mutex_regions(cfg: nx.DiGraph):
    mutex_pairs = []
    for s in cfg.nodes:
        succs = list(cfg.successors(s))
        if len(succs) < 2:
            continue

        a, b = succs[0], succs[1]

        dist_from_a = dict(nx.single_source_shortest_path_length(cfg, a))
        dist_from_b = dict(nx.single_source_shortest_path_length(cfg, b))
        dist_from_s = dict(nx.single_source_shortest_path_length(cfg, s))

        reachA = set(dist_from_a.keys())
        reachB = set(dist_from_b.keys())
        inter = (reachA & reachB)
        inter.discard(s)

        if not inter:
            continue

        merge = None
        best = 1 << 30
        for m in inter:
            dsm = dist_from_s.get(m, None)
            if dsm is None:
                continue
            if dsm < best:
                best = dsm
                merge = m
        if merge is None:
            continue

        da_merge = dist_from_a.get(merge, None)
        db_merge = dist_from_b.get(merge, None)
        if da_merge is None or db_merge is None:
            continue

        regionA = {x for x, d in dist_from_a.items() if d < da_merge}
        regionB = {x for x, d in dist_from_b.items() if d < db_merge}

        if regionA and regionB:
            mutex_pairs.append((regionA, regionB))

    return mutex_pairs


def _is_mutex(bb_i, bb_j, mutex_regions) -> bool:
    if bb_i == bb_j:
        return False
    for regionA, regionB in mutex_regions:
        if (bb_i in regionA and bb_j in regionB) or (bb_i in regionB and bb_j in regionA):
            return True
    return False


def compute_cfg_token_rel_and_mask(
    bb_ids, cfg, max_rel_dist: int,
    bb_last_cti_tok=None,
    use_mutex: bool = True,
    tokens=None,
    use_operand_anchor: bool = True,
    cti_operand_edges: Optional[bool] = None,
    **kwargs,
):
    L = len(bb_ids)
    no_relation_id = max_rel_dist + 1

    bb2idxs = defaultdict(list)
    for idx, bb in enumerate(bb_ids):
        bb2idxs[bb].append(idx)

    adj = [set() for _ in range(L)]

    for bb, idxs in bb2idxs.items():
        idxs_sorted = sorted(idxs)
        for a, b in zip(idxs_sorted, idxs_sorted[1:]):
            adj[a].add(b)
            adj[b].add(a)
    for u, v in cfg.edges():
        if u not in bb2idxs or v not in bb2idxs:
            continue

        first_v = min(bb2idxs[v])
        anchor_u = None
        anchor_src = None

        if use_operand_anchor and tokens is not None:
            anchor_u = _find_operand_anchor(u, v, bb2idxs, tokens=tokens)
            if anchor_u is not None:
                anchor_src = "operand"

        if anchor_u is None and bb_last_cti_tok is not None and u in bb_last_cti_tok:
            cand = bb_last_cti_tok[u]
            if (0 <= cand < L) and (cand in bb2idxs[u]):
                anchor_u = cand
                anchor_src = "cti_op"

        if anchor_u is None:
            anchor_u = max(bb2idxs[u])
            anchor_src = "last_tok"

        adj[anchor_u].add(first_v)
        adj[first_v].add(anchor_u)

    rel = np.full((L, L), no_relation_id, dtype=np.int64)
    mask = np.zeros((L, L), dtype=np.int64)

    mutex_regions = _compute_branch_mutex_regions(cfg) if use_mutex else []

    for src in range(L):
        dist = [-1] * L
        dist[src] = 0
        q = deque([src])

        while q:
            uu = q.popleft()
            if dist[uu] >= max_rel_dist:
                continue
            for nei in adj[uu]:
                if dist[nei] == -1:
                    dist[nei] = dist[uu] + 1
                    q.append(nei)

        bb_src = bb_ids[src]
        for j, d in enumerate(dist):
            if d == -1:
                continue
            if use_mutex and _is_mutex(bb_src, bb_ids[j], mutex_regions):
                continue
            if d > max_rel_dist:
                d = max_rel_dist
            rel[src, j] = d
            mask[src, j] = 1

    return rel, mask, no_relation_id


def _find_cfg_back_edges(cfg: nx.DiGraph) -> set:
    visited = set()
    rec_stack = set()
    back_edges = set()

    for start in cfg.nodes():
        if start in visited:
            continue
        stack = [(start, iter(cfg.successors(start)))]
        visited.add(start)
        rec_stack.add(start)

        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    visited.add(child)
                    rec_stack.add(child)
                    stack.append((child, iter(cfg.successors(child))))
                elif child in rec_stack:
                    back_edges.add((node, child))
            except StopIteration:
                stack.pop()
                rec_stack.discard(node)

    return back_edges


def compute_cfg_token_rel_and_mask_longest(
    bb_ids, cfg, max_rel_dist: int,
    bb_last_cti_tok=None,
    use_mutex: bool = True,
    tokens=None,
    use_operand_anchor: bool = True,
    **kwargs,
):
    L = len(bb_ids)
    no_relation_id = max_rel_dist + 1

    bb2idxs = defaultdict(list)
    for idx, bb in enumerate(bb_ids):
        bb2idxs[bb].append(idx)

    back_edges_cfg = _find_cfg_back_edges(cfg)

    adj_fwd = [[] for _ in range(L)]
    in_degree = [0] * L

    for bb, idxs in bb2idxs.items():
        idxs_sorted = sorted(idxs)
        for a, b in zip(idxs_sorted, idxs_sorted[1:]):
            adj_fwd[a].append(b)
            in_degree[b] += 1

    for u, v in cfg.edges():
        if (u, v) in back_edges_cfg:
            continue
        if u not in bb2idxs or v not in bb2idxs:
            continue

        first_v = min(bb2idxs[v])
        anchor_u = None

        if use_operand_anchor and tokens is not None:
            anchor_u = _find_operand_anchor(u, v, bb2idxs, tokens=tokens)

        if anchor_u is None and bb_last_cti_tok is not None and u in bb_last_cti_tok:
            cand = bb_last_cti_tok[u]
            if (0 <= cand < L) and (cand in bb2idxs[u]):
                anchor_u = cand

        if anchor_u is None:
            anchor_u = max(bb2idxs[u])

        adj_fwd[anchor_u].append(first_v)
        in_degree[first_v] += 1

    topo_order = []
    temp_in = in_degree[:]
    queue = deque([i for i in range(L) if temp_in[i] == 0])
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for nei in adj_fwd[node]:
            temp_in[nei] -= 1
            if temp_in[nei] == 0:
                queue.append(nei)

    topo_pos = {node: pos for pos, node in enumerate(topo_order)}

    mutex_regions = _compute_branch_mutex_regions(cfg) if use_mutex else []

    rel = np.full((L, L), no_relation_id, dtype=np.int64)
    mask = np.zeros((L, L), dtype=np.int64)

    for src in range(L):
        if src not in topo_pos:
            rel[src, src] = 0
            mask[src, src] = 1
            continue

        dist = [-1] * L
        dist[src] = 0
        src_topo = topo_pos[src]

        for node in topo_order[src_topo:]:
            if dist[node] < 0:
                continue
            for nei in adj_fwd[node]:
                new_d = dist[node] + 1
                if new_d > dist[nei]:
                    dist[nei] = new_d

        bb_src = bb_ids[src]
        for j, d in enumerate(dist):
            if d < 0:
                continue
            if use_mutex and _is_mutex(bb_src, bb_ids[j], mutex_regions):
                continue
            d_capped = min(d, max_rel_dist)
            if rel[src, j] == no_relation_id or d_capped > rel[src, j]:
                rel[src, j] = d_capped
                mask[src, j] = 1

    rel_T = rel.T.copy()
    mask_T = mask.T.copy()

    both_valid = (rel != no_relation_id) & (rel_T != no_relation_id)
    only_rev = (rel == no_relation_id) & (rel_T != no_relation_id)

    rel[both_valid] = np.maximum(rel[both_valid], rel_T[both_valid])
    mask[both_valid] = 1

    rel[only_rev] = rel_T[only_rev]
    mask[only_rev] = 1

    np.fill_diagonal(rel, 0)
    np.fill_diagonal(mask, 1)

    return rel, mask, no_relation_id


def gen_funcstr_with_rel_no_dup(
    f,
    max_rel_dist: int = 16,
    max_raw_tokens: int = 509,
    no_call_cti: bool = True,
    use_mutex: bool = True,
    use_operand_anchor: bool = False,
    use_longest_path: bool = False,
):
    tokens, bb_ids, pos_in_bb, bb_last_cti_tok, bb_last_cti_tgt, cti_operand_edges = linearize_cfg_no_dup(
        f, no_call_cti=no_call_cti
    )
    cfg, _ = _extract_cfg_and_asm_source(f)

    if len(tokens) > max_raw_tokens:
        tokens = tokens[:max_raw_tokens]
        bb_ids = bb_ids[:max_raw_tokens]
        pos_in_bb = pos_in_bb[:max_raw_tokens]

    if use_longest_path:
        rel_ids, path_mask, no_relation_id = compute_cfg_token_rel_and_mask_longest(
            bb_ids,
            cfg,
            max_rel_dist=max_rel_dist,
            bb_last_cti_tok=bb_last_cti_tok,
            use_mutex=use_mutex,
            use_operand_anchor=use_operand_anchor,
            tokens=tokens,
        )
    else:
        rel_ids, path_mask, no_relation_id = compute_cfg_token_rel_and_mask(
            bb_ids,
            cfg,
            max_rel_dist=max_rel_dist,
            bb_last_cti_tok=bb_last_cti_tok,
            use_mutex=use_mutex,
            use_operand_anchor=use_operand_anchor,
            tokens=tokens,
        )

    aux = {
        "tokens": tokens,
        "bb_ids": bb_ids,
        "pos_in_bb": pos_in_bb,
        "no_relation_id": no_relation_id,
        "bb_last_cti_tok": bb_last_cti_tok,
        "bb_last_cti_tgt": bb_last_cti_tgt,
        "cti_operand_edges": cti_operand_edges,
    }
    return " ".join(tokens), rel_ids, path_mask, aux


def _make_datasetbase(datapath, filt, alldata, convert_jump, opt):
    tries = [
        lambda: DatasetBase(datapath, filt, alldata, opt=opt),
        lambda: DatasetBase(datapath, filt, alldata, opt),
        lambda: DatasetBase(datapath, filt, alldata),
        lambda: DatasetBase(datapath, opt=opt),
        lambda: DatasetBase(datapath),
    ]
    last_err = None
    for fn in tries:
        try:
            ds = fn()
            return ds
        except Exception as e:
            last_err = e
    raise last_err


def _iter_pairs(dataset) -> Iterable:
    return dataset.get_paired_data_iter()


def _process_one_pair(args):
    (
        idx, proj, funcname, funcs_dict,
        opt, add_ebd,
        max_rel_dist, max_raw_tokens,
        no_call_cti,
        use_mutex,
        use_operand_anchor,
        use_longest_path,
    ) = args

    local_functions, local_rel_masks = [], []
    local_ebd = {'proj': proj, 'funcname': funcname} if add_ebd else None

    for o in opt:
        if not funcs_dict.get(o):
            continue
        f = funcs_dict[o]

        try:
            func_str, rel_small, mask_small, aux = gen_funcstr_with_rel_no_dup(
                f,
                max_rel_dist=max_rel_dist,
                max_raw_tokens=max_raw_tokens,
                no_call_cti=no_call_cti,
                use_mutex=use_mutex,
                use_operand_anchor=use_operand_anchor,
                use_longest_path=use_longest_path,
            )
        except Exception:
            continue

        if not func_str or rel_small is None or mask_small is None:
            continue

        no_rel_id = aux["no_relation_id"]
        local_rel_masks.append((rel_small.astype(np.int16), mask_small.astype(np.int8), no_rel_id))
        if add_ebd:
            local_ebd[o] = len(local_functions)
        local_functions.append(func_str)

    return idx, local_functions, local_rel_masks, local_ebd, len(local_functions)


def load_paired_data_with_rel(
    datapath,
    filt=None,
    alldata=True,
    convert_jump=False,
    opt=None,
    add_ebd=False,
    max_rel_dist=44,
    max_raw_tokens=509,
    num_workers: int = 0,
    chunksize: int = 8,
    show_pbar: bool = True,
    no_call_cti: bool = True,
    use_mutex: bool = True,
    use_operand_anchor: bool = False,
    use_longest_path: bool = False,
):
    if opt is None:
        opt = ["O0", "O1", "O2", "O3", "Os"]

    dataset = _make_datasetbase(datapath, filt, alldata, convert_jump, opt)
    pairs_iter = _iter_pairs(dataset)

    if num_workers <= 0:
        functions, rel_masks = [], []
        ebds = [] if add_ebd else []
        for idx, item in enumerate(tqdm(pairs_iter, desc="Build REL cache (single)", disable=not show_pbar)):
            proj, funcname, funcs_dict = item[0], item[1], item[2]
            t = (
                idx, proj, funcname, funcs_dict,
                opt, add_ebd,
                max_rel_dist, max_raw_tokens,
                no_call_cti,
                use_mutex,
                use_operand_anchor,
                use_longest_path,
            )
            _, lf, lr, le, _ = _process_one_pair(t)
            functions.append(lf)
            rel_masks.append(lr)
            if add_ebd:
                ebds.append(le)
        return functions, rel_masks, ebds

    pairs = list(pairs_iter)
    tasks = []
    for idx, item in enumerate(pairs):
        proj, funcname, funcs_dict = item[0], item[1], item[2]
        tasks.append(
            (
                idx, proj, funcname, funcs_dict,
                opt, add_ebd,
                max_rel_dist, max_raw_tokens,
                no_call_cti,
                use_mutex,
                use_operand_anchor,
                use_longest_path,
            )
        )

    functions = [None] * len(tasks)
    rel_masks = [None] * len(tasks)
    ebds = [None] * len(tasks) if add_ebd else []

    from multiprocessing import Pool
    with Pool(processes=num_workers) as pool:
        pbar = tqdm(total=len(tasks), desc="Build REL cache (mp)", disable=not show_pbar)
        for idx, lf, lr, le, _ in pool.imap_unordered(_process_one_pair, tasks, chunksize=chunksize):
            functions[idx] = lf
            rel_masks[idx] = lr
            if add_ebd:
                ebds[idx] = le
            pbar.update(1)
        pbar.close()

    return functions, rel_masks, ebds


def pad_rel_ids_only(rel_small: np.ndarray,
                     max_seq_len: int = 512,
                     no_relation_id: int = 0):
    L = int(rel_small.shape[0])

    if L > max_seq_len - 2:
        L = max_seq_len - 2
        rel_small = rel_small[:L, :L]

    full_rel = np.full((max_seq_len, max_seq_len), no_relation_id, dtype=np.int64)

    full_rel[1:L+1, 1:L+1] = rel_small
    full_rel[0, :L+2] = 0
    full_rel[:L+2, 0] = 0
    full_rel[L+1, :L+2] = 0
    full_rel[:L+2, L+1] = 0

    return full_rel



