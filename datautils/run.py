#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import multiprocessing
import time

from util.pairdata import pairdata


ROOT = os.path.dirname(os.path.abspath(__file__))


def get_targets(path, prefixfilter=None):
    targets = []
    for root, _dirs, files in os.walk(path):
        for file in files:
            if prefixfilter is None or any(file.startswith(prefix) for prefix in prefixfilter):
                targets.append(os.path.join(root, file))
    return targets


def build_parser():
    parser = argparse.ArgumentParser("Extract BinaryCorp-style function features with IDA")
    parser.add_argument("--ida-path", default=os.environ.get("IDA_PATH", "idat64"))
    parser.add_argument("--dataset-dir", default=os.path.join(ROOT, "dataset"))
    parser.add_argument("--script-path", default=os.path.join(ROOT, "process.py"))
    parser.add_argument("--save-root", default=os.path.join(ROOT, "extract"))
    parser.add_argument("--log-dir", default=os.path.join(ROOT, "log"))
    parser.add_argument("--idb-dir", default=os.path.join(ROOT, "idb"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefix", action="append", default=None)
    return parser


def run_one(args):
    ida_path, script_path, log_dir, idb_dir, ida_input = args
    filename = os.path.basename(ida_input)
    log_path = os.path.join(log_dir, f"{filename}.log")
    idb_path = os.path.join(idb_dir, f"{filename}.idb")
    cmd = [
        ida_path,
        f"-L{log_path}",
        "-c",
        "-A",
        f"-S{script_path}",
        f"-o{idb_path}",
        ida_input,
    ]
    return subprocess.call(cmd)


def main():
    args = build_parser().parse_args()
    for path in (args.dataset_dir, args.save_root, args.log_dir, args.idb_dir):
        os.makedirs(path, exist_ok=True)

    start = time.time()
    targets = get_targets(args.dataset_dir, args.prefix)
    jobs = [
        (args.ida_path, args.script_path, args.log_dir, args.idb_dir, target)
        for target in targets
    ]

    with multiprocessing.Pool(processes=args.workers) as pool:
        list(pool.imap_unordered(run_one, jobs))

    pairdata(args.save_root)
    print(f"processed={len(targets)} time={time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
