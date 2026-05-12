# SemFormer

SemFormer is a research prototype for binary code similarity detection (BCSD).
It focuses on execution-oriented contextual semantic modeling for instruction
sequences. The implementation includes semantic-link relative distance modeling,
Relative Distance Prediction (RDP) pre-training, and weighted contrastive
fine-tuning for cross-optimization-level function matching.

This repository contains the core model and training code used in the paper.
Datasets and BinaryCorp preprocessing artifacts are not included. Pre-trained
and fine-tuned checkpoints are released separately on Hugging Face Hub.

## Repository Layout

```text
.
|-- code/
|   |-- model.py       # Unified SemFormer model with t5/qr/qkr attention modes
|   |-- data.py        # Relative-distance data loading for evaluation/export
|   |-- pretrain.py    # MLM + RDP pre-training
|   |-- shard_produce.py # Fine-tuning shard generation
|   |-- finetune.py    # Weighted InfoNCE fine-tuning
|   |-- eval_save.py   # Embedding export for retrieval evaluation
|   `-- fasteval.py    # MRR and Recall@1 evaluation from exported embeddings
|-- requirements.txt
`-- README.md
```

## Model Modes

`model.py` provides one interface for the three semantic-link attention variants:

- `t5`: scalar logit bias per relation bucket and attention head. This is the
  default final model.
- `qr`: query-relation interaction, corresponding to QK + QR attention.
- `qkr`: disentangled query-key-relation attention, corresponding to QK + QR + KR.

Use `--model-mode` in `pretrain.py`, `finetune.py`, and `eval_save.py` to select the mode. Keep the same mode when loading a checkpoint.

## Environment

The experiments were run on Ubuntu 20.04 with CUDA driver 535.86.10 and two
NVIDIA A6000/A100 GPUs. The verified Python environment uses Python 3.8.20,
PyTorch 1.12.1 with CUDA 11.3, Transformers 4.31.0, Tokenizers 0.13.3,
NetworkX 3.1, NumPy 1.24.3, Pandas 2.0.3, scikit-learn 1.1.3, SciPy 1.10.1,
and TensorBoard 2.12.3.

Create an independent SemFormer environment with:

```bash
conda create -n semformer python=3.8 -y
conda activate semformer

conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
```

If your GPU driver or CUDA runtime is different, install the PyTorch build that
matches your machine first, then install the remaining dependencies from
`requirements.txt`.

For a quick dependency check:

```bash
python - <<'PY'
import torch, transformers, networkx, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("networkx", networkx.__version__)
print("numpy", numpy.__version__)
PY
```

The repository includes the tokenizer vocabulary and the lightweight dataset
loader used by the training scripts. If you use a custom tokenizer, either set
`SEMFORMER_TOKENIZER_DIR` to the tokenizer directory or pass `--tokenizer-dir`
to the scripts that expose this option.

## Data Preparation

Prepare BinaryCorp-style pickle files before training. The pre-training script
expects a directory of function-level pickle files through `--pkl-dir`. The
fine-tuning script expects paired training shards through `--train_shard_glob`
and, optionally, validation shards through `--eval_shard_glob`. Shards can be
generated with `shard_produce.py` from a processed BinaryCorp directory.

Large data files should stay outside the Git repository. A common layout is:

```text
SemFormer/
|-- code/
|-- checkpoints/       # downloaded checkpoints
|-- data/              # ignored by git
`-- outputs/           # ignored by git
```

## Checkpoints

After the checkpoints are available on Hugging Face Hub, download them with:

```bash
mkdir -p checkpoints

hf download SemFormer/semformer-rdp-pretrain \
  --repo-type model \
  --local-dir checkpoints/semformer-rdp-pretrain

hf download SemFormer/semformer-finetune-bs64 \
  --repo-type model \
  --local-dir checkpoints/semformer-finetune-bs64
```

The pre-training checkpoint is used as the initialization for fine-tuning. The
fine-tuned checkpoint can be used directly for embedding export and retrieval
evaluation.

## Pre-training

Run RDP pre-training from the `code/` directory. The default arguments in
`pretrain.py` follow the configuration used for the final T5-style RDP
pre-training run: scalar semantic-link bias, MLM+RDP, `max_rel_dist=512`,
`max_raw_tokens=400`, `pairs_per_seq=8`, mixed RDP pair sampling,
`nontrivial_frac=0.9`, and `max_steps=680000`.

For a two-GPU run:

```bash
cd code
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_P2P_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
torchrun --standalone --nproc_per_node=2 pretrain.py \
  --pkl-dir ../data/BinaryCorp/train \
  --out ../outputs/pretrain_rdp_t5
```

The script defaults to `--tokenizer-dir ./tokenizer`, `--batch-size 24`,
`--grad-accum-steps 4`, `--num-workers 8`, `--fp16`, and `--pe-equals-we`.
Use `--use-longest-path` to reproduce the longest-distance ablation; otherwise
the shortest reachable distance is used.

## Fine-tuning Shard Generation

Generate fine-tuning shards before running `finetune.py`:

```bash
cd code
python shard_produce.py \
  --data-path ../data/binarycorp_train \
  --out-dir ../data/rel_shards_train_shortest \
  --max-rel-dist 512 \
  --max-pairs-per-shard 6000 \
  --num-workers 32
```

Each shard is a pickle file containing `(functions_chunk, rel_chunk, ebds_chunk)`,
which is the format consumed by `finetune.py`.

## Fine-tuning

Fine-tune the pre-trained checkpoint with weighted in-batch InfoNCE. The default
arguments in `finetune.py` follow the final batch-size-64 configuration used in
the paper: `batch_size=64`, `gradient_accumulation_steps=2`, `lr=3e-6`,
`temperature=0.03`, `weight_base=1.0`, `weight_hard=3.0`, `freeze_cnt=0`,
mixed precision, and gradient checkpointing.

```bash
cd code
CUDA_VISIBLE_DEVICES=0 python finetune.py
```

By default, the script reads the pre-training checkpoint from
`../checkpoints/semformer-rdp-pretrain`, training shards from
`../data/rel_shards_train_shortest/rel_shard_*.pkl`, validation shards from
`../data/rel_shards_test_shortest/rel_shard_*.pkl`, and writes outputs to
`../outputs/finetune_rdp_shortest_bs64`. Override these paths if your data
layout is different.

The fine-tuning objective increases the training weight of harder positive
pairs, especially pairs compiled under more distant optimization levels.

## Embedding Export and Retrieval Evaluation

`eval_save.py` exports function embeddings for downstream retrieval evaluation:

```bash
cd code
python eval_save.py \
  --model-dir ../checkpoints/semformer-finetune-bs64 \
  --dataset-dir ../data/binarycorp_eval \
  --out-pkl ../outputs/eval_embeddings.pkl \
  --model-mode t5
```

The exported pickle can be evaluated with `fasteval.py`:

```bash
cd code
python fasteval.py \
  --experiment-path ../outputs/eval_embeddings.pkl \
  --poolsize 10000
```

The script reports MRR and Recall@1 for the default optimization-level pairs:
O0-O3, O0-Os, O1-O3, O1-Os, O2-O3, and O2-Os.

The public release focuses on the BCSD pre-training and fine-tuning pipeline.
The vulnerability-search experiments in the paper use the same exported
function embeddings but depend on additional CVE-specific data preparation that
is not included in this repository.

## Notes

- `t5` is the default final architecture.
- `qr` and `qkr` are preserved in `model.py` for ablation experiments.
- Training outputs, checkpoints, logs, and BinaryCorp data are ignored by git by
  default.
- The public repository should not include local IDA databases, extracted
  binaries, large pickle files, checkpoints, or logs.



