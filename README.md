# SemFormer

SemFormer is a research prototype for binary code similarity detection (BCSD).
It focuses on execution-oriented contextual semantic modeling for instruction
sequences. The implementation includes semantic-link relative distance modeling,
Relative Distance Prediction (RDP) pre-training, and weighted contrastive
fine-tuning for cross-optimization-level function matching.

This repository contains the core model, data processing utilities, and training
code for SemFormer. Large training corpora and model checkpoints should be kept
outside the Git repository and can be distributed separately.

## Repository Layout

```text
.
|-- model.py           # Unified SemFormer model with t5/qr/qkr attention modes
|-- data.py            # Relative-distance data loading for evaluation/export
|-- pretrain.py        # MLM + RDP pre-training
|-- shard_produce.py   # Fine-tuning shard generation
|-- finetune.py        # Weighted InfoNCE fine-tuning
|-- eval_save.py       # Embedding export for retrieval evaluation
|-- fasteval.py        # MRR and Recall@1 evaluation from exported embeddings
|-- tokenizer/         # Default tokenizer files
|-- datautils/         # Binary feature extraction and loading utilities
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

SemFormer is developed for a PyTorch and HuggingFace Transformers environment.
The recommended setup uses Python 3.8, PyTorch 1.12.1, Transformers 4.31.0,
Tokenizers 0.13.3, NetworkX 3.1, NumPy 1.24.3, Pandas 2.0.3, scikit-learn
1.1.3, SciPy 1.10.1, and TensorBoard 2.12.3.

Create an independent SemFormer environment with:

```bash
conda create -n semformer python=3.8 -y
conda activate semformer

conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA runtime if the example above
does not match your machine, then install the remaining dependencies from
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
|-- checkpoints/       # downloaded checkpoints
|-- data/              # ignored by git
`-- outputs/           # ignored by git
```

The default script paths are relative to the repository root.

## Checkpoints

If released checkpoints are available on Hugging Face Hub, download them with:

```bash
mkdir -p checkpoints

hf download <namespace>/semformer-rdp-pretrain \
  --repo-type model \
  --local-dir checkpoints/semformer-rdp-pretrain

hf download <namespace>/semformer-finetune-bs64 \
  --repo-type model \
  --local-dir checkpoints/semformer-finetune-bs64
```

The pre-training checkpoint is used as the initialization for fine-tuning. The
fine-tuned checkpoint can be used directly for embedding export and retrieval
evaluation.

## Pre-training

Run RDP pre-training with the default SemFormer configuration:

For single-node distributed training:

```bash
torchrun --standalone --nproc_per_node=<num_gpus> pretrain.py
```

Adjust the default paths in `pretrain.py` or pass command-line arguments if your
data layout is different.

## Fine-tuning Shard Generation

Generate fine-tuning shards before running `finetune.py`:

```bash
python shard_produce.py
```

Each shard is a pickle file containing `(functions_chunk, rel_chunk, ebds_chunk)`,
which is the format consumed by `finetune.py`.

## Fine-tuning

Fine-tune the pre-trained checkpoint with weighted in-batch InfoNCE. The default
arguments in `finetune.py` provide the standard SemFormer fine-tuning
configuration.

```bash
python finetune.py
```

Adjust the default paths in `finetune.py` or pass command-line arguments if your
data layout is different.

The fine-tuning objective increases the training weight of harder positive
pairs, especially pairs compiled under more distant optimization levels.

## Embedding Export and Retrieval Evaluation

`eval_save.py` exports function embeddings for downstream retrieval evaluation:

```bash
python eval_save.py
```

The exported pickle can be evaluated with `fasteval.py`:

```bash
python fasteval.py
```

The script reports MRR and Recall@1 for the default optimization-level pairs:
O0-O3, O0-Os, O1-O3, O1-Os, O2-O3, and O2-Os.

This release focuses on the BCSD pre-training and fine-tuning pipeline. Other
downstream retrieval tasks can reuse the exported function embeddings with
task-specific query and gallery construction.

## Notes

- `t5` is the default final architecture.
- `qr` and `qkr` are preserved in `model.py` for ablation experiments.
- Training outputs, checkpoints, large pickle files, and temporary caches should
  be kept outside version control.



