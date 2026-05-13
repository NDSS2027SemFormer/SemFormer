# SemFormer

SemFormer is a research prototype for binary code similarity detection (BCSD).
It focuses on execution-oriented contextual semantic modeling for instruction
sequences. The implementation includes semantic-link relative distance modeling,
Relative Distance Prediction (RDP) pre-training, and weighted contrastive
fine-tuning for cross-optimization-level function matching.

This repository contains the core model, data processing utilities, and training
code for SemFormer. Model checkpoints and evaluation artifacts are distributed
through Hugging Face Hub.

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
  default mode used by the provided checkpoints.
- `qr`: query-relation interaction, corresponding to QK + QR attention.
- `qkr`: disentangled query-key-relation attention, corresponding to QK + QR + KR.

Use `--model-mode` in `pretrain.py`, `finetune.py`, and `eval_save.py` to select the mode. Keep the same mode when loading a checkpoint.

## Environment

SemFormer is developed for a PyTorch and HuggingFace Transformers environment.
The recommended setup uses Python 3.8, PyTorch 1.12.1, Transformers 4.31.0,
Tokenizers 0.13.3, NetworkX 3.1, NumPy 1.24.3, scikit-learn 1.1.3, and
TQDM 4.66.5.

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

The IDA feature extraction utilities under `datautils/` additionally require
IDA Python and BinaryAI when raw binaries are processed from scratch.

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

The default layout keeps large data files outside the tracked source tree:

```text
SemFormer/
|-- checkpoints/       # downloaded checkpoints
|-- data/              # ignored by git
`-- outputs/           # ignored by git
```

The default script paths are relative to the repository root.

## Released Artifacts

Download the released checkpoints from Hugging Face Hub:

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

BinaryCorp-3M embeddings exported by `eval_save.py` are also provided for direct
retrieval evaluation:

```bash
mkdir -p outputs

hf download SemFormer/semformer-eval-artifacts \
  --repo-type dataset \
  --local-dir outputs
```

## Pre-training

Run MLM + RDP pre-training with the default SemFormer configuration:

For single-node distributed training:

```bash
torchrun --standalone --nproc_per_node=<num_gpus> pretrain.py
```

By default, `pretrain.py` reads BinaryCorp-style pickle files from
`./data/BinaryCorp/train`, uses the tokenizer in `./tokenizer`, and writes
checkpoints to `./outputs/pretrain_rdp_t5`.

The pre-training data pipeline builds instruction-level relative distance
matrices from the control-flow graph. The default setting uses shortest-path
distances with operand-anchor links enabled. Use `--use-longest-path` for the
longest-path variant and `--no-use-operand-anchor` to disable operand-anchor
links. Adjust the default paths in `pretrain.py` or pass command-line arguments
if your data layout is different.

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

To evaluate a released artifact directly:

```bash
python fasteval.py --experiment-path ./outputs/semformer-rdp-finetune.pkl
```

The script reports MRR and Recall@1 for the default optimization-level pairs:
O0-O3, O0-Os, O1-O3, O1-Os, O2-O3, and O2-Os.

This release focuses on the BCSD pre-training and fine-tuning pipeline. Other
downstream retrieval tasks can reuse the exported function embeddings with
task-specific query and gallery construction.

## Configuration

- The default model mode is `t5`.
- `qr` and `qkr` are available through `--model-mode` for attention-mode
  ablation.
- Generated checkpoints, training outputs, large pickle files, and temporary
  caches are excluded from version control.



