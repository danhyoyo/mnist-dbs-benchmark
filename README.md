# DBS: Quantized Zeroth-Order Training Benchmark

## What this project does

This project develops and evaluates a neural-network training method called **DBS**. DBS updates quantized weights by comparing the loss from two forward passes, so it can learn without calculating gradients with backpropagation. The goal is to test whether a small, discrete parameter state can be useful for training models when gradient computation, memory, or hardware resources are limited.

The repository provides a reproducible MNIST experiment. It trains the same `784 → 256 → 10` MLP with four methods—SGD, INT8 quantization-aware training, Binary/Bop-style updates, and DBS—and compares their accuracy, compute cost, memory use, and runtime. The main deliverable is an evidence-based assessment of where DBS works, what it costs, and which design changes improve it.

In one sentence: **this is a research prototype for gradient-free, quantized neural-network training, together with the benchmarks needed to evaluate it fairly.**

The project is not a production deep-learning framework. The current code is an experimental reference implementation used to test the DBS idea and its variants.

The repository is intended as a reproducible research prototype. It measures accuracy, training forward evaluations, wall time, CUDA memory, and (when the GPU driver exposes power data) estimated energy. The documentation also records the persistent parameter-state representation.

## What is DBS?

DBS stores each parameter as a quantization-bin index and keeps a small integer evidence accumulator:

```text
q: uint8       quantized bin index for each parameter
e: int16       accumulated signed evidence
```

For each training minibatch, DBS:

1. Selects a block of parameters.
2. Samples sparse directions `r ∈ {-1, 0, +1}`.
3. Evaluates the loss at `q + r` and `q - r`.
4. Converts the paired loss comparison into signed votes.
5. Moves a parameter by one quantization bin when its evidence reaches a threshold.

The default `layer_balanced` variant uses layer-specific scales and initialization, plus a per-layer block schedule. The `residual` evidence rule keeps votes that exceed the threshold after a bin move; `reset` is the original rule and remains the required control condition.

## Current results

Full MNIST, MLP `784 → 256 → 10`, 10 epochs, 8 paired probes, block size 4096, threshold 3, 256 bins, one fixed seed. The test set was evaluated after selecting the best checkpoint on a validation split.

| Method | Test accuracy | Training forward evaluations | Wall time |
|---|---:|---:|---:|
| SGD | 98.03% | 2,150 | 98.8 s |
| DBS (`reset`) | 86.10% | 34,400 | 197.6 s |
| DBS (`residual`) | **86.55%** | 34,400 | 200.1 s |

The residual rule is a promising incremental improvement, not a final claim of superiority. A three-seed subset study also favored residual DBS, but the full-data result still needs more seeds and additional datasets.

## Research position

The paired loss comparison and signed zeroth-order update are related to SPSA, sign-m-SPSA, ZOQO, and other quantized zeroth-order methods. The project should therefore be presented as an experimental DBS variant whose possible contribution is the combination of integer evidence accumulation, one-bin transitions, residual voting, and layer-balanced sparse scheduling.

The current implementation does not yet prove integer-only execution or lower energy use: it temporarily decodes weights to floating point and uses dense PyTorch kernels for the forward pass. It is also currently evaluated mainly on MNIST. See [DBS_LITERATURE_REVIEW.md](DBS_LITERATURE_REVIEW.md) for the literature comparison and novelty limits.

## Repository layout

```text
benchmark.py                Main benchmark: SGD, QAT, Binary/Bop, and DBS
train_compare.py            Validation split, checkpoint selection, and comparison
tests/test_dbs.py           Unit tests for DBS state and update rules
EXPERIMENTS.md              Layer-balanced DBS experiments and ablations
RESIDUAL_EXPERIMENT.md      Residual evidence experiment and results
DBS_LITERATURE_REVIEW.md    Literature review and novelty assessment
requirements.txt            Python dependencies
```

Datasets, checkpoints, plots, and run logs are intentionally excluded from Git by `.gitignore`.

## Installation

Python 3.10 or newer is recommended. Install the dependencies in a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

For CUDA, install a PyTorch build matching the CUDA version on your machine before installing the remaining requirements.

## Quick start

Run a small smoke test first:

```bash
python benchmark.py --method all --epochs 1 --train-limit 2048 --test-limit 1000 --probes 2 --device cpu --skip-plots --output-dir runs/smoke
```

Run the complete benchmark:

```bash
python benchmark.py --method all --epochs 10 --batch-size 256 --device cuda --output-dir runs/full
```

If CUDA is unavailable, replace `--device cuda` with `--device cpu`.

## Reproducible DBS comparison

`train_compare.py` creates a fixed validation split from MNIST, rewinds the minibatch order for each method, saves the best checkpoint, and evaluates the test set after model selection:

```bash
python train_compare.py \
  --methods sgd dbs_reset dbs_residual \
  --seeds 1 2 3 \
  --epochs 10 \
  --device cuda \
  --output-dir runs/compare
```

The command writes `comparison.csv`, `results.json`, `summary.json`, and one checkpoint per method and seed. To evaluate a saved checkpoint:

```bash
python train_compare.py \
  --eval-checkpoint runs/compare/dbs_residual_seed1.pt \
  --device cuda
```

For a fast CPU experiment:

```bash
python train_compare.py \
  --methods dbs_reset dbs_residual \
  --seeds 1 2 3 \
  --epochs 10 \
  --train-limit 4096 \
  --val-size 512 \
  --test-limit 2000 \
  --hidden 64 \
  --device cpu \
  --output-dir runs/subset
```

## Main options

```text
--method {sgd,qat,binary,dbs,all}
--epochs N --batch-size N --hidden N --seed N --device {cpu,cuda}
--probes R --block-size N --perturb-fraction P
--threshold N --bins N
--dbs-variant {legacy,layer_scaled,layer_scheduled,layer_balanced}
--dbs-evidence-rule {reset,residual}
--dbs-init-levels X
```

The important DBS compute trade-off is that each probe uses two minibatch forward evaluations. Therefore `forward_evaluations`, wall time, and accuracy should be reported together; epoch count alone is not a fair compute comparison.

## Testing

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

The tests cover reproducible initialization, layer-balanced block coverage, finite outputs, integer state types, reset versus residual evidence, clipping at bin boundaries, and negative-direction updates.

## Limitations and next research steps

- Run 3–5 seeds on full MNIST and report mean and standard deviation.
- Add close zeroth-order baselines such as sign-m-SPSA or ZOQO under the same forward budget.
- Evaluate on Fashion-MNIST or another held-out dataset.
- Replace floating-point weight materialization with an integer or embedded-device kernel before claiming hardware or energy benefits.
- Measure memory and energy on the target device rather than inferring them from parameter storage alone.

## License

No license has been selected yet. Add a license before publishing the repository if you want others to reuse or redistribute the code.
