# DNM Paper

Reference implementation for the experiments used in the DNM paper. The repository contains the mutation model, the ECKITY integration layer, the artificial ant benchmark, and a runnable experiment entry point for reproducing results.

## About the paper

This repository accompanies the paper `BERT Mutation: Deep Transformer Model for Masked Uniform Mutation in Genetic Algorithms`.

The paper proposes a domain-independent mutation operator for genetic algorithms that uses a BERT-style masked model to predict beneficial gene replacements from context. To make this work for fixed-length GA representations, it adds an elite-guided data augmentation mechanism that creates additional learning signal from strong historical solutions.

In the paper, the method is evaluated on four domains: Frozen Lake, Artificial Ant, Graph Coloring, and Unweighted Set Cover. The reported results show faster convergence and better final fitness than standard mutation baselines and an adaptive operator-selection baseline, while maintaining meaningful population diversity.

## Requirements

- Python 3.9 or newer
- A working PyTorch installation

The code depends on:

- `numpy`
- `torch`
- `transformers`
- `eckity`

## Installation

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Running the experiments

The main entry point is the artificial ant experiment runner:

```bash
python example_runner.py
```

You can also run it as a module:

```bash
python -m dnm_paper.experiments.artificial_ant
```

Useful options:

```bash
python -m dnm_paper.experiments.artificial_ant --generations 100 --runs 3 --population-size 6
python -m dnm_paper.experiments.artificial_ant --maps-dir artifical_ant_maps --output-dir experiments/artificial_ant/runs
```

By default, results are written under `experiments/artificial_ant/runs/<map_name>/bert_mutation/<run_id>/results.json`.

## Project structure

```text
dnm_paper/
  config.py                     Experiment configuration and default paths
  individuals.py               Custom ECKITY individual creator
  logging_utils.py             JSON statistics logger
  experiments/
    artificial_ant.py          CLI entry point and experiment orchestration
  mutation/
    bert.py                    BERT-based mutation operator
    eckity_adapter.py          Adapter that plugs the mutation operator into ECKITY
  problems/
    artificial_ant.py          Artificial ant map loader and evaluator
problems/
artifical_ant_maps/            Benchmark map files
example_runner.py              Thin compatibility wrapper for running the main experiment
pyproject.toml                 Package metadata and dependencies
```

## Notes

- The repository now uses `pathlib`-based paths, so it can be run from the repository root without editing hard-coded relative paths.
- Generated files such as experiment outputs, IDE settings, and Python cache directories are excluded through `.gitignore`.
