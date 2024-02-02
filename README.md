

[![Build-and-Test](https://github.com/recursionpharma/gflownet/actions/workflows/build-and-test.yaml/badge.svg)](https://github.com/recursionpharma/gflownet/actions/workflows/build-and-test.yaml)
[![Code Quality](https://github.com/recursionpharma/gflownet/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/recursionpharma/gflownet/actions/workflows/code-quality.yaml)
[![Python versions](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

# gflownet

GFlowNet-related training and environment code on graphs.

**Primer**

[GFlowNet](https://yoshuabengio.org/2022/03/05/generative-flow-networks/), short for Generative Flow Network, is a novel generative modeling framework, particularly suited for discrete, combinatorial objects. Here in particular it is implemented for graph generation.

The idea behind GFN is to estimate flows in a (graph-theoretic) directed acyclic network*. The network represents all possible ways of constructing an object, and so knowing the flow gives us a policy which we can follow to sequentially construct objects. Such a sequence of partially constructed objects is a _trajectory_. *Perhaps confusingly, the _network_ in GFN refers to the state space, not a neural network architecture.

Here the objects we construct are themselves graphs (e.g. graphs of atoms), which are constructed node by node. To make policy predictions, we use a graph neural network. This GNN outputs per-node logits (e.g. add an atom to this atom, or add a bond between these two atoms), as well as per-graph logits (e.g. stop/"done constructing this object").

The GNN model can be trained on a mix of existing data (offline) and self-generated data (online), the latter being obtained by querying the model sequentially to obtain trajectories. For offline data, we can easily generate trajectories since we know the end state.

## Repo overview

- [algo](src/gflownet/algo), contains GFlowNet algorithms implementations ([Trajectory Balance](https://arxiv.org/abs/2201.13259), [SubTB](https://arxiv.org/abs/2209.12782), [Flow Matching](https://arxiv.org/abs/2106.04399)), as well as some baselines. These implement how to sample trajectories from a model and compute the loss from trajectories.
- [data](src/gflownet/data), contains dataset definitions, data loading and data sampling utilities.
- [envs](src/gflownet/envs), contains environment classes; a graph-building environment base, and a molecular graph context class. The base environment is agnostic to what kind of graph is being made, and the context class specifies mappings from graphs to objects (e.g. molecules) and torch geometric Data.
- [examples](docs/examples), contains simple example implementations of GFlowNet.
- [models](src/gflownet/models), contains model definitions.
- [tasks](src/gflownet/tasks), contains training code.
    -  [qm9](src/gflownet/tasks/qm9/qm9.py), temperature-conditional molecule sampler based on QM9's HOMO-LUMO gap data as a reward.
    -  [seh_frag](src/gflownet/tasks/seh_frag.py), reproducing Bengio et al. 2021, fragment-based molecule design targeting the sEH protein
    -  [seh_frag_moo](src/gflownet/tasks/seh_frag_moo.py), same as the above, but with multi-objective optimization (incl. QED, SA, and molecule weight objectives).
- [utils](src/gflownet/utils), contains utilities (multiprocessing, metrics, conditioning).
- [`trainer.py`](src/gflownet/trainer.py), defines a general harness for training GFlowNet models.
- [`online_trainer.py`](src/gflownet/online_trainer.py), defines a typical online-GFN training loop.

See [implementation notes](docs/implementation_notes.md) for more.

## Getting started

A good place to get started is with the [sEH fragment-based MOO task](src/gflownet/tasks/seh_frag_moo.py). The file `seh_frag_moo.py` is runnable as-is (although you may want to change the default configuration in `main()`).

## Installation

### PIP

This package is installable as a PIP package, but since it depends on some torch-geometric package wheels, the `--find-links` arguments must be specified as well:

```bash
pip install -e . --find-links https://data.pyg.org/whl/torch-1.13.1+cu117.html
```
Or for CPU use:

```bash
pip install -e . --find-links https://data.pyg.org/whl/torch-1.13.1+cpu.html
```

To install or [depend on](https://matiascodesal.com/blog/how-use-git-repository-pip-dependency/) a specific tag, for example here `v0.0.10`, use the following scheme:
```bash
pip install git+https://github.com/recursionpharma/gflownet.git@v0.0.10 --find-links ...
```

If package dependencies seem not to work, you may need to install the exact frozen versions listed `requirements/`, i.e. `pip install -r requirements/main_3.9.txt`.

## Antibiotics

For antibiotics applications, install additional dependencies:
```bash
pip install chemprop==1.6.1
pip install chemfunc==1.0.5
pip install descriptastorus==2.6.1
pip install typed-argument-parser==1.9.0
```

TODO: chemfunc Python version issue

TODO: move these commands to SyntheMol README

**Note:** If you get the issue `ImportError: libXrender.so.1: cannot open shared object file: No such file or directory`, run `conda install -c conda-forge xorg-libxrender`.

Then run the following experiments to generate molecules.

```bash
python src/gflownet/tasks/seh_frag_moo.py \
    --objectives s_aureus solubility \
    --log_dir logs/s_aureus_solubility

python src/gflownet/tasks/seh_frag_moo.py \
    --objectives s_aureus solubility sa \
    --log_dir logs/s_aureus_solubility_sa
```

Next, extract the results from the sqlite database to CSV.

```bash
for MODEL in s_aureus_solubility s_aureus_solubility_sa
do
python scripts/extract_results.py \
    --results_path logs/${MODEL}/final/generated_mols_0.db \
    --save_path logs/${MODEL}/final/molecules.csv
done
```

Rename columns and rescale solubility and SA score.

```bash
for MODEL in s_aureus_solubility s_aureus_solubility_sa
do
python -c "import pandas as pd;
path = 'logs/${MODEL}/final/molecules.csv';
data = pd.read_csv(path);
data = data.rename(columns={'smi': 'smiles', 'fr_0': 'S. aureus', 'fr_1': 'Solubility', 'fr_2': 'sa_score'});
data['Solubility'] = 14 * data['Solubility'] - 10;
if 'sa_score' in data: data['sa_score'] = -9 * data['sa_score'] + 10;
data.to_csv(path, index=False)"
done
```

Compute novelty of the generated molecules.

```bash
for MODEL in s_aureus_solubility s_aureus_solubility_sa
do
chemfunc nearest_neighbor \
    --data_path logs/${MODEL}/final/molecules.csv \
    --reference_data_path ../SyntheMol/rl/data/s_aureus/s_aureus_hits.csv \
    --reference_name train_hits \
    --metric tversky

chemfunc nearest_neighbor \
    --data_path logs/${MODEL}/final/molecules.csv \
    --reference_data_path ../SyntheMol/rl/data/chembl/chembl.csv \
    --reference_name chembl \
    --metric tversky
done
```

Select hit molecules that satisfy novelty, diversity, and efficacy thresholds (optionally including synthesiability).

```bash
python ../SyntheMol/scripts/data/select_molecules.py \
    --data_path logs/s_aureus_solubility/final/molecules.csv \
    --save_molecules_path logs/s_aureus_solubility/final/hits.csv \
    --save_analysis_path logs/s_aureus_solubility/final/analysis.csv \
    --score_columns "S. aureus" "Solubility" \
    --score_comparators ">=0.5" ">=-4" \
    --novelty_threshold 0.6 \
    --similarity_threshold 0.6 \
    --select_num 150 \
    --sort_column "S. aureus" \
    --descending

python ../SyntheMol/scripts/data/select_molecules.py \
    --data_path logs/s_aureus_solubility_sa/final/molecules.csv \
    --save_molecules_path logs/s_aureus_solubility_sa/final/hits.csv \
    --save_analysis_path logs/s_aureus_solubility_sa/final/analysis.csv \
    --score_columns "S. aureus" "Solubility" "sa_score" \
    --score_comparators ">=0.5" ">=-4" "<=4" \
    --novelty_threshold 0.6 \
    --similarity_threshold 0.6 \
    --select_num 150 \
    --sort_column "S. aureus" \
    --descending
```

Visualize hits.

```bash
for MODEL in s_aureus_solubility s_aureus_solubility_sa
do
chemfunc visualize_molecules \
    --data_path logs/${MODEL}/final/hits.csv \
    --save_dir logs/${MODEL}/final/hits
done
```

## Developing & Contributing

TODO: Write Contributing.md.
