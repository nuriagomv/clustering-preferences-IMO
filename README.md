# Clustering preferences in inverse multiobjective optimization

Estimate a small set of preference vectors that explain observed decisions of a multiobjective linear program. Two settings:

* **Known objectives.** The cost matrix `C` is given. The model clusters decisions and estimates one weight vector per cluster.
* **Unknown objectives.** Weights are chosen from a catalogue. The model estimates `C` and the assignments together.

Both are mixed-integer programs (Gurobi), warm-started by an alternating heuristic. The forward problem is either the sustainable-diet model or a synthetic LP.

## Layout

```
data/                  workbook and the zipped synthetic sample
src/data/              diet and synthetic instance generators
src/models/            Gurobi formulations
src/heuristics/        alternating warm starts
src/metrics.py         preference-recovery scores and cluster cleanup
src/catalogues.py      interpretable weight catalogues (K = 2 and K = 3)
experiments/           drivers. Edit the grid at the top of main().
outputs/               created on the first run; one pickle per configuration
```

| Previously | Now |
| --- | --- |
| `auxfuncs.py` | `src/metrics.py` |
| `group_diet_instances.py` | `src/data/diet.py` |
| `synthetic_instances.py` | `src/data/synthetic.py` |
| `formulation_fknown.py` | `src/models/known_objectives.py` |
| `formulation_fUNknown.py` | `src/models/unknown_objectives.py` |
| `heuristic_fknown.py` | `src/heuristics/known_objectives.py` |
| `heuristic_fUNknown.py` | `src/heuristics/unknown_objectives.py` |
| `main_fknown.py` | `experiments/run_known_objectives.py` |
| `main_fUNknown.py` | `experiments/run_unknown_objectives.py` |
| `main_fknown_synthetic.py` | `experiments/run_known_objectives_synthetic.py` |

Function names of the formulations are unchanged: `problem_fknown` and `problem_fUNknown`.

## Run

From the repository root, with a Gurobi license available:

```bash
python experiments/run_known_objectives.py
python experiments/run_unknown_objectives.py
python experiments/run_known_objectives_synthetic.py
python -m src.data.synthetic
```

The grids (seeds, `N`, `K`, `d`, number of clusters, time limit) are the constants at the top of each `main`. A configuration whose pickle is already in the matching `outputs/` folder is skipped.

`python -m src.data.synthetic` writes one synthetic dataset to `outputs/synthetic_datasets/`.

## Dependencies

```bash
pip install -r requirements.txt
```

Gurobi is required for every solve, including the forward diet and synthetic LPs. The diet generator reads `data/sustainable_diet_plan_data.xlsx`.

## What a run records

Each pickle holds the estimated weights `W`, the assignments `X`, solver status, gap, runtime, and the mean recovery scores on the train and test splits: top-objective match, RMSE, Spearman, Kendall, cosine similarity, and 1D Wasserstein distance. The unknown-objective pickles also store the estimated cost matrix and its cosine similarity to the true `C`.
