# SA-SPINN

Code and results for *Joint estimation of battery state of charge and health robust to sensor errors with a sensor-aligned separable physics-informed neural network*.

SA-SPINN estimates state of charge (SOC) and state of health (SOH) with one sub-network per sensor, a log-domain fusion, a two-way Coulomb-counting coupling and a training penalty on the sensor-induced uncertainty. The uncertainty is evaluated by Monte Carlo over declared sensor error budgets.

## Layout

| Folder | Content |
|---|---|
| `src/` | data parsing, health indicators, model, training, sensor-error tests |
| `analysis/` | aggregation of all runs into `results/numbers.json` and the result figures |
| `runs/` | every run: `log.txt`, `results/` (JSON metrics), `models/` (trained weights), `status/` |
| `results/` | `numbers.json` and `figures/`, produced by `analysis/` |

## Setup

Python 3.12 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## Data

Both datasets are public. Download them into `data/raw/`:

| Dataset | Source | Folder |
|---|---|---|
| XJTU | [10.5281/zenodo.10963339](https://doi.org/10.5281/zenodo.10963339) | `data/raw/xjtu/Batch-1` to `Batch-6` |
| TJU | [10.5281/zenodo.6405084](https://doi.org/10.5281/zenodo.6405084) | `data/raw/tju/Dataset_1_NCA_battery`, `Dataset_2_NCM_battery`, `Dataset_3_NCM_NCA_battery` |

## Reproduce

Run from the repository root.

```
# 1. XJTU: parse, health indicators, SOC cache, health-indicator draws under the budget
python src/xjtu.py
python src/features.py
python src/data.py
python src/gum.py --hi-draws 20

# 2. TJU: same steps with DS=tju
python src/tju.py
DS=tju python src/features.py
DS=tju python src/data.py
DS=tju python src/gum.py --hi-draws 20

# 3. Train a grid of models (grids are defined in src/train.py)
python src/train.py --grid s12_xjtu_tdrop --alpha 1e-3

# 4. Sensor-error tests of trained models
python src/fusion.py s12 [--tju]

# 5. Numbers and figures from all runs
python analysis/aggregate.py
python analysis/figures.py
```

`python src/monitor.py runs/<folder>` shows the progress of a running grid.

The trained models are included, so step 5 works without retraining, and step 4 needs only the processed data.

## Runs

Each folder is `runs/s<step>_<what>/`. The final models and the tests behind the reported results are:

| Folder | Content |
|---|---|
| `s12_xjtu_tdrop`, `s12_tju_final` | final SA-SPINN, 5 seeds |
| `s10_xjtu_base`, `s10_tju_base` | retrained baselines, 5 seeds |
| `s11_*`, `s14_*` | ablation models (penalty, grouping, plug-in) |
| `s12_eval[_tju]` | clean accuracy, budget A, stress, failed sensor, slow faults, fault isolation |
| `s11_ladder[_tju]`, `s11_budget[_tju]`, `s11_sweep[_tju]` | ablation ladder, slow faults, budget sweep |
| `s13_*` | penalty weight |
| `s15_budgetb[_tju]` | budget B |
| `s3_gum`, `s5_*`, `s6_*`, `s7_*`, `s9_*` | uncertainty budget, fusion rules, training methods, coupling |
| `s1_*`, `s2_*`, `s4_*` | earlier design runs |
| `s8_xjtu`, `s8_tju` | other published splits, not used in the reported results |
