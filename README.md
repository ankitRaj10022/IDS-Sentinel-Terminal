# IDS Automation Console

This repository now has a real application layer on top of the original IDS scripts:

- a FastAPI backend
- a browser dashboard served by that backend
- dockerized train/test automation for the bundled NSL-KDD style datasets
- legacy artifact evaluation for the prediction files already stored in the repo

## What It Does

The original repo was a set of standalone Python scripts for:

- classical ML training on `kddtrain.csv` and `kddtest.csv`
- dense neural network training on the duplicated files under `dnn/kdd/binary`
- saving predictions, probabilities, and checkpoints into `classical/` and `dnn/`

The new app wraps that into:

- `GET /api/overview` for dataset and benchmark visibility
- `POST /api/jobs/classical` to launch classical model jobs
- `POST /api/jobs/dnn` to launch DNN training jobs
- `POST /api/jobs/legacy-evaluation` to snapshot the legacy artifacts

Run output is written into `automation/`.

## Run In WSL

From WSL in this repo root:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8000
```

## Terminal-Only App

The terminal console uses the same training and testing code without the browser UI.

```bash
bash scripts/ids-terminal.sh
```

It opens an `ids>` prompt with Linux-style commands such as `help`, `ls`, `head`, `datasets`, `attacks`, `features`, `legacy`, `runs`, `best`, and `train`.

Useful non-interactive commands:

```bash
bash scripts/ids-terminal.sh overview
bash scripts/ids-terminal.sh datasets
bash scripts/ids-terminal.sh attacks
bash scripts/ids-terminal.sh features
bash scripts/ids-terminal.sh legacy
bash scripts/ids-terminal.sh runs
bash scripts/ids-terminal.sh reports
bash scripts/ids-terminal.sh best
bash scripts/ids-terminal.sh auto-train --auto-profile quick
bash scripts/ids-terminal.sh train-classical --profile fast
bash scripts/ids-terminal.sh train-dnn --architectures 1 --epochs 1 --train-sample 5000 --test-sample 2000
```

Inside the `ids>` prompt:

```text
datasets
attacks
features
train auto quick
train classical balanced
train dnn quick
runs 10
head kddtrain.csv 3
exit
```

## Job Profiles

Classical jobs:

- `fast`: smaller sampled run for quick validation
- `balanced`: larger sampled run including KNN
- `full`: full dataset run for the practical classical models

DNN jobs:

- `fast`: sampled dataset, 3 epochs
- `balanced`: larger sample, 5 epochs
- `full`: full dataset, 10 epochs

These defaults are intentionally safer than the original raw scripts, which tried to run very expensive full-data routines directly.

## Notes

- The root CSVs and `dnn/kdd/binary` CSVs are duplicate copies of the same train/test data.
- The legacy SVM and KNN outputs are still readable by the dashboard, but fresh retraining of those exact legacy choices on the entire dataset is not the recommended default path.
- For WSL-only source checks without Docker, use:

```bash
bash scripts/wsl-smoke-check.sh classical
```
