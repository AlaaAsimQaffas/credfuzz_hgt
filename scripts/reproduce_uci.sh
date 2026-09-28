#!/usr/bin/env bash
set -euo pipefail
python scripts/download_data.py --dataset uci
python scripts/run_experiment.py --config configs/uci.yaml
python scripts/run_baselines.py --config configs/uci.yaml
python scripts/make_tables.py --results-dir results
