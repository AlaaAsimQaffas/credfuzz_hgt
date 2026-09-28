# CredFuzz-HGT


## Data

Data are not redistributed. Downloading implies acceptance of each provider's terms.

| Key | Source | Expected files |
|---|---|---|
| `uci` | [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | Downloaded automatically, or `data/raw/uci/default_of_credit_card_clients.xls` |
| `home_credit` | [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) | Competition CSV files in `data/raw/home_credit/` |
| `lendingclub` | [LendingClub Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) | Loan CSV in `data/raw/lendingclub/` |

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/download_data.py --dataset uci
python scripts/run_experiment.py --config configs/uci.yaml
```

For a quick installation check:

```bash
python scripts/run_experiment.py --config configs/uci.yaml --smoke-test
pytest -q
```

Run all folds, baselines, ablations, and tables:

```bash
bash scripts/reproduce_uci.sh
python scripts/run_baselines.py --config configs/uci.yaml
python scripts/run_ablations.py --config configs/uci.yaml
python scripts/make_tables.py --results-dir results
```


## Outputs

Each run records fold predictions, metrics, selected thresholds, configuration, software versions, seed, parameter count, and runtime under `results/`. All preprocessing statistics, graph construction parameters, class weights, and thresholds are fitted without outer-test labels.

## Reproducibility

Deterministic algorithms are enabled where supported. Exact equality across CUDA hardware is not guaranteed. For archival releases, create a Git tag, attach the ZIP to GitHub Releases, archive the release with Zenodo, and insert the DOI into `CITATION.cff` and `.zenodo.json`.

## License and citation

Code is released under the MIT License. Dataset licenses and terms remain with their providers. See `CITATION.cff`.
