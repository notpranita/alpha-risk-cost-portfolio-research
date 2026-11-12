# Replication, Test Scope, and GitHub Publication

## Recommended local environment

Python 3.11–3.13 (standard scientific stack), 2 GB RAM. In a virtual environment run `python -m pip install -e .`, then `python -m acadian_real.study --root . --end 2025-11`, `python -m acadian_real.reporting .`, and `python -m pytest -q`. `OPENBLAS_NUM_THREADS=1` often prevents unnecessary oversubscription during monthly regression/optimization.

The original four downloaded datasets are provided in `data/raw`, as provided. `data/processed/panel.pkl` is an intermediate cache; delete it when raw data or feature definitions change. A standard local `pyarrow` install reads the Parquet data. The `parquet_native.py` fallback is for the exact simple 8-column SNAPPY format supplied and uses the Linux `libsnappy` shared library via ctypes; it is not supported as a general-purpose reader.

## Tested and untested claims

Validated with pytest: correct file parsing and percent-to-decimal conversion; missing-value sentinel replacement; record counts, ticker count, date coverage, absence of duplicate ticker/date keys and nonpositive adjusted prices; historical target alignment; no false fundamental alpha columns; rank score transformation; regression forecasting; positive-definite risk covariance; cost positivity and monotonic scale sensitivity; numerical cost-gradient finite differences; portfolio weight limits, factor caps, infeasible cap rejection; weight drift; monthly drawdown; research CSV counts, sum-to-one holdings, cost nonnegativity, and nonmissing outcomes; statistical report presence.

The backtest has **71 months × 40 assets × 4 strategies** and 2,840 monthly stock forecasts. Do not interpret these software tests as economic strategy validation.

## Repository upload for GitHub beginners

Create a new public GitHub repository; in GitHub Desktop choose `File → Add Local Repository` and select this extracted project directory. If you instead already created an empty repository, open `Repository → Show in Explorer` and paste the **contents** of this extracted project into that folder. Commit to main and select Publish/Push origin. GitHub shows `README.md` automatically. This package includes a ~43 MB source dataset, under GitHub's 100 MB single-file limit; users downloading source data must respect the source provider's research/education license. See `LICENSE_DATA.md`.

**Public GitHub version:** Raw price and Kenneth French archives and processed panel are excluded through `.gitignore`/ZIP packaging to avoid indiscriminate redistribution; dataset-dependent tests use pytest.skip when these source files are not present. The private/local project archive includes the user-provided source data and all 23 tests pass there.
