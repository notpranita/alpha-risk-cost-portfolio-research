# Local validation record

## Executed checks

The following checks were completed in this working environment:

| Validation | Result | Scope |
|---|---|---|
| Full `pytest` suite | **47 passed** | Test cases under `tests/` including CLI smoke test |
| Statement coverage | **94%** | `pytest-cov` source package `acadian_arc` |
| Full synthetic CLI run | **Completed** | `--months 42 --seed 17 --output-dir examples/synthetic` |
| Out-of-sample monthly decisions | **27** | 42 total dates less 15 warm-up dates |
| Artifact generation | **Completed** | CSVs, JSON, report Markdown, PNG plots |
| Python/runtime | **Python 3.13.5 in Linux** | Package requires Python >= 3.10; GitHub Actions configured for 3.10, 3.12 but remote status not verified locally |

Command line used for full suite:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
PYTHONPATH=src python -m pytest -q --cov=acadian_arc --cov-report=term-missing
```

Output: `47 passed` and `TOTAL 522 statements, 30 missed, 94%`.

Command for the synthetic demonstration:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
PYTHONPATH=src python -m acadian_arc.cli \
--months 42 --seed 17 --output-dir examples/synthetic
```

Console output includes `OOS periods: 27`, benchmark synthetic CAGR `2.14%`, alpha-only synthetic CAGR `6.37%`, alpha-risk synthetic CAGR `5.70%`, and alpha-risk-cost synthetic CAGR `5.97%`.

## Do not overinterpret this validation

- This validates **the code under the tested circumstances**; no testing methodology guarantees a defect-free system.
- Numerical results depend on Python/BLAS/scikit-learn/SciPy versions; tolerances should be used to evaluate minor cross-version changes.
- GitHub Actions is included, but has **not been run on the user's GitHub account**.
- The testing universe is fictional, the labels are generated, and the forecasting structure is built around them. No actual economic alpha is proven.
- No public-data connector, exchange order routing, real execution-cost calibration or live portfolio is present.
