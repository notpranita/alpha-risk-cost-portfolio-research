# Publish and present this research on GitHub

This file is for the repository owner, not for investors. The most important goal is to make the project legible **without requiring the reviewer to run code first**.

## 1. What a visitor sees first

- **Repository title**: `alpha-risk-cost-portfolio-research` (suggested; choose any available name).
- **Description**: `Five-layer equity alpha forecasting, factor risk, nonlinear transaction costs and benchmark-relative portfolio optimization — tested Python research framework.`
- **Topics**: `quantitative-finance`, `portfolio-optimization`, `systematic-investing`, `factor-models`, `ridge-regression`, `transaction-costs`, `backtesting`, `python`, `scipy`.
- **README top section**: research question, independent-project disclosure, architecture diagram, model table and sample charts.
- **Commit examples**: the deterministic simulated dataset/reports, not gigabytes of raw or private market data.
- **Tests**: GitHub Actions will run automatically on each push; do not call a workflow green before it has actually run on GitHub.

## 2. Exact upload steps (terminal)

First extract the ZIP and navigate to the directory that contains `README.md`:

```bash
cd acadian-alpha-risk-cost
python -m pip install -e '.[dev]'
python -m pytest -q
python -m acadian_arc.cli --months 42 --seed 17 --output-dir reports/local-validation

git init
git add .
git commit -m "Build auditable alpha-risk-cost quantitative research framework"
git branch -M main
```

Then create an **empty** repository on [github.com/new](https://github.com/new) (do **not** initialize its README/license because these already exist locally). Replace `YOUR_USERNAME` with your own GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/alpha-risk-cost-portfolio-research.git
git push -u origin main
```

GitHub may prompt for a browser sign-in, credential manager or personal-access token; normal account passwords are not accepted for Git HTTPS authentication. Alternatively, drag-and-drop the extracted contents using GitHub's **Add file → Upload files** interface, preserving the directory structure; Git commands are better for a multi-file project with future changes.

## 3. Recommended GitHub landing-page order

1. Project one-line research question and **synthetic/independent status**.
2. Mermaid pipeline diagram (rendered natively by GitHub Markdown).
3. Model-to-code mapping table: what math is used, which `.py` implements it.
4. Sample charts from `examples/synthetic/` with captions **SYNTHETIC, not actual results**.
5. Reproduction commands, test status and numerical constraints.
6. Real methodological limitations and future study plan.
7. Public SEC source link and explicit attribution of **inspiration only**.

The existing README already follows this order. The detailed derivations live in `docs/` so the landing page stays scannable.

## 4. What the commit demonstrates to a quant reviewer

| Demonstrated capability | Tangible evidence |
|---|---|
| Reproducible applied modeling | `configs/research.json`, fixed random seed, deterministic sample data |
| Regression and feature engineering | `signals.py`, `forecast.py`, and tests |
| Matrix statistics / risk modeling | `risk.py` and variance attribution audit |
| Nonlinear optimization | `optimizer.py`, explicit constraints and gradient |
| Market microstructure assumptions | `costs.py`, participation bounds and order-profile CSV |
| Experimental design | Three ablations and walk-forward testing in `backtest.py` |
| Scientific honesty | Sourced SEC context, transparent approximations, caveats and missing empirical claims |
| Software engineering | Python package, typed-ish modular functions, pytest, GitHub Actions |

## 5. After the first public release

- Check the **Actions** tab to confirm CI actually passes on GitHub (Python 3.10 and 3.12).
- Tag a stable release, e.g. `v1.0.0-synthetic-prototype`; this can be a GitHub Release with the ZIP or source archive.
- Keep the README consistent with the real source. When upgrading models, revise methods and tests together.
- Add provenance and an appropriate citation/license only for any new datasets you have legal rights to redistribute.
- The ideal next version is **v2: point-in-time real-data research** with a validated changing constituent set, realistic execution timestamp, holdout tests, and statistical uncertainty. Treat that as a separate milestone, not a feature already delivered.

## 6. Suggested résumé description

> Implemented an independent Python systematic-equity research framework with five cross-sectional Ridge forecasting layers (stock, peers, country, industry and intersections), a shrinkage factor-covariance risk model, participation-sensitive nonlinear transaction costs, and SLSQP benchmark-relative constrained portfolio optimization. Built a lagged monthly backtesting and reporting pipeline with strategy ablations, risk attribution, automated tests and CI, and demonstrated reproducibility on synthetic data.

Avoid calling the synthetic demonstration **market outperformance** or representing Acadian's actual signals/weights/calibrations as reconstructed.
