# Experiments, ablations, and honest interpretation

## 1. What is being tested?

The reproducible demonstration has two purposes:

1. **Software correctness:** verify that a source-to-output portfolio pipeline runs, constraints are satisfied, costs are allocated, and return accounting is internally consistent.
2. **Illustrative causal structure:** compare three different *objective functions* on identical generated data and constraints, to see how cost- and risk-awareness can influence turnover and portfolio holdings.

It **cannot** test whether the chosen signals predict actual market returns because the simulated dataset deliberately implants a predictive signal/return relationship.

## 2. Experiment design

The generator (`data.py`) creates a fixed universe of 16 fictional securities with 4 sectors, 2 countries and 2 simulated names per sector×country combination, repeated across 42 synthetic monthly dates. A seeded random number generator (`seed=17`) evolves five stock features and three aggregate cycle features. These are used by a hidden label-generating formula with random market, sector and stock shocks.

The model **does not see the hidden label-generation coefficients**. It is trained on `next_return` from preceding months only. After 15 initial warm-up months, the backtest evaluates 27 monthly decisions.

### Three strategy variants

| Strategy | Objective | Evaluation |
|---|---|---|
| `alpha_only` | Forecast alpha; no explicit optimizer risk or cost term | Actual portfolio cost estimates still deducted ex post |
| `alpha_risk` | Forecast alpha minus benchmark-relative variance penalty | Actual portfolio cost estimates still deducted ex post |
| `alpha_risk_cost` | Forecast alpha minus risk and modeled spread/impact penalties | Actual portfolio cost estimates still deducted ex post |

The monthly benchmark series is also calculated. All variants share long-only, budget, single-name, sector, turnover and ADV participation bounds; therefore differences come from objectives, not a different feasible set.

## 3. Precommitted summary statistics

- Out-of-sample cross-sectional Spearman **rank IC** each month and its mean/dispersion.
- Annualized CAGR (compounded), active simple monthly mean return × 12, ex-post active tracking error, information ratio and max drawdown.
- Mean monthly **one-way** turnover = half of the sum of absolute portfolio weight changes.
- Spread costs and nonlinear market-impact costs: per period and summed over the horizon (summed fractions are **not** compounded wealth effects).
- Ex-ante active tracking error and variance attribution to factor groups and stock-specific residuals.
- Forecast active alpha (model expectation) versus subsequent realized active performance.

See [`examples/synthetic/EXPERIMENT_REPORT.md`](../examples/synthetic/EXPERIMENT_REPORT.md) for the actual generated table and [`summary.json`](../examples/synthetic/summary.json) for machine-readable values.

## 4. What the current synthetic example shows

The included seed's reported synthetic 27-month out-of-sample results:

| Strategy | CAGR | Information ratio | Mean one-way monthly turnover | Sum of estimated monthly cost fractions |
|---|---:|---:|---:|---:|
| benchmark | 2.14% | n/a | 0.00% | 0.00% |
| alpha_only | 6.37% | 1.039 | 32.12% | 1.17% |
| alpha_risk | 5.70% | 0.883 | 31.64% | 1.11% |
| alpha_risk_cost | 5.97% | 0.944 | 30.58% | 1.07% |

Here, cost-aware optimization reduces turnover and simulated forecast transaction costs relative to the cost-blind alpha+ risk variant, but does **not** generate the highest CAGR in this simulation. The objective itself is a **forecast utility function**, not a guarantee of maximum ex-post geometric growth.

**No statistical inference about real investable alpha follows from these figures.** They are conditional on one artificial signal-generating process, one random seed, and arbitrarily chosen nuisance parameters. No p-values or market performance are claimed.

## 5. Automated testing

Latest local validation on Python 3.13: **47 passed**, **94% statement coverage**, including a CLI smoke test, across tests in `tests/`. Tests include:

- Fail-closed data validation for duplicate names, bad dates, invalid liquidity, benchmark sums, varying constituent sets, delayed feature availability.
- Leave-one-out peer calculations, z-score finite output, and other signal transformations.
- Ridge forecasts trained only on completed prior dates; explicit five-layer blend and expected output shape.
- PSD and symmetry of the estimated stock covariance; factor-group variance attribution adds up.
- Zero-trade cost, spread/impact nonnegativity and cost scaling with participation.
- Long-only/full-investment/sector/active/turnover/ADV constraints; infeasible inputs rejected.
- Cost-aware versus cost-blind portfolio trade comparison on a controlled fixture.
- Strict forecast training cutoff, month-end weight drift, net cost accounting and generated outputs.

Run: `python -m pytest -q --cov=acadian_arc --cov-report=term-missing`. See CI configuration in `.github/workflows/ci.yml`.

## 6. Key unperformed empirical studies

**Real historical backtest:** Obtain point-in-time security identifiers, index membership, fundamental releases, price/volume, spread histories and delisting outcomes. Rework the fixed-universe data contract before running a changing index. Publish provenance and data rights.

**Alpha ablations:** Compare stock only, peers only, macro only, all five with fixed blending, all five with training-period-only blend estimation. Use rolling one-month rank IC and long-short decile spread before portfolio optimization.

**Cost sensitivity:** Sweep AUM, impact coefficient `k`, execution days, ADV caps, portfolio turnover limits, and spreads. Plot cost versus gross alpha and risk exposures. *Do not optimize the hyperparameters on held-out test outcomes.*

**Risk decomposition:** Evaluate the effect of industry/style neutrality, risk penalty strength, covariance lookback, and shrinkage versus sample covariance. Report both mean and tail exposure.

**Statistical robustness:** Multiple non-overlapping market regimes, block bootstrap confidence intervals for active returns, turnover confidence bands, robustness to sector concentration and transaction cost assumptions.

**Execution realism:** Use genuine intraday quotes/trades to calibrate spread and impact; report implementation shortfall against arrival price. The present `order_profile.csv` is a bucket *illustration*, not recorded fills.

## 7. Responsible claims for a GitHub résumé

**Accurate:** “Developed a five-layer Ridge equity alpha forecasting system, shrinkage factor-risk estimator, nonlinear liquidity-aware optimizer and monthly walk-forward backtester; validated the software with automated unit and integration tests on a reproducible synthetic research dataset.”

**Inaccurate:** “Replicated Acadian's trading algorithm,” “discovered 6% annual alpha,” “proved transaction-cost optimization outperforms,” or “backtested a real global equity universe.”

By separating prototype validation from empirical financial validation, the project becomes more credible to quant researchers and graduate admissions reviewers.
