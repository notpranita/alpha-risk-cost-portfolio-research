# Mathematical methodology and implementation audit

**Scope:** Independent replication of the **architecture**, not of Acadian's proprietary forecasts, calibrated risk model or execution systems. This document explains the actual code. See the SEC [Investor Forum slides 19–23](https://www.sec.gov/Archives/edgar/data/1748824/000162828026036160/a2026-masterfilexirforum.htm) for the public conceptual inspiration.

## A. Research protocol and time conventions

Each observation is one `ticker` at decision date `t` (month-end). Numeric predictor values are assumed available **by** `t`; `next_return` is simple total return earned **after** `t` and before the next decision. Backtest as-of date `t`:

1. `validate_panel()` checks schema, uniqueness, finite fields, full benchmark weights, panel consistency and optional `available_date <= date`.
2. `design_matrix()` creates normalized signals using **only same-date features**, with no reference to the future label.
3. `predict_asof()` trains on `date < t` and predicts for date `t`. In this panel contract, the next-period outcome for prior date `t−1` has completed by `t`.
4. `estimate_factor_risk()` fits historical *cross-sectional* factor returns to past realized returns only and computes the **monthly** covariance available at `t`.
5. `optimize()` determines portfolio weights at `t`. It never receives the `t` label.
6. The backtest then uses realized `t` → `t+1` returns **for evaluation**, deducts forecast trading costs once, and drifts target weights to the beginning-of-next-period composition.

This is an abstract **decision-date close / next-month return** model, not an intraday execution simulator. Public live-data implementations need to specify when features were known and when orders can actually be filled.

## B. Feature preprocessing and five return-forecast models

### Stock-specific signals

Five numeric raw features are provided by the panel: `value`, `quality`, `growth`, `momentum`, and `low_vol`. They are *precomputed inputs*, not formulas deriving point-in-time accounting ratios. On each date `t` and within each sector:

1. Winsorize each feature at 2.5th and 97.5th percentile (small groups will have weak percentile estimates).
2. Compute `z = (x − mean(x)) / std(x)` with population standard deviation, returning zero if variance is negligible.
3. Clip normalized feature scores to `[−3,3]`.

`signals.py` provides normalization. Sign conventions for each factor must be consistently positive for stronger expected alpha in an externally sourced data pipeline.

### Peer layer

For `quality` and `momentum`, compute leave-one-out means among other stocks in the **same sector**: `peer_i = (sum_sector(x) − x_i)/(N_sector − 1)`. The peer mean is cross-sectionally winsorized/z-scored. This is a simple reproducible industry-peer **proxy**, not Acadian's proprietary networks, alternative peer definitions, or published number of signals. A sector with one constituent receives a zero fallback.

### Country, industry and intersection layers

The caller supplies `country_cycle`, `industry_cycle`, and `intersection_cycle` as numerical proxy signals, and the code z-scores them cross-sectionally at each date. These variables are synthetic in the bundled generator. The program does **not** itself estimate macroeconomic cycles from inflation, rates, or other releases.

### Regression

For each of the five layers `m` (stock, peer, country, industry, country×industry), fit a separate `StandardScaler` followed by L2-regularized `Ridge` on only preceding decision-date rows:

`beta_m = argmin_beta ||y_train − X_train,m beta||² + rho ||beta||²`

where the pipeline centers/scales features before fitting. Default `ridge_penalty=50` is an independent assumption. The stock, peer and macro submodels all target realized next-month stock returns but use disjoint input columns. The **component predictions are cross-sectionally demeaned** on the date being forecast, removing intercept/market-level common predictions from active-alpha calculations.

Aggregate monthly forecast:

`alpha_t = 0.55 f_stock + 0.20 f_peer + 0.08 f_country + 0.09 f_industry + 0.08 f_intersection`.

The predefined weights sum to one. They are **not learned or disclosed Acadian weights**. Final values are clipped to `[-0.10,0.10]` monthly forecast return. No tuning/validation search of blend weights is performed: these are controls for the software demonstration, not researched optimal settings. `forecast.py` stores `train_end` for auditability.

**Caution:** The panel can have correlated, redundant signals; separate Ridge models and weighted averaging are not guaranteed to produce an efficient forecast. A real study should estimate each signal's out-of-sample IC and test model combination weights on earlier validation windows only.

## C. Risk model

The model makes benchmark-relative ex-ante covariance explicit, using *historical* stock returns and *historical* exposures.

At each historical month `s < t`, the matrix `B_s` has columns:

- global-market constant (`1` for all securities),
- sector dummy variables (with first sector omitted),
- country dummy variables (with first country omitted),
- `style:value` and `style:momentum` normalized features,
- `alpha:stock_specific` = `0.25 quality_z + 0.25 growth_z + 0.50 low_vol_z`.

Cross-sectional factor returns are estimated using lightly regularized least squares:

`f_s = (B_s' B_s + 0.10 I)^(-1) B_s' r_(s→s+1)`.

The factor covariance `F` is estimated from the most recent `risk_lookback=18` months of `f_s` with scikit-learn's `LedoitWolf` covariance shrinkage; a 1e-9 diagonal jitter protects against near singularity. Stock-specific variances are sample variances of past residuals, floored at `0.00002`. The total **monthly** covariance becomes:

`Sigma_t = B_t F B_t' + diag(specific_variance)`.

For active weights `a = w − b`, ex-ante monthly active variance is `a' Sigma a`. Factor contribution is `q_j (F q)_j`, with `q = B_t' a`. Specific contribution is `sum_i a_i² D_ii`. The sum of factor contributions and specific contribution equals total active variance (up to floating point tolerances). Some *individual* factor contributions may be negative because correlated-factor variance attribution is signed.

The implementation groups contributions as market, industry, country, generic style, alpha factor and stock-specific. **This is an illustrative grouping, not Acadian's risk policy**. Factor exposures may be non-orthogonal and a small example universe makes factor-return estimation noisy. Risk beta to rates and FX is **not explicitly modeled**.

## D. Nonlinear transaction cost model

For each stock let `delta_i = w_i − p_i` be the signed target minus pretrade weight. The model uses:

`SpreadCost_i = (spread_bps_i / 20000) * |delta_i|`

(half of **full** bid–ask spread expressed as a fraction) and

`ImpactCost_i = k * (sigma_month_i / sqrt(21)) * sqrt(AUM / (ADV_i * execution_days)) * |delta_i|^(3/2)`.

`TC = sum_i (SpreadCost_i + ImpactCost_i)`.

This is a square-root *participation-sensitive* illustrative impact model. Assumes `ADV` in **USD/day**, monthly volatility measured in fraction-of-return units, 21 trading days/month, and `AUM` in USD. `spread_bps` is the full quoted spread, not half spread. The exponent `3/2` appears after combining trade dollars `|delta|` with square-root size/ADV participation. The coefficient `impact_k=0.12` is **not empirically calibrated**. Cost and participation are evaluated using intended aggregate trading over `execution_days`; observed time-varying intraday liquidity and fills are unavailable.

## E. Optimization

Let `b` be benchmark weights, `p` pretrade portfolio weights, `w` target weights, `alpha` blended monthly return forecast, and `Sigma` the monthly covariance.

`max_w alpha'(w−b) − (lambda/2)(w−b)'Sigma(w−b) − gamma TC(w−p)`

Constraints:

- Long only: `w_i >= 0`.
- Fully invested: `sum_i w_i = 1`.
- Maximum position: `w_i <= max_weight` (configuration: `0.25`).
- Maximum active weight: `|w_i−b_i| <= max_active` (configuration: `0.11`).
- Total *two-way* turnover: `sum_i |w_i−p_i| <= max_turnover` (configuration: `0.65`). `mean_one_way_turnover` in the report divides the realized two-way figure by 2.
- ADV participation: `AUM * |w_i−p_i| / (ADV_i * execution_days) <= max_participation` (configuration: `0.08`).
- Sector active tilt: `|sum_(i∈sector) (w_i−b_i)| <= sector_tolerance` (configuration: `0.10`).

The optimizer uses SciPy's **SLSQP** with an explicit derivative of the objective. Auxiliary variables `u_i >= |w_i−p_i|` linearize the absolute-value constraints. Their nonnegative bounds and constraints are enforced with SciPy `Bounds` and `LinearConstraint`.

**Important nuance:** When `gamma=0` (cost-unaware), optimizing with auxiliaries is still feasible because cost terms are not included in the objective; actual realized trades are recomputed using `w−p`. The three strategies use identical feasible sets; they differ only in alpha/risk/cost objective weights:

| Strategy | Risk penalty | Cost penalty |
|---|---:|---:|
| `alpha_only` | 0 | 0 |
| `alpha_risk` | `lambda` | 0 |
| `alpha_risk_cost` | `lambda` | 1 |

Trading costs **are still deducted in the backtest** for all strategies. Thus `alpha_only` means *cost-blind decision*, **not free trading**. This isolates the contribution of cost awareness.

**Solver risks:** Optimizer success alone is not proof of correctness. Code checks full-investment, holding, sector, active, turnover and ADV feasibility. SLSQP may fail on legitimate inputs; it then raises a `RuntimeError` rather than silently committing a portfolio.

## F. Monthly walk-forward accounting

At each decision month, target weights are applied to next month's **gross simple returns** `r`:

`gross = sum_i w_i r_i`.

First estimate one-off portfolio execution costs `C` from `delta = w−p`, then compute:

`net = (1−C) * (1+gross) − 1`.

The multiplicative form avoids inconsistent simple-return treatment across periods. When the next month passes, weights are drifted *before* the next rebalance:

`p_(t+1),i = w_(t,i) * (1+r_(t,i)) / (1+gross_t)`.

Benchmark weights and benchmark month returns are separately provided per decision date. Benchmark trade costs are assumed zero; this is a chosen comparison convention, **not necessarily fair for live active/passive cost comparisons**.

Annualization of monthly results:

- `CAGR = (prod_t(1+r_net,t))^(12/T) − 1`.
- `annualized mean active return = 12 * mean(r_strategy − r_benchmark)`.
- `TE = sqrt(12) * stdev(r_strategy − r_benchmark)`.
- `IR = annualized mean active / TE` (may be undefined).
- `max drawdown` derived from running compounded-wealth peaks.
- `monthly rank IC = Spearman(predicted stock alpha, realized next-month stock return)` within each date.

The generated report includes both expected ex-ante alpha and achieved ex-post returns so the two are never conflated.

## G. Tests and boundaries

The test suite checks coefficient preparation, future-feature rejection (when `available_date` exists), dated regression training, factor/covariance properties, transaction costs, optimization feasibility, accounting identity, drift and file generation. **No test can prove absence of all forms of data leakage in a user-supplied dataset**: e.g., an incorrectly tagged earnings date can pass a syntactic date check. See [DATA_SCHEMA.md](DATA_SCHEMA.md).
