# Formal Methodology and Model Specifications

## 1. Objective, hypotheses, and scope

An empirical study asks whether incorporating explicit risk and modeled implementation costs improves a forecast-driven US large-cap portfolio. Four strategies are evaluated on the **same dynamically reselected 40-name liquid basket**: equal-weight benchmark, alpha-only, alpha plus factor risk, and alpha plus risk and trading costs. The historical stock selection universe contains the current S&P 500 members, and this matters to every inference.

The disclosure basis is Acadian Asset Management, Investor Forum (19 May 2026), slides 19–23: stock-specific, peer and macro return forecasting; explicit uncompensated and style risk; proprietary alpha risk; dynamic transaction-cost models; and alpha-preserving trading. These are an architecture reference, not reproducible proprietary specifications.

H1: rolling cross-sectional return forecasts have positive mean rank information coefficient. H2: factor-aware optimization improves downside or risk-adjusted outcomes relative to alpha-only. H3: transaction-cost optimization reduces turnover and predicted cost. H4: net historical active performance is significantly greater than zero. All hypotheses can fail.

## 2. Market data provenance

Stock-price and OHLCV dataset: Aethereal current-member S&P 500 (2015-01-02 to 2026-11-10; 1,444,124 rows; 503 distinct tickers, USD, adjusted close). Fama/French 5 monthly factor portfolios, one-month risk-free rate, and momentum factor and 49 value-weight industry return portfolios from the author's Sept 2026 CRSP output. Raw factor files report returns in **percent**; ingestion divides by 100 and replaces missing-value sentinels `-99.99` and `-999` with null. All four source archives are preserved under `data/raw`.

Survivorship: a stock is present historically only if it belongs to the current index constituent dataset. That is **not** equivalent to historical index composition, and historic delistings and former constituents are not rescued by dynamic liquidity ranking.

## 3. Timestamp alignment and security selection

Decision timestamp: after month *t* final trading-day close. A target weight is not presumed executable at that observed closing print. Entry occurs at the first session's **adjusted open** in *t+1*, and end-of-hold/next rebalance at the first session's adjusted open in *t+2*. The dataset contains open, close, and adjusted close but not a separate adjusted open. Accordingly, the operational adjusted-open proxy is `open × (adj_close / close)` on that calendar date. This incorporates split/dividend adjustments from the vendor's data and cannot reproduce ex-ante dividend knowledge or actual open auction fills.

For each stock i and signal month t, the training and realized target is:

`R_(i,t+1) = adjusted_open_(i, first trading day of t+2) / adjusted_open_(i, first trading day of t+1) − 1`.

No missing future holding return is replaced with zero; missing returns cause the backtest to stop instead of silently overstating performance. Eligible securities must have 252 previously observed trading dates, sufficient valid features and a trailing ADV estimate. Of the available current-member candidates each month, choose 40 ranked by trailing 20-day dollar ADV, computed using information available by month-end t. Benchmark is equal-weight in these same 40 stocks, rebalanced monthly; its fees are not set to zero. The sample is 71 decisions (2020-01 to 2025-11), corresponding to open-to-open realized holdings approximately February 2020 to December 2025.

## 4. Cross-sectional return features

Signals are evaluated at month t close (before entering at t+1 open) and percentile-ranked cross-sectionally to [−0.5, 0.5]. `mom12 = P_(t−1)/P_(t−12) − 1` (longer-horizon momentum excluding the latest month); `mom6 = P_(t−1)/P_(t−6) − 1`; reversal is negative close-to-close return during t; low-volatility is minus trailing daily return standard deviation (63 sessions) times √21; low-range is minus trailing 20-day median relative high-to-low range; log-liquidity is log trailing 20-session ADV.

Peer proxy: for every signal month, construct trailing 12-month return correlations among the 100 historically most liquid current-member candidates, requiring at least eight paired historical monthly returns. For each stock, take the five other stocks with highest correlations and average their publicly measurable six-month momentum. This is a *statistical peer network*, **not** a fundamental supply-chain or economic industry network.

Macro context: the Fama/French market excess return's past six-month cumulative return is interacted with a stock's past six-month momentum. Separately, trailing three-month mean returns of the 49 published industry portfolios are used to compute cross-industry dispersion; this is interacted with momentum. These are simple empirical proxies for changing market/industry environments; no geographic forecast or country–industry interaction model is claimed. The market-beta signal is deliberately **excluded from stand-alone alpha features**, consistent with the disclosure's distinction between market risk and rewarded selection risk.

**No accounting value/quality/earnings/growth factors are estimated:** point-in-time financial statements and publication dates have not been supplied.

## 5. Ridge return forecasting

Within each rebalance, train pooled stock/month rows over approximately 48 historical months, excluding months t and t−1. A feature observation from month t−1 has a target return ending at the opening of month t+1, which is **not** known at the close of month t. The latest admissible training feature month is t−2, whose target is complete by the beginning of month t. The model uses scikit-learn StandardScaler fitted **only on training rows** then Ridge(alpha=100) on the 9-ranked feature columns. Its intercept is estimated; forecast output is clipped to [−0.15, 0.15] monthly. No optimization of ridge penalties against the eventual test returns is performed. Predict all currently selected names; use **Spearman cross-sectional correlation** of each month's predicted and next-month realized 40 stock returns to assess forecasting skill. Monthly IC t-statistics and p-values use HAC/Newey–West standard errors (3 lags). A negative or insignificant IC invalidates claims of demonstrated selection alpha.

## 6. Benchmark-relative risk decomposition

Model excess stock returns on trailing 36 monthly Fama/French factors: Mkt−RF, SMB, HML, RMW, CMA, MOM. Each security i is fitted by time-series OLS with an intercept and factor exposures β_i; floor residual variance to 0.0004 (monthly squared return). Shrink the six-factor covariance with Ledoit–Wolf estimator. Then:

`Sigma = B Sigma_F B^T + diag(sigma_epsilon_i^2)`.

The intended risk penalty is **active** variance `(w−b)^T Sigma (w−b)`, not absolute total variance. Explicit risk constraints limit active market β to [−0.15,0.15] and each active five-style-factor β to [−0.30,0.30]. All factors are estimates using only historical monthly returns available by t close. Residual model risk is *not* equivalent to proprietary alpha risk: idiosyncratic expected returns remain unverified, and statistical residuals can contain omitted factors.

## 7. Return–risk–cost optimization

For each decision t, solve:

`max_w  forecast^T w − (12/2)(w−b)^T Sigma (w−b) − TC(w−w_prev)`.

Constraints: all weights nonnegative, sum(weights)=1, each ≤0.10, plus the active-factor limits for the risk-aware and cost-aware variants. SLSQP in SciPy uses an analytical objective gradient and analytical Jacobians for the equality and linear inequalities. Alpha-only is fitted with a small quadratic stabilization term (0.001 ||w−b||^2) and no factor constraints. Risk-only omits predicted cost from its objective. All portfolios are evaluated *net of trading costs*, including the equal-weight benchmark.

`TC = Σ_i [ 0.25 × Range20_i × |Δw_i| + 0.08 × σ_month_i × sqrt(AUM/ADV20_i) × |Δw_i|^(3/2) ]`.

AUM=USD 10 million. This cost is a **scenario/proxy model** only. The first term estimates a fraction of daily high-to-low range; intraday range is neither a quoted nor effective half-spread. The second term is convex in absolute order size with an ADV participation nonlinearity, not fitted to real impact. For dropped securities, additional liquidation cost of 50 bps × exited weight is charged. No commission, funding, market-maker rebate, bid/ask quote, trade fills, tax, or borrow charges are present. Optimizer result is rejected if it violates numerical feasibility.

## 8. Backtest accounting and evaluation

Next-month holdings are drifted only using that month’s observable first-open to final-close price change before its month-end decision, normalized into next-decision pre-trade holdings, compared to new targets, and charged costs. Because orders execute at the following first open, this is a last-known-close pre-trade approximation; the overnight opening gap and subsequent price slippage are not captured. Gross portfolio return is `w^T R`; modeled net return is `(1−TC)(1+gross_return)−1`. Initial funding incurs estimated opening costs. Monthly one-way turnover is half of the total absolute trade weight, including exits. Return distributions, CAGR, standard deviation ×√12, maximum drawdown and forecast tracking error are reported. Sharpe without risk-free deduction is explicitly labeled as such; factor-regression alphas use an aligned next-period monthly risk-free rate.

The six-factor monthly intercept regression uses strategy return for the holding month (t+1) against Fama/French factors for t+1, with three-lag HAC standard errors. Active return inference uses strategy-minus-equal-weight benchmark monthly net return, not against a stock index. Costs are re-scaled on **fixed historical weights** for sensitivity; that is not a full reoptimization under alternative assumed AUM or costs.

## 9. Research safeguards, remaining biases, and planned improvements

Safeguards implemented: historical windows without future returns; month-end-close signal to subsequent month-open execution; no silent missing returns; separate code modules; explicit constraints and finite-difference derivative tests; hypothesis tests rather than cherry-picking cumulative growth; fixed research configuration and preserved raw input files.

Remaining risks: complete survivorship bias; adjusted open proxy; vendor corporate-action revisions; Fama/French data revisions (Sept 2026 version used to study 2020); uncalibrated transaction costs; lookback/hyperparameter selection without nested cross-validation; small and concentrated 40-stock benchmark; actual microstructure spreads and capacity unavailable; no sector tags or point-in-time fundamentals; and no live order implementation. None of these limitations is cured by 23 automated tests.
