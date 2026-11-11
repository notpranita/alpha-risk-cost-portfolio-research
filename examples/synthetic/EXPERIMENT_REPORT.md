# Automated experiment report — SYNTHETIC DATA ONLY

This report validates the software workflow. It does **not** establish investment alpha or forecast accuracy in real markets.

## Out-of-sample backtest metrics

| Strategy | CAGR | Information ratio | Mean monthly one-way turnover | Summed model costs |
|---|---:|---:|---:|---:|
| benchmark | 2.14% | n/a | 0.00% | 0.00% |
| alpha_only | 6.37% | 1.039 | 32.12% | 1.17% |
| alpha_risk | 5.70% | 0.883 | 31.64% | 1.11% |
| alpha_risk_cost | 5.97% | 0.944 | 30.58% | 1.07% |

## Research audit trail

- The forecast for t was trained only on `next_return` labels from decision dates strictly earlier than t.
- Factor risk was estimated from previously realized returns only, with time-specific historical exposures.
- Portfolio turnover uses beginning-of-period drifted weights; trading costs are deducted once at rebalance.
- Benchmark is rebalanced monthly without modeled benchmark trading fees (a deliberate comparison assumption).
- The reported execution cost is **estimated**, not a realized fill / implementation shortfall.
- Synthetic next-period returns were deliberately generated with predictive signals and must never be treated as empirical validation.

## Files

- `stock_forecasts.csv`: as-of dates, training cutoff, all five component forecasts, ex-post labels
- `risk_attribution.csv`: signed factor contributions (variance units) and total ex-ante active variance
- `transaction_costs.csv`: spread and nonlinear market impact estimates
- `portfolio_weights.csv`: pretrade, benchmark, optimized weights, and orders
- `monthly_returns.csv`: gross and net strategy returns / benchmark per period
- `order_profile.csv`: illustrative participation-based order buckets (not routing/fills)

See `docs/METHODOLOGY.md` for parameter definitions, hypotheses, and limitations.
