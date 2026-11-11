# Monthly data schema and point-in-time requirements

The CLI `--input-csv` expects **one row per (`date`,`ticker`)** with each row's *predictor values available at the stated decision date*. The target `next_return` is ex-post, **only for training completed periods and evaluating test periods**; it must not be constructed from information observed at the decision timestamp.

## Required columns

| Column | Type | Meaning / units | Critical caution |
|---|---|---|---|
| `date` | ISO date | Month-end decision timestamp | Use actual as-of date; identify execution cutoff |
| `ticker` | String | Stable panel identifier | Ticker reuse/renames require permanent identifiers |
| `sector` | String | Sector for normalization, dummies, peers and bounds | This prototype assumes sector doesn't change |
| `country` | String | Country grouping | Country classification is static in prototype |
| `value` | Float | Value signal, positive meaning stronger value opportunity | Precomputed by user; **not** automatically P/B or earnings yield |
| `quality` | Float | Positive quality factor score | Publication lags mandatory |
| `growth` | Float | Positive growth factor score | Forecast/filing timestamps mandatory |
| `momentum` | Float | Price-momentum signal | Should not use return after decision date |
| `low_vol` | Float | Positive low-volatility signal | Estimated from preceding prices only |
| `country_cycle` | Float | Country macro-cycle proxy | Release date must precede decision |
| `industry_cycle` | Float | Industry-cycle proxy | Release date must precede decision |
| `intersection_cycle` | Float | Country×industry proxy | Release date must precede decision |
| `benchmark_weight` | Float | Nonnegative weight in comparison index | Weights per month must sum to one |
| `adv_usd` | Float | Average daily **USD trade value** | Calculated from prior daily prices/volume |
| `spread_bps` | Float | Full quoted bid–ask spread **in basis points** | Not half-spread |
| `market_cap_usd` | Float | As-of market cap, USD | Included for reporting/profiling; not in model objective |
| `next_return` | Float | Subsequent month simple total return | Includes dividends; should include delisting if real data |
| `available_date` | ISO date, optional | Last time all row features were known | Must be `<= date`; conservative dates recommended |

All numerical fields must be finite; ADV and market cap positive; spreads nonnegative; benchmark weights nonnegative and sum to 1 each date; `next_return > -1`. This prototype expects **the same securities at every date** and fixed sector/country classifications. These strict restrictions intentionally surface unsupported real-world universe problems, rather than claim they are solved.

## Bias controls required before using live prices and fundamentals

1. **Survivorship bias.** Do not select today's index members and backfill historical returns. Need full historical constituents, including bankrupt/delisted names; this code does not supply them.
2. **Earnings announcement and restatement leakage.** Use the release date on which a fact was publicly knowable, not fiscal quarter-end. Historical amended filings cannot overwrite earlier information sets.
3. **Price availability and tradability.** Do not use the current month's close to compute both a signal and an order filled *at the same close* without latency/slippage assumptions. Consider next-session execution using separate daily histories.
4. **Return alignment.** At `t`, `next_return[t]` is not yet known. Model fits only `date < t`; data engineer must ensure that previous return windows have truly completed and no revised labels leak.
5. **Adjusted prices / delisting.** Use economically meaningful total return and corporate-action adjustments; cash/stock mergers and suspensions need accounting.
6. **Benchmark holdings.** Real index weights vary and rebalance, while this prototype demands a fixed full-history universe. You must explicitly redesign validation/backtesting before claiming a true benchmark test.
7. **Macro vintages.** Use first-release macro data (or historical vintages); don't plug modern revised values into previous decision dates.
8. **Liquidity availability.** Historical ADV and spreads require pretrade estimation. A present-day spread applied to 2019 is leakage.
9. **Hyperparameter fitting.** If choosing factor weights, risk penalty, forecast horizon, universe, or impact coefficient based on test returns, use a separate validation period, walk-forward nested selection, or preregistration.
10. **Licensing and provenance.** Record vendor, retrieval date, dataset revision, exchange/currency conversion, and permissible open-source redistribution. Do not upload paid market data to public GitHub without permission.

## Example conceptual CSV rows

Numbers below are merely illustrative and **not a complete valid panel** (a real file must include sufficient securities, dates and benchmark weights summing to one):

```csv
date,ticker,sector,country,value,quality,growth,momentum,low_vol,country_cycle,industry_cycle,intersection_cycle,benchmark_weight,adv_usd,spread_bps,market_cap_usd,next_return,available_date
2024-01-31,EXAMPLE1,Technology,US,0.4,0.7,0.2,0.1,0.4,0.1,0.2,0.0,0.05,35000000,11.0,7000000000,0.012,2024-01-31
```

`next_return` must be retained in the historical panel for backtest scoring but **never used to derive feature values at the same date**. The generated synthetic panel fulfills the schema automatically.

## What this program does not fetch

There is no automatic SEC XBRL, CRSP, WRDS, Compustat, Bloomberg, Refinitiv, FRED, index-constituent, or intraday TAQ integration. A public SEC report describes an investment process; it does **not** furnish point-in-time equity factor data. A substantive empirical claim requires a separate, licensed data-engineering pipeline.
