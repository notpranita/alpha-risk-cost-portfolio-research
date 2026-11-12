# Data Quality and Provenance Audit

Source archives preserved byte-for-byte in `data/raw`; SHA-256 manifest below.

```json
{
  "F-F_Research_Data_5_Factors_2x3_CSV.zip": {
    "bytes": 11971,
    "sha256": "17d7a575e71d6764a70ddec3f8d887756c0eebf3a60f64d6ba95d2adde3fed26"
  },
  "F-F_Momentum_Factor_CSV.zip": {
    "bytes": 5603,
    "sha256": "27ed60e5b2caa9ea334a261cddfa864a9926f6bf45e9677c95dfd835c5abb315"
  },
  "sp500-stock-prices.parquet": {
    "bytes": 44703673,
    "sha256": "2df8fcf0a440eb418bd7df4d614add6f8a10684ac0e19d5ed81685ae7203bf82"
  },
  "49_Industry_Portfolios_CSV.zip": {
    "bytes": 489861,
    "sha256": "d1a3b0b9896e1e633ed2aa85e24ca144d7998f8ab2ff344e2379383ebeb7b836"
  }
}
```

The uploaded S&P price archive decodes to **1,444,124 daily ticker/date records, 503 symbols**, from 2015-01-02 through 2026-11-10. Its eight columns are date, ticker, open, high, low, close, adj_close, volume. No missing fields were found in these 8 columns when loaded, and adjusted close is strictly positive. This **does not** mean the panel is balanced or that delisted stocks are represented.

The uploaded Kenneth French archives are labeled September 2026 CRSP output. Parsed monthly 5 factors + one risk-free rate + momentum returns contain 759 months, July 1963–September 2026; 49 value-weighted industry return portfolios contain 1,203 months, July 1926–September 2026. All percent returns were divided by 100 and missing-value sentinels replaced. The code reads only the first *value-weighted monthly industry* section; it does not combine later equal-weight/annual sections.

**Data definitions:** adjusted open is derived as `open*adj_close/close`, trailing ADV is 20-day mean `close*volume`, and the intraday-range proxy is 20-day median `(high-low)/close`. No bid/ask prices, execution data, sector mapping, point-in-time fundamentals, or delisted-name membership file were supplied.
