"""Strict panel validation and deterministic simulated equity research universe."""
from __future__ import annotations
import numpy as np
import pandas as pd

STOCK_FEATURES = ("value", "quality", "growth", "momentum", "low_vol")
MACRO_FEATURES = ("country_cycle", "industry_cycle", "intersection_cycle")
REQUIRED = ("date", "ticker", "sector", "country", *STOCK_FEATURES,
            *MACRO_FEATURES, "benchmark_weight", "adv_usd", "spread_bps",
            "market_cap_usd", "next_return")


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Input means as-of features at t, and realized t→t+1 label (NOT a feature)."""
    missing = set(REQUIRED) - set(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].isna().any():
        raise ValueError("Invalid dates")
    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate (date,ticker) rows")
    if df[list(REQUIRED)[4:]].select_dtypes(include="number").empty:
        raise ValueError("Numeric fields are missing")
    numeric = [*STOCK_FEATURES, *MACRO_FEATURES, "benchmark_weight", "adv_usd", "spread_bps", "market_cap_usd", "next_return"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="raise")
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise ValueError("NaN or infinite numeric input")
    if (df["adv_usd"] <= 0).any() or (df["market_cap_usd"] <= 0).any():
        raise ValueError("ADV and market capitalization must be positive")
    if (df["spread_bps"] < 0).any() or (df["benchmark_weight"] < 0).any():
        raise ValueError("Spreads and benchmark weights must be non-negative")
    if (df["next_return"] <= -1).any():
        raise ValueError("Next-period simple return must exceed -100%")
    dates = sorted(df["date"].unique())
    ref = set(df.loc[df["date"] == dates[0], "ticker"])
    if any(set(df.loc[df["date"] == d, "ticker"]) != ref for d in dates):
        raise ValueError("Prototype expects a fixed full-history universe; dynamic universe not supported")
    if not np.allclose(df.groupby("date")["benchmark_weight"].sum().to_numpy(), 1, atol=1e-7):
        raise ValueError("Benchmark weights must sum to 1 on each date")
    if "available_date" in df.columns:
        df["available_date"] = pd.to_datetime(df["available_date"])
        if df["available_date"].isna().any() or (df["available_date"] > df["date"]).any():
            raise ValueError("Features contain look-ahead: available_date after decision date")
    if (df.groupby("ticker")["sector"].nunique() > 1).any() or (df.groupby("ticker")["country"].nunique() > 1).any():
        raise ValueError("Prototype expects static sector/country classification")
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def generate_synthetic_panel(months: int = 48, seed: int = 17,
                             names_per_group: int = 2) -> pd.DataFrame:
    """A *fictional* universe with a deliberately planted weak factor-return relationship.

    Demonstrates engineering only. It must not be interpreted as discovered alpha.
    """
    if months < 18 or names_per_group < 2:
        raise ValueError("Use at least 18 months and two names per country/industry group")
    rng = np.random.default_rng(seed)
    sectors = ("Technology", "Financials", "Healthcare", "Industrials")
    countries = ("US", "GB")
    identities = [(f"SIM{i:03d}", sector, country)
                  for i, (sector, country) in enumerate((s, c) for s in sectors for c in countries for _ in range(names_per_group))]
    n = len(identities)
    dates = pd.date_range("2019-01-31", periods=months, freq="ME")
    latent = rng.normal(size=(n, len(STOCK_FEATURES)))
    cap = rng.lognormal(mean=np.log(4e9), sigma=.55, size=n)
    liquidity = rng.lognormal(mean=np.log(35e6), sigma=.40, size=n)
    ids = pd.DataFrame(identities, columns=["ticker", "sector", "country"])
    weights = cap / cap.sum()
    rows = []
    sector_cycle = {s: 0.0 for s in sectors}
    country_cycle = {c: 0.0 for c in countries}
    cross_cycle = {(s, c): 0.0 for s in sectors for c in countries}
    for date in dates:
        latent = .62 * latent + .78 * rng.normal(size=latent.shape)
        for s in sectors:
            sector_cycle[s] = .55 * sector_cycle[s] + rng.normal(scale=.9)
        for c in countries:
            country_cycle[c] = .55 * country_cycle[c] + rng.normal(scale=.9)
        for k in cross_cycle:
            cross_cycle[k] = .45 * cross_cycle[k] + rng.normal(scale=.75)
        for i, (ticker, sector, country) in enumerate(identities):
            row = {"date": date, "ticker": ticker, "sector": sector,
                   "country": country, "benchmark_weight": weights[i],
                   "market_cap_usd": cap[i], "adv_usd": liquidity[i] * rng.uniform(.8, 1.2),
                   "spread_bps": float(np.clip(8 + 30e6 / liquidity[i] * 2 + rng.normal(0, 1), 2, 35)),
                   "country_cycle": country_cycle[country], "industry_cycle": sector_cycle[sector],
                   "intersection_cycle": cross_cycle[sector, country]}
            row.update({k: float(latent[i, j]) for j, k in enumerate(STOCK_FEATURES)})
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    # Label generation ONLY. No research model gets this function's coefficients.
    from .signals import design_matrix
    design = design_matrix(frame, allow_unlabeled=True)
    signal = (.22 * design["value_z"] + .27 * design["quality_z"] +
              .17 * design["growth_z"] + .26 * design["momentum_z"] +
              .08 * design["low_vol_z"] + .12 * design["peer_quality_z"] +
              .10 * design["peer_momentum_z"] + .12 * design["country_cycle_z"] +
              .10 * design["industry_cycle_z"] + .08 * design["intersection_cycle_z"])
    next_returns = []
    for date, group in design.groupby("date", sort=True):
        common = rng.normal(.005, .025)
        sector_shock = {s: rng.normal(0, .013) for s in sectors}
        # Synthetic forecasting relationship: not observed in real financial data.
        ids_for_date = group.index
        y = (common + .010 * signal.loc[ids_for_date].to_numpy() +
             np.array([sector_shock[s] for s in group["sector"]]) +
             rng.normal(0, .040, size=len(group)))
        next_returns.extend(np.maximum(y, -.85))
    frame["next_return"] = next_returns
    frame["available_date"] = frame["date"]
    return validate_panel(frame)
