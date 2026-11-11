"""Forecast execution costs: half spread plus 3/2-power dollar-ADV impact."""
from dataclasses import dataclass
import numpy as np


@dataclass
class CostEstimate:
    spread: float
    impact: float
    total: float


def cost_coefficients(spread_bps, volatility_monthly, adv_usd, *, aum_usd=10_000_000,
                      impact_k=.12, execution_days=2):
    spread = np.asarray(spread_bps, dtype=float) / 20000
    vol_daily = np.asarray(volatility_monthly, dtype=float) / np.sqrt(21)
    adv = np.asarray(adv_usd, dtype=float)
    if (spread < 0).any() or (vol_daily < 0).any() or (adv <= 0).any():
        raise ValueError("Invalid costs inputs")
    if aum_usd <= 0 or execution_days <= 0 or impact_k < 0:
        raise ValueError("Invalid cost settings")
    coeff = impact_k * vol_daily * np.sqrt(aum_usd / (adv * execution_days))
    return spread, coeff


def transaction_cost(trades, spread_bps, volatility_monthly, adv_usd, *,
                     aum_usd=10_000_000, impact_k=.12, execution_days=2) -> CostEstimate:
    trade = np.abs(np.asarray(trades, dtype=float))
    linear, impact_coeff = cost_coefficients(spread_bps, volatility_monthly, adv_usd,
                                          aum_usd=aum_usd, impact_k=impact_k,
                                          execution_days=execution_days)
    if trade.shape != linear.shape:
        raise ValueError("Trade dimension mismatch")
    spread = float(linear @ trade)
    impact = float(impact_coeff @ np.power(trade, 1.5))
    return CostEstimate(spread, impact, spread + impact)


def profile_orders(tickers, trades, adv_usd, *, aum_usd=10_000_000, execution_days=2):
    """Illustrative order profiling; no broker router or fill simulation."""
    import pandas as pd
    dollars = aum_usd * np.asarray(trades, float)
    participation = np.abs(dollars) / (np.asarray(adv_usd, float) * execution_days)
    style = np.where(participation < .01, "patient", np.where(participation < .05, "balanced", "high-touch"))
    return pd.DataFrame({"ticker": list(tickers), "trade_usd": dollars,
                         "participation": participation, "execution_bucket": style})
