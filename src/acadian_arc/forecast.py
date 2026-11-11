"""Five disjoint Ridge models: stock, peer, country, industry, country×industry.

Ridge models are academic proxies; their coefficients are estimated only on
completed earlier periods. Blending weights are fixed in advance.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .signals import STOCK_COLUMNS, PEER_COLUMNS

LAYER_FEATURES = {
    "stock": STOCK_COLUMNS,
    "peer": PEER_COLUMNS,
    "country": ["country_cycle_z"],
    "industry": ["industry_cycle_z"],
    "intersection": ["intersection_cycle_z"],
}
DEFAULT_BLEND = {"stock": .55, "peer": .20, "country": .08, "industry": .09, "intersection": .08}


@dataclass
class ForecastResult:
    forecasts: pd.DataFrame
    train_end: pd.Timestamp
    train_rows: int


def predict_asof(design: pd.DataFrame, decision_date, *,
                 min_train_months: int = 12, ridge_penalty: float = 50.,
                 blend: dict | None = None) -> ForecastResult:
    t = pd.Timestamp(decision_date)
    dates = pd.to_datetime(design["date"])
    prior = design.loc[dates < t]
    current = design.loc[dates == t].sort_values("ticker")
    if current.empty:
        raise ValueError("Decision date not in panel")
    if prior["date"].nunique() < min_train_months:
        raise ValueError("Insufficient completed prior months")
    weights = dict(DEFAULT_BLEND if blend is None else blend)
    if set(weights) != set(LAYER_FEATURES) or not np.isclose(sum(weights.values()), 1):
        raise ValueError("Blend weights must contain 5 layers and sum to one")
    if (np.array(list(weights.values())) < 0).any():
        raise ValueError("Blend weights must be nonnegative")
    if not np.isfinite(prior["next_return"].to_numpy()).all():
        raise ValueError("Training labels must be known at decision date")
    forecasts = current[["ticker", "date"]].reset_index(drop=True).copy()
    for name, columns in LAYER_FEATURES.items():
        model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_penalty))
        model.fit(prior[columns].to_numpy(), prior["next_return"].to_numpy())
        pred = model.predict(current[columns].to_numpy())
        # The benchmark-neutral optimization needs only cross-sectional alpha.
        forecasts[name] = pred - pred.mean()
    forecasts["alpha_monthly"] = sum(weights[k] * forecasts[k] for k in LAYER_FEATURES)
    forecasts["alpha_monthly"] = forecasts["alpha_monthly"].clip(-.10, .10)
    return ForecastResult(forecasts, pd.Timestamp(prior["date"].max()), len(prior))
