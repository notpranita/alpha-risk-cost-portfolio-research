"""Rolling cross-sectional factor-return estimation + PSD factor covariance.

For each historical month, uses its own exposures, not the current exposure
matrix applied backward to old realized returns.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass
class FactorRisk:
    covariance: np.ndarray
    exposures: np.ndarray
    factor_covariance: np.ndarray
    specific_variance: np.ndarray
    factors: list[str]

    def breakdown(self, active: np.ndarray) -> dict[str, float]:
        active = np.asarray(active, dtype=float)
        e = self.exposures.T @ active
        marginal = e * (self.factor_covariance @ e)
        parts: dict[str, float] = {}
        for name, contribution in zip(self.factors, marginal):
            group = ("market" if name == "market" else
                     "industry" if name.startswith("sector:") else
                     "country" if name.startswith("country:") else
                     "generic_style" if name.startswith("style:") else "alpha_factor")
            parts[group] = parts.get(group, 0.) + float(contribution)
        parts["specific"] = float((active**2) @ self.specific_variance)
        parts["total_variance"] = float(active @ self.covariance @ active)
        return parts


def _factor_names(frame: pd.DataFrame) -> list[str]:
    sectors = sorted(frame["sector"].unique())
    countries = sorted(frame["country"].unique())
    return (["market"] + [f"sector:{x}" for x in sectors[1:]] +
            [f"country:{x}" for x in countries[1:]] +
            ["style:value", "style:momentum", "alpha:stock_specific"])


def exposures_for(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    cols = []
    for name in names:
        if name == "market":
            cols.append(np.ones(len(frame)))
        elif name.startswith("sector:"):
            cols.append((frame["sector"] == name.split(":", 1)[1]).astype(float).to_numpy())
        elif name.startswith("country:"):
            cols.append((frame["country"] == name.split(":", 1)[1]).astype(float).to_numpy())
        elif name == "style:value":
            cols.append(frame["value_z"].to_numpy(dtype=float))
        elif name == "style:momentum":
            cols.append(frame["momentum_z"].to_numpy(dtype=float))
        elif name == "alpha:stock_specific":
            cols.append((.25 * frame["quality_z"] + .25 * frame["growth_z"] + .50 * frame["low_vol_z"]).to_numpy(dtype=float))
        else:
            raise ValueError(f"Unknown factor {name}")
    return np.column_stack(cols)


def estimate_factor_risk(design: pd.DataFrame, date, *, lookback: int = 18,
                         min_observations: int = 12, residual_floor: float = .00002) -> FactorRisk:
    t = pd.Timestamp(date)
    history = design.loc[design["date"] < t]
    current = design.loc[design["date"] == t].sort_values("ticker")
    if current.empty:
        raise ValueError("Date absent")
    dates = sorted(history["date"].unique())[-lookback:]
    if len(dates) < min_observations:
        raise ValueError("Insufficient prior realized returns")
    names = _factor_names(design)
    ordered_tickers = current["ticker"].tolist()
    f_returns = []
    residuals = []
    for past in dates:
        rows = history.loc[history["date"] == past].set_index("ticker").loc[ordered_tickers].reset_index()
        X = exposures_for(rows, names)
        y = rows["next_return"].to_numpy(dtype=float)
        # Gentle regularization prevents unstable factor-return regression.
        coef = np.linalg.solve(X.T @ X + .10 * np.eye(X.shape[1]), X.T @ y)
        f_returns.append(coef)
        residuals.append(y - X @ coef)
    f_returns = np.asarray(f_returns)
    residuals = np.asarray(residuals)
    F = LedoitWolf().fit(f_returns).covariance_ + np.eye(len(names)) * 1e-9
    spec = np.maximum(residuals.var(axis=0, ddof=1), residual_floor)
    X_now = exposures_for(current, names)
    cov = X_now @ F @ X_now.T + np.diag(spec)
    return FactorRisk((cov + cov.T) / 2, X_now, F, spec, names)
