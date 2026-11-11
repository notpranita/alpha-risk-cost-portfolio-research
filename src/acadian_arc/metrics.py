"""Return attribution and risk measures. Input returns must be per-period simple."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def summary_metrics(returns, benchmark, *, annualization=12):
    r = np.asarray(returns, float); b = np.asarray(benchmark, float)
    if r.ndim != 1 or r.shape != b.shape or not len(r):
        raise ValueError("Return vectors must have same nonzero length")
    if (r <= -1).any() or (b <= -1).any():
        raise ValueError("Simple returns cannot be <= -100%")
    active = r-b
    wealth = np.cumprod(1+r)
    peaks = np.maximum.accumulate(np.r_[1., wealth])[1:]
    te = active.std(ddof=1)*np.sqrt(annualization) if len(r)>1 else float("nan")
    return {
        "cagr": float(wealth[-1]**(annualization/len(r)) - 1),
        "benchmark_cagr": float(np.prod(1+b)**(annualization/len(b)) - 1),
        "annualized_mean_active_return": float(active.mean()*annualization),
        "tracking_error": float(te),
        "information_ratio": float(active.mean()*annualization/te) if te>1e-12 else None,
        "max_drawdown": float(np.min(wealth/peaks-1)),
        "cumulative_return": float(wealth[-1]-1),
    }


def rank_ic(prediction, realized):
    pred = np.asarray(prediction, float); actual = np.asarray(realized, float)
    if len(pred) < 3 or np.std(pred)<1e-12 or np.std(actual)<1e-12:
        return float("nan")
    return float(spearmanr(pred, actual).statistic)
