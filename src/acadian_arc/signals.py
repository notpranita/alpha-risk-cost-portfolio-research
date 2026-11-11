"""As-of feature engineering; peer signals are leave-one-out by industry."""
import numpy as np
import pandas as pd
from .data import STOCK_FEATURES, MACRO_FEATURES

STOCK_COLUMNS = [f"{k}_z" for k in STOCK_FEATURES]
PEER_COLUMNS = ["peer_quality_z", "peer_momentum_z"]
MACRO_COLUMNS = ["country_cycle_z", "industry_cycle_z", "intersection_cycle_z"]
ALL_COLUMNS = STOCK_COLUMNS + PEER_COLUMNS + MACRO_COLUMNS


def _z(x: pd.Series) -> pd.Series:
    arr = x.astype(float)
    lo, hi = arr.quantile([.025, .975])
    arr = arr.clip(lo, hi)
    sd = arr.std(ddof=0)
    return (arr - arr.mean()) / sd if sd > 1e-12 else pd.Series(0., index=arr.index)


def design_matrix(panel: pd.DataFrame, allow_unlabeled: bool = False) -> pd.DataFrame:
    """No access to the target column; uses only simultaneous, as-of predictors."""
    req = {"date", "ticker", "sector", *STOCK_FEATURES, *MACRO_FEATURES}
    if not req.issubset(panel.columns):
        raise ValueError(f"Missing feature columns: {sorted(req - set(panel.columns))}")
    frames = []
    for _, date_group in panel.groupby("date", sort=True):
        g = date_group.sort_values("ticker").copy()
        for f in STOCK_FEATURES:
            g[f"{f}_z"] = g.groupby("sector")[f].transform(_z).clip(-3, 3)
        for f in ("quality", "momentum"):
            # Peers are in the same industry; own observation is excluded.
            total = g.groupby("sector")[f].transform("sum")
            count = g.groupby("sector")[f].transform("count")
            peer = ((total - g[f]) / (count - 1).replace(0, np.nan)).fillna(0)
            g[f"peer_{f}_z"] = _z(peer).clip(-3, 3)
        for f in MACRO_FEATURES:
            g[f"{f}_z"] = _z(g[f]).clip(-3, 3)
        frames.append(g)
    return pd.concat(frames, ignore_index=True) if frames else panel.copy()
