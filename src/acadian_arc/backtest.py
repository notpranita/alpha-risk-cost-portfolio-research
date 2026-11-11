"""Monthly walk-forward accounting; no same-period training labels.

At decision date t: all feature data are stamped <= t, regressions use labels
from dates < t, risk uses periods < t, then t->t+1 returns are realized.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import validate_panel
from .signals import design_matrix
from .forecast import predict_asof
from .risk import estimate_factor_risk
from .optimizer import optimize
from .costs import profile_orders
from .metrics import summary_metrics, rank_ic

STRATEGIES = ("alpha_only", "alpha_risk", "alpha_risk_cost")


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    positions: pd.DataFrame
    forecasts: pd.DataFrame
    costs: pd.DataFrame
    factor_risk: pd.DataFrame
    orders: pd.DataFrame
    summary: dict


def run_backtest(panel: pd.DataFrame, *, min_train_months=15, risk_lookback=18,
                 risk_aversion=6., aum_usd=10_000_000, impact_k=.12,
                 execution_days=2, max_weight=.16, max_active=.11,
                 max_turnover=.65, max_participation=.08, sector_tolerance=.10) -> BacktestResult:
    panel = validate_panel(panel)
    design = design_matrix(panel)
    dates = sorted(design["date"].unique())
    tickers = sorted(design["ticker"].unique())
    if len(dates) <= min_train_months:
        raise ValueError("No out-of-sample periods")
    history_end = max(min_train_months, 12)
    current_weights = {s: None for s in STRATEGIES}
    records = []; positions=[]; predictions=[]; costs=[]; factors=[]; orders=[]
    for date in dates[history_end:]:
        dt = pd.Timestamp(date)
        g = design.loc[design["date"] == dt].set_index("ticker").loc[tickers].reset_index()
        forecast = predict_asof(design, dt, min_train_months=min_train_months)
        forecast_df = forecast.forecasts.set_index("ticker").loc[tickers]
        risk = estimate_factor_risk(design, dt, lookback=risk_lookback)
        alpha = forecast_df["alpha_monthly"].to_numpy()
        b = g["benchmark_weight"].to_numpy(dtype=float)
        y = g["next_return"].to_numpy(dtype=float)  # used AFTER decisions
        vol = np.sqrt(np.diag(risk.covariance))
        spread = g["spread_bps"].to_numpy(dtype=float)
        adv = g["adv_usd"].to_numpy(dtype=float)
        sectors = g["sector"].to_numpy()
        realized_benchmark = float(b@y)
        records.append({"date": dt, "strategy": "benchmark", "gross_return": realized_benchmark,
                        "net_return": realized_benchmark, "benchmark_return": realized_benchmark,
                        "turnover": 0., "exante_te": 0., "forecast_active_alpha": 0.})
        for i, ticker in enumerate(tickers):
            prediction = {"date":dt, "ticker":ticker, "realized_next_return":y[i],
                          "train_end":forecast.train_end, "train_rows": forecast.train_rows}
            prediction.update({col: float(forecast_df.iloc[i][col]) for col in
                               ["alpha_monthly", "stock", "peer", "country", "industry", "intersection"]})
            predictions.append(prediction)
        for strategy in STRATEGIES:
            prev = b if current_weights[strategy] is None else current_weights[strategy]
            res = optimize(alpha, risk.covariance, b, prev, spread, vol, adv, sectors,
                           aum_usd=aum_usd,
                           risk_aversion=0. if strategy=="alpha_only" else risk_aversion,
                           impact_k=impact_k,
                           cost_strength=1. if strategy=="alpha_risk_cost" else 0.,
                           execution_days=execution_days, max_weight=max_weight,
                           max_active=max_active, max_turnover=max_turnover,
                           max_participation=max_participation, sector_tolerance=sector_tolerance)
            gross = float(res.weights @ y)
            net = float((1 - res.forecast_cost.total)*(1 + gross)-1)
            if net <= -1:
                raise ValueError("Portfolio loss exceeds 100%")
            records.append({"date":dt, "strategy":strategy, "gross_return":gross,
                            "net_return":net, "benchmark_return":realized_benchmark,
                            "turnover":res.turnover, "exante_te":res.exante_tracking_error,
                            "forecast_active_alpha":res.expected_active_alpha})
            costs.append({"date":dt, "strategy":strategy,
                          "spread_cost":res.forecast_cost.spread,
                          "impact_cost":res.forecast_cost.impact,
                          "estimated_total_cost":res.forecast_cost.total})
            breakdown = risk.breakdown(res.weights-b)
            breakdown.update({"date": dt, "strategy":strategy})
            factors.append(breakdown)
            for i, ticker in enumerate(tickers):
                positions.append({"date":dt, "strategy":strategy, "ticker":ticker,
                                  "benchmark_weight":b[i], "pretrade_weight":prev[i],
                                  "target_weight":res.weights[i],
                                  "trade_weight":res.weights[i]-prev[i]})
            if strategy=="alpha_risk_cost":
                prof = profile_orders(tickers, res.weights-prev, adv,
                                      aum_usd=aum_usd, execution_days=execution_days)
                prof["date"] = dt
                orders.extend(prof.to_dict("records"))
            # Weights drift by realized price returns before next rebalance.
            current_weights[strategy] = res.weights*(1+y)/(1+gross)
    ret = pd.DataFrame(records)
    position_df = pd.DataFrame(positions)
    forecast_df = pd.DataFrame(predictions)
    cost_df = pd.DataFrame(costs)
    factor_df = pd.DataFrame(factors)
    summaries = {}
    for strategy in ("benchmark", *STRATEGIES):
        d = ret.loc[ret["strategy"] == strategy].sort_values("date")
        m = summary_metrics(d["net_return"].to_numpy(), d["benchmark_return"].to_numpy())
        m["mean_one_way_turnover"] = float(d["turnover"].mean()/2)
        m["mean_exante_te_monthly"] = float(d["exante_te"].mean())
        c = cost_df.loc[cost_df["strategy"] == strategy]
        m["total_estimated_cost_fraction_summed"] = float(c["estimated_total_cost"].sum())
        m["n_periods"] = len(d)
        summaries[strategy] = m
    ic = forecast_df.groupby("date").apply(
        lambda z: rank_ic(z["alpha_monthly"], z["realized_next_return"]),
        include_groups=False)
    summary = {"strategies": summaries, "forecast_diagnostics": {
        "mean_monthly_rank_ic": float(ic.mean()) if ic.notna().any() else None,
        "monthly_rank_ic_sd": float(ic.std(ddof=1)) if ic.notna().sum()>1 else None,
        "n_oos_months": int(ic.notna().sum())},
        "data_type": "synthetic" if all(str(x).startswith("SIM") for x in tickers) else "user_supplied",
        "important": "Synthetic outcomes are NOT evidence of real-world investment returns. Estimated costs are not observed fills."}
    return BacktestResult(ret, position_df, forecast_df, cost_df, factor_df,
                          pd.DataFrame(orders), summary)
