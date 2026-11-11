"""GitHub-ready reproducible sample artifacts and plain-language model card."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .backtest import BacktestResult


def _json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def save_report(result: BacktestResult, destination, *, config: dict | None = None):
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("monthly_returns",result.returns), ("portfolio_weights",result.positions),
                     ("stock_forecasts",result.forecasts), ("transaction_costs",result.costs),
                     ("risk_attribution",result.factor_risk), ("order_profile",result.orders)):
        df.to_csv(out/f"{name}.csv", index=False)
    metadata = {"config":config or {}, "results":result.summary}
    (out/"summary.json").write_text(json.dumps(_json_safe(metadata), indent=2, allow_nan=False))
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, g in result.returns.groupby("strategy"):
        g = g.sort_values("date")
        ax.plot(g["date"], np.cumprod(1+g["net_return"]), label=strategy)
    ax.set(title="SYNTHETIC DEMO — simulated net-of-modeled-cost wealth",
           ylabel="Wealth index ($1 starting value)", xlabel="Decision date")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(out/"synthetic_wealth.png", dpi=155);plt.close(fig)
    fig,ax = plt.subplots(figsize=(9,4))
    aggregate = result.costs.groupby("strategy")[["spread_cost", "impact_cost"]].sum()
    aggregate.plot(kind="bar", stacked=True, ax=ax)
    ax.set(title="SYNTHETIC DEMO — sum of modeled trading cost fractions",
           ylabel="Sum of monthly portfolio cost fractions", xlabel="Strategy")
    ax.tick_params(axis="x", rotation=0);fig.tight_layout()
    fig.savefig(out/"synthetic_costs.png", dpi=155);plt.close(fig)
    m = result.summary["strategies"]
    table = pd.DataFrame({s: {"CAGR (synthetic)":m[s]["cagr"],
                               "Information ratio (synthetic)":m[s]["information_ratio"],
                               "Mean one-way turnover":m[s]["mean_one_way_turnover"],
                               "Summed modeled costs":m[s]["total_estimated_cost_fraction_summed"]}
                          for s in m}).T
    markdown = ["# Automated experiment report — SYNTHETIC DATA ONLY", "",
                "This report validates the software workflow. It does **not** establish investment alpha or forecast accuracy in real markets.",
                "", "## Out-of-sample backtest metrics", "",
                "| Strategy | CAGR | Information ratio | Mean monthly one-way turnover | Summed model costs |",
                "|---|---:|---:|---:|---:|"]
    for name in ("benchmark", "alpha_only", "alpha_risk", "alpha_risk_cost"):
        v = m[name]
        ir = "n/a" if v["information_ratio"] is None else f'{v["information_ratio"]:.3f}'
        markdown.append(f'| {name} | {v["cagr"]:.2%} | {ir} | {v["mean_one_way_turnover"]:.2%} | {v["total_estimated_cost_fraction_summed"]:.2%} |')
    markdown += ["", "## Research audit trail", "",
                 "- The forecast for t was trained only on `next_return` labels from decision dates strictly earlier than t.",
                 "- Factor risk was estimated from previously realized returns only, with time-specific historical exposures.",
                 "- Portfolio turnover uses beginning-of-period drifted weights; trading costs are deducted once at rebalance.",
                 "- Benchmark is rebalanced monthly without modeled benchmark trading fees (a deliberate comparison assumption).",
                 "- The reported execution cost is **estimated**, not a realized fill / implementation shortfall.",
                 "- Synthetic next-period returns were deliberately generated with predictive signals and must never be treated as empirical validation.",
                 "", "## Files", "",
                 "- `stock_forecasts.csv`: as-of dates, training cutoff, all five component forecasts, ex-post labels",
                 "- `risk_attribution.csv`: signed factor contributions (variance units) and total ex-ante active variance",
                 "- `transaction_costs.csv`: spread and nonlinear market impact estimates",
                 "- `portfolio_weights.csv`: pretrade, benchmark, optimized weights, and orders",
                 "- `monthly_returns.csv`: gross and net strategy returns / benchmark per period",
                 "- `order_profile.csv`: illustrative participation-based order buckets (not routing/fills)",
                 "", "See `docs/METHODOLOGY.md` for parameter definitions, hypotheses, and limitations."]
    (out/"EXPERIMENT_REPORT.md").write_text("\n".join(markdown)+"\n", encoding="utf8")
    return out
