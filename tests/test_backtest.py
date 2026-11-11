import json
import numpy as np
import pandas as pd
import pytest
from acadian_arc.backtest import run_backtest
from acadian_arc.reporting import save_report


@pytest.fixture(scope="module")
def result():
    from acadian_arc.data import generate_synthetic_panel
    return run_backtest(generate_synthetic_panel(months=21,seed=21),min_train_months=15)


def test_oos_dates_count(result):
    assert len(result.returns)==(21-15)*4
    assert result.forecasts.train_end.lt(result.forecasts.date).all()


def test_gross_net_model_cost_accounting(result):
    ret=result.returns.merge(result.costs,on=["date","strategy"],how="left")
    nonbench=ret[ret.strategy!="benchmark"]
    expect=(1-nonbench.estimated_total_cost)*(1+nonbench.gross_return)-1
    np.testing.assert_allclose(expect,nonbench.net_return,atol=1e-12)


def test_portfolios_sum_to_one(result):
    sums = result.positions.groupby(["date","strategy"]).target_weight.sum()
    np.testing.assert_allclose(sums,1,atol=1e-6)


def test_drift_between_months(result):
    dates=sorted(result.returns.date.unique())
    s="alpha_risk_cost"
    day1=result.positions[(result.positions.date==dates[0]) & (result.positions.strategy==s)].sort_values("ticker")
    day2=result.positions[(result.positions.date==dates[1]) & (result.positions.strategy==s)].sort_values("ticker")
    actual=result.forecasts[result.forecasts.date==dates[0]].sort_values("ticker").realized_next_return.to_numpy()
    post=day1.target_weight.to_numpy()*(1+actual)/(1+day1.target_weight.to_numpy()@actual)
    np.testing.assert_allclose(post,day2.pretrade_weight.to_numpy(),atol=1e-8)


def test_risk_factor_contributions_sum(result):
    df=result.factor_risk
    groupcols=["market","industry","country","generic_style","alpha_factor","specific"]
    observed=df[[k for k in groupcols if k in df.columns]].sum(axis=1)
    np.testing.assert_allclose(observed,df.total_variance,atol=1e-8)


def test_run_report_files(result,tmp_path):
    saved=save_report(result,tmp_path/"out")
    required=("summary.json","EXPERIMENT_REPORT.md","synthetic_wealth.png",
              "synthetic_costs.png","monthly_returns.csv","stock_forecasts.csv",
              "transaction_costs.csv","risk_attribution.csv","order_profile.csv")
    for name in required:
        assert (saved/name).is_file(),name
        assert (saved/name).stat().st_size>0
    js=json.loads((saved/"summary.json").read_text())
    assert js["results"]["data_type"]=="synthetic"
    assert "SYNTHETIC" in (saved/"EXPERIMENT_REPORT.md").read_text()
