import numpy as np
import pytest
from acadian_arc.optimizer import optimize


def sample():
    n=8; b=np.ones(n)/n
    return {"alpha":np.linspace(-.025,.025,n),"cov":np.eye(n)*.003,
            "benchmark":b, "previous":b.copy(),"spread_bps":np.ones(n)*10,
            "volatility_monthly":np.ones(n)*.065, "adv_usd":np.ones(n)*3e7,
            "sectors":["A"]*4+["B"]*4, "max_weight":.25,
            "max_active":.08, "max_turnover":.50,"sector_tolerance":.05}


def test_long_only_budget_active_sector_turnover_and_adv():
    p=sample();r=optimize(**p)
    w=r.weights;b=p["benchmark"]
    assert np.isclose(w.sum(),1,atol=1e-6)
    assert (w>=-1e-7).all()
    assert (w<=.250001).all()
    assert (np.abs(w-b)<=.080001).all()
    assert np.abs(w[:4].sum()-b[:4].sum()) <= .050001
    assert r.turnover <= .500001
    assert (np.abs(w-b)*1e7 <= .08*3e7*2+1).all()
    assert r.forecast_cost.total>=0


def test_risk_only_case_stays_at_benchmark():
    p=sample();p["alpha"]=np.zeros(8)
    r=optimize(**p)
    np.testing.assert_allclose(r.weights,p["benchmark"],atol=1e-5)


def test_bad_covariance_rejected():
    p=sample();p["cov"][0,0]=-.1
    with pytest.raises(ValueError, match="positive semidefinite"):
        optimize(**p)


def test_infeasible_benchmark_rejected():
    p=sample();p["max_weight"] = .09
    with pytest.raises(ValueError,match="infeasible"):
        optimize(**p)


def test_nonlinear_cost_aware_trades_no_more_than_blind():
    p=sample();p["alpha"]=np.linspace(-.007,.007,8)
    p["spread_bps"]=np.ones(8)*70
    aware=optimize(**p,cost_strength=1)
    blind=optimize(**p,cost_strength=0)
    assert aware.turnover <= blind.turnover+1e-5
