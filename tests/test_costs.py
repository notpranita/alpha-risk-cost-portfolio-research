import numpy as np
import pytest
from acadian_arc.costs import transaction_cost, cost_coefficients, profile_orders


def kwargs():
    return {"spread_bps":[10,20], "volatility_monthly":[.06,.08], "adv_usd":[2e7,4e7]}


def test_zero_trade_zero_cost():
    assert transaction_cost([0,0],**kwargs()).total==0


def test_market_impact_increases_with_size():
    small=transaction_cost([.01,0],**kwargs())
    large=transaction_cost([.02,0],**kwargs())
    assert large.impact>2*small.impact
    assert np.isclose(large.spread,2*small.spread)


def test_cost_decreases_with_liquidity():
    low=transaction_cost([.08,0],**kwargs())
    high=transaction_cost([.08,0],spread_bps=[10,20],volatility_monthly=[.06,.08],adv_usd=[8e7,4e7])
    assert high.impact<low.impact


def test_cost_fraction_math():
    cost=transaction_cost([.1,0],spread_bps=[10,10],volatility_monthly=[0,0],adv_usd=[1e7,1e7])
    assert np.isclose(cost.spread, .1*10/20000)
    assert cost.impact==0


def test_invalid_adv_rejected():
    with pytest.raises(ValueError,match="Invalid"):
        cost_coefficients([5],[.1],[0])


def test_order_profile_participation():
    orders=profile_orders(["A","B"],[.01,.2],[1e7,1e7],aum_usd=1e7,execution_days=1)
    assert np.isclose(orders.participation.iloc[0],.01)
    assert orders.execution_bucket.iloc[1]=="high-touch"
