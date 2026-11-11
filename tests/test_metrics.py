import numpy as np
import pytest
from acadian_arc.metrics import summary_metrics,rank_ic


def test_zero_active_performance():
    m=summary_metrics([.01,.01,.01],[.01,.01,.01])
    assert m["information_ratio"] is None
    assert np.isclose(m["annualized_mean_active_return"],0)


def test_known_drawdown():
    m=summary_metrics([.1,-.5,.2],[0,0,0])
    assert np.isclose(m["max_drawdown"],-.5)


def test_rank_ic_perfect():
    assert np.isclose(rank_ic([1,2,3],[2,4,5]),1)
    assert np.isclose(rank_ic([1,2,3],[5,4,2]),-1)


def test_return_mismatch():
    with pytest.raises(ValueError):
        summary_metrics([.01],[.01,.02])
