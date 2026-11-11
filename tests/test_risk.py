import numpy as np
import pandas as pd
import pytest
from acadian_arc.risk import estimate_factor_risk


def test_risk_psd(design):
    date=sorted(design.date.unique())[18]
    risk=estimate_factor_risk(design, date)
    assert risk.covariance.shape[0] == len(design.ticker.unique())
    assert np.allclose(risk.covariance,risk.covariance.T)
    assert np.linalg.eigvalsh(risk.covariance).min()>0


def test_factor_variance_decomposition(design):
    date=sorted(design.date.unique())[18]
    risk=estimate_factor_risk(design,date)
    active=np.linspace(-.03,.03,risk.covariance.shape[0]);active-=active.mean()
    parts=risk.breakdown(active)
    assert np.isclose(sum(v for k,v in parts.items() if k!="total_variance"),parts["total_variance"],atol=1e-9)
    assert parts["specific"]>=0


def test_future_labels_do_not_change_current_risk(design):
    date=sorted(design.date.unique())[18]
    base=estimate_factor_risk(design,date).covariance
    copy=design.copy();copy.loc[copy.date>=date,"next_return"]+=1000
    candidate=estimate_factor_risk(copy,date).covariance
    np.testing.assert_allclose(base,candidate)


def test_insufficient_history(design):
    with pytest.raises(ValueError,match="Insufficient"):
        estimate_factor_risk(design,sorted(design.date.unique())[5])
