import numpy as np
import pandas as pd
import pytest
from acadian_arc.forecast import predict_asof


def test_forecast_train_cutoff_and_components(design):
    decision = sorted(design.date.unique())[16]
    f = predict_asof(design, decision)
    assert f.train_end < decision
    assert f.train_rows == len(design[design.date < decision])
    for col in ("stock", "peer", "country", "industry", "intersection", "alpha_monthly"):
        assert col in f.forecasts.columns
        assert np.isfinite(f.forecasts[col]).all()
    assert abs(f.forecasts["alpha_monthly"].mean()) < 1e-7


def test_no_current_or_future_label_leakage(design):
    decision = sorted(design.date.unique())[17]
    baseline=predict_asof(design, decision).forecasts
    new=design.copy()
    new.loc[new.date>=decision,"next_return"] = 1000  # impossible but should not be read
    candidate=predict_asof(new, decision).forecasts
    pd.testing.assert_frame_equal(baseline,candidate)


def test_prior_training_label_affects_forecast(design):
    decision = sorted(design.date.unique())[17]
    baseline = predict_asof(design,decision).forecasts["alpha_monthly"]
    modified=design.copy()
    idx=modified.index[modified.date==sorted(design.date.unique())[10]][0]
    modified.loc[idx,"next_return"]+=2
    changed=predict_asof(modified, decision).forecasts["alpha_monthly"]
    assert not np.allclose(baseline,changed)


def test_cannot_fit_on_current_date_only(design):
    with pytest.raises(ValueError,match="Insufficient"):
        predict_asof(design, sorted(design.date.unique())[8])


def test_blend_weights_validate(design):
    date=sorted(design.date.unique())[16]
    with pytest.raises(ValueError, match="sum to one"):
        predict_asof(design,date, blend={"stock":1.0})
