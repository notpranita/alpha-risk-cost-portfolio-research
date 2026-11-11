import numpy as np
import pandas as pd
from acadian_arc.signals import design_matrix, ALL_COLUMNS


def test_features_no_nan(design):
    assert np.isfinite(design[ALL_COLUMNS].to_numpy()).all()


def test_future_label_does_not_perturb_features(panel):
    x = design_matrix(panel)
    modified = panel.copy()
    modified["next_return"] = 0.2
    after = design_matrix(modified)
    pd.testing.assert_frame_equal(x[ALL_COLUMNS], after[ALL_COLUMNS])


def test_peer_excludes_own_observation(panel):
    g = panel[panel.date == panel.date.iloc[0]].copy()
    identity=g.iloc[0]
    candidate = g[g.sector == identity.sector].copy()
    from acadian_arc.signals import _z
    expected_raw = (g.groupby("sector").quality.transform("sum") - g.quality) / (g.groupby("sector").quality.transform("count") - 1)
    expected = _z(expected_raw)
    actual = design_matrix(g)["peer_quality_z"].to_numpy()
    np.testing.assert_allclose(expected.to_numpy(), actual, atol=1e-12)
    assert len(candidate)>1


def test_zero_variance_is_finite(panel):
    copy=panel.copy();copy["industry_cycle"]=5.0
    matrix=design_matrix(copy)
    assert np.isfinite(matrix["industry_cycle_z"]).all()
    assert np.allclose(matrix["industry_cycle_z"],0)
