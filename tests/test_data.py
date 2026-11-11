import numpy as np
import pandas as pd
import pytest
from acadian_arc.data import generate_synthetic_panel, validate_panel


def test_synthetic_deterministic(panel):
    same = generate_synthetic_panel(months=22, seed=42)
    pd.testing.assert_frame_equal(panel, same)


def test_synthetic_different_seeds(panel):
    other = generate_synthetic_panel(months=22, seed=41)
    assert not np.allclose(panel.next_return, other.next_return)


def test_benchmark_weights_sum_one(panel):
    assert np.allclose(panel.groupby("date").benchmark_weight.sum(), 1)

@pytest.mark.parametrize("field", ["quality", "next_return", "adv_usd"])
def test_missing_numeric_rejected(panel, field):
    df = panel.copy();df.loc[0,field] = np.nan
    with pytest.raises(ValueError,match="NaN|infinite"):
        validate_panel(df)


def test_missing_columns_rejected(panel):
    with pytest.raises(ValueError,match="Missing required"):
        validate_panel(panel.drop(columns="growth"))


def test_duplicate_row_rejected(panel):
    with pytest.raises(ValueError,match="Duplicate"):
        validate_panel(pd.concat([panel,panel.iloc[[0]]]))


def test_future_available_date_rejected(panel):
    altered = panel.copy();altered.loc[0,"available_date"] = altered.loc[0,"date"] + pd.Timedelta(days=1)
    with pytest.raises(ValueError,match="look-ahead"):
        validate_panel(altered)


def test_fixed_universe_rejected(panel):
    with pytest.raises(ValueError,match="fixed full-history"):
        validate_panel(panel.drop(index=0))


def test_invalid_price_return_rejected(panel):
    bad=panel.copy();bad.loc[0,"next_return"]=-1.0
    with pytest.raises(ValueError,match="-100%"):
        validate_panel(bad)


def test_bad_benchmark_rejected(panel):
    bad=panel.copy();bad.loc[0,"benchmark_weight"]=.99
    with pytest.raises(ValueError,match="sum to 1"):
        validate_panel(bad)
