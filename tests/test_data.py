from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from acadian_real.data import load_factors,french_zip,prices
ROOT=Path(__file__).resolve().parents[1]
RAW_PRESENT=(ROOT/'data/raw/sp500-stock-prices.parquet').exists() and (ROOT/'data/raw/F-F_Momentum_Factor_CSV.zip').exists()
PANEL_PRESENT=(ROOT/'data/processed/panel.pkl').exists()

@pytest.mark.skipif(not RAW_PRESENT,reason="download the four original market datasets first")
def test_five_factor_monthly_units():
 f,i=load_factors(ROOT/'data/raw')
 assert list(f.columns)==['Mkt-RF','SMB','HML','RMW','CMA','RF','MOM']
 assert len(f)>700 and len(i)>1000 and i.shape[1]==49
 assert f.index.is_unique and i.index.is_unique
 assert f.loc['2025-01':'2025-12'].shape[0]==12

@pytest.mark.skipif(not RAW_PRESENT,reason="market source files are not committed")
def test_french_2026_sep_positive_rf():
 f,_=load_factors(ROOT/'data/raw');assert 0<f.loc['2026-09-30','RF']<.01

@pytest.mark.skipif(not RAW_PRESENT,reason="market source files are not committed")
def test_parquet_real_rows_and_bounds():
 p=ROOT/'data/raw/sp500-stock-prices.parquet'
 d=prices(p)
 assert len(d)==1_444_124
 assert d.ticker.nunique()==503
 assert d.date.min()==pd.Timestamp('2015-01-02')
 assert d.date.max()==pd.Timestamp('2026-11-10')
 assert (d.adj_close>0).all() and (d.volume>=0).all()
 assert not d.duplicated(['date','ticker']).any()

@pytest.mark.skipif(not PANEL_PRESENT,reason="derived panel is not committed")
def test_no_lookahead_month_label_alignment():
 d=pd.read_pickle(ROOT/'data/processed/panel.pkl')[0]
 df=d.xs('AAPL',level='ticker')
 assert pd.notna(df.loc['2020-02-29','next_ret']) # next-month OPEN-to-OPEN holding period
 assert -.5 <= df.loc['2020-02-29','reversal'] <= .5 # ranked signal

def test_no_fundamental_feature_masquerading():
 from acadian_real.models import FEATURES
 assert 'earnings' not in FEATURES and 'value' not in FEATURES and 'beta12' not in FEATURES
 assert 'peer_mom' in FEATURES

@pytest.mark.skipif(not RAW_PRESENT,reason="market source files are not committed")
def test_missing_sentinel_replaced():
 f,_=load_factors(ROOT/'data/raw')
 assert np.isfinite(f.loc['2019-01':'2025-12'].to_numpy()).all()
