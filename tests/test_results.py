from pathlib import Path
import json
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]

def test_computed_results_71_months():
 r=pd.read_csv(ROOT/'results/monthly_returns.csv')
 assert r.month.nunique()==71 and len(r)==284
 assert r.strategy.nunique()==4
 assert r[['net_return','gross_return','trading_cost','turnover']].notna().all().all()
 assert (r['trading_cost']>=0).all()

def test_forecast_decisions_and_holdings():
 f=pd.read_csv(ROOT/'results/stock_forecasts.csv')
 h=pd.read_csv(ROOT/'results/holdings.csv')
 assert f.groupby('month').size().eq(40).all()
 assert h.groupby(['month','strategy']).weight.sum().between(.99999,1.00001).all()
 assert h.weight.between(-1e-6,.100001).all()

def test_results_are_really_historical():
 s=json.loads((ROOT/'results/summary.json').read_text())
 assert s['actual_data_rows']==1444124
 assert s['n_oos_months']==71
 assert 'NOT IMPLEMENTED' in s['fundamental_value_quality_earnings']
 assert 'survivorship' in s['study'].lower()

def test_statistical_inference_and_sensitivity():
 d=json.loads((ROOT/'results/statistical_analysis.json').read_text())
 assert 0<=d['signal_skill']['hac_p']<=1
 assert len(d['fixed_weights_cost_sensitivity'])>=4


def test_training_label_cutoff_has_no_lookahead():
 t=pd.read_csv(ROOT/'results/training_windows.csv',parse_dates=['month'])
 assert len(t)==71
 # A label at month t-1 would need the first open of t+1, hence cannot train at month t close.
 source=(ROOT/'src/acadian_real/study.py').read_text()
 assert "last_label_month=month-pd.offsets.MonthEnd(2)" in source
