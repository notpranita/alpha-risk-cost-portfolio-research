import numpy as np
import pandas as pd
import pytest
from acadian_real.models import FEATURES,FACTORS,rank_features,train_predict,factor_risk

def test_rank_features_preserve_order():
 ix=pd.MultiIndex.from_product([[pd.Timestamp('2020-01-31')],['A','B','C']],names=['month','ticker'])
 f=pd.DataFrame({k:[1.,2.,3.] for k in FEATURES},index=ix)
 o=rank_features(f)
 assert o['mom12'].iloc[2]==pytest.approx(.5)
 assert o['mom12'].iloc[0]==pytest.approx(-1/6)

def test_ridge_uses_only_passed_training_labels():
 dates=pd.date_range('2017-01-31',periods=18,freq='ME')
 ix=pd.MultiIndex.from_product([dates,['A','B','C','D','E','F','G','H']],names=['month','ticker'])
 rng=np.random.default_rng(22)
 f=pd.DataFrame({k:rng.normal(size=len(ix)) for k in FEATURES},index=ix)
 f['next_ret']=f['mom12']*.02+rng.normal(0,.002,len(ix))
 p,m=train_predict(f.iloc[:-8],f.iloc[-8:])
 assert len(p)==8 and np.all(np.isfinite(p))
 assert p.iloc[-1]>p.iloc[np.argmin(f.iloc[-8:]['mom12'].to_numpy())]

def test_factor_risk_psd():
 dates=pd.date_range('2018-01-31',periods=40,freq='ME');rng=np.random.default_rng(2)
 factors=pd.DataFrame(rng.normal(0,.03,(40,6)),columns=FACTORS,index=dates)
 r=pd.DataFrame(rng.normal(0,.04,(40,4)),columns=list('ABCD'),index=dates)
 sigma,B,rv,F=factor_risk(r,factors,list('ABCD'),dates[-1]);assert sigma.shape==(4,4)
 assert np.linalg.eigvalsh(sigma).min()>0
 assert (rv>0).all()
