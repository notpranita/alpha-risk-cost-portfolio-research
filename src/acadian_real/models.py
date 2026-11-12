"""Out-of-sample stock return forecasting and covariance estimation."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf

FEATURES=['mom12','mom6','reversal','lowvol','lowrange','liq','peer_mom','regime_mom','disp_mom']
FACTORS=['Mkt-RF','SMB','HML','RMW','CMA','MOM']

def rank_features(frame):
    """Within-date robust percentile rank, centered on zero; avoids scale domination."""
    out=frame.copy()
    for x in FEATURES:out[x]=out.groupby(level='month')[x].rank(pct=True)-0.5
    return out

def train_predict(train,holdout,alpha=100.0):
    """Pooled rolling Ridge; train contains only labels that had arrived at decision time."""
    train=train.dropna(subset=FEATURES+['next_ret'])
    holdout=holdout.dropna(subset=FEATURES)
    if len(train)<100 or len(holdout)==0:raise ValueError('Insufficient training records')
    mdl=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
    mdl.fit(train[FEATURES].to_numpy(),train['next_ret'].to_numpy())
    forecast=np.clip(mdl.predict(holdout[FEATURES].to_numpy()),-.15,.15)
    return pd.Series(forecast,index=holdout.index,name='alpha'),mdl

def factor_risk(returns,factors,tickers,month,window=36):
    """Stock exposure to six published risk factors; LedoitWolf shrinkage of factor covariance.
    Residual variances contribute a diagonal specific-risk matrix (no forward observations).
    """
    start=(pd.Timestamp(month)-pd.DateOffset(months=window+2))
    r=returns.loc[(returns.index>=start)&(returns.index<=month),tickers].tail(window)
    f=factors.reindex(r.index)[FACTORS]
    B=np.zeros((len(tickers),len(FACTORS)));rv=np.zeros(len(tickers))
    for i,ticker in enumerate(tickers):
        x=f.to_numpy();y=r[ticker].to_numpy();mask=np.isfinite(x).all(axis=1)&np.isfinite(y)
        if mask.sum()<18:
            B[i,0]=1.0;rv[i]=.005;continue
        xx=np.c_[np.ones(mask.sum()),x[mask]]
        coef=np.linalg.lstsq(xx,y[mask],rcond=None)[0]
        B[i]=np.clip(coef[1:],-4,4)
        errors=y[mask]-xx@coef
        rv[i]=max(np.var(errors,ddof=1),.0004)
    xf=f.dropna().to_numpy()
    C=LedoitWolf().fit(xf).covariance_ if len(xf)>12 else np.diag([.003]*6)
    cov=B@C@B.T+np.diag(rv)
    cov=(cov+cov.T)/2
    return cov,B,rv,C
