"""Loader for public historical prices and Kenneth French monthly archives."""
import re, zipfile
from pathlib import Path
import pandas as pd
import numpy as np

def prices(path):
    try:
        d = pd.read_parquet(path)
    except ImportError:
        from .parquet_native import read
        d = read(path)
    required={'date','ticker','open','high','low','close','adj_close','volume'}
    if not required.issubset(d): raise ValueError(f'missing columns: {required-set(d)}')
    d['date']=pd.to_datetime(d['date'])
    d=d.sort_values(['date','ticker']).drop_duplicates(['date','ticker'])
    return d

def french_zip(path, industry=False):
    with zipfile.ZipFile(path) as z:
        txt=z.read(z.namelist()[0]).decode('latin1')
    lines=txt.splitlines();found=False;records=[];cols=None
    for line in lines:
        s=line.strip()
        if industry:
            if 'Average Value Weighted Returns -- Monthly' in s: found=True;continue
        else:found=True
        if not found:continue
        parts=[x.strip() for x in line.split(',')]
        if len(parts)<2:
            if industry and records:break
            continue
        if re.fullmatch(r'\d{6}',parts[0]):
            if cols is None: continue
            if len(parts)==len(cols):records.append(parts)
        elif parts[0]=='' and len(parts)>1 and cols is None:
            cols=['month']+parts[1:]
    if not records:raise ValueError(f'No monthly entries in {path}')
    df=pd.DataFrame(records,columns=cols);df['month']=pd.to_datetime(df['month'],format='%Y%m').dt.to_period('M').dt.to_timestamp('M')
    for col in cols[1:]:df[col]=pd.to_numeric(df[col],errors='coerce').replace([-99.99,-999],np.nan)/100
    return df.set_index('month').sort_index()

def load_factors(base):
    base=Path(base)
    five=french_zip(base/'F-F_Research_Data_5_Factors_2x3_CSV.zip');mom=french_zip(base/'F-F_Momentum_Factor_CSV.zip');ind=french_zip(base/'49_Industry_Portfolios_CSV.zip',True)
    mom_col=next(x for x in mom.columns if 'Mom' in x)
    factors=five.join(mom[[mom_col]].rename(columns={mom_col:'MOM'}),how='inner')
    return factors,ind

def prepare_panel(price_df,factors,industries):
    d=price_df.copy();d['month']=d.date.dt.to_period('M').dt.to_timestamp('M')
    d['dollar_volume']=d['close']*d['volume']
    d['range']=((d.high-d.low)/d.close.replace(0,np.nan)).clip(0,1)
    d['daily_ret']=d.groupby('ticker',sort=False).adj_close.pct_change(fill_method=None)
    # Adjust the observed opening quote using the adjustment factor on that same date.
    d['adjusted_open']=d.open*(d.adj_close/d.close)
    g=d.groupby('ticker',sort=False)
    d['adv20']=g.dollar_volume.transform(lambda s:s.rolling(20,min_periods=15).mean())
    d['vol63']=g.daily_ret.transform(lambda s:s.rolling(63,min_periods=40).std()*np.sqrt(21))
    d['range20']=g['range'].transform(lambda s:s.rolling(20,min_periods=15).median())
    d['observed_days']=g.cumcount()+1
    # All monthly features evaluated at the final available trading close of that month.
    m=d.groupby(['month','ticker'],sort=True).tail(1).set_index(['month','ticker']).sort_index()
    m=m[['adj_close','adv20','vol63','range20','observed_days']].rename(columns={'adj_close':'price'})
    monthly=m['price'].unstack('ticker')
    ret=monthly.pct_change(fill_method=None)
    # no forward fill of unavailable prices
    for k,periods in [('mom12',12),('mom6',6)]:
        m[k]=(monthly.shift(1)/monthly.shift(periods)-1).stack(dropna=False).reindex(m.index)
    m['reversal']=(-ret).stack(dropna=False).reindex(m.index)
    m['liq']=np.log(m.adv20.clip(lower=1))
    m['lowvol']=-m.vol63
    m['lowrange']=-m.range20
    first_open=d.groupby(['month','ticker'],sort=True).head(1).set_index(['month','ticker'])['adjusted_open'].unstack('ticker')
    # Signal at t month-end close; execute first opening quote in t+1 month;
    # exit/rebalance at first opening quote in t+2 month. No same-close execution.
    # Month-t open-to-close is observable at t close; used to estimate
    # pretrade drift at the subsequent decision, never at the current one.
    m['month_open_to_close']=(monthly/first_open-1).stack(dropna=False).reindex(m.index)
    forward_open_return=first_open.shift(-2)/first_open.shift(-1)-1
    m['next_ret']=forward_open_return.stack(dropna=False).reindex(m.index)
    market=factors['Mkt-RF'].reindex(ret.index)
    # trailing 12-month beta uses ONLY returns observed by the rebalance date
    cov=ret.rolling(12,min_periods=9).cov(market);var=market.rolling(12,min_periods=9).var()
    beta=cov.div(var,axis=0).clip(-3,4)
    m['beta12']=beta.stack(dropna=False).reindex(m.index)
    # 49-industry market breadth (known at month end) enters via beta interaction
    disp=industries.reindex(ret.index).rolling(3,min_periods=3).mean().std(axis=1)
    markettrend=market.rolling(6,min_periods=4).sum()
    beta_long=m['beta12'].unstack('ticker')
    m['disp_mom']=(m['mom6'].unstack('ticker').mul(disp,axis=0)).stack(dropna=False).reindex(m.index)
    m['regime_mom']=(m['mom6'].unstack('ticker').mul(markettrend,axis=0)).stack(dropna=False).reindex(m.index)
    # Statistical peer network: trailing 12-month returns, 5 most-correlated other stocks.
    # Candidate network at every date contains ONLY historically observable stocks.
    peers=[]
    for month in ret.index:
        history=ret.loc[:month].tail(12)
        if len(history)<9:continue
        present=m.loc[month].nlargest(100,'adv20').index
        corr=history[present].corr(min_periods=8)
        mom=m.loc[month,'mom6']
        corr.values[np.diag_indices_from(corr)]=-np.inf
        for ticker in present:
            scores=corr[ticker].dropna().nlargest(5).index
            peers.append((month,ticker,mom.reindex(scores).mean()))
    pp=pd.Series({(t,s):v for t,s,v in peers},name='peer_mom')
    pp.index=pd.MultiIndex.from_tuples(pp.index,names=['month','ticker'])
    m=m.join(pp,how='left')
    return m,ret
