"""Transparent diagnostics, HAC inference, sensitivity, and publication-quality figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import load_factors
from .models import FACTORS
from .portfolio import performance

def analyze(root):
    root=Path(root);d=root/'results'; fig=d/'figures';fig.mkdir(exist_ok=True,parents=True)
    r=pd.read_csv(d/'monthly_returns.csv',parse_dates=['month'])
    ic=pd.read_csv(d/'forecast_diagnostics.csv',parse_dates=['month'])
    h=pd.read_csv(d/'holdings.csv',parse_dates=['month'])
    exp=pd.read_csv(d/'factor_exposures.csv',parse_dates=['month'])
    factors,industry=load_factors(root/'data/raw')
    factors.index=pd.to_datetime(factors.index)
    net=r.pivot(index='month',columns='strategy',values='net_return').sort_index()
    gross=r.pivot(index='month',columns='strategy',values='gross_return').sort_index()
    cost=r.pivot(index='month',columns='strategy',values='trading_cost').sort_index()
    turnover=r.pivot(index='month',columns='strategy',values='turnover').sort_index()
    def chart(name,title,xlabel=None,ylabel=None):
        plt.title(title,fontsize=13,pad=12)
        if xlabel:plt.xlabel(xlabel)
        if ylabel:plt.ylabel(ylabel)
        plt.grid(alpha=.22)
        plt.tight_layout();plt.savefig(fig/name,dpi=190,bbox_inches='tight');plt.close()
    plt.figure(figsize=(9,4.5));(1+net).cumprod().plot(ax=plt.gca(),linewidth=1.8)
    chart('01_wealth.png','Net wealth: $1 invested in each strategy',ylabel='Cumulative wealth (USD)')
    plt.figure(figsize=(9,4.2));wealth=(1+net).cumprod();dd=wealth/wealth.cummax()-1;dd.plot(ax=plt.gca())
    chart('02_drawdowns.png','Peak-to-trough drawdowns',ylabel='Drawdown')
    plt.figure(figsize=(9,4.2));annual=((1+net).groupby(net.index.year).prod()-1)*100
    annual.plot.bar(ax=plt.gca(),width=.8);plt.xticks(rotation=0)
    chart('03_annual.png','Realized calendar-year returns (net of modeled costs)',ylabel='Calendar-year return (%)')
    plt.figure(figsize=(9,4.2));turnover.rolling(6,min_periods=3).mean().plot(ax=plt.gca())
    chart('04_turnover.png','Six-month average one-way turnover',ylabel='Monthly turnover')
    plt.figure(figsize=(9,4.2));cost.cumsum().plot(ax=plt.gca())
    chart('05_costs.png','Cumulative modeled trading-cost fractions',ylabel='Cumulative estimated cost')
    plt.figure(figsize=(9,3.6));plt.bar(ic.month,ic.ic,width=22);plt.axhline(0,color='black',linewidth=.7)
    chart('06_ic.png','Out-of-sample cross-sectional Spearman rank IC',ylabel='Rank IC')
    plt.figure(figsize=(9,3.6));(net['cost']-net['benchmark']).rolling(12,min_periods=6).sum().plot(ax=plt.gca())
    plt.axhline(0,color='black',linewidth=.7);chart('07_active.png','Rolling 12-month cumulative active return (approximation)',ylabel='Sum of monthly return differences')
    plt.figure(figsize=(9,4.0));ex=exp[exp.strategy=='cost'][FACTORS].mean().sort_values()
    ex.plot.barh(ax=plt.gca());chart('08_exposures.png','Average active exposures: cost-aware strategy',xlabel='Regression-estimated factor beta')
    plt.figure(figsize=(9,3.8));(gross-net).mean().sort_values().mul(1e4).plot.bar(ax=plt.gca());plt.xticks(rotation=0)
    chart('09_costdrag.png','Average monthly performance drag from costs',ylabel='Basis points per month')
    plt.figure(figsize=(9,3.8));h[h.strategy=='cost'].groupby('month').weight.max().plot(ax=plt.gca())
    chart('10_position.png','Largest portfolio weight at each rebalance',ylabel='Maximum single-stock weight')
    # Statistical inference: Newey-West heteroskedasticity/autocorrelation-consistent SEs.
    analysis={};arr=ic.ic.dropna().to_numpy();
    m=sm.OLS(arr,np.ones((len(arr),1))).fit(cov_type='HAC',cov_kwds={'maxlags':3})
    analysis['signal_skill']={'mean_spearman_ic':float(arr.mean()),'hac_t':float(m.tvalues[0]),'hac_p':float(m.pvalues[0]),
                             'ic_positive_months':int((arr>0).sum()),'months':len(arr)}
    for st in ['alpha','risk','cost']:
        active=(net[st]-net['benchmark']).to_numpy()
        mm=sm.OLS(active,np.ones((len(active),1))).fit(cov_type='HAC',cov_kwds={'maxlags':3})
        analysis[st]={'monthly_active_mean':float(active.mean()),
                      'annualized_active_mean_arithmetic':float(active.mean()*12),
                      'active_hac_t':float(mm.tvalues[0]),'active_hac_p':float(mm.pvalues[0]),
                      'active_mean_95pct_hac_ci_monthly':list(map(float,mm.conf_int()[0])),
                      'tracking_error_realized_annual':float(active.std(ddof=1)*np.sqrt(12)),
                      'information_ratio':float(active.mean()/active.std(ddof=1)*np.sqrt(12))}
    analysis['factor_regressions']={}
    for st in ['benchmark','alpha','risk','cost']:
        ff=factors.reindex(net.index+pd.offsets.MonthEnd(1)).copy();ff.index=net.index
        y=net[st]-ff['RF'];x=ff[FACTORS]
        aligned=pd.concat([y.rename('y'),x],axis=1).dropna()
        reg=sm.OLS(aligned.y,sm.add_constant(aligned[FACTORS])).fit(cov_type='HAC',cov_kwds={'maxlags':3})
        analysis['factor_regressions'][st]={'annualized_intercept':float(reg.params['const']*12),'intercept_p':float(reg.pvalues['const']),
            'r_squared':float(reg.rsquared),'betas':{k:float(reg.params[k]) for k in FACTORS}}
    analysis['subperiods']={}
    for label,sel in {'2020_2022':net.loc[:'2022-12'],'2023_2025':net.loc['2023-01':]}.items():
        analysis['subperiods'][label]={st:performance(sel[st]) for st in net.columns}
    analysis['fixed_weights_cost_sensitivity']={}
    for scalar in [0,0.5,1,1.5,2,3]:
        analysis['fixed_weights_cost_sensitivity'][str(scalar)]={st:performance((1-scalar*cost[st])*(1+gross[st])-1)['cagr'] for st in net.columns}
    analysis['data_check']={'monthly_rows':len(net),'strategy_count':len(net.columns),
         'total_stock_forecasts':len(pd.read_csv(d/'stock_forecasts.csv')),
         'stock_holdings_rows':len(h),'equal_weight_basket_size':int(len(h[(h.month==h.month.min())&(h.strategy=='benchmark')]))}
    (d/'statistical_analysis.json').write_text(json.dumps(analysis,indent=2))
    return analysis
if __name__=='__main__':
    import sys;print(json.dumps(analyze(sys.argv[1] if len(sys.argv)>1 else '.'),indent=2))
