"""End-to-end historical study. All inputs in raw/ are user-supplied real market data."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .data import prices,load_factors,prepare_panel
from .models import FEATURES,FACTORS,rank_features,train_predict,factor_risk
from .portfolio import solve,cost_and_gradient,before_trade,performance

def run(root,start='2020-01',end='2025-12',universe_size=40,train_months=48,model_alpha=100):
    root=Path(root);raw=root/'data/raw';output=root/'results';output.mkdir(parents=True,exist_ok=True)
    prices_df=prices(raw/'sp500-stock-prices.parquet')
    factors,industries=load_factors(raw)
    cache=root/'data/processed/panel.pkl'
    if cache.exists():
        panel,returns=pd.read_pickle(cache)
    else:
        panel,returns=prepare_panel(prices_df,factors,industries)
        panel=rank_features(panel)
        pd.to_pickle((panel,returns),cache)
    print(f'Loaded {len(prices_df):,} stock-days, {len(panel):,} stock-months; {len(returns.columns)} tickers',flush=True)
    starts=pd.Period(start,'M').to_timestamp('M');ends=pd.Period(end,'M').to_timestamp('M')
    months=[x for x in returns.index if starts<=x<=ends and x+pd.offsets.MonthEnd(1) in returns.index]
    prev={n:{} for n in ['benchmark','alpha','risk','cost']};rows=[];diagnostics=[];alpharows=[];holdings=[];exposures=[];training=[]
    for mi,month in enumerate(months):
        if mi%6==0:print(f'Out-of-sample evaluation {month.date()} ({mi+1}/{len(months)})',flush=True)
        if month not in panel.index.get_level_values(0):continue
        current=panel.loc[month].copy()
        # Construct liquid available universe using only information at decision date.
        valid=current.dropna(subset=FEATURES+['adv20','vol63','range20'])
        valid=valid.loc[valid.observed_days>=252]
        assets=valid.nlargest(universe_size,'adv20').index.tolist()
        if len(assets)<universe_size:continue
        ts=pd.Timestamp(month)-pd.DateOffset(months=train_months)
        # An observation at t-1 is labeled by t+1 opening price, which is NOT known
        # at decision t close. Train only on signal months <= t-2.
        last_label_month=month-pd.offsets.MonthEnd(2)
        hist=panel.loc[(panel.index.get_level_values('month')<=last_label_month)&(panel.index.get_level_values('month')>=ts)].copy()
        # All training labels end no later than the opening of decision month t.
        eligible=hist.dropna(subset=FEATURES+['next_ret'])
        if len(eligible)<500:continue
        mu,mod=train_predict(eligible,valid.loc[assets],alpha=model_alpha)
        mu=mu.reindex(assets)
        cov,B,rv,F=factor_risk(returns,factors,assets,month)
        b=np.ones(len(assets))/len(assets)
        adv=valid.loc[assets,'adv20'].to_numpy()
        vol=valid.loc[assets,'vol63'].to_numpy()
        spread=valid.loc[assets,'range20'].to_numpy()
        nxt=panel.loc[month].loc[assets,'next_ret']
        miss=int(nxt.isna().sum())
        # Missing future return => refuse to imply complete validated accounting.
        if miss:raise ValueError(f'{month}: missing future return for {miss} holdings; resolve stale data/delisting instead of assuming zero')
        realized=nxt.to_numpy()
        obs=spearmanr(mu.to_numpy(),realized).statistic
        diagnostics.append({'month':str(month.date()),'ic':float(obs),'train_rows':len(eligible),'avg_pred':float(mu.mean()),'n_assets':len(assets)})
        for t in assets:alpharows.append({'month':str(month.date()),'ticker':t,'alpha_forecast':float(mu[t]),'next_return':float(nxt[t])})
        for name in prev:
            old=prev[name]
            # Trade current selected basket from prior drifted weights, including liquidation costs on exits.
            wprev=np.array([old.get(t,0.0) for t in assets]); b_local=b
            if name=='benchmark':w=b_local
            else:w,_=solve(mu.to_numpy(),cov,b_local,wprev,adv,vol,spread,method=name,B=B)
            c, _=cost_and_gradient(w,wprev,adv,vol,spread)
            exit_weight=sum(weight for ticker,weight in old.items() if ticker not in assets)
            # Liquidation of removed names uses a clearly identified conservative proxy, not zero cost.
            exit_cost=exit_weight*0.005
            c+=exit_cost
            gross=float(w@realized)
            net=float((1-c)*(1+gross)-1)
            active= w-b
            rows.append({'month':str(month.date()),'strategy':name,'gross_return':gross,'net_return':net,
                         'trading_cost':float(c),'turnover':float((np.abs(w-wprev).sum()+exit_weight)/2),
                         'tracking_error_forecast':float(np.sqrt(active@cov@active)),
                         'realized_active_gross':float(active@realized),
                         'exited_weight':float(exit_weight)})
            for k,t in enumerate(assets):
                holdings.append({'month':str(month.date()),'strategy':name,'ticker':t,'weight':float(w[k]),
                                 'benchmark_weight':float(b[k]),'alpha_forecast':float(mu[t]),
                                 'ADV_USD':float(adv[k]),'monthly_vol':float(vol[k])})
            exposure=(w-b)@B
            exposures.append({'month':str(month.date()),'strategy':name,**{k:float(v) for k,v in zip(FACTORS,exposure)}})
            # For the next decision (month t+1 close), drift only through
            # the open-to-close return observed by THAT decision; do not
            # sneak t+2 opening prices into future portfolio optimization.
            next_month=month+pd.offsets.MonthEnd(1)
            known_at_next_decision=panel.loc[next_month].loc[assets,'month_open_to_close']
            if known_at_next_decision.isna().any():
                raise ValueError('Missing observed close-to-open holding drift')
            prev[name]=before_trade(dict(zip(assets,w)),known_at_next_decision.to_dict())
        training.append({'month':str(month.date()),'train_start':str(ts.date()),'train_rows':len(eligible)})
    ret=pd.DataFrame(rows); ret.to_csv(output/'monthly_returns.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(output/'forecast_diagnostics.csv',index=False)
    pd.DataFrame(alpharows).to_csv(output/'stock_forecasts.csv',index=False)
    pd.DataFrame(holdings).to_csv(output/'holdings.csv',index=False)
    pd.DataFrame(exposures).to_csv(output/'factor_exposures.csv',index=False)
    pd.DataFrame(training).to_csv(output/'training_windows.csv',index=False)
    stats={}
    for strat,chunk in ret.groupby('strategy'):
        stats[strat]={**performance(chunk.net_return),
                      'average_monthly_turnover':float(chunk.turnover.mean()),
                      'total_trading_cost_fraction':float(chunk.trading_cost.sum()),
                      'avg_forecast_tracking_error':float(chunk.tracking_error_forecast.mean())}
    for strat in ['alpha','risk','cost']:
        this=ret[ret.strategy==strat].set_index('month').net_return
        benchmark=ret[ret.strategy=='benchmark'].set_index('month').net_return
        diff=this-benchmark
        stats[strat]['information_ratio_vs_equal_weight']=float(diff.mean()/diff.std(ddof=1)*np.sqrt(12)) if diff.std(ddof=1)>0 else None
        stats[strat]['mean_monthly_active_return']=float(diff.mean())
    ics=np.array(pd.DataFrame(diagnostics).ic)
    summary={'study':'Real historical equities; CURRENT S&P 500 CONSTITUENTS only (survivorship biased)',
             'data_date_max':str(prices_df.date.max().date()),'actual_data_rows':int(len(prices_df)),
             'unique_tickers':int(prices_df.ticker.nunique()),'sample_start':start,'sample_end':end,
             'universe_size':universe_size,'n_oos_months':len(diagnostics),
             'forecast_spearman_ic_mean':float(np.nanmean(ics)),
             'forecast_ic_positive_fraction':float(np.mean(ics>0)),
             'risk_factors':FACTORS,'forecast_features':FEATURES,
             'fundamental_value_quality_earnings':'NOT IMPLEMENTED: no point-in-time fundamental data supplied',
             'benchmark':'rebalanced equal-weight basket of the same selected 40 liquid surviving stocks',
             'market_index_reference':'Fama-French market portfolio, not S&P 500',
             'trade_timing':'signals at month-end closing prices; enter next month first trading session open; exit following first trading session open, adjusted-price approximation',
             'cost_model':'quarter of daily high-low range plus nonlinear volatility/ADV proxy; not calibrated to trade prints',
             'cost_accounting':'actual realized net return charged to all strategies; liquidation of dropped holdings at 50bps assumed',
             'strategies':stats}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary

def cli():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--start',default='2020-01');p.add_argument('--end',default='2025-12');p.add_argument('--universe',type=int,default=40);a=p.parse_args()
    print(json.dumps(run(a.root,a.start,a.end,a.universe),indent=2))
if __name__=='__main__':cli()
