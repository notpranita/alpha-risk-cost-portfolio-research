"""Constrained optimization and interpretable transaction cost accounting."""
import numpy as np
from scipy.optimize import minimize

def cost_and_gradient(w,wprev,adv,vol,range_proxy,aum=10_000_000,impact=0.08):
    """Nonlinear cost fraction: (half-range proxy)*|trade| + impact*monthly-volatility*sqrt(order/ADV)*|trade|.
    NOTE: observed intraday range is NOT an observed bid/ask spread; cost is scenario-calibrated.
    """
    trade=w-wprev;absw=np.abs(trade)
    rate=0.25*np.clip(range_proxy,0,.15) # quarter daily range, NOT quoted half-spread
    coeff=impact*np.clip(vol,.005,.30)*np.sqrt(aum/np.maximum(adv,1))
    cost=np.sum(rate*absw+coeff*absw**1.5)
    grad=np.sign(trade)*(rate+1.5*coeff*np.sqrt(absw))
    return float(cost),grad

def solve(alpha,Sigma,b,wprev,adv,vol,range_proxy,method='cost',risk_aversion=12.,aum=10_000_000,max_weight=.10,B=None):
    alpha=np.asarray(alpha,float);n=len(alpha)
    b=np.asarray(b,float);wprev=np.asarray(wprev,float)
    if n*max_weight<1:raise ValueError('position cap too small for fully-invested portfolio')
    if method not in ('alpha','risk','cost'):raise ValueError(method)
    def objective(w):
        diff=w-b
        risk=Sigma@diff
        val=-alpha@w+(.5*risk_aversion*diff@risk if method!='alpha' else 0.0)
        jac=-alpha+(risk_aversion*risk if method!='alpha' else 0.0)
        if method=='cost':
            cost,dc=cost_and_gradient(w,wprev,adv,vol,range_proxy,aum)
            val+=cost;jac+=dc
        elif method=='alpha':
            val+=0.001*np.sum(diff**2);jac+=0.002*diff
        return val,np.asarray(jac)
    cons=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(n)}]
    if B is not None and method != 'alpha':
        # Market and generic style factors are monitored separately from intended selection risk.
        for k in range(B.shape[1]):
            limit=.15 if k==0 else .30
            v=B[:,k].copy()
            cons += [{'type':'ineq','fun':lambda w,v=v,limit=limit:limit-(w-b)@v,'jac':lambda w,v=v:-v},
                     {'type':'ineq','fun':lambda w,v=v,limit=limit:limit+(w-b)@v,'jac':lambda w,v=v:v}]
    res=minimize(lambda w:objective(w)[0],b,jac=lambda w:objective(w)[1],method='SLSQP',
        bounds=[(0,max_weight)]*n,constraints=cons,
        options={'maxiter':170,'ftol':1e-9})
    if not res.success or abs(res.x.sum()-1)>1e-5 or np.min(res.x)<-1e-6:
        raise RuntimeError(f'optimization failed: {res.message}')
    return np.clip(res.x,0,max_weight),res

def before_trade(weights,ret):
    # post-period drift, before the following rebalance
    nextpos={k:float(v)*(1+float(ret.get(k,0))) for k,v in weights.items()}
    total=sum(nextpos.values())
    return {k:v/total for k,v in nextpos.items()} if total>0 else {}

def performance(r):
    r=np.asarray(r,float);n=len(r)
    if n<3:return {}
    wealth=np.cumprod(1+r)
    return {'months':int(n),'cagr':float(wealth[-1]**(12/n)-1),
            'volatility':float(r.std(ddof=1)*np.sqrt(12)),
            'sharpe_zero_rf':float(r.mean()/r.std(ddof=1)*np.sqrt(12)) if r.std(ddof=1)>0 else None,
            'max_drawdown':float(np.min(wealth/np.maximum.accumulate(np.r_[1.,wealth])[1:]-1)),
            'final_wealth':float(wealth[-1])}
