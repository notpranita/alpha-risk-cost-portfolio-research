import numpy as np
import pytest
from acadian_real.portfolio import cost_and_gradient,solve,before_trade,performance

def cfg():
 n=20; rng=np.random.default_rng(6);X=rng.normal(0,.04,(n,5));Sigma=X@X.T+np.eye(n)*.0005
 return n,Sigma,np.ones(n)/n,np.ones(n)*.01,np.ones(n)*20e6,np.ones(n)*.05,np.ones(n)*.02

def test_zero_trade_cost():
 _,_,b,_,adv,v,sp=cfg();c,g=cost_and_gradient(b,b,adv,v,sp);assert c==0
 assert np.allclose(g,0)

def test_transaction_cost_nonlinear():
 _,_,b,_,adv,v,sp=cfg();w=b.copy();w[0]+=.02;w[1]-=.02
 a,_=cost_and_gradient(w,b,adv,v,sp,aum=1e6)
 z,_=cost_and_gradient(w,b,adv,v,sp,aum=1e8)
 assert a>0 and z>a

def test_cost_derivative_finite_difference():
 _,_,b,_,adv,v,sp=cfg();w=b.copy();w[0]+=.03;w[1]-=.03
 c,g=cost_and_gradient(w,b,adv,v,sp)
 eps=1e-6;we=w.copy();we[0]+=eps;wp=w.copy();wp[0]-=eps
 finite=(cost_and_gradient(we,b,adv,v,sp)[0]-cost_and_gradient(wp,b,adv,v,sp)[0])/(2*eps)
 assert g[0]==pytest.approx(finite,rel=1e-3)

@pytest.mark.parametrize('mode',['alpha','risk','cost'])
def test_portfolio_constraints(mode):
 n,S,b,alpha,adv,v,sp=cfg();rng=np.random.default_rng(7)
 B=rng.normal(size=(n,6));B[:,0]+=1
 w,_=solve(alpha,S,b,b,adv,v,sp,method=mode,B=B)
 assert abs(w.sum()-1)<1e-5
 assert w.min()>=-1e-6 and w.max()<=.100001
 if mode!='alpha':
  expos=(w-b)@B;assert abs(expos[0])<=.150001 and np.max(abs(expos[1:]))<=.300001

def test_invalid_position_cap():
 n,S,b,a,adv,v,sp=cfg()
 with pytest.raises(ValueError):solve(a,S,b,b,adv,v,sp,max_weight=.01)

def test_drift_normalization():
 d=before_trade({'A':.5,'B':.5},{'A':.1,'B':-.1})
 assert d['A']==pytest.approx(.55) and d['B']==pytest.approx(.45)

def test_performance_constant_earnings():
 m=performance(np.array([.01]*12));assert m['cagr']==pytest.approx((1.01)**12-1)
 assert m['max_drawdown']==pytest.approx(0)
