"""SLSQP nonlinear portfolio optimizer: active alpha, factor risk, trading cost.

Auxiliary absolute-trade variables u_i >= |w_i - previous_i| provide linear
turnover / participation constraints. This is a convex program when covariance
is positive semidefinite; SciPy SLSQP numerically solves the resulting problem.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from .costs import cost_coefficients, transaction_cost, CostEstimate


@dataclass
class OptimizationResult:
    weights: np.ndarray
    turnover: float
    expected_active_alpha: float
    exante_tracking_error: float
    forecast_cost: CostEstimate
    objective: float
    status: str


def optimize(alpha, cov, benchmark, previous, spread_bps, volatility_monthly,
             adv_usd, sectors, *, aum_usd=10_000_000, risk_aversion=6.,
             impact_k=.12, cost_strength=1., max_weight=.16,
             max_active=.11, max_turnover=.65, max_participation=.08,
             execution_days=2, sector_tolerance=.10,
             maxiter=700) -> OptimizationResult:
    a = np.asarray(alpha, float)
    b = np.asarray(benchmark, float)
    prev = np.asarray(previous, float)
    sigma = np.asarray(cov, float)
    adv = np.asarray(adv_usd, float)
    n = len(a)
    if n < 2 or any(len(v) != n for v in (b, prev, adv, sectors)) or sigma.shape != (n, n):
        raise ValueError("Dimensions do not match")
    if not np.isfinite(np.r_[a, b, prev, sigma.flatten()]).all():
        raise ValueError("Non-finite optimizer inputs")
    if not np.isclose(b.sum(), 1, atol=1e-7) or not np.isclose(prev.sum(), 1, atol=1e-7):
        raise ValueError("Weights must sum to one")
    if (b < 0).any() or (prev < 0).any() or (adv <= 0).any() or cost_strength < 0 or risk_aversion < 0:
        raise ValueError("Invalid optimizer parameters")
    if np.linalg.eigvalsh((sigma + sigma.T)/2).min() < -1e-8:
        raise ValueError("Risk covariance must be positive semidefinite")
    if max_turnover <= 0 or max_participation <= 0:
        raise ValueError("Turnover and participation limits must be positive")
    sigma = (sigma + sigma.T) / 2
    spread, impact = cost_coefficients(spread_bps, volatility_monthly, adv,
                                      aum_usd=aum_usd, impact_k=impact_k,
                                      execution_days=execution_days)
    cap = max_participation * adv * execution_days / aum_usd
    lo = np.maximum(0, b - max_active)
    hi = np.minimum(max_weight, b + max_active)
    if (lo > hi).any() or (b > hi).any():
        raise ValueError("Benchmark is infeasible under position bounds")
    low_bounds = np.r_[lo, np.zeros(n)]
    high_bounds = np.r_[hi, cap]
    constraints = []
    summation = np.zeros((1, 2*n)); summation[0, :n] = 1
    constraints.append(LinearConstraint(summation, [1.], [1.]))
    # Positive auxiliary variables bound each signed trade.
    plus = np.c_[np.eye(n), -np.eye(n)]
    minus = np.c_[-np.eye(n), -np.eye(n)]
    constraints.append(LinearConstraint(plus, -np.inf*np.ones(n), prev))
    constraints.append(LinearConstraint(minus, -np.inf*np.ones(n), -prev))
    turn = np.zeros((1, 2*n)); turn[0, n:] = 1
    constraints.append(LinearConstraint(turn, [-np.inf], [max_turnover]))
    for sector in sorted(set(sectors)):
        z = (np.asarray(sectors) == sector).astype(float)
        mat = np.r_[z, np.zeros(n)].reshape(1,-1)
        exposure = float(z@b)
        constraints.append(LinearConstraint(mat, [exposure-sector_tolerance], [exposure+sector_tolerance]))

    def fun(v):
        w = v[:n]; u = np.maximum(v[n:], 0)
        delta = w-b
        gross = a @ delta - .5*risk_aversion * (delta @ sigma @ delta)
        forecast_tc = spread @ u + impact @ (u ** 1.5)
        return -gross + cost_strength * forecast_tc

    def jac(v):
        w = v[:n]; u = np.maximum(v[n:], 0)
        grad_w = -a + risk_aversion * sigma @ (w-b)
        grad_u = cost_strength * (spread + 1.5*impact*np.sqrt(u))
        return np.r_[grad_w, grad_u]

    # Start at the benchmark. It is position/sector feasible (the trade from
    # prev might not be; the solver is responsible for satisfying those caps).
    x0 = np.r_[b, np.abs(b-prev)]
    result = minimize(fun, x0, method="SLSQP", jac=jac,
                      bounds=Bounds(low_bounds, high_bounds),
                      constraints=constraints,
                      options={"maxiter": maxiter, "ftol": 1e-11, "disp": False})
    if not result.success:
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    weights = np.asarray(result.x[:n])
    true_trades = np.abs(weights-prev)
    feasible = (abs(weights.sum()-1) < 1e-5 and weights.min() >= -1e-6 and
                (weights <= hi+1e-6).all() and (weights >= lo-1e-6).all() and
                true_trades.sum() <= max_turnover+1e-5 and (true_trades <= cap+1e-5).all())
    for sector in sorted(set(sectors)):
        mask = np.asarray(sectors) == sector
        feasible &= abs((weights-b)[mask].sum()) <= sector_tolerance+1e-5
    if not feasible:
        raise RuntimeError("Solver reported success but violated constraints")
    cost = transaction_cost(weights-prev, spread_bps, volatility_monthly, adv,
                            aum_usd=aum_usd, impact_k=impact_k, execution_days=execution_days)
    active = weights-b
    return OptimizationResult(weights, float(true_trades.sum()), float(a@active),
                              float(np.sqrt(max(active@sigma@active, 0))),
                              cost, float(-fun(np.r_[weights, true_trades])),
                              str(result.message))
