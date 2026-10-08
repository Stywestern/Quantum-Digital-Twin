"""Classical outer loop of QAOA: parameter initialisation, optimisation, depth sweep.

Sign convention (matches circuits.py): U = prod_l exp(-i beta_l X) exp(-i gamma_l H_C), minimising <H_C>.
Trotterised annealing  H(s) = -(1-s) sum X + s H_C  gives  gamma_l = dt*s_l > 0  and  beta_l = -dt*(1-s_l) < 0,
so the "ramp" start has POSITIVE gammas and NEGATIVE betas. (The landscape is symmetric under
(gamma, beta) -> (-gamma, -beta), so the optimum may come back with either overall sign.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .circuits import QAOABase, make_circuit
from .ising import Landscape


@dataclass
class QAOAResult:
    p: int
    params: np.ndarray
    energy_norm: float
    energy: float              # in original QUBO units
    approx_ratio: float
    approx_ratio_random: float
    p_ground: float
    n_evals: int               # expectation evaluations used by THIS optimisation (all restarts)
    success: bool
    message: str
    history: list = field(default_factory=list)


# ---- initialisation --------------------------------------------------------------------------
def ramp_init(p: int, dt: float = 0.75) -> np.ndarray:
    """Linear-ramp (Trotterised annealing) start. dt is the Trotter step in normalised units."""
    s = (np.arange(p) + 0.5) / p
    return np.concatenate([dt * s, -dt * (1.0 - s)])


def random_init(p: int, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([rng.uniform(0.0, 1.5, p), rng.uniform(-1.0, 0.0, p)])


def interp_init(params: np.ndarray, p_prev: int) -> np.ndarray:
    """INTERP warm start (Zhou et al. 2020): linearly interpolate optimal depth-p parameters into a
    depth-(p+1) starting point. Applied separately to gammas and betas."""
    def stretch(x):
        p = len(x)
        old = np.concatenate([[0.0], x, [0.0]])                  # old[0] = old[p+1] = 0
        return np.array([((i - 1) / p) * old[i - 1] + ((p - i + 1) / p) * old[i] for i in range(1, p + 2)])
    return np.concatenate([stretch(params[:p_prev]), stretch(params[p_prev:])])


# ---- optimisation ----------------------------------------------------------------------------
def optimize_qaoa(circ: QAOABase, x0: np.ndarray | None = None, method: str = "COBYLA",
                  maxiter: int = 200, n_restarts: int = 1, seed: int = 0, dt: float = 0.75,
                  options: dict | None = None) -> QAOAResult:
    """Minimise <H_C> over the 2p parameters. Gradient-free by default (COBYLA): no autodiff needed,
    works identically for both backends. Restart 0 uses x0 (or the ramp); extra restarts are random."""
    rng = np.random.default_rng(seed)
    starts = [np.asarray(x0, float) if x0 is not None else ramp_init(circ.p, dt)]
    starts += [random_init(circ.p, rng) for _ in range(max(0, n_restarts - 1))]

    evals_before = circ.n_evals
    best, best_hist = None, []
    for x in starts:
        hist: list = []

        def f(v, hist=hist):
            val = circ.expectation(v)
            hist.append(val)
            return val

        opts = {"maxiter": maxiter}
        if method.upper() == "COBYLA":
            opts["rhobeg"] = 0.3
        if options:
            opts.update(options)
        res = minimize(f, x, method=method, options=opts)
        if best is None or res.fun < best.fun:
            best, best_hist = res, hist

    land: Landscape = circ.land
    pr = circ.probs(best.x)
    e_norm = float(pr @ land.E_norm)
    return QAOAResult(
        p=circ.p, params=np.asarray(best.x, float), energy_norm=e_norm,
        energy=e_norm * land.scale + land.prob.offset,
        approx_ratio=land.approx_ratio(e_norm), approx_ratio_random=land.approx_ratio_random(),
        p_ground=land.p_ground(pr), n_evals=circ.n_evals - evals_before,
        success=bool(best.success), message=str(best.message), history=best_hist)


def depth_sweep(landscape: Landscape, p_values=(1, 2, 3), backend: str = "numpy", method: str = "COBYLA",
                maxiter: int = 200, n_restarts: int = 1, seed: int = 0, dt: float = 0.75,
                backend_kwargs: dict | None = None):
    """Optimise at increasing depth, warm-starting p+1 from the interpolated optimum of p.
    Returns a list of (QAOAResult, circuit) -- one per depth. This is the basic scaling experiment:
    approximation ratio / ground-state probability / evaluations vs p."""
    out, prev = [], None
    for p in p_values:
        circ = make_circuit(landscape, p, backend, **(backend_kwargs or {}))
        x0 = interp_init(prev.params, prev.p) if (prev is not None and prev.p == p - 1) else None
        res = optimize_qaoa(circ, x0=x0, method=method, maxiter=maxiter, n_restarts=n_restarts,
                            seed=seed, dt=dt)
        out.append((res, circ))
        prev = res
    return out