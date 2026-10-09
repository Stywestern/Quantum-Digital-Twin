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

from solvers.simulated_gate_based.s2_circuits import QAOABase, make_circuit
from solvers.simulated_gate_based.s1_ising import Landscape


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


if __name__ == "__main__":
    import numpy as np
    import time

    try:
        from solvers.qubo_formulator import QuboFormulator
        from homemade_grids.small_grids import case3_low_gen
        
        # Adjust imports based on your actual folder structure
        from solvers.simulated_gate_based.s1_ising import IsingProblem, Landscape
        from solvers.simulated_gate_based.s2_circuits import make_circuit, resource_estimate
        from solvers.simulated_gate_based.s3_optimize import depth_sweep
    except ImportError as e:
        print(f"[-] Import Error: {e}")
        print("Ensure you are running this from the project root.")
        exit(1)

    print("\n" + "="*80)
    print(" QAOA PIPELINE: IDEAL TRAINING & NOISY HARDWARE TRANSFER ".center(80, "="))
    print("="*80 + "\n")

    # 1. Grid & QUBO Initialization
    print("[1] Formulating QUBO for case3_low_gen (50 MW precision)...")
    net = case3_low_gen()
    if hasattr(net, 'ext_grid') and not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0

    formulator = QuboFormulator(formulation="dc_ptdf", mw_precision=50.0)
    bqm, _ = formulator._formulate_qubo(net)
    
    prob = IsingProblem.from_bqm(bqm)
    landscape = Landscape(prob, normalize=True)
    print(f"    -> Logical Qubits: {prob.n}")
    
    if prob.n > 10:
        print("    [!] FATAL: Too many qubits for density matrix simulation. Aborting.")
        exit(1)

    # 2. Train on the Ideal Baseline (Numpy)
    print("\n[2] Training QAOA on Ideal State-Vector (Numpy)...")
    t_start = time.time()
    
    # - Cap depth at p=3 to keep physical CNOTs under ~200
    # - Maxiter=1000 gives COBYLA room to breathe
    # - n_restarts=25 forces a wide global search so it doesn't get stuck
    sweep_results = depth_sweep(
        landscape, 
        p_values=(1, 2, 3),  
        backend="numpy", 
        method="COBYLA", 
        maxiter=1000,
        n_restarts=25  
    )
    
    t_end = time.time()
    
    target_p = 3
    best_p_result, _ = sweep_results[-1]
    optimal_angles = best_p_result.params
    
    print(f"    -> Training Complete in {t_end - t_start:.2f} seconds.")
    print(f"    -> Best p={target_p} Angles (Gamma, Beta): \n       {np.round(optimal_angles, 4)}")
    print(f"    -> [Ideal] P(Ground State): {best_p_result.p_ground * 100:.2f}%")
    print(f"    -> [Ideal] QUBO Energy (€) : {best_p_result.energy:.2f}")

    # 3. Hardware Resource Estimation
    print("\n[3] Compiling for IBM Eagle (Heavy-Hex)...")
    ibm_metrics = resource_estimate(prob, p=target_p, target_hardware="ibm_eagle")
    print(f"    -> SWAP-Routed Circuit Depth : {ibm_metrics['depth_estimate']} layers")
    print(f"    -> Physical CNOT Gates       : {ibm_metrics['physical_two_qubit_gates']}")
    print(f"    -> Routing Overhead          : {ibm_metrics['routing_overhead_multiplier']:.2f}x")

    # 4. Parameter Transfer to Noisy Emulator
    print("\n[4] Transferring Optimal Angles to Noisy Hardware Emulator...")
    
    qaoa_ibm = make_circuit(
        landscape, 
        p=target_p, 
        backend="pennylane", 
        device="default.mixed", 
        cost_layer="gates",
        hardware_preset="ibm_eagle"
    )
    
    t0 = time.time()
    # CRITICAL: We do NOT optimize here. We simply measure the circuit ONCE using the trained angles.
    probs_ibm = qaoa_ibm.probs(optimal_angles)
    t1 = time.time()
    
    # 5. Extract Degraded Metrics
    ibm_energy_norm = float(probs_ibm @ landscape.E_norm)
    ibm_energy_raw = ibm_energy_norm * landscape.scale + landscape.prob.offset
    ibm_p_ground = landscape.p_ground(probs_ibm)
    
    print(f"    -> Hardware Execution Time : {t1 - t0:.2f} seconds")
    print(f"    -> [Noisy] P(Ground State): {ibm_p_ground * 100:.2f}%")
    print(f"    -> [Noisy] QUBO Energy (€) : {ibm_energy_raw:.2f}")

    # 6. Conclusion
    print("\n[5] Degradation Summary:")
    prob_drop = (best_p_result.p_ground - ibm_p_ground) * 100
    energy_penalty = ibm_energy_raw - best_p_result.energy
    
    print(f"    -> Ground State Probability fell by {prob_drop:.2f}% due to CNOT/Readout noise.")
    print(f"    -> Economic Penalty (Cost Increase) : +€{energy_penalty:.2f}/hr")
    
    print("\n" + "="*80)