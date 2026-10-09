"""Sampling from the trained QAOA state and decoding with the formulator's own decoder.

Same selection rule as the fixed SimulatedAnnealingSolver: decode distinct samples in increasing QUBO
energy and take the FIRST FEASIBLE one; never report "Success" for an infeasible dispatch.
"""
from __future__ import annotations

import numpy as np

from solvers.simulated_gate_based.s2_circuits import QAOABase


def sample_indices(circ: QAOABase, params, shots: int, seed: int = 0):
    """Draw `shots` bitstrings from the exact output distribution (= ideal shot noise only; hardware
    noise is not included unless the circuit itself was built with a noise model).
    Returns (unique basis-state indices, counts)."""
    pr = circ.probs(params)
    pr = pr / pr.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(pr.size, size=int(shots), p=pr)
    return np.unique(idx, return_counts=True)


def evaluate_samples(circ: QAOABase, formulator, net, params, shots: int = 2000, seed: int = 0,
                     reference_cost: float | None = None, max_decode: int | None = None) -> dict:
    """formulator._formulate_qubo(net) must have been called on this same instance (fills the registry)."""
    land = circ.land
    uniq, cnt = sample_indices(circ, params, shots, seed)
    energies = land.raw_energy(uniq)
    order = np.argsort(energies)
    if max_decode:
        order = order[:max_decode]

    best_feasible, best_any = None, None
    feas_shots, decoded_shots, n_decoded = 0, 0, 0
    for j in order:
        sample = land.index_to_sample(int(uniq[j]))
        dispatch, sgen, slack, cost, feas = formulator._decode_solution(sample, net)
        n_decoded += 1
        decoded_shots += int(cnt[j])
        row = {"dispatch": dispatch, "sgen_dispatch": sgen, "slack_dispatch": slack, "cost": cost,
               "feasibility": feas, "qubo_energy": float(energies[j]), "shots": int(cnt[j])}
        if best_any is None:
            best_any = row
        if feas["is_feasible"]:
            feas_shots += int(cnt[j])
            if best_feasible is None:          # energy-sorted -> first feasible = lowest-energy feasible
                best_feasible = row

    out = {
        "shots": int(shots), "unique_samples": int(len(uniq)), "decoded_unique": n_decoded,
        "feasible_fraction": feas_shots / decoded_shots if decoded_shots else 0.0,
        "feasible_found": best_feasible is not None,
        "best_feasible": best_feasible, "lowest_energy_sample": best_any,
        "p_ground_sampled": float(cnt[np.isin(uniq, land.ground)].sum() / shots),
    }
    if best_feasible is not None and reference_cost:
        out["gap_percent"] = 100.0 * (best_feasible["cost"] - reference_cost) / abs(reference_cost)
    return out


if __name__ == "__main__":
    import numpy as np
    import time

    try:
        from solvers.qubo_formulator import QuboFormulator
        from homemade_grids.small_grids import case3_low_gen
        
        # Adjust imports based on your actual folder structure
        from solvers.simulated_gate_based.s1_ising import IsingProblem, Landscape
        from solvers.simulated_gate_based.s2_circuits import make_circuit
        from solvers.simulated_gate_based.s3_optimize import depth_sweep
        from solvers.simulated_gate_based.s5_evaluate import evaluate_samples
    except ImportError as e:
        print(f"[-] Import Error: {e}")
        exit(1)

    print("\n" + "="*80)
    print(" END-TO-END QAOA PIPELINE: MEASUREMENT & DECODING ".center(80, "="))
    print("="*80 + "\n")

    # 1. Formulation
    print("[1] Formulating QUBO for case3_low_gen (50 MW precision)...")
    net = case3_low_gen()
    if hasattr(net, 'ext_grid') and not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0

    formulator = QuboFormulator(formulation="dc_ptdf", mw_precision=50.0)
    bqm, _ = formulator._formulate_qubo(net)
    
    prob = IsingProblem.from_bqm(bqm)
    landscape = Landscape(prob, normalize=True)
    
    if prob.n > 12:
        print("    [!] FATAL: Too many qubits for density matrix simulation. Aborting.")
        exit(1)

    # 2. Training (Ideal Baseline)
    print(f"\n[2] Training QAOA (p=3, Powell Optimizer, {prob.n} qubits)...")
    target_p = 3
    sweep_results = depth_sweep(
        landscape, 
        p_values=(1, 2, target_p),  
        backend="numpy", 
        method="Powell", 
        maxiter=1000,
        n_restarts=15  
    )
    
    best_p_result, _ = sweep_results[-1]
    optimal_angles = best_p_result.params
    print(f"    -> [Ideal] P(Ground State) Achieved: {best_p_result.p_ground * 100:.2f}%")

    # 3. Initialization of Emulators
    qaoa_numpy = make_circuit(landscape, p=target_p, backend="numpy")
    qaoa_ibm = make_circuit(
        landscape, 
        p=target_p, 
        backend="pennylane", 
        device="default.mixed", 
        cost_layer="gates",
        hardware_preset="ibm_eagle"
    )

    # Known exact minimum for case3_low_gen is €2800
    exact_cost = 2800.0

    # 4. Measurement & Decoding (Ideal)
    print("\n[3] Taking 2,000 Shots from Ideal Simulator...")
    t0 = time.time()
    ideal_eval = evaluate_samples(
        circ=qaoa_numpy, 
        formulator=formulator, 
        net=net, 
        params=optimal_angles, 
        shots=2000, 
        reference_cost=exact_cost
    )
    t1 = time.time()
    
    print(f"    -> Decoding Time      : {t1 - t0:.2f} seconds")
    print(f"    -> Feasible Shots     : {ideal_eval['feasible_fraction'] * 100:.1f}%")
    if ideal_eval['feasible_found']:
        print(f"    -> Best Feasible Cost : €{ideal_eval['best_feasible']['cost']:.2f}")
        print(f"    -> Gap to Exact Min   : {ideal_eval.get('gap_percent', 0):.2f}%")
    else:
        print("    -> [!] No feasible states found in ideal shots.")

    # 5. Measurement & Decoding (Noisy IBM)
    print("\n[4] Taking 2,000 Shots from Noisy IBM Emulator...")
    t0 = time.time()
    ibm_eval = evaluate_samples(
        circ=qaoa_ibm, 
        formulator=formulator, 
        net=net, 
        params=optimal_angles, 
        shots=2000, 
        reference_cost=exact_cost
    )
    t1 = time.time()
    
    print(f"    -> Decoding Time      : {t1 - t0:.2f} seconds")
    print(f"    -> Feasible Shots     : {ibm_eval['feasible_fraction'] * 100:.1f}%")
    if ibm_eval['feasible_found']:
        print(f"    -> Best Feasible Cost : €{ibm_eval['best_feasible']['cost']:.2f}")
        print(f"    -> Gap to Exact Min   : {ibm_eval.get('gap_percent', 0):.2f}%")
    else:
        print("    -> [!] No feasible states found due to hardware noise.")

    print("\n" + "="*80)