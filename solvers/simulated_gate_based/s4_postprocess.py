"""
Steepest-descent postprocessing for gate-based QAOA readouts.

Mitigates depolarizing and readout bit-flip noise by pushing measured samples 
into their nearest local minimum. Identical in principle to D-Wave's dwave-greedy,
adapted for the IsingProblem format and basis-state indices.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from solvers.simulated_gate_based.s1_ising import IsingProblem, Landscape

def steepest_descent(prob: IsingProblem, samples: np.ndarray, max_steps: int | None = None, tol: float = 1e-12):
    """Vectorized steepest descent. samples: (R, n) of +/-1."""
    n, R = prob.n, samples.shape[0]
    if n == 0 or R == 0:
        return samples, 0
    
    Jm = sp.coo_matrix((np.r_[prob.J, prob.J], (np.r_[prob.rows, prob.cols], np.r_[prob.cols, prob.rows])),
                       shape=(n, n)).tocsr()
    s = samples.T.astype(np.float64)
    h = prob.h[:, None]
    cols = np.arange(R)
    flips = 0
    
    for _ in range(max_steps or 10 * n):
        gain = s * (h + Jm @ s)  # > 0 where flipping lowers the energy
        i = gain.argmax(axis=0)  # steepest move per read
        active = gain[i, cols] > tol
        if not active.any():
            break
        s[i[active], cols[active]] *= -1.0
        flips += int(active.sum())
        
    return s.T.astype(np.int8), flips

def postprocess_shots(landscape: Landscape, uniq_indices: np.ndarray, counts: np.ndarray):
    """
    Translates quantum basis indices to spins, applies greedy descent, and 
    groups the repaired states. Returns (new_unique_indices, new_counts, total_flips).
    """
    n = landscape.n
    shifts = np.arange(n - 1, -1, -1).astype(np.int64)

    # 1. Translate Indices to Spins (+1, -1)
    bits = (np.asarray(uniq_indices, dtype=np.int64)[:, None] >> shifts) & 1
    spins = 1.0 - 2.0 * bits

    # 2. Apply Matrix Steepest Descent (we ignore its internal unweighted flip count)
    new_spins, _ = steepest_descent(landscape.prob, spins)

    # 3. Calculate true total physical flips weighted by shot counts
    flips_per_unique = (spins != new_spins).sum(axis=1)
    total_flips = int(np.sum(flips_per_unique * counts))

    # 4. Translate Spins back to Indices
    new_bits = ((1 - new_spins) // 2).astype(np.int64)
    new_indices = (new_bits << shifts).sum(axis=1)

    # 5. Collapse duplicate states that fell into the same valley
    uniq_out, inverse_idx = np.unique(new_indices, return_inverse=True)
    new_counts = np.zeros(len(uniq_out), dtype=counts.dtype)
    np.add.at(new_counts, inverse_idx, counts)

    return uniq_out, new_counts, total_flips


if __name__ == "__main__":
    import time
    
    try:
        from solvers.qubo_formulator import QuboFormulator
        from homemade_grids.small_grids import case3_low_gen
        
        # Adjust imports based on your actual folder structure
        from solvers.simulated_gate_based.s1_ising import IsingProblem, Landscape
    except ImportError as e:
        print(f"[-] Import Error: {e}")
        exit(1)

    print("\n" + "="*80)
    print(" TESTING STEEPEST DESCENT ERROR MITIGATION ".center(80, "="))
    print("="*80 + "\n")

    # 1. Initialize Problem
    net = case3_low_gen()
    if hasattr(net, 'ext_grid') and not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0

    formulator = QuboFormulator(formulation="dc_ptdf", mw_precision=50.0)
    bqm, _ = formulator._formulate_qubo(net)
    prob = IsingProblem.from_bqm(bqm)
    land = Landscape(prob, normalize=True)
    
    print(f"[1] Problem Loaded: {prob.n} Logical Qubits.")
    
    # 2. Find the true ground state to use as a baseline
    true_gs_idx = land.ground[0]
    true_gs_energy = land.raw_energy(true_gs_idx)
    print(f"    -> True Ground State Index : {true_gs_idx}")
    print(f"    -> True Ground State Energy: €{true_gs_energy:.2f}")

    # 3. Inject a single bit-flip error (simulating a hardware Readout error)
    # We XOR the index with 1 to flip the least significant bit
    error_bit = 1 
    corrupted_idx = true_gs_idx ^ error_bit
    corrupted_energy = land.raw_energy(corrupted_idx)
    
    print("\n[2] Injecting a 1-Bit Hardware Error...")
    print(f"    -> Corrupted State Index   : {corrupted_idx}")
    print(f"    -> Corrupted State Energy  : €{corrupted_energy:.2f}")
    
    # 4. Create a mock sample set (e.g., 500 physical shots of this corrupted state)
    uniq_raw = np.array([corrupted_idx], dtype=np.int64)
    cnt_raw = np.array([500], dtype=np.int64)
    
    # 5. Run the Post-Processor
    print("\n[3] Running Steepest Descent Post-Processing...")
    t0 = time.time()
    uniq_fixed, cnt_fixed, total_flips = postprocess_shots(land, uniq_raw, cnt_raw)
    t1 = time.time()
    
    fixed_idx = uniq_fixed[0]
    fixed_energy = land.raw_energy(fixed_idx)
    
    print(f"    -> Post-Processing Time    : {t1 - t0:.4f} seconds")
    print(f"    -> Total Bits Flipped      : {total_flips} (Should be exactly 500)")
    print(f"    -> Fixed State Index       : {fixed_idx}")
    print(f"    -> Fixed State Energy      : €{fixed_energy:.2f}")
    
    if fixed_idx == true_gs_idx:
        print("\n    -> SUCCESS: The post-processor correctly calculated the local gradients")
        print("                and flipped the erroneous bit back to the ground state!")
    else:
        print("\n    -> FAILED: The post-processor got trapped or flipped the wrong bit.")
        
    print("\n" + "="*80)