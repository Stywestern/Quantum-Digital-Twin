"""Optional greedy steepest-descent postprocessing (numpy, vectorised over reads).

For each sample, repeatedly flip the single spin whose flip lowers the Ising energy the most, until no
single flip helps (a local minimum w.r.t. one-spin moves). Same idea as D-Wave's SteepestDescentSolver
(dwave-greedy), reimplemented so that sample order and array layout stay under our control.

Ising energy convention: E = sum_i h_i s_i + sum_(ij) J_ij s_i s_j (each coupler once).
Flipping spin i changes E by -2 * s_i * (h_i + sum_j J_ij s_j); gain = s_i * field_i > 0 means E drops.
"""
from __future__ import annotations

import dimod
import numpy as np
import scipy.sparse as sp

from solvers.simulated_quantum_annealing.s2_mapping import IsingArrays


def steepest_descent(ar: IsingArrays, samples: np.ndarray, max_steps: int | None = None, tol: float = 1e-12):
    """samples: (num_reads, n) of +/-1 in ar.labels order. Returns (new_samples int8, total_flips)."""
    n, R = len(ar.labels), samples.shape[0]
    if n == 0 or R == 0:
        return samples, 0
    Jm = sp.coo_matrix((np.r_[ar.J, ar.J], (np.r_[ar.rows, ar.cols], np.r_[ar.cols, ar.rows])),
                       shape=(n, n)).tocsr()
    s = samples.T.astype(np.float64)              # (n, R), copy
    h = ar.h[:, None]
    cols = np.arange(R)
    flips = 0
    for _ in range(max_steps or 10 * n):
        gain = s * (h + Jm @ s)                   # > 0 where flipping lowers the energy
        i = gain.argmax(axis=0)                   # steepest move per read
        active = gain[i, cols] > tol
        if not active.any():
            break
        s[i[active], cols[active]] *= -1.0
        flips += int(active.sum())
    return s.T.astype(np.int8), flips


def descend_sampleset(sampleset: dimod.SampleSet, spin_bqm: dimod.BinaryQuadraticModel):
    """Steepest descent on an already-unembedded LOGICAL SPIN sampleset. Keeps num_occurrences and
    extra vectors (e.g. chain_break_fraction); recomputes energies. Returns (sampleset, total_flips)."""
    ar = IsingArrays.from_bqm(spin_bqm)
    pos = {v: i for i, v in enumerate(sampleset.variables)}
    arr = np.asarray(sampleset.record.sample)[:, [pos[v] for v in ar.labels]]
    new, flips = steepest_descent(ar, arr)
    energies = spin_bqm.energies((new, ar.labels))
    extra = {k: sampleset.record[k] for k in sampleset.record.dtype.names
             if k not in ("sample", "energy", "num_occurrences")}
    out = dimod.SampleSet.from_samples((new, ar.labels), energy=energies, vartype=dimod.SPIN,
                                       num_occurrences=sampleset.record.num_occurrences, **extra)
    return out, flips


# =========================================================================
# Execution Block: Self-Test
# =========================================================================
if __name__ == "__main__":
    import dimod
    import numpy as np
    from solvers.simulated_quantum_annealing.s2_mapping import IsingArrays

    print("=== Testing Greedy Steepest Descent ===")

    # 1. Create a logical BQM (Ferromagnetic chain: prefers all +1 or all -1)
    # Ground state energy is -2.0 for [1, 1, 1] or [-1, -1, -1]
    bqm = dimod.BinaryQuadraticModel(
        {'v0': 0.0, 'v1': 0.0, 'v2': 0.0},
        {('v0', 'v1'): -1.0, ('v1', 'v2'): -1.0},
        0.0, dimod.SPIN
    )

    # 2. Create a sub-optimal sample
    # We provide [1, -1, 1], which has an energy of +2.0
    ar = IsingArrays.from_bqm(bqm)
    suboptimal_samples = np.array([[1, -1, 1]], dtype=np.int8)

    print("[1] Initial State:")
    print(f"    -> Spins: {suboptimal_samples[0]}")
    print(f"    -> Energy: {bqm.energy({'v0': 1, 'v1': -1, 'v2': 1})}")

    # 3. Run raw steepest descent
    print("\n[2] Running Vectorized Steepest Descent...")
    new_samples, total_flips = steepest_descent(ar, suboptimal_samples)

    print(f"    -> Final Spins: {new_samples[0]}")
    print(f"    -> Total Flips Executed: {total_flips}")

    # 4. Run SampleSet wrapper
    print("\n[3] Testing descend_sampleset wrapper...")
    sampleset = dimod.SampleSet.from_samples(
        [{'v0': 1, 'v1': -1, 'v2': 1}],  # Wrapped in a list, trailing comma removed
        energy=[2.0],
        vartype=dimod.SPIN
    )

    optimized_sampleset, wrapper_flips = descend_sampleset(sampleset, bqm)
    best_record = optimized_sampleset.first

    print(f"    -> Final Energy: {best_record.energy}")
    print(f"    -> Final State: {best_record.sample}")