"""BQM -> Ising problem, plus the exact energy table over all 2^n bitstrings.

Conventions (all verified by selftest.py -- run it before trusting anything else):
  * qubit i  <->  BQM variable labels[i]
  * index k of an n-qubit basis state: wire 0 is the MOST significant bit (PennyLane convention)
  * measured bit b_i in {0,1}  ->  Z eigenvalue z_i = 1 - 2*b_i   (|0> -> +1)  ->  spin s_i = z_i
  * dimod: SPIN +1 <-> BINARY 1, therefore the QUBO variable value is  x_i = 1 - b_i

Cost Hamiltonian:  H_C = sum_i h_i Z_i + sum_(i<j) J_ij Z_i Z_j  (+ offset, irrelevant for the dynamics).
"""
from __future__ import annotations

from dataclasses import dataclass

import dimod
import numpy as np


@dataclass
class IsingProblem:
    labels: list            # qubit i <-> labels[i]
    h: np.ndarray           # (n,)
    rows: np.ndarray        # (m,) int, rows < cols
    cols: np.ndarray        # (m,) int
    J: np.ndarray           # (m,)
    offset: float

    @property
    def n(self) -> int:
        return len(self.labels)

    @classmethod
    def from_bqm(cls, bqm: dimod.BinaryQuadraticModel) -> "IsingProblem":
        spin = bqm.change_vartype(dimod.SPIN, inplace=False)
        labels = list(spin.variables)
        pos = {v: i for i, v in enumerate(labels)}
        h = np.array([spin.get_linear(v) for v in labels], dtype=float)
        rows, cols, vals = [], [], []
        for u, v, b in spin.iter_quadratic():
            i, j = sorted((pos[u], pos[v]))
            rows.append(i)
            cols.append(j)
            vals.append(b)
        return cls(labels, h, np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64),
                   np.array(vals, dtype=float), float(spin.offset))

    def energy_scale(self) -> float:
        """Standard deviation of H_C - offset under the uniform distribution over bitstrings:
        sqrt(sum h^2 + sum J^2) (the Z_i and Z_i Z_j terms are uncorrelated, each with variance 1)."""
        s = float(np.sqrt(np.sum(self.h ** 2) + np.sum(self.J ** 2)))
        return s if s > 0.0 else 1.0


def memory_estimate_gb(n: int) -> float:
    """Rough peak memory of the numpy backend: energy table (8 B), state (16 B), phase buffer (16 B),
    temporaries (~16 B) per amplitude."""
    return 56.0 * (1 << n) / 1e9


class Landscape:
    """Exact normalised energy table  E_norm[k] = (E(k) - offset) / scale  for all k in [0, 2^n).

    Why normalise: QAOA applies exp(-i*gamma*H_C), so gamma is an angle (periodic). With raw QUBO
    coefficients (penalties up to ~1e5-1e6) the landscape in gamma is hopelessly oscillatory, and the
    optimiser has no sensible range to search. Dividing by the energy standard deviation puts the
    useful gamma range at O(1) and leaves the minimiser unchanged. The mean of E_norm is exactly 0.
    """

    def __init__(self, prob: IsingProblem, normalize: bool = True, max_qubits: int = 24,
                 chunk_bits: int = 12):
        n = prob.n
        if n > max_qubits:
            raise ValueError(
                f"{n} qubits exceeds max_qubits={max_qubits} (needs ~{memory_estimate_gb(n):.1f} GB and "
                f"a 2^{n} energy table). Reduce the QUBO: coarser mw_precision, larger "
                f"slack_precision_factor, fewer assets/lines, or raise max_qubits knowingly.")
        self.prob, self.n, self.N = prob, n, 1 << n
        self.scale = prob.energy_scale() if normalize else 1.0
        self.E_norm = self._build_table(chunk_bits)
        self.e_min = float(self.E_norm.min())
        self.e_max = float(self.E_norm.max())
        tol = 1e-9 * max(1.0, abs(self.e_min), abs(self.e_max))
        self.ground = np.flatnonzero(self.E_norm <= self.e_min + tol)

    def _build_table(self, chunk_bits: int) -> np.ndarray:
        p, n = self.prob, self.n
        h = p.h / self.scale
        J = p.J / self.scale
        out = np.empty(self.N)
        shifts = (n - 1 - np.arange(n)).astype(np.int64)
        chunk = 1 << min(chunk_bits, n)
        for start in range(0, self.N, chunk):
            idx = np.arange(start, min(start + chunk, self.N), dtype=np.int64)
            z = 1.0 - 2.0 * ((idx[:, None] >> shifts[None, :]) & 1)
            e = z @ h
            if J.size:
                e += (z[:, p.rows] * z[:, p.cols]) @ J
            out[start:start + len(idx)] = e
        return out

    # ---- conversions -------------------------------------------------------
    def raw_energy(self, idx) -> np.ndarray:
        """Original QUBO energy (incl. offset) of basis-state index/indices."""
        return self.E_norm[np.asarray(idx)] * self.scale + self.prob.offset

    def index_to_sample(self, k: int) -> dict:
        """Basis index -> {variable label: QUBO value in {0,1}} (x = 1 - b)."""
        n = self.n
        return {lab: 1 - ((int(k) >> (n - 1 - i)) & 1) for i, lab in enumerate(self.prob.labels)}

    # ---- metrics -----------------------------------------------------------
    def approx_ratio(self, e_mean_norm: float) -> float:
        """(E_max - <E>) / (E_max - E_min) in [0, 1]; 1 = ground state, uniform random state gives
        approx_ratio_random()."""
        span = self.e_max - self.e_min
        return float((self.e_max - e_mean_norm) / span) if span > 0 else 1.0

    def approx_ratio_random(self) -> float:
        return self.approx_ratio(0.0)

    def p_ground(self, probs: np.ndarray) -> float:
        return float(probs[self.ground].sum())


if __name__ == "__main__":
    import pandapower.networks as nw
    import time
    
    # Adjust this import based on your actual project structure
    try:
        from solvers.qubo_formulator import QuboFormulator
    except ImportError:
        print("[-] Could not import QuboFormulator. Ensure script is run from project root.")
        exit(1)

    print("=== Testing Ising Landscape Generation ===")
    
    # 1. Generate a small test grid
    print("[1] Loading case5 network...")
    net = nw.case5()
    if not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0
    
    # 2. Formulate the BQM using your existing pipeline
    # We MUST use coarse precision (50MW) to keep variables < 24 to avoid RAM crashes
    print("[2] Formulating DC-PTDF QUBO (Radix, 50 MW precision)...")
    formulator = QuboFormulator(formulation="dc_ptdf", mw_precision=50.0)
    bqm, _ = formulator._formulate_qubo(net)
    
    logical_vars = len(bqm.variables)
    print(f"    -> BQM generated successfully with {logical_vars} logical variables.")
    
    if logical_vars > 24:
        print("    [!] ERROR: Too many variables for full state-vector Landscape test. Aborting.")
        exit(1)

    # 3. Convert to Ising and Build the Landscape
    print("\n[3] Converting BQM to Ising Hamiltonian...")
    t0 = time.time()
    prob = IsingProblem.from_bqm(bqm)
    print(f"    -> Ising Model created. Energy Scale (Normalization Factor): {prob.energy_scale():.2f}")
    
    print(f"\n[4] Building Exact Energy Landscape (Brute Force 2^{logical_vars} states)...")
    landscape = Landscape(prob, normalize=True)
    t1 = time.time()
    
    print(f"    -> Landscape built in {t1-t0:.4f} seconds!")
    print(f"    -> Total states analyzed: {landscape.N:,}")
    print(f"    -> Normalized Ground State Energy: {landscape.e_min:.4f}")
    print(f"    -> Normalized Max Energy (Worst State): {landscape.e_max:.4f}")
    print(f"    -> Number of degenerate ground states found: {len(landscape.ground)}")
    
    # 4. Check the true global minimum
    print("\n[5] True Global Minimum Decoding:")
    best_idx = landscape.ground[0]
    raw_e = landscape.raw_energy(best_idx)
    
    print(f"    -> Raw QUBO Energy (Cost + Penalties): {raw_e:.2f}")
    
    # Convert the integer index back into a binary dictionary
    sample = landscape.index_to_sample(best_idx)
    
    # Print the first 5 variable assignments to verify the dictionary mapping works
    preview = {k: sample[k] for k in list(sample.keys())[:5]}
    print(f"    -> Optimal Bitstring Sample (Preview): {preview}")
    print("\n=== Test Complete ===")