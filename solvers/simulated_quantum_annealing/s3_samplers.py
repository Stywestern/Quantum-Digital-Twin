"""Step 3: sampler back-ends acting on the *programmed* (scaled, quantised, noisy) Ising model.

Both return an int8 array (num_reads, n) of +/-1 spins, columns ordered like IsingArrays.labels.

  SAFreezeout : simulated annealing whose final inverse temperature is the physical freeze-out
                beta,  beta = (B(s*)/2) / kT.   Cheap; no quantum-inspired dynamics.
  SVMCSampler : spin-vector Monte Carlo (Shin, Smith, Smolin, Vazirani 2014). Each qubit is a planar
                rotor angle theta in [0, pi]; the schedule A(s), B(s) drives a Metropolis anneal
                with  E = -A/2 sum sin(theta_i) + B/2 [sum h_i cos(theta_i) + sum J_ij cos(theta_i) cos(theta_j)].
                Updates are vectorised over reads and over graph-colour classes.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp

from solvers.simulated_quantum_annealing.s1_hardware import KB_OVER_H_GHZ_PER_K, Schedule, beta_eff
from solvers.simulated_quantum_annealing.s2_mapping import IsingArrays


class SAFreezeout:
    def __init__(self, schedule: Schedule, temperature_mk: float = 12.0, freeze_s: float = 0.6,
                 num_sweeps: int = 1000, beta_hot: float = 0.1):
        """freeze_s is the point on the schedule where dynamics stop (a calibration parameter)."""
        self.schedule, self.temperature_mk = schedule, temperature_mk
        self.freeze_s, self.num_sweeps, self.beta_hot = freeze_s, num_sweeps, beta_hot

    def beta_final(self) -> float:
        _, B = self.schedule.at(self.freeze_s)
        return beta_eff(B, self.temperature_mk)

    def sample(self, ar: IsingArrays, num_reads: int, seed: int) -> np.ndarray:
        try:
            from dwave.samplers import SimulatedAnnealingSampler
        except ImportError:
            from neal import SimulatedAnnealingSampler
        bqm = ar.to_bqm()
        bf = self.beta_final()
        res = SimulatedAnnealingSampler().sample(
            bqm, num_reads=num_reads, num_sweeps=self.num_sweeps,
            beta_range=(min(self.beta_hot, bf / 10.0), bf), seed=int(seed))
        lab = list(res.variables)
        pos = {v: i for i, v in enumerate(lab)}
        order = [pos[v] for v in ar.labels]
        return np.asarray(res.record.sample)[:, order].astype(np.int8)


class SVMCSampler:
    def __init__(self, schedule: Schedule, temperature_mk: float = 12.0, num_sweeps: int = 1000):
        """num_sweeps is a stand-in for annealing time (calibration parameter)."""
        self.schedule, self.temperature_mk, self.num_sweeps = schedule, temperature_mk, num_sweeps

    def sample(self, ar: IsingArrays, num_reads: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n, R = len(ar.labels), num_reads
        kT = KB_OVER_H_GHZ_PER_K * self.temperature_mk * 1e-3     # GHz

        # symmetric sparse coupling matrix and graph-colour blocks (no two nodes in a block are adjacent,
        # so a whole block can be Metropolis-updated simultaneously without violating detailed balance)
        Jm = sp.coo_matrix((np.r_[ar.J, ar.J], (np.r_[ar.rows, ar.cols], np.r_[ar.cols, ar.rows])),
                           shape=(n, n)).tocsr()
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(zip(ar.rows.tolist(), ar.cols.tolist()))
        colour = nx.greedy_color(g, strategy="largest_first")
        blocks = {}
        for v, c in colour.items():
            blocks.setdefault(c, []).append(v)
        blocks = [np.array(b) for b in blocks.values()]
        Jrows = [Jm[b] for b in blocks]
        hb = [ar.h[b][:, None] for b in blocks]

        theta = np.full((n, R), np.pi / 2)                        # start in the transverse-field ground state
        for t in range(self.num_sweeps):
            A, B = self.schedule.at((t + 0.5) / self.num_sweeps)
            for b, Jb, h in zip(blocks, Jrows, hb):
                c = np.cos(theta)
                field = h + Jb @ c                                # (|b|, R): h_i + sum_j J_ij cos(theta_j)
                old = theta[b]
                new = rng.uniform(0.0, np.pi, old.shape)
                dE = (-0.5 * A * (np.sin(new) - np.sin(old))
                      + 0.5 * B * field * (np.cos(new) - np.cos(old)))
                accept = (dE <= 0.0) | (rng.random(dE.shape) < np.exp(-np.clip(dE / kT, 0.0, 700.0)))
                theta[b] = np.where(accept, new, old)
        return np.where(np.cos(theta) > 0.0, 1, -1).T.astype(np.int8)


def make_sampler(kind: str, schedule: Schedule, temperature_mk: float, **kw):
    if kind == "sa":
        return SAFreezeout(schedule, temperature_mk, **kw)
    if kind == "svmc":
        return SVMCSampler(schedule, temperature_mk, **kw)
    raise ValueError(f"unknown sampler {kind!r}; use 'sa' or 'svmc'")


# =========================================================================
# Execution Block: Self-Test
# =========================================================================
if __name__ == "__main__":
    import dimod
    import numpy as np

    # Ensure these paths match your actual directory structure
    from solvers.simulated_quantum_annealing.s1_hardware import Schedule
    from solvers.simulated_quantum_annealing.s2_mapping import (
        IsingArrays, scale_to_hardware, quantise, add_ice_noise
    )

    print("=== Testing SVMC Sampler Pipeline ===")
    
    # 1. Create a synthetic frustrated logical problem (triangle graph)
    # Using dimod.SPIN directly to avoid the vartype conversion error
    print("[1] Creating logical SPIN BQM...")
    bqm = dimod.BinaryQuadraticModel(
        {'v0': 0.1, 'v1': -0.2, 'v2': 0.0},
        {('v0', 'v1'): 1.5, ('v1', 'v2'): 2.0, ('v0', 'v2'): 2.5},
        0.0, dimod.SPIN
    )
    
    # 2. Convert to IsingArrays
    print("[2] Converting to IsingArrays...")
    ar_logical = IsingArrays.from_bqm(bqm)
    
    # 3. Map to Analog Hardware (Scale, Quantize, ICE)
    print("[3] Mapping to Analog Hardware (Z4 ranges, 5-bit DAC, 0.01 ICE)...")
    h_range = (-4.0, 4.0)
    j_range = (-1.0, 1.0)
    
    ar_scaled, scale_factor = scale_to_hardware(ar_logical, h_range, j_range)
    ar_quantized = quantise(ar_scaled, h_range, j_range, bits=5)
    
    rng = np.random.default_rng(seed=42)
    ar_physical = add_ice_noise(ar_quantized, sigma_h=0.01, sigma_j=0.01, rng=rng)
    
    # 4. Initialize SVMC Sampler
    print("\n[4] Initializing SVMC Sampler...")
    schedule = Schedule.placeholder()
    temperature_mk = 15.0
    num_sweeps = 500  # Number of simulation steps per read
    num_reads = 10
    
    svmc_sampler = SVMCSampler(
        schedule=schedule, 
        temperature_mk=temperature_mk, 
        num_sweeps=num_sweeps
    )
    
    # 5. Execute Sampling
    print(f"[5] Executing SVMC for {num_reads} reads...")
    samples = svmc_sampler.sample(ar_physical, num_reads=num_reads, seed=42)
    
    print("\n[+] SVMC Execution Successful!")
    print(f"    -> Output Shape: {samples.shape} (reads, variables)")
    print(f"    -> First Sample (Spins): {samples[0]}")
    print(f"    -> Variable Order: {ar_physical.labels}")