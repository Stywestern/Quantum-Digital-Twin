"""Steps 2-4 glued together: EmulatedQPU.sample(logical_bqm, embedding, ...) -> logical SampleSet."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import dimod
import numpy as np
from dwave.embedding import (MinimizeEnergy, discard, embed_bqm, majority_vote,
                             unembed_sampleset, weighted_random)
from dwave.embedding.chain_strength import uniform_torque_compensation

from solvers.simulated_quantum_annealing.s1b_embedding import embedding_stats
from solvers.simulated_quantum_annealing.s1_hardware import Hardware, Schedule
from solvers.simulated_quantum_annealing.s2_mapping import IsingArrays, add_ice_noise, quantise, scale_to_hardware
from solvers.simulated_quantum_annealing.s3_samplers import make_sampler


@dataclass
class EmulationResult:
    sampleset: dimod.SampleSet            # logical, BINARY, has 'chain_break_fraction'
    chain_strength: float                 # in problem units, before hardware scaling
    scale_factor: float                   # problem units -> device units
    embedding_stats: dict
    target_sampleset: dimod.SampleSet | None = None
    info: dict = field(default_factory=dict)


class EmulatedQPU:
    def __init__(self, hardware: Hardware, schedule: Schedule | None = None, sampler: str = "sa",
                 sampler_kwargs: dict | None = None, dac_bits: int | None = 5,
                 ice_sigma_h: float = 0.02, ice_sigma_j: float = 0.01,
                 use_extended_j: bool = True, seed: int = 0):
        """
        dac_bits, ice_sigma_h, ice_sigma_j : PLACEHOLDER defaults. They are the knobs step 5 calibrates
        against real QPU runs; do not read physical meaning into them yet.
        use_extended_j : allow chain couplers (strongly negative J) to use the extended J range.
        """
        self.hw = hardware
        self.schedule = schedule or Schedule.placeholder()
        self.sampler_name = sampler
        self.sampler = make_sampler(sampler, self.schedule, hardware.temperature_mk, **(sampler_kwargs or {}))
        self.dac_bits, self.sigma_h, self.sigma_j = dac_bits, ice_sigma_h, ice_sigma_j
        self.use_extended_j = use_extended_j
        self.rng = np.random.default_rng(seed)

    def _chain_break_method(self, name, logical_spin, embedding):
        return {"majority": majority_vote, "discard": discard, "weighted": weighted_random,
                "minimize_energy": MinimizeEnergy(logical_spin, embedding)}[name]

    def sample(self, logical_bqm: dimod.BinaryQuadraticModel, embedding: dict,
               chain_strength: float | None = None, prefactor: float = 1.414,
               num_reads: int = 1000, reads_per_programming: int = 100,
               chain_break_method: str = "majority", keep_target: bool = False) -> EmulationResult:
        """
        reads_per_programming : the ICE noise realisation is redrawn every this many reads, mimicking
                                re-programming of the device between batches.
        """
        logical_spin = logical_bqm.change_vartype(dimod.SPIN, inplace=False)
        if chain_strength is None:
            chain_strength = float(uniform_torque_compensation(logical_spin, prefactor=prefactor))

        # 1) embed: chain couplers = -chain_strength, biases spread over chains
        embedded = embed_bqm(logical_spin, embedding, self.hw.graph, chain_strength=chain_strength)
        ideal = IsingArrays.from_bqm(embedded)

        # 2) map to device: scale, quantise
        j_range = self.hw.extended_j_range if self.use_extended_j else self.hw.j_range
        scaled, factor = scale_to_hardware(ideal, self.hw.h_range, j_range)
        programmed = quantise(scaled, self.hw.h_range, j_range, self.dac_bits)

        # 3) sample, redrawing control noise for each programming cycle
        batches, done = [], 0
        while done < num_reads:
            r = min(reads_per_programming, num_reads - done)
            noisy = add_ice_noise(programmed, self.sigma_h, self.sigma_j, self.rng)
            batches.append(self.sampler.sample(noisy, r, seed=int(self.rng.integers(2 ** 31 - 1))))
            done += r
        samples = np.vstack(batches)

        # energies are reported w.r.t. the IDEAL embedded model (what the user actually asked for)
        energies = embedded.energies((samples, ideal.labels))
        target = dimod.SampleSet.from_samples((samples, ideal.labels), energy=energies, vartype=dimod.SPIN)

        # 4) resolve chains -> logical samples
        method = self._chain_break_method(chain_break_method, logical_spin, embedding)
        logical = unembed_sampleset(target, embedding, logical_spin,
                                    chain_break_method=method, chain_break_fraction=True)
        logical = logical.change_vartype(dimod.BINARY, inplace=False)

        info = {"sampler": self.sampler_name, "dac_bits": self.dac_bits, "sigma_h": self.sigma_h,
                "sigma_j": self.sigma_j, "num_reads": num_reads,
                "reads_per_programming": reads_per_programming,
                "chain_break_method": chain_break_method,
                "placeholder_schedule": self.schedule.is_placeholder}
        return EmulationResult(logical, chain_strength, factor, embedding_stats(embedding),
                               target if keep_target else None, info)


# =========================================================================
# Execution Block: End-to-End Self-Test
# =========================================================================
if __name__ == "__main__":
    import pandapower.networks as nw
    
    try:
        from solvers.qubo_formulator import QuboFormulator
        from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings, select_best
    except ImportError as e:
        print(f"[-] Import error: {e}. Ensure script is run from project root.")
        exit(1)

    print("=== Testing Full EmulatedQPU Pipeline ===")
    
    # 1. Grid & Formulation
    print("[1] Loading case5 and formulating BQM...")
    net = nw.case5()
    if not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0
    
    # Using 10.0 MW precision for a balanced test size
    formulator = QuboFormulator(formulation="dc_ptdf", mw_precision=10.0)
    logical_bqm, _ = formulator._formulate_qubo(net)
    print(f"    -> Logical BQM: {len(logical_bqm.variables)} vars, {len(logical_bqm.quadratic)} edges.")
    
    if len(logical_bqm.variables) == 0:
        print("[-] BQM is empty. Exiting.")
        exit(1)

    # 2. Hardware & Minor-Embedding
    print("\n[2] Initializing Z4 Hardware and finding embedding...")
    hw = Hardware.ideal_zephyr(m=4, t=4)
    # Fast single-seed minorminer test to save time
    emb_results = find_embeddings(logical_bqm, hw, methods=["minorminer"], seeds=[0], verbose=False)
    _, embedding = select_best(emb_results)
    print(f"    -> Embedded into {sum(len(c) for c in embedding.values())} physical qubits.")

    # 3. Emulated QPU Execution
    print("\n[3] Booting EmulatedQPU (SVMC sampler)...")
    qpu = EmulatedQPU(
        hardware=hw,
        sampler="svmc",
        dac_bits=5,
        ice_sigma_h=0.01,
        ice_sigma_j=0.01,
        use_extended_j=True
    )

    print("[4] Submitting problem to EmulatedQPU (20 reads)...")
    # Using reads_per_programming=10 to test the ICE noise redraw logic
    result = qpu.sample(
        logical_bqm=logical_bqm,
        embedding=embedding,
        num_reads=20,
        reads_per_programming=10,
        chain_break_method="majority"
    )

    # 4. Results
    print("\n[+] EmulatedQPU Execution Successful!")
    print(f"    -> Hardware Scale Factor: {result.scale_factor:e}")
    print(f"    -> Chain Strength Applied: {result.chain_strength:.2f}")
    
    sampleset = result.sampleset
    best_energy = sampleset.first.energy
    
    # Safely extract chain break fraction
    cbf = 0.0
    if hasattr(sampleset.record, 'chain_break_fraction'):
        cbf = sampleset.record.chain_break_fraction[0]
        
    print(f"    -> Best Energy (Logical): {best_energy:.2f}")
    print(f"    -> Chain Break Fraction (Best Sample): {cbf:.4f}")
    print(f"    -> Total Unique Samples Found: {len(sampleset)}")
    
    # 5. Decoding validation
    print("\n[5] Decoding back to Power Grid...")
    best_sample_dict = dict(sampleset.first.sample)
    dispatch, sgen, slack, cost, feasibility = formulator._decode_solution(best_sample_dict, net)
    print(f"    -> Feasible: {feasibility['is_feasible']}")
    if feasibility['is_feasible']:
        print(f"    -> Cost: {cost:.2f} EUR/hr")
    else:
        print("    -> Grid State: Constraint violations present (expected due to quantization/noise).")