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
from solvers.simulated_quantum_annealing.s4_postprocess import descend_sampleset, steepest_descent

@dataclass
class EmulationResult:
    sampleset: dimod.SampleSet            # logical, BINARY, has 'chain_break_fraction'
    chain_strength: float                 # in problem units, before hardware scaling
    scale_factor: float                   # problem units -> device units
    embedding_stats: dict
    target_sampleset: dimod.SampleSet | None = None
    info: dict = field(default_factory=dict)


class EmulatedQPU:
    def __init__(self, hardware: Hardware, schedule: Schedule, sampler: str = "sa",
                 sampler_kwargs: dict | None = None, dac_bits: int | None = 5,
                 ice_sigma_h: float = 0.02, ice_sigma_j: float = 0.01,
                 use_extended_j: bool = True, seed: int = 0):
        """
        dac_bits, ice_sigma_h, ice_sigma_j : PLACEHOLDER defaults. They are the knobs step 5 calibrates
        against real QPU runs; do not read physical meaning into them yet.
        use_extended_j : allow chain couplers (strongly negative J) to use the extended J range.
        """

        self.hw = hardware
        self.schedule = schedule
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
               relative_chain_multiplier: float | None = 1,
               num_reads: int = 1000, reads_per_programming: int = 100,
               chain_break_method: str = "majority", keep_target: bool = False,
               postprocess: str | None = "both") -> EmulationResult:
        """
        relative_chain_multiplier : If provided, caps the calculated chain strength at this
                                    multiplier times the maximum logical problem weight.
                                    e.g., 1.0 means chains are exactly as strong as the max constraint.
        reads_per_programming : the ICE noise realisation is redrawn every this many reads, mimicking
                                re-programming of the device between batches.
        postprocess : None (default) | "physical" | "logical" | "both". Greedy steepest descent.
                      "physical": on the embedded problem, before chain resolution.
                      "logical" : on the logical problem after chain resolution.
        """
        if postprocess not in (None, "physical", "logical", "both"):
            raise ValueError("postprocess must be None, 'physical', 'logical' or 'both'")
            
        logical_spin = logical_bqm.change_vartype(dimod.SPIN, inplace=False)
        
        # --- Capped Chain Strength Logic ---
        if chain_strength is None:
            raw_cs = float(uniform_torque_compensation(logical_spin, embedding=embedding, prefactor=prefactor))
            
            if relative_chain_multiplier is not None:
                max_logical_weight = max(
                    max((abs(v) for v in logical_spin.linear.values()), default=0.0),
                    max((abs(v) for v in logical_spin.quadratic.values()), default=0.0)
                )
                capped_cs = float(max_logical_weight * relative_chain_multiplier)
                chain_strength = min(raw_cs, capped_cs)
            else:
                chain_strength = raw_cs
        else:
            raw_cs = chain_strength # Fallback if user explicitly provided a chain_strength
        # ---------------------------------------------

        # 1A) The QPU Matrix (Capped): For scaling and hardware sampling
        embedded_qpu = embed_bqm(logical_spin, embedding, self.hw.graph, chain_strength=chain_strength)
        ideal_qpu = IsingArrays.from_bqm(embedded_qpu)

        # 1B) The Healing Matrix (Uncapped): For the physical post-processor
        embedded_healing = embed_bqm(logical_spin, embedding, self.hw.graph, chain_strength=raw_cs)
        ideal_healing = IsingArrays.from_bqm(embedded_healing)

        # 2) Map to device: Scale the QPU MATRIX (Preserves DAC dynamic range!)
        j_range = self.hw.extended_j_range if self.use_extended_j else self.hw.j_range
        scaled, factor = scale_to_hardware(ideal_qpu, self.hw.h_range, j_range)
        programmed = quantise(scaled, self.hw.h_range, j_range, self.dac_bits)

        # 3) Sample, redrawing control noise for each programming cycle
        batches, done = [], 0
        while done < num_reads:
            r = min(reads_per_programming, num_reads - done)
            noisy = add_ice_noise(programmed, self.sigma_h, self.sigma_j, self.rng)
            batches.append(self.sampler.sample(noisy, r, seed=int(self.rng.integers(2 ** 31 - 1))))
            done += r
        samples = np.vstack(batches)

        flips_phys = flips_log = 0
        
        # 4) Physical Post-Processing: Feed it the HEALING matrix to force chain repair
        if postprocess in ("physical", "both"):
            samples, flips_phys = steepest_descent(ideal_healing, samples) 

        # Energies are reported w.r.t. the CAPPED embedded model (what you actually submitted to QPU)
        energies = embedded_qpu.energies((samples, ideal_qpu.labels))
        target = dimod.SampleSet.from_samples((samples, ideal_qpu.labels), energy=energies, vartype=dimod.SPIN)

        # 5) Resolve chains -> logical samples
        method = self._chain_break_method(chain_break_method, logical_spin, embedding)
        logical = unembed_sampleset(target, embedding, logical_spin,
                                    chain_break_method=method, chain_break_fraction=True)
                                    
        # 6) Logical post-processing (fine-tunes the resolved economic variables)
        if postprocess in ("logical", "both"):
            logical, flips_log = descend_sampleset(logical, logical_spin)
            
        logical = logical.change_vartype(dimod.BINARY, inplace=False)

        info = {"sampler": self.sampler_name, "dac_bits": self.dac_bits, "sigma_h": self.sigma_h,
                "sigma_j": self.sigma_j, "num_reads": num_reads,
                "reads_per_programming": reads_per_programming,
                "chain_break_method": chain_break_method,
                "postprocess": postprocess, "postprocess_flips": (flips_phys, flips_log),
                "placeholder_schedule": self.schedule.is_placeholder}
                
        return EmulationResult(logical, chain_strength, factor, embedding_stats(embedding),
                               target if keep_target else None, info)


# =========================================================================
# Execution Block: Exhaustive Combinatorial Sweep & Architecture Benchmark
# =========================================================================
if __name__ == "__main__":
    import time
    import itertools
    import dimod
    import numpy as np
    import pandapower.networks as nw

    try:
        from solvers.qubo_formulator import QuboFormulator
        from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings, select_best
        from solvers.simulated_quantum_annealing.s1_hardware import Hardware, Schedule
        from solvers.simulated_quantum_annealing.s5_emulators import EmulatedQPU
    except ImportError as e:
        print(f"[-] Import error: {e}. Ensure script is run from project root.")
        exit(1)

    print("=== Initializing Exhaustive Combinatorial Emulation Sweep on case5 ===")
    
    # 1. Grid & Formulation
    net = nw.case5()
    if not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0
    
    formulator = QuboFormulator(formulation="dc_ptdf", mw_precision=10.0)
    logical_bqm, _ = formulator._formulate_qubo(net)

    # 2. Hardware & Minor-Embedding (Fixed to keep graph routing constant)
    hw = Hardware.ideal_zephyr(m=4, t=4)
    emb_results = find_embeddings(logical_bqm, hw, methods=["minorminer"], seeds=[0], verbose=False)
    _, embedding = select_best(emb_results)

    # 3. Define Combinatorial Parameter Space
    samplers = ["sqa", "svmc", "sa"]
    dac_bits_list = [3, 5, 8]
    multipliers = [None, 0.5, 1.0, 2.0]
    cb_methods = ["majority", "discard"]
    post_modes = [None, "physical", "logical", "both"]

    schedule = Schedule.placeholder()
    num_reads = 100
    seed_val = 42

    combinations = list(itertools.product(samplers, dac_bits_list, multipliers, cb_methods, post_modes))
    total_runs = len(combinations)
    
    print(f"[-] Total configurations in combinatorial space: {total_runs}")
    print("\n" + "="*125)
    print(f"{'Sampler':<6} | {'DAC':<4} | {'Mult':<6} | {'CB Method':<9} | {'Post-Proc':<9} | {'Energy':<10} | {'CBF (%)':<8} | {'Phys':<5} | {'Log':<5} | {'Time (s)'}")
    print("="*125)

    results = []

    for idx, (sampler_name, dac, mult, cb, pp) in enumerate(combinations, 1):
        # Configure sampler-specific arguments
        sampler_kwargs = {}
        if sampler_name == "sqa":
            sampler_kwargs["trotter_slices"] = 16
        elif sampler_name == "svmc":
            sampler_kwargs["num_sweeps"] = 1000

        qpu = EmulatedQPU(
            hardware=hw,
            schedule=schedule,
            sampler=sampler_name,
            sampler_kwargs=sampler_kwargs,
            dac_bits=dac,
            ice_sigma_h=0.01,
            ice_sigma_j=0.01,
            seed=seed_val
        )

        t0 = time.time()
        try:
            res = qpu.sample(
                logical_bqm=logical_bqm,
                embedding=embedding,
                num_reads=num_reads,
                reads_per_programming=50,
                relative_chain_multiplier=mult,
                chain_break_method=cb,
                postprocess=pp
            )
            t_exec = time.time() - t0

            if len(res.sampleset) > 0:
                best_energy = float(res.sampleset.first.energy)
                cbf = float(np.mean(res.sampleset.record.chain_break_fraction) * 100)
            else:
                best_energy = float('nan')
                cbf = 100.0

            phys_flips, log_flips = res.info.get("postprocess_flips", (0, 0))
            status = "Success"

        except Exception as e:
            t_exec = time.time() - t0
            best_energy = float('nan')
            cbf = 100.0
            phys_flips, log_flips = 0, 0
            status = f"Failed: {str(e)}"

        results.append({
            "sampler": sampler_name,
            "dac": dac,
            "mult": mult,
            "cb": cb,
            "pp": str(pp),
            "energy": best_energy,
            "cbf": cbf,
            "phys": phys_flips,
            "log": log_flips,
            "time": t_exec,
            "status": status
        })

        mult_str = str(mult) if mult is not None else "RMS"
        pp_str = str(pp) if pp is not None else "None"
        
        print(f"{sampler_name.upper():<6} | {dac:<4} | {mult_str:<6} | {cb:<9} | {pp_str:<9} | {best_energy:<10.2f} | {cbf:<8.2f} | {phys_flips:<5} | {log_flips:<5} | {t_exec:<8.4f}")

    print("="*125)
    print(f"[+] Completed all {total_runs} emulation runs.")