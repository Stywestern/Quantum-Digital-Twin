"""QAOA Solvers: Drop-in siblings of SimulatedAnnealingSolver and EmulatedQPUSolver.

Splits QAOA into two dedicated solvers:
  - IdealQAOA: Fast statevector simulation (Numpy backend), all-to-all connectivity baseline.
  - EmuQAOA:   Angle transfer to a noisy gate-level emulator (PennyLane density matrix),
               modeling IBM Heavy-Hex routing, CNOT depolarizing errors, and readout noise.
"""
from __future__ import annotations

import time
import numpy as np

from solvers.qubo_formulator import QuboFormulator
from solvers.simulated_gate_based.s2_circuits import make_circuit, resource_estimate
from solvers.simulated_gate_based.s5_evaluate import evaluate_samples
from solvers.simulated_gate_based.s1_ising import IsingProblem, Landscape
from solvers.simulated_gate_based.s3_optimize import depth_sweep


class _BaseQAOASolver(QuboFormulator):
    """Internal base class providing shared QUBO formulation and OPF decoding infrastructure."""

    def __init__(self, formulation="dc_ptdf", mw_precision=20.0, p=2, p_sweep=None,
                 shots=2000, method="COBYLA", maxiter=200, n_restarts=1, dt=0.75,
                 seed=0, max_qubits=24, max_decode=None, **kwargs):
        super().__init__(formulation=formulation, mw_precision=mw_precision, **kwargs)
        self.p = p
        self.p_sweep = tuple(p_sweep) if p_sweep else None
        self.shots = shots
        self.method = method
        self.maxiter = maxiter
        self.n_restarts = n_restarts
        self.dt = dt
        self.seed = seed
        self.max_qubits = max_qubits
        self.max_decode = max_decode

    def _build_opf_payload(self, cost, dispatch, sgen_dispatch, slack_dispatch, feasibility,
                           complexity, status, solver_name, form_t, opt_t, samp_t, start,
                           qaoa_meta, resources, n_qubits, hardware_metrics):
        max_load_pct = None
        if feasibility and getattr(self, "_lines", None):
            loads = [abs(feasibility["line_flows_mw"].get(ln["idx"], 0.0)) / ln["p_max"] * 100.0
                     for ln in self._lines if ln["p_max"] > 0.0]
            if loads:
                max_load_pct = round(max(loads), 3)

        solution = {
            "cost_eur_per_hr": cost,
            "generator_dispatch_mw": dispatch,
            "static_generator_dispatch_mw": sgen_dispatch,
            "slack_dispatch_mw": slack_dispatch,
            "grid_state": {
                "max_line_loading_percent": max_load_pct,
                "max_voltage_pu": 1.0,
                "min_voltage_pu": 1.0,
                "feasibility": feasibility,
            },
        }

        metadata = {
            "status": status,
            "solver_name": solver_name,
            "execution_time_seconds": {
                "total": round(time.time() - start, 4),
                "formulation_cpu": round(form_t, 4),
                "parameter_optimisation": round(opt_t, 4),
                "sampling_and_decoding": round(samp_t, 4),
            },
            "problem_complexity": complexity,
            "qubo_parameters": {
                "mw_precision": self.mw_precision,
                "penalty_balance": getattr(self, "penalty_balance", None),
                "penalty_line": getattr(self, "penalty_line", None),
                "seed": self.seed,
            },
            "algorithmic_metrics": {
                "framework": "gate_based_qaoa",
                "num_iterations": 1,
                "qaoa": qaoa_meta,
            },
            "hardware_metrics": hardware_metrics,
        }
        return solution, metadata


class IdealQAOA(_BaseQAOASolver):
    """Ideal statevector QAOA solver using the high-speed NumPy backend.

    Serves as the theoretical mathematical baseline with infinite coherence
    and all-to-all logical connectivity.
    """

    def __init__(self, formulation="dc_ptdf", mw_precision=20.0, p=2, p_sweep=None,
                 shots=2000, method="COBYLA", maxiter=200, n_restarts=1, dt=0.75,
                 seed=0, max_qubits=24, max_decode=None, **kwargs):
        super().__init__(
            formulation=formulation, mw_precision=mw_precision, p=p, p_sweep=p_sweep,
            shots=shots, method=method, maxiter=maxiter, n_restarts=n_restarts, dt=dt,
            seed=seed, max_qubits=max_qubits, max_decode=max_decode, **kwargs
        )

    def solve_opf(self, net):
        start = time.time()
        status, cost, dispatch, sgen_dispatch, slack_dispatch = "Success", None, {}, {}, {}
        feasibility, complexity, qaoa_meta, resources, n_qubits = None, {}, {}, {}, 0
        form_t = opt_t = samp_t = 0.0

        try:
            # 1. Formulation & Landscape
            t0 = time.time()
            bqm, complexity = self._formulate_qubo(net)
            form_t = time.time() - t0
            n_qubits = len(bqm.variables)

            prob = IsingProblem.from_bqm(bqm)
            land = Landscape(prob, max_qubits=self.max_qubits)

            # 2. Angle Training on Ideal Statevector
            t1 = time.time()
            p_values = self.p_sweep or (self.p,)
            sweep = depth_sweep(land, p_values, backend="numpy", method=self.method,
                                maxiter=self.maxiter, n_restarts=self.n_restarts,
                                seed=self.seed, dt=self.dt)
            opt_t = time.time() - t1
            res, circ = sweep[-1]

            # 3. Sampling from Ideal Distribution
            t2 = time.time()
            stats = evaluate_samples(circ, self, net, res.params, shots=self.shots,
                                     seed=self.seed, max_decode=self.max_decode)
            samp_t = time.time() - t2

            chosen = stats["best_feasible"] or stats["lowest_energy_sample"]
            if stats["best_feasible"] is None:
                f = chosen["feasibility"]
                status = (f"Infeasible: no feasible sample among {stats['decoded_unique']} decoded unique "
                          f"samples (balance_ok={f['balance_ok']}, lines_ok={f['lines_ok']})")

            dispatch, sgen_dispatch = chosen["dispatch"], chosen["sgen_dispatch"]
            slack_dispatch, cost, feasibility = chosen["slack_dispatch"], chosen["cost"], chosen["feasibility"]

            resources = resource_estimate(prob, res.p, target_hardware="ideal")
            qaoa_meta = {
                "depths": [r.p for r, _ in sweep],
                "per_depth": [{"p": r.p, "approx_ratio": r.approx_ratio, "p_ground": r.p_ground,
                               "n_evals": r.n_evals, "converged": r.success} for r, _ in sweep],
                "approx_ratio": res.approx_ratio,
                "approx_ratio_random_baseline": res.approx_ratio_random,
                "p_ground_exact": res.p_ground,
                "p_ground_sampled": stats["p_ground_sampled"],
                "feasible_fraction": stats["feasible_fraction"],
                "feasible_found": stats["feasible_found"],
                "energy_scale": land.scale,
                "total_circuit_evals": int(sum(r.n_evals for r, _ in sweep)),
                "optimizer": self.method,
                "backend": "numpy_statevector",
            }
        except Exception as e:
            status = f"Failed: {e}"
            print(f"\n[IdealQAOA Error]: {e}\n")

        hardware_metrics = {
            "qpu_target": "ideal_statevector",
            "logical_qubits": n_qubits,
            "physical_qubits": n_qubits,
            "circuit_depth": resources.get("depth_estimate", 0),
            "two_qubit_gates": resources.get("logical_two_qubit_gates", 0),
            "routing_overhead_multiplier": 1.0,
            "resources": resources,
        }

        return self._build_opf_payload(
            cost, dispatch, sgen_dispatch, slack_dispatch, feasibility, complexity, status,
            f"ideal_qaoa_{self.formulation}", form_t, opt_t, samp_t, start,
            qaoa_meta, resources, n_qubits, hardware_metrics
        )


class EmuQAOA(_BaseQAOASolver):
    """Hardware-aware QAOA emulator modeling physical gate-based chips.

    Workflow:
      1. Solves for optimal (gamma, beta) angles using the fast NumPy simulator.
      2. Transfers parameters into a PennyLane density matrix (`default.mixed`).
      3. Applies heavy-hex routing SWAP inflation, two-qubit depolarizing noise,
         and readout bit-flip channels.
      4. Extracted shots undergo Steepest Descent error mitigation to fix readout/CNOT flips.
    """

    def __init__(self, formulation="dc_ptdf", mw_precision=20.0, p=2, p_sweep=None,
                 hardware_preset="ibm_eagle", shots=2000, method="COBYLA", maxiter=200,
                 n_restarts=1, dt=0.75, seed=0, max_qubits=12, max_decode=None, **kwargs):
        super().__init__(
            formulation=formulation, mw_precision=mw_precision, p=p, p_sweep=p_sweep,
            shots=shots, method=method, maxiter=maxiter, n_restarts=n_restarts, dt=dt,
            seed=seed, max_qubits=max_qubits, max_decode=max_decode, **kwargs
        )
        self.hardware_preset = hardware_preset

    def solve_opf(self, net):
        # We need the index sampler and post-processor specifically for Emulation
        from solvers.simulated_gate_based.s5_evaluate import sample_indices
        from solvers.simulated_gate_based.s4_postprocess import postprocess_shots

        start = time.time()
        status, cost, dispatch, sgen_dispatch, slack_dispatch = "Success", None, {}, {}, {}
        feasibility, complexity, qaoa_meta, resources, n_qubits = None, {}, {}, {}, 0
        form_t = opt_t = samp_t = 0.0

        try:
            # 1. Formulation & Landscape
            t0 = time.time()
            bqm, complexity = self._formulate_qubo(net)
            form_t = time.time() - t0
            n_qubits = len(bqm.variables)

            if n_qubits > self.max_qubits:
                raise ValueError(
                    f"{n_qubits} qubits exceeds density matrix ceiling ({self.max_qubits}). "
                    f"Simulating 2^(2*{n_qubits}) state elements will cause an OOM error."
                )

            prob = IsingProblem.from_bqm(bqm)
            land = Landscape(prob, max_qubits=self.max_qubits)

            # 2. Angle Training via Parameter Transfer (Fast NumPy backend)
            t1 = time.time()
            p_values = self.p_sweep or (self.p,)
            sweep = depth_sweep(land, p_values, backend="numpy", method=self.method,
                                maxiter=self.maxiter, n_restarts=self.n_restarts,
                                seed=self.seed, dt=self.dt)
            opt_t = time.time() - t1
            res, _ = sweep[-1]

            # 3. Instantiate Noisy Hardware-Mapped Circuit
            circ_emu = make_circuit(
                land, res.p, backend="pennylane", device="default.mixed",
                cost_layer="gates", hardware_preset=self.hardware_preset
            )

            # 4. Sampling & Error Mitigation
            t2 = time.time()
            
            # 4a. Raw Measurement (Hardware readouts before classical fixes)
            uniq_raw, cnt_raw = sample_indices(circ_emu, res.params, shots=self.shots, seed=self.seed)
            
            raw_energies = land.raw_energy(uniq_raw)
            best_raw_idx = uniq_raw[np.argmin(raw_energies)]
            best_raw_energy = float(np.min(raw_energies))
            
            # 4b. Steepest Descent Mitigation
            uniq_fixed, cnt_fixed, total_flips = postprocess_shots(land, uniq_raw, cnt_raw)
            
            # 4c. Decode the Mitigated Shots
            # (We bypass the internal sampler in evaluate_samples by overwriting it momentarily, 
            # or we just manually decode the fixed shots here as evaluate_samples does)
            order = np.argsort(land.raw_energy(uniq_fixed))
            if self.max_decode: order = order[:self.max_decode]
            
            best_feasible, best_any = None, None
            feas_shots, decoded_shots, n_decoded = 0, 0, 0
            
            for j in order:
                sample = land.index_to_sample(int(uniq_fixed[j]))
                d_p, d_sgen, d_slack, d_cost, d_feas = self._decode_solution(sample, net)
                n_decoded += 1
                decoded_shots += int(cnt_fixed[j])
                
                row = {"dispatch": d_p, "sgen_dispatch": d_sgen, "slack_dispatch": d_slack, "cost": d_cost,
                       "feasibility": d_feas, "qubo_energy": float(land.raw_energy(uniq_fixed[j])), "shots": int(cnt_fixed[j])}
                
                if best_any is None: best_any = row
                if d_feas["is_feasible"]:
                    feas_shots += int(cnt_fixed[j])
                    if best_feasible is None: best_feasible = row
            
            samp_t = time.time() - t2

            chosen = best_feasible or best_any
            if best_feasible is None:
                f = chosen["feasibility"]
                status = (f"Infeasible: no feasible sample among {n_decoded} decoded unique "
                          f"samples under {self.hardware_preset} noise (balance_ok={f['balance_ok']}, lines_ok={f['lines_ok']})")

            dispatch, sgen_dispatch = chosen["dispatch"], chosen["sgen_dispatch"]
            slack_dispatch, cost, feasibility = chosen["slack_dispatch"], chosen["cost"], chosen["feasibility"]
            
            # Calculate P(ground) for both sets
            p_ground_raw = float(cnt_raw[np.isin(uniq_raw, land.ground)].sum() / self.shots)
            p_ground_fixed = float(cnt_fixed[np.isin(uniq_fixed, land.ground)].sum() / self.shots)

            resources = resource_estimate(prob, res.p, target_hardware=self.hardware_preset)
            
            # 5. Metadata Construction (Tracking Pre/Post Mitigation)
            qaoa_meta = {
                "depths": [r.p for r, _ in sweep],
                "trained_optimal_angles": res.params.tolist(),
                "ideal_approx_ratio": res.approx_ratio,
                "ideal_p_ground": res.p_ground,
                
                # Mitigation Tracking Block
                "raw_readout": {
                    "p_ground_sampled": p_ground_raw,
                    "lowest_energy_sampled": best_raw_energy,
                    "unique_states_measured": int(len(uniq_raw))
                },
                "mitigated_readout": {
                    "p_ground_sampled": p_ground_fixed,
                    "lowest_energy_sampled": best_any["qubo_energy"] if best_any else None,
                    "unique_states_collapsed": int(len(uniq_fixed)),
                    "total_hardware_errors_fixed": total_flips,
                    "feasible_fraction": feas_shots / decoded_shots if decoded_shots else 0.0,
                    "feasible_found": best_feasible is not None
                },
                
                "noise_model": getattr(circ_emu, "noise", {}),
                "energy_scale": land.scale,
                "optimizer": self.method,
                "backend": f"pennylane_noisy_{self.hardware_preset}",
            }
        except Exception as e:
            status = f"Failed: {e}"
            print(f"\n[EmuQAOA Error]: {e}\n")

        hardware_metrics = {
            "qpu_target": f"noisy_{self.hardware_preset}_heavy_hex",
            "logical_qubits": n_qubits,
            "physical_qubits": resources.get("qubits", n_qubits),
            "circuit_depth": resources.get("depth_estimate", 0),
            "logical_two_qubit_gates": resources.get("logical_two_qubit_gates", 0),
            "physical_two_qubit_gates": resources.get("physical_two_qubit_gates", 0),
            "routing_overhead_multiplier": resources.get("routing_overhead_multiplier", 1.0),
            "noise_profile": getattr(circ_emu, "noise", {}) if 'circ_emu' in locals() else {},
            "resources": resources,
        }

        return self._build_opf_payload(
            cost, dispatch, sgen_dispatch, slack_dispatch, feasibility, complexity, status,
            f"emu_qaoa_{self.hardware_preset}_{self.formulation}", form_t, opt_t, samp_t, start,
            qaoa_meta, resources, n_qubits, hardware_metrics
        )


# Backward compatibility aliases
IdealQAOASolver = IdealQAOA
EmuQAOASolver = EmuQAOA

if __name__ == "__main__":
    import json
    
    try:
        from homemade_grids.small_grids import case3_low_gen
    except ImportError as e:
        print(f"[-] Import Error: {e}")
        exit(1)

    print("\n" + "="*80)
    print(" QAOA SOLVERS: END-TO-END PIPELINE TEST ".center(80, "="))
    print("="*80 + "\n")

    # 1. Setup Grid
    print("[1] Initializing 3-Bus Grid...")
    net = case3_low_gen()
    if hasattr(net, 'ext_grid') and not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0

    # 2. Test IdealQAOA
    print("\n[2] Executing IdealQAOA (Statevector, p=3)...")
    ideal_solver = IdealQAOA(
        formulation="dc_ptdf", mw_precision=50.0, p=3, 
        method="Powell", maxiter=1000, n_restarts=15
    )
    ideal_sol, ideal_meta = ideal_solver.solve_opf(net)
    
    print(f"    -> Status: {ideal_meta['status']}")
    if ideal_sol['cost_eur_per_hr'] is not None:
        print(f"    -> Best Cost: €{ideal_sol['cost_eur_per_hr']:.2f}")
        qaoa_metrics = ideal_meta['algorithmic_metrics']['qaoa']
        print(f"    -> P(Ground Exact): {qaoa_metrics['p_ground_exact']*100:.2f}%")

    # 3. Test EmuQAOA
    print("\n[3] Executing EmuQAOA (IBM Eagle + Steepest Descent Mitigation, p=3)...")
    emu_solver = EmuQAOA(
        formulation="dc_ptdf", mw_precision=50.0, p=3, hardware_preset="ibm_eagle", 
        method="Powell", maxiter=1000, n_restarts=15
    )
    emu_sol, emu_meta = emu_solver.solve_opf(net)

    print(f"    -> Status: {emu_meta['status']}")
    if emu_sol['cost_eur_per_hr'] is not None:
        print(f"    -> Best Cost: €{emu_sol['cost_eur_per_hr']:.2f}")
        
        # 4. Extract and Display Mitigation Metrics
        meta = emu_meta['algorithmic_metrics']['qaoa']
        print("\n[4] Mitigation Metrics Breakdown:")
        
        raw = meta.get("raw_readout", {})
        mitigated = meta.get("mitigated_readout", {})
        
        if raw and mitigated:
            print(f"    -> Raw Readout Best Energy  : €{raw.get('lowest_energy_sampled', 0):.2f}")
            print(f"    -> Raw P(Ground)            : {raw.get('p_ground_sampled', 0)*100:.2f}%\n")
            
            print(f"    -> Hardware Errors Fixed    : {mitigated.get('total_hardware_errors_fixed', 0)} bit-flips")
            
            print(f"\n    -> Mitigated Best Energy    : €{mitigated.get('lowest_energy_sampled', 0):.2f}")
            print(f"    -> Mitigated P(Ground)      : {mitigated.get('p_ground_sampled', 0)*100:.2f}%")
            print(f"    -> Final Feasible Fraction  : {mitigated.get('feasible_fraction', 0)*100:.2f}%")
        else:
            print("    -> [!] Mitigation metrics missing from metadata.")
        
    print("\n" + "="*80)