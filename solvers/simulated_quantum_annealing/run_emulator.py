import time
import numpy as np
import dimod

from solvers.qubo_formulator import QuboFormulator

from solvers.simulated_quantum_annealing.s1_hardware import Hardware
from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings, select_best
from solvers.simulated_quantum_annealing.s4_emulators import EmulatedQPU

class EmulatedQPUSolver(QuboFormulator):
    """
    End-to-end digital twin of a D-Wave QPU.
    Executes Logical Formulation -> Minor-Embedding -> Hardware Scaling -> DAC Quantization -> 
    ICE Noise Injection -> SVMC Physics Simulation -> Chain Resolution -> Grid Decoding.
    """
    def __init__(self, formulation="dc_ptdf", num_reads=500, num_sweeps=5000, 
                 mw_precision=10.0, hardware_profile=None, sampler="svmc",
                 dac_bits=5, ice_sigma_h=0.01, ice_sigma_j=0.01, use_extended_j=True,
                 reads_per_programming=100, chain_break_method="majority",
                 embedding_methods=("clique", "minorminer"), embedding_seeds=(0, 1, 2),
                 trotter_slices=32, seed=42, **kwargs):
        
        super().__init__(formulation=formulation, mw_precision=mw_precision, **kwargs)
        
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.seed = seed
        self.trotter_slices = trotter_slices  # Added
        
        # Hardware & Analog parameters
        self.hardware = hardware_profile or Hardware.ideal_zephyr(m=4, t=4)
        self.sampler_type = sampler
        self.dac_bits = dac_bits
        self.ice_sigma_h = ice_sigma_h
        self.ice_sigma_j = ice_sigma_j
        self.use_extended_j = use_extended_j
        
        # Execution parameters
        self.reads_per_programming = reads_per_programming
        self.chain_break_method = chain_break_method
        self.embedding_methods = embedding_methods
        self.embedding_seeds = embedding_seeds

    def solve_opf(self, net):
        start_time = time.time()
        status, cost, dispatch, sgen_dispatch, slack_dispatch = "Success", None, {}, {}, {}
        form_time, embed_time, samp_time = 0.0, 0.0, 0.0
        complexity, sampler_stats, hw_metrics = {}, {}, {}
        
        try:
            # 1. Formulation Phase
            t0 = time.time()
            bqm, complexity = self._formulate_qubo(net)
            form_time = time.time() - t0
            logical_vars = len(bqm.variables)

            # --- Calculate Hardware Feasibility (At-Risk Fraction) ---
            at_risk_fraction = 0.0
            if logical_vars > 0:
                max_coeff = max(
                    abs(max(bqm.linear.values(), key=abs, default=0)),
                    abs(max(bqm.quadratic.values(), key=abs, default=0))
                )
                dac_step = max_coeff / (2 ** self.dac_bits)
                at_risk_count = sum(1 for v in list(bqm.linear.values()) + list(bqm.quadratic.values()) 
                                    if 1e-10 < abs(v) < dac_step)
                total_non_zero = sum(1 for v in list(bqm.linear.values()) + list(bqm.quadratic.values()) 
                                     if abs(v) > 1e-10)
                at_risk_fraction = (at_risk_count / total_non_zero) if total_non_zero > 0 else 0.0

            feas_report = {"at_risk_fraction": round(at_risk_fraction, 4)}

            if logical_vars == 0:
                best_sample = {}
                cost = float(bqm.offset)
                feasibility = {"is_feasible": True, "line_flows_mw": {}}
                status = "Success (0 variables, trivial)"
            else:
                # 2. Embedding Phase
                t1 = time.time()
                emb_results = find_embeddings(
                    bqm, self.hardware, methods=self.embedding_methods, 
                    seeds=self.embedding_seeds, verbose=False
                )
                if not emb_results:
                    raise RuntimeError("All embedding heuristics failed.")
                best_label, embedding = select_best(emb_results)
                emb_stats = emb_results[best_label]["stats"]
                embed_time = time.time() - t1

                # 3. Hardware Mapping & Sampling Phase
                t2 = time.time()
                
                sampler_kwargs = {"num_sweeps": self.num_sweeps}
                if self.sampler_type == "sqa":
                    sampler_kwargs["trotter_slices"] = self.trotter_slices

                qpu = EmulatedQPU(
                    hardware=self.hardware, sampler=self.sampler_type,
                    sampler_kwargs=sampler_kwargs, dac_bits=self.dac_bits,
                    ice_sigma_h=self.ice_sigma_h, ice_sigma_j=self.ice_sigma_j,
                    use_extended_j=self.use_extended_j, seed=self.seed
                )
                
                # --- RUN 1: RAW ANALOG PHYSICS (No Post-Processing) ---
                # Fix seed to ensure identical noise and path integrals
                qpu.rng = np.random.default_rng(self.seed) 
                result_raw = qpu.sample(
                    logical_bqm=bqm, embedding=embedding, num_reads=self.num_reads,
                    reads_per_programming=self.reads_per_programming,
                    chain_break_method=self.chain_break_method, postprocess=None
                )
                
                # --- RUN 2: ANALOG PHYSICS + STEEPEST DESCENT ---
                qpu.rng = np.random.default_rng(self.seed) # Reset seed to match Run 1
                result_sds = qpu.sample(
                    logical_bqm=bqm, embedding=embedding, num_reads=self.num_reads,
                    reads_per_programming=self.reads_per_programming,
                    chain_break_method=self.chain_break_method, postprocess="both"
                )
                samp_time = time.time() - t2
                
                # Extract RAW physics metrics
                energies_raw = result_raw.sampleset.record.energy
                cbf_array_raw = getattr(result_raw.sampleset.record, 'chain_break_fraction', np.zeros(len(energies_raw)))
                
                # Extract SDS physics metrics (This is what we decode and keep)
                energies_sds = result_sds.sampleset.record.energy
                best_sample = result_sds.sampleset.first.sample
                cbf_array_sds = getattr(result_sds.sampleset.record, 'chain_break_fraction', np.zeros(len(energies_sds)))
                
                phys_flips, log_flips = result_sds.info.get("postprocess_flips", (0, 0))
                
                sampler_stats = {
                    "num_reads_requested": self.num_reads,
                    "num_sweeps_per_read": self.num_sweeps,
                    "unique_states_found_sds": len(result_sds.sampleset),
                    
                    # Pre-SDS (Raw Analog) Stats
                    "raw_energy_best_logical": round(float(np.min(energies_raw)), 2),
                    "raw_energy_mean_logical": round(float(np.mean(energies_raw)), 2),
                    "raw_chain_break_fraction_mean": round(float(np.mean(cbf_array_raw)), 4),
                    
                    # Post-SDS Stats
                    "sds_energy_best_logical": round(float(np.min(energies_sds)), 2),
                    "sds_energy_mean_logical": round(float(np.mean(energies_sds)), 2),
                    "sds_chain_break_fraction_mean": round(float(np.mean(cbf_array_sds)), 4),
                    
                    "postprocess_physical_flips": int(phys_flips),
                    "postprocess_logical_flips": int(log_flips)
                }
                
                hw_metrics = {
                    "qpu_target": self.hardware.name,
                    "topology": self.hardware.topology,
                    "logical_qubits": logical_vars,
                    "physical_qubits": emb_stats['physical'],
                    "qubit_overhead_factor": round(emb_stats['physical'] / logical_vars, 2) if logical_vars else 0,
                    "max_chain_length": emb_stats['max_chain'],
                    "embedding_heuristic_winner": best_label,
                    "hardware_scale_factor": result_sds.scale_factor,
                    "chain_strength_applied": result_sds.chain_strength,
                    "dac_resolution_bits": self.dac_bits,
                    "ice_sigma_h": self.ice_sigma_h,
                    "ice_sigma_j": self.ice_sigma_j
                }

                # 4. Decoding Phase (Decoding the SDS optimized state)
                dispatch, sgen_dispatch, slack_dispatch, cost, feasibility = self._decode_solution(best_sample, net)
            
        except Exception as e:
            status = f"Failed: {str(e)}"
            feasibility = {"is_feasible": False, "line_flows_mw": {}}
            print(f"\nCRITICAL ERROR IN EMULATED QPU PIPELINE: {e}\n")

        exec_time = round(time.time() - start_time, 4)
        
        # Safely calculate max line loading percentage
        max_load_pct = None
        if hasattr(self, '_lines') and self._lines:
            loadings = [
                (abs(feasibility.get("line_flows_mw", {}).get(ln['idx'], 0.0)) / ln['p_max']) * 100.0
                for ln in self._lines if ln['p_max'] > 0.0
            ]
            if loadings:
                max_load_pct = round(max(loadings), 3)

        solution = {
            "cost_eur_per_hr": cost,
            "generator_dispatch_mw": dispatch,
            "static_generator_dispatch_mw": sgen_dispatch,
            "slack_dispatch_mw": slack_dispatch,
            "grid_state": {
                "max_line_loading_percent": max_load_pct,
                "max_voltage_pu": 1.0 if self.formulation == "dc_ptdf" else None,
                "min_voltage_pu": 1.0 if self.formulation == "dc_ptdf" else None,
                "feasibility": feasibility
            }
        }
        
        metadata = {
            "status": status,
            "solver_name": f"emulated_qpu_{self.sampler_type}_{self.formulation}",
            "execution_time_seconds": {
                "total": exec_time,
                "formulation_cpu": round(form_time, 4),
                "embedding_cpu": round(embed_time, 4),
                "sampling_analog_sim": round(samp_time, 4)
            },
            "problem_complexity": complexity,
            "hardware_feasibility_report": feas_report,  # Now correctly populated
            "qubo_parameters": {
                "mw_precision": self.mw_precision,
                "penalty_balance": getattr(self, 'penalty_balance', None),
                "penalty_line": getattr(self, 'penalty_line', None),
            },
            "hardware_metrics": hw_metrics,
            "algorithmic_metrics": {
                "framework": "hardware_aware_analog_simulation",
                "sampler_stats": sampler_stats
            }
        }
        
        return solution, metadata