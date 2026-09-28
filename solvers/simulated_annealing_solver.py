import time
import copy
import numpy as np
import neal
from solvers.qubo_formulator import QuboFormulator

import time
import copy
import numpy as np
import neal
from solvers.qubo_formulator import QuboFormulator
from solvers.simulated_quantum_annealing.s5_postprocess import descend_sampleset # Add SDS

class SimulatedAnnealingSolver(QuboFormulator):
    def __init__(self, formulation="dc_ptdf", num_reads=500, num_sweeps=3000, max_time=1800, 
                 mw_precision=10.0, seed=None, **kwargs):
        
        super().__init__(
            formulation=formulation, max_time=max_time, mw_precision=mw_precision, **kwargs)
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.seed = seed
        self.sampler = neal.SimulatedAnnealingSampler()

    def solve_opf(self, net):
        start_time = time.time()
        status, cost, dispatch, sgen_dispatch, slack_dispatch = "Success", None, {}, {}, {}
        form_time, samp_time = 0.0, 0.0
        bqm, complexity, sampler_stats = None, {}, {}
        
        try:
            # 1. Formulation Phase
            t0 = time.time()
            bqm, complexity = self._formulate_qubo(net)
            form_time = time.time() - t0
            
            # 2. Sampling Phase
            t1 = time.time()
            response = self.sampler.sample(
                bqm, 
                num_reads=self.num_reads, 
                num_sweeps=self.num_sweeps,
                seed=self.seed
            )
            
            # Post-Process: Apply Greedy Steepest Descent
            # This ensures SA isn't trapped in tiny thermal divots near the optimum
            optimized_response, total_flips = descend_sampleset(response, bqm.change_vartype('SPIN', inplace=False))
            optimized_response = optimized_response.change_vartype('BINARY', inplace=False)
            samp_time = time.time() - t1
            
            energies = optimized_response.record.energy
            sampler_stats = {
                "num_reads_requested": self.num_reads,
                "num_sweeps_per_read": self.num_sweeps,
                "unique_states_found": len(optimized_response.record),
                "energy_best": round(float(np.min(energies)), 2),
                "energy_mean": round(float(np.mean(energies)), 2),
                "energy_worst": round(float(np.max(energies)), 2),
                "energy_std_dev": round(float(np.std(energies)), 2),
                "postprocess_logical_flips": total_flips
            }
    
            # 3. Decoding Phase
            best_sample = optimized_response.first.sample
            dispatch, sgen_dispatch, slack_dispatch, cost, feasibility = self._decode_solution(best_sample, net)
            
        except Exception as e:
            status = f"Failed: {str(e)}"
            print(f"\nCRITICAL FORMULATION ERROR: {e}\n")
            raise

        exec_time = round(time.time() - start_time, 4)
        
        max_load_pct = None
        if hasattr(self, '_lines') and self._lines:
            loadings = []
            for ln in self._lines:
                if ln['p_max'] > 0.0:
                    flow = abs(feasibility["line_flows_mw"].get(ln['idx'], 0.0))
                    loadings.append((flow / ln['p_max']) * 100.0)
            if loadings:
                max_load_pct = round(max(loadings), 3)

        # Ensure optimality gap explicitly flags when the solver cheated
        optimality_gap = 0.0
        if not feasibility.get("is_feasible", False):
            status = "Success (Infeasible)"
            # Negative gap implies it found a "cheating" state cheaper than the true constrained minimum
            # But technically gap is undefined for infeasible states
            optimality_gap = None 

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
            "solver_name": f"neal_simulated_annealing_{self.formulation}",
            "execution_time_seconds": {
                "total": exec_time,
                "formulation_cpu": round(form_time, 4),
                "sampling_cpu_sa": round(samp_time, 4)
            },
            "problem_complexity": complexity,
            "qubo_parameters": {
                "mw_precision": self.mw_precision,
                "angle_precision": getattr(self, 'angle_precision', None),
                "penalty_balance": getattr(self, 'penalty_balance', None),
                "penalty_line": getattr(self, 'penalty_line', None),
                "seed": self.seed
            },
            "algorithmic_metrics": {
                "framework": "pure_qubo_discretization",
                "num_iterations": 1,
                "optimality_gap_percent": optimality_gap,
                "sampler_stats": sampler_stats
            },
            "hardware_metrics": {
                "qpu_target": "classical_cpu_simulated",
                "logical_qubits": len(bqm.variables) if bqm else 0,
                "physical_qubits": len(bqm.variables) if bqm else 0,
                "max_chain_length": 1,
                "circuit_depth": 0
            }
        }
        
        return solution, metadata