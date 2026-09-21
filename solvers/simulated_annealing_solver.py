import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import numpy as np
import neal
from solvers.base_qubo_formulator import BaseQuboFormulator

class SimulatedAnnealingSolver(BaseQuboFormulator):
    def __init__(self, formulation="dc", num_reads=500, num_sweeps=1000, max_time=1800, 
                 mw_precision=1.0, enforce_line_limits=True, seed=None, **kwargs):
        
        super().__init__(
            formulation=formulation, max_time=max_time, mw_precision=mw_precision,
            enforce_line_limits=enforce_line_limits, **kwargs
        )
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.seed = seed
        self.sampler = neal.SimulatedAnnealingSampler()

    def solve(self, net):
        start_time = time.time()
        status, cost, dispatch, sgen_dispatch, slack_dispatch = "Success", None, {}, {}, {}
        form_time, samp_time = 0.0, 0.0
        bqm, complexity, sampler_stats = None, {}, {}
        
        try:
            # 1. Formulation Phase (CPU) via BaseQuboFormulator
            t0 = time.time()
            bqm, complexity = self._formulate_qubo(net)
            form_time = time.time() - t0
            
            # 2. Sampling Phase (Classical SA mimicking QPU API)
            t1 = time.time()
            response = self.sampler.sample(
                bqm, 
                num_reads=self.num_reads, 
                num_sweeps=self.num_sweeps,
                seed=self.seed
            )
            samp_time = time.time() - t1
            
            energies = response.record.energy
            sampler_stats = {
                "num_reads_requested": self.num_reads,
                "num_sweeps_per_read": self.num_sweeps,
                "unique_states_found": len(response.record),
                "energy_best": round(float(np.min(energies)), 2),
                "energy_mean": round(float(np.mean(energies)), 2),
                "energy_worst": round(float(np.max(energies)), 2),
                "energy_std_dev": round(float(np.std(energies)), 2)
            }
    
            # 3. Decoding Phase (CPU) via BaseQuboFormulator
            best_sample = response.first.sample
            dispatch, sgen_dispatch, slack_dispatch, cost, feasibility = self._decode_solution(best_sample, net)
            
        except Exception as e:
            status = f"Failed: {str(e)}"
            print(f"\nCRITICAL FORMULATION ERROR: {e}\n")
            raise

        exec_time = round(time.time() - start_time, 4)
        
        # 4. Final Solution and Metadata Assembly
        solution = {
            "cost_eur_per_hr": cost,
            "generator_dispatch_mw": dispatch,
            "static_generator_dispatch_mw": sgen_dispatch,
            "slack_dispatch_mw": slack_dispatch,
            "grid_state": {
                "max_line_loading_percent": None,
                "max_voltage_pu": 1.0 if self.formulation == "dc" else None,
                "min_voltage_pu": 1.0 if self.formulation == "dc" else None,
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
                "angle_precision": self.angle_precision,
                "penalty_balance": self.penalty_balance,
                "penalty_line": self.penalty_line,
                "enforce_line_limits": self.enforce_line_limits,
                "seed": self.seed
            },
            "algorithmic_metrics": {
                "framework": "pure_qubo_discretization",
                "num_iterations": 1,
                "optimality_gap_percent": 0.0,
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

# =============================================================================
# Execution Block to inspect Case5
# =============================================================================
if __name__ == "__main__":
    import pandapower.networks as nw
    print("Loading case5...")
    net = nw.case5()
    
    print("Instantiating QUBO SA Solver (DC)...")
    solver = SimulatedAnnealingSolver(
        formulation="dc", 
        mw_precision=1.0, 
        enforce_line_limits=True  
    )
    
    print("Extracting QUBO matrix parameters without solving...")
    bqm, complexity = solver.view_problem_definition(net)