import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import numpy as np
from solvers.base_qubo_formulator import BaseQuboFormulator

try:
    import openjij as oj
except ImportError:
    raise ImportError("OpenJij is required for SQA. Run: pip install openjij")

class SimulatedQuantumAnnealingSolver(BaseQuboFormulator):
    def __init__(self, formulation="dc", num_reads=500, num_sweeps=1000, max_time=1800, 
                     mw_precision=1.0, trotter_slices=8, enforce_line_limits=True, seed=None, **kwargs):
            
        super().__init__(
                formulation=formulation, max_time=max_time, mw_precision=mw_precision,
                enforce_line_limits=enforce_line_limits, **kwargs
            )
        
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.trotter_slices = trotter_slices
        self.seed = seed  # Enable fixed-seed reproducible runs
        
        # Instantiate OpenJij's SQA sampler
        self.sampler = oj.SQASampler()

    def solve(self, net):
        start_time = time.time()
        status, cost, dispatch, sgen_dispatch, slack_dispatch, feasibility = "Success", None, {}, {}, {}, {}
        form_time, samp_time = 0.0, 0.0
        bqm, complexity, sampler_stats = None, {}, {}
        
        try:
            # 1. Formulation Phase (CPU) via BaseQuboFormulator
            t0 = time.time()
            bqm, complexity = self._formulate_qubo(net)
            form_time = time.time() - t0
            
            # Shrink the mountains so quantum tunneling (gamma=1.0) can see them
            scale_factor = 10000.0
            bqm.scale(1.0 / scale_factor)
            
            # 2. Sampling Phase (SQA via OpenJij)
            t1 = time.time()
            
            if self.seed is not None:
                np.random.seed(self.seed)
                
            energies = []
            unique_states = set()
            best_sample = None
            best_energy = float('inf')
            
            # Manual read loop to force true stochastic diversity
            for _ in range(self.num_reads):
                current_seed = int(np.random.randint(0, 2**31 - 1)) if self.seed is not None else None
                
                response = self.sampler.sample(
                    bqm, 
                    num_reads=1, 
                    num_sweeps=self.num_sweeps, 
                    trotter=self.trotter_slices,
                    seed=current_seed,
                    # [CRITICAL FIX] Scale beta up so the solver actually freezes at the end
                    # Without this, the solver remains too hot and ignores the scaled penalties
                    beta=scale_factor  
                )
                
                e = response.first.energy
                energies.append(e)
                unique_states.add(frozenset(response.first.sample.items()))
                
                if e < best_energy:
                    best_energy = e
                    best_sample = response.first.sample
                    
            samp_time = time.time() - t1
            
            sampler_stats = {
                "num_reads_requested": self.num_reads,
                "num_sweeps_per_read": self.num_sweeps,
                "trotter_slices": self.trotter_slices,
                "unique_states_found": len(unique_states),
                "energy_best": round(float(np.min(energies)), 2),
                "energy_mean": round(float(np.mean(energies)), 2),
                "energy_worst": round(float(np.max(energies)), 2),
                "energy_std_dev": round(float(np.std(energies)), 2)
            }
    
            # 3. Decoding Phase (CPU) via BaseQuboFormulator
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
                "feasibility": feasibility  # Inject feasibility into the payload
            }
        }
        
        metadata = {
            "status": status,
            "solver_name": f"openjij_sqa_{self.formulation}",
            "execution_time_seconds": {
                "total": exec_time,
                "formulation_cpu": round(form_time, 4),
                "sampling_cpu_sqa": round(samp_time, 4)
            },
            "problem_complexity": complexity,
            "qubo_parameters": {
                "mw_precision": self.mw_precision,
                "angle_precision": self.angle_precision,
                "penalty_balance": self.penalty_balance,
                "penalty_line": self.penalty_line,
                "enforce_line_limits": self.enforce_line_limits,
                "seed": self.seed  # Log the seed for post-analysis
            },
            "algorithmic_metrics": {
                "framework": "pure_qubo_discretization",
                "num_iterations": 1,
                "optimality_gap_percent": 0.0,
                "sampler_stats": sampler_stats
            },
            "hardware_metrics": {
                "qpu_target": "classical_cpu_sqa_simulated",
                "logical_qubits": len(bqm.variables) if bqm else 0,
                "physical_qubits": len(bqm.variables) if bqm else 0,
                "max_chain_length": 1,
                "circuit_depth": 0
            }
        }
        
        return solution, metadata