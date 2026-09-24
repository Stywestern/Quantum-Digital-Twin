import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import copy
import numpy as np
import neal
from solvers.qubo_formulator import QuboFormulator

class SimulatedAnnealingSolver(QuboFormulator):
    def __init__(self, formulation="dc", num_reads=500, num_sweeps=1000, max_time=1800, 
                 mw_precision=1.0, seed=None, **kwargs):
        
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
        
        # Safely calculate max line loading percentage from the decoded flows
        max_load_pct = None
        if hasattr(self, '_lines') and self._lines:
            loadings = []
            for ln in self._lines:
                if ln['p_max'] > 0.0:
                    flow = abs(feasibility["line_flows_mw"].get(ln['idx'], 0.0))
                    loadings.append((flow / ln['p_max']) * 100.0)
            if loadings:
                max_load_pct = round(max(loadings), 3)

        # 4. Final Solution and Metadata Assembly
        solution = {
            "cost_eur_per_hr": cost,
            "generator_dispatch_mw": dispatch,
            "static_generator_dispatch_mw": sgen_dispatch,
            "slack_dispatch_mw": slack_dispatch,
            "grid_state": {
                "max_line_loading_percent": max_load_pct,
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
                "angle_precision": getattr(self, 'angle_precision', None),
                "penalty_balance": getattr(self, 'penalty_balance', None),
                "penalty_line": getattr(self, 'penalty_line', None),
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

    def solve_pf(self, net):
        """Executes a QUBO Power Flow by locking economic variables."""
        net_pf = copy.deepcopy(net)
        
        # 1. Lock dispatchable assets to eliminate decision variables
        for idx in net_pf.gen.index:
            p = float(net_pf.gen.at[idx, 'p_mw']) if not np.isnan(net_pf.gen.at[idx, 'p_mw']) else 0.0
            net_pf.gen.at[idx, 'min_p_mw'] = p
            net_pf.gen.at[idx, 'max_p_mw'] = p
            
        for idx in net_pf.ext_grid.index:
            if 'p_mw' in net_pf.ext_grid.columns and not np.isnan(net_pf.ext_grid.at[idx, 'p_mw']):
                p = float(net_pf.ext_grid.at[idx, 'p_mw'])
            else:
                p = 0.0
            net_pf.ext_grid.at[idx, 'min_p_mw'] = p
            net_pf.ext_grid.at[idx, 'max_p_mw'] = p

        # 2. Strip cost polynomials
        net_pf.poly_cost = net_pf.poly_cost.iloc[0:0]
        if hasattr(net_pf, 'pwl_cost'):
            net_pf.pwl_cost = net_pf.pwl_cost.iloc[0:0]

        # 3. Solve using the core QUBO workflow
        solution, metadata = self.solve_opf(net_pf)
        
        # 4. Tweak outputs for PF context
        solution["cost_eur_per_hr"] = None
        metadata["solver_name"] = f"neal_simulated_annealing_pf_{self.formulation}"
        metadata["algorithmic_metrics"]["framework"] = "pure_qubo_discretization_pf"
        
        return solution, metadata