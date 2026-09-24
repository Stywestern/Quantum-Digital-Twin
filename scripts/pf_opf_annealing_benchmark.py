###############################################################################################################
#                                            SETUP
###############################################################################################################
# Library imports
import pandapower as pp
import copy
import os
import pandas as pd

# Module imports
from homemade_grids.small_grids import case3_low_gen, case3_high_gen, case4_low_gen, case4_high_gen, extract_network_info
from solvers.qubo_formulator import QuboFormulator
from solvers.classical_ip_solver import ClassicalIPSolver
from solvers.simulated_annealing_solver import SimulatedAnnealingSolver
from solvers.simulated_quantum_annealing_solver import SimulatedQuantumAnnealingSolver

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Safely converts numpy types to native Python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (bool, np.bool_)): return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def save_payload(data: dict, filename: str):
    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    file_path = f"{save_dir}/{filename}"

    # FIX: Use file_path here instead of filename
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)
    print(f"[I/O] Payload saved to {file_path}")

def print_execution_summary(payloads: dict, config: dict = None):
    """Parses saved payloads and prints pivot tables of key OPF/PF metrics and hardware complexity."""
    print("\n" + "="*110)
    print(" PIPELINE EXECUTION SUMMARY ".center(110, "="))
    print("="*110)
    
    rows_perf = []
    rows_res = []
    
    for solver_name, payload in payloads.items():
        opf_sol = payload.get("opf_results", {}).get("solution", {})
        opf_meta = payload.get("opf_results", {}).get("metadata", {})
        pf_sol = payload.get("pf_validation_results", {}).get("solution", {})
        pf_meta = payload.get("pf_validation_results", {}).get("metadata", {})
        
        # --- 1. Performance & Economics Table ---
        opf_cost = opf_sol.get("cost_eur_per_hr") if opf_sol else None
        
        slack_mw = None
        if opf_sol and opf_sol.get("slack_dispatch_mw"):
            slack_mw = sum(d.get("p_mw", 0) for d in opf_sol["slack_dispatch_mw"].values())
            
        pf_loading = None
        if pf_sol and pf_sol.get("grid_state"):
            pf_loading = pf_sol["grid_state"].get("max_line_loading_percent")
            
        exec_time = round(
            opf_meta.get("execution_time_seconds", {}).get("total", 0.0) + 
            pf_meta.get("execution_time_seconds", {}).get("total", 0.0), 4
        )
        
        rows_perf.append({
            "Solver": solver_name,
            "OPF Status": opf_meta.get("status", "Failed"),
            "Cost (EUR/hr)": round(opf_cost, 2) if opf_cost is not None else "N/A",
            "Slack (MW)": round(slack_mw, 2) if slack_mw is not None else "N/A",
            "PF Status": pf_meta.get("status", "Failed"),
            "PF Max Load (%)": round(pf_loading, 2) if pf_loading is not None else "N/A",
            "Time (s)": exec_time
        })

        # --- 2. Hardware & Complexity Table ---
        if "Classical IP" not in solver_name:
            comp = opf_meta.get("problem_complexity", {})
            hw = opf_meta.get("hardware_metrics", {})
            alg = opf_meta.get("algorithmic_metrics", {})
            feas = opf_meta.get("hardware_feasibility_report", {})
            
            form = comp.get("formulation", "N/A")
            enc = comp.get("encoding", "N/A")
            prec = config.get("mw_precision", "N/A") if config else "N/A"
            
            q_domain = comp.get("quantum_domain_qubo", {})
            l_qub = q_domain.get("total_logical_qubits", "N/A")
            dens = q_domain.get("density_percent", "N/A")
            dr = comp.get("hardware_limits", {}).get("dynamic_range", "N/A")
            
            p_qub = hw.get("physical_qubits", "N/A")
            max_chain = hw.get("max_chain_length", "N/A")
            
            # Formatting fractional metrics
            cb_frac = alg.get("sampler_stats", {}).get("mean_chain_break_fraction")
            cb = f"{round(cb_frac * 100, 3)}%" if cb_frac is not None else "N/A"
            
            risk_frac = feas.get("at_risk_fraction")
            risk = f"{round(risk_frac * 100, 2)}%" if risk_frac is not None else "N/A"
            
            rows_res.append({
                "Solver": solver_name,
                "Formulation": form,
                "Encoding": enc,
                "Prec(MW)": prec,
                "L-Qubits": l_qub,
                "P-Qubits": p_qub,
                "Max Chain": max_chain,
                "Density": f"{dens}%" if dens != "N/A" else dens,
                "Dyn Range": dr,
                "At-Risk": risk,
                "Chain Break": cb
            })

    # Print First Table
    df_perf = pd.DataFrame(rows_perf)
    print(df_perf.to_string(index=False))
    
    # Print Second Table
    print("\n" + "-"*110)
    print(" QUBO COMPLEXITY & HARDWARE METRICS ".center(110, "-"))
    print("-" * 110)
    if rows_res:
        df_res = pd.DataFrame(rows_res)
        print(df_res.to_string(index=False))
    else:
        print("No QUBO solvers executed to report hardware metrics.")
    print("="*110 + "\n")

###############################################################################################################
#                                        EXECUTION BLOCK
###############################################################################################################

if __name__ == "__main__":
    import time
    
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 1: Initialize the problem and record the original problem parameters
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    print("\n" + "-"*80)
    print(" STEP 1: GRID INITIALIZATION ")
    print("-"*80)
    t_start = time.time()
    
    pristine_net = case3_low_gen()
    original_problem_parameters = extract_network_info(pristine_net)
    grid_name = original_problem_parameters.get("network_name", "unknown_grid").replace(" ", "_").lower()
    
    print(f"Grid loaded: {original_problem_parameters.get('network_name')}")
    print(f"Step 1 Time: {round(time.time() - t_start, 4)} seconds")

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 2: Continuous Ground Truth (OPF -> PF Validation)
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    print("\n" + "-"*80)
    print(" STEP 2: CLASSICAL IP SOLVER (Continuous Ground Truth) ")
    print("-"*80)
    t_start = time.time()
    
    ip_solver = ClassicalIPSolver(formulation="dc")
    net_ip_opf = copy.deepcopy(pristine_net)
    ip_opf_solution, ip_opf_metadata = ip_solver.solve_opf(net_ip_opf)
    
    ip_opf_cost = None
    if ip_opf_metadata["status"] == "Success":
        ip_opf_cost = ip_opf_solution['cost_eur_per_hr']
        print(f"[IP OPF] Optimal Cost: {ip_opf_cost} EUR/hr")
        
        net_ip_pf = copy.deepcopy(pristine_net)
        for idx_str, dispatch_data in ip_opf_solution['generator_dispatch_mw'].items():
            net_ip_pf.gen.loc[int(idx_str), 'p_mw'] = float(dispatch_data['p_mw'])
            
        ip_pf_solution, ip_pf_metadata = ip_solver.solve_pf(net_ip_pf)
        print(f"[IP PF] Validation Feasible: {ip_pf_metadata['status']}")
    else:
        print("[OPF] Classical solver failed to converge. The grid may be physically infeasible.")
        ip_pf_solution, ip_pf_metadata = {}, {}

    ground_truth_payload = {
        "problem_parameters": original_problem_parameters,
        "opf_results": {"solution": ip_opf_solution, "metadata": ip_opf_metadata},
        "pf_validation_results": {"solution": ip_pf_solution, "metadata": ip_pf_metadata}
    }
    
    filename_ip = f"{grid_name}_classical_ip_{ip_solver.formulation}.json"
    save_payload(ground_truth_payload, filename_ip)
    print(f"Step 2 Time: {round(time.time() - t_start, 4)} seconds")

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 3: Setup the qubo formulation configuration
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    print("\n" + "-"*80)
    print(" STEP 3: QUBO CONFIGURATION ")
    print("-"*80)
    
    formulator_config = {
        "formulation": "dc_ptdf",
        "encoding": "radix",
        "mw_precision": 10.0, 
        "penalty_balance": None, 
        "penalty_line": None
    }
    
    form_type = formulator_config["formulation"]
    prec = formulator_config["mw_precision"]
    encoding_type = formulator_config["encoding"]
    print(f"Formulation: {form_type} | Encoding: {encoding_type} | Precision: {prec} MW")

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 4: Discrete Ground truth (Simulated Annealing)
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    num_reads = 500
    num_sweeps = 2000

    print("\n" + "-"*80)
    print(" STEP 4: SIMULATED ANNEALING SOLVER (Discrete Baseline) ")
    print("-"*80)
    t_start = time.time()
    
    sa_solver = SimulatedAnnealingSolver(
        **formulator_config,
        num_reads=num_reads,
        num_sweeps=num_sweeps
    )
    
    net_discrete_opf = copy.deepcopy(pristine_net)
    discrete_opf_solution, discrete_opf_metadata = sa_solver.solve_opf(net_discrete_opf)
    
    if discrete_opf_metadata["status"] == "Success":
        print(f"[SA OPF] Optimal Cost: {discrete_opf_solution['cost_eur_per_hr']} EUR/hr")
        net_discrete_pf = copy.deepcopy(pristine_net)
        
        for idx_str, dispatch_data in discrete_opf_solution['generator_dispatch_mw'].items():
            net_discrete_pf.gen.loc[int(idx_str), 'p_mw'] = float(dispatch_data['p_mw'])
            
        for idx_str, dispatch_data in discrete_opf_solution['slack_dispatch_mw'].items():
            net_discrete_pf.ext_grid.loc[int(idx_str), 'p_mw'] = float(dispatch_data['p_mw'])
            
        discrete_pf_solution, discrete_pf_metadata = sa_solver.solve_pf(net_discrete_pf)
        print(f"[SA PF] Validation Feasible: {discrete_pf_metadata['status']}")
    else:
        print("[SA OPF] Solver failed.")
        discrete_pf_solution, discrete_pf_metadata = {}, {}

    discrete_payload = {
        "problem_parameters": original_problem_parameters,
        "opf_results": {"solution": discrete_opf_solution, "metadata": discrete_opf_metadata},
        "pf_validation_results": {"solution": discrete_pf_solution, "metadata": discrete_pf_metadata}
    }
    
    filename_discrete = f"{grid_name}_sa_{form_type}_prec{prec}.json"
    save_payload(discrete_payload, filename_discrete)
    print(f"Step 4 Time: {round(time.time() - t_start, 4)} seconds")

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 5: Hardware-Aware Simulated Quantum Annealing (SQA)
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    print("\n" + "-"*80)
    print(" STEP 5: HARDWARE-AWARE SQA (QPU Simulation) ")
    print("-"*80)
    t_start = time.time()
    
    sqa_solver = SimulatedQuantumAnnealingSolver(
        **formulator_config, 
        num_reads=num_reads, 
        num_sweeps=num_sweeps, 

        # SQA special params
        trotter_slices=16,        
        chain_strength_scale=2.5,    
        beta=10.0,
    )
    
    net_sqa_opf = copy.deepcopy(pristine_net)
    
    # PASS IN THE IP COST FOR THE OPTIMALITY GAP METRIC
    sqa_opf_solution, sqa_opf_metadata = sqa_solver.solve_opf(net_sqa_opf, reference_cost_eur_per_hr=ip_opf_cost)
    
    if sqa_opf_metadata["status"] == "Success":
        print(f"[SQA OPF] Optimal Cost: {sqa_opf_solution['cost_eur_per_hr']} EUR/hr")
        net_sqa_pf = copy.deepcopy(pristine_net)
        
        for idx_str, dispatch_data in sqa_opf_solution['generator_dispatch_mw'].items():
            net_sqa_pf.gen.loc[int(idx_str), 'p_mw'] = float(dispatch_data['p_mw'])
        for idx_str, dispatch_data in sqa_opf_solution['slack_dispatch_mw'].items():
            net_sqa_pf.ext_grid.loc[int(idx_str), 'p_mw'] = float(dispatch_data['p_mw'])
            
        sqa_pf_solution, sqa_pf_metadata = sqa_solver.solve_pf(net_sqa_pf)
        print(f"[SQA PF] Validation Feasible: {sqa_pf_metadata['status']}")
    else:
        print("[SQA OPF] Solver failed.")
        sqa_pf_solution, sqa_pf_metadata = {}, {}

    sqa_payload = {
        "problem_parameters": original_problem_parameters,
        "opf_results": {"solution": sqa_opf_solution, "metadata": sqa_opf_metadata},
        "pf_validation_results": {"solution": sqa_pf_solution, "metadata": sqa_pf_metadata}
    }
    
    filename_sqa = f"{grid_name}_sqa_{form_type}_prec{prec}.json"
    save_payload(sqa_payload, filename_sqa)
    print(f"Step 5 Time: {round(time.time() - t_start, 4)} seconds")

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP X: Execution Summary
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    all_payloads = {
        f"Classical IP ({ip_solver.formulation})": ground_truth_payload,
        f"Simulated Annealing ({form_type})": discrete_payload,
        f"Hardware-Aware SQA ({form_type})": sqa_payload
    }
    print_execution_summary(all_payloads, formulator_config)