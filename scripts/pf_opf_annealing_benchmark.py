###############################################################################################################
#                                            SETUP
###############################################################################################################
# Library imports
import pandapower as pp
import copy
import os
import pandas as pd
import json
import numpy as np

# Module imports
from homemade_grids.small_grids import case3_low_gen, case3_high_gen, case4_low_gen, case4_high_gen, extract_network_info
from solvers.qubo_formulator import QuboFormulator
from solvers.classical_ip_solver import ClassicalIPSolver
from solvers.simulated_annealing_solver import SimulatedAnnealingSolver
from solvers.simulated_quantum_annealing_solver import SimulatedQuantumAnnealingSolver

from solvers.temp import SimulatedQuantumAnnealingSolver1

class NumpyEncoder(json.JSONEncoder):
    """Safely converts numpy types to native Python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (bool, np.bool_)): return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def save_payload(data: dict, filepath: str):
    save_dir = os.path.dirname(filepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)
    print(f"[I/O] Payload saved to {filepath}")

def print_execution_summary(payloads: dict, config: dict = None):
    """Parses saved payloads and prints pivot tables of key OPF metrics and hardware complexity."""
    print("\n" + "="*110)
    print(" PIPELINE EXECUTION SUMMARY ".center(110, "="))
    print("="*110)
    
    rows_perf = []
    rows_res = []
    
    for solver_name, payload in payloads.items():
        opf_sol = payload.get("opf_results", {}).get("solution", {})
        opf_meta = payload.get("opf_results", {}).get("metadata", {})
        
        # --- 1. Performance & Economics Table ---
        opf_cost = opf_sol.get("cost_eur_per_hr") if opf_sol else None
        
        slack_mw = None
        if opf_sol and opf_sol.get("slack_dispatch_mw"):
            slack_mw = sum(d.get("p_mw", 0) for d in opf_sol["slack_dispatch_mw"].values())
            
        pf_loading = None
        if opf_sol and opf_sol.get("grid_state"):
            pf_loading = opf_sol["grid_state"].get("max_line_loading_percent")
            
        exec_time = round(opf_meta.get("execution_time_seconds", {}).get("total", 0.0), 4)
        
        rows_perf.append({
            "Solver": solver_name,
            "OPF Status": opf_meta.get("status", "Failed"),
            "Cost (EUR/hr)": round(opf_cost, 2) if opf_cost is not None else "N/A",
            "Slack (MW)": round(slack_mw, 2) if slack_mw is not None else "N/A",
            "Max Load (%)": round(pf_loading, 2) if pf_loading is not None else "N/A",
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
#                                            Runner
###############################################################################################################

def run_pipeline(formulator_config, pristine_net, num_reads=500, num_sweeps=2000, trotter_slices=16):
    import time
    import copy
    
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 1: Initialize the problem and enforce rules
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    original_problem_parameters = extract_network_info(pristine_net)
    grid_name = original_problem_parameters.get("network_name", "unknown_grid").replace(" ", "_").lower()

    output_dir = os.path.join("output", grid_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Enforce No-Export rule universally before handing grid to solvers
    if not pristine_net.ext_grid.empty:
        pristine_net.ext_grid['min_p_mw'] = 0.0

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 2: Continuous Ground Truth (Classical IP OPF)
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    t_start = time.time()
    
    ip_solver = ClassicalIPSolver(formulation="dc")
    net_ip_opf = copy.deepcopy(pristine_net)
    ip_opf_solution, ip_opf_metadata = ip_solver.solve_opf(net_ip_opf)
    
    ip_opf_cost = None
    if ip_opf_metadata["status"] == "Success":
        ip_opf_cost = ip_opf_solution['cost_eur_per_hr']
    else:
        print("[OPF] Classical solver failed to converge. The grid may be physically infeasible.")

    # Notice pf_validation_results is now omitted since OPF output contains feasibility data natively
    ground_truth_payload = {
        "problem_parameters": original_problem_parameters,
        "opf_results": {"solution": ip_opf_solution, "metadata": ip_opf_metadata}
    }
    
    filename_ip = os.path.join(output_dir, f"classical_ip_{ip_solver.formulation}.json")
    save_payload(ground_truth_payload, filename_ip)
    print(f"Step 2 (classical exact solver) Time: {round(time.time() - t_start, 4)} seconds")

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 3: Setup the qubo formulation configuration
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    form_type = formulator_config["formulation"]
    prec = formulator_config["mw_precision"]
    encoding_type = formulator_config["encoding"]

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 4: Discrete Ground truth (Simulated Annealing OPF)
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    t_start = time.time()
    
    sa_solver = SimulatedAnnealingSolver(
        **formulator_config,
        num_reads=num_reads,
        num_sweeps=num_sweeps
    )
    
    net_discrete_opf = copy.deepcopy(pristine_net)
    discrete_opf_solution, discrete_opf_metadata = sa_solver.solve_opf(net_discrete_opf)
    
    if discrete_opf_metadata["status"] != "Success":
        print("[SA OPF] Solver failed.")

    discrete_payload = {
        "problem_parameters": original_problem_parameters,
        "opf_results": {"solution": discrete_opf_solution, "metadata": discrete_opf_metadata}
    }
    
    filename_discrete = os.path.join(output_dir, f"sa_{form_type}_enc_{encoding_type}_prec_{prec}.json")
    save_payload(discrete_payload, filename_discrete)
    print(f"Step 4 (sa solver) Time: {round(time.time() - t_start, 4)} seconds")

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 5: Hardware-Aware Simulated Quantum Annealing (SQA OPF)
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    t_start = time.time()
    
    sqa_solver = SimulatedQuantumAnnealingSolver1(
        **formulator_config, 
        num_reads=num_reads, 
        num_sweeps=num_sweeps, 
        trotter_slices=trotter_slices,        
    )
    
    net_sqa_opf = copy.deepcopy(pristine_net)
    
    sqa_opf_solution, sqa_opf_metadata = sqa_solver.solve_opf(net_sqa_opf, reference_cost_eur_per_hr=ip_opf_cost)
    
    if sqa_opf_metadata["status"] != "Success":
        print("[SQA OPF] Solver failed.")

    sqa_payload = {
        "problem_parameters": original_problem_parameters,
        "opf_results": {"solution": sqa_opf_solution, "metadata": sqa_opf_metadata}
    }
    
    filename_sqa = os.path.join(output_dir, f"sqa_{form_type}_enc_{encoding_type}_prec_{prec}.json")
    save_payload(sqa_payload, filename_sqa)
    print(f"Step 5 (sqa solver) Time: {round(time.time() - t_start, 4)} seconds")

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP X: Execution Summary
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    all_payloads = {
        f"Classical IP ({ip_solver.formulation})": ground_truth_payload,
        f"Simulated Annealing ({form_type})": discrete_payload,
        f"Hardware-Aware SQA ({form_type})": sqa_payload
    }
    print_execution_summary(all_payloads, formulator_config)

###############################################################################################################
#                                        Execution Block
###############################################################################################################

if __name__ == "__main__":
    # If you run main.py directly, it just tests one configuration natively.
    default_config = {
        "formulation": "dc_ptdf",
        "encoding": "radix",
        "mw_precision": 10.0, 
    }

    num_reads = 300
    num_sweeps = 5000

    run_pipeline(default_config, case3_low_gen(), num_reads=num_reads, num_sweeps=num_sweeps)