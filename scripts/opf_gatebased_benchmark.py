###############################################################################################################
#                                                SETUP
###############################################################################################################
import pandapower.networks as pn
import os
import copy
import time
import json
import numpy as np
import pandas as pd

from homemade_grids.small_grids import extract_network_info
from solvers.classical_ip_solver import ClassicalIPSolver

# Import the dual-solvers we built
from solvers.simulated_gate_based.run_emulator import IdealQAOA, EmuQAOA

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

def load_payload(filepath: str) -> dict:
    """Loads an existing JSON payload from disk."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_execution_summary(payloads: dict, config: dict = None):
    """Parses saved payloads and prints pivot tables of key OPF metrics and QAOA gate complexity."""
    print("\n" + "="*115)
    print(" QAOA PIPELINE EXECUTION SUMMARY ".center(115, "="))
    print("="*115)

    rows_perf = []
    rows_res = []

    for solver_name, payload in payloads.items():
        opf_sol = payload.get("opf_results", {}).get("solution", {}) or {}
        opf_meta = payload.get("opf_results", {}).get("metadata", {}) or {}

        # --- 1. Performance & Economics Table ---
        opf_cost = opf_sol.get("cost_eur_per_hr")
        slack_mw = None
        if opf_sol.get("slack_dispatch_mw"):
            slack_mw = sum(d.get("p_mw", 0.0) for d in opf_sol["slack_dispatch_mw"].values())

        pf_loading = None
        if opf_sol.get("grid_state"):
            pf_loading = opf_sol["grid_state"].get("max_line_loading_percent")

        exec_time = round(opf_meta.get("execution_time_seconds", {}).get("total", 0.0), 4)

        rows_perf.append({
            "Solver": solver_name,
            "OPF Status": opf_meta.get("status", "Failed"),
            "Cost (EUR/hr)": f"{opf_cost:.2f}" if opf_cost is not None else "N/A",
            "Slack (MW)": f"{slack_mw:.2f}" if slack_mw is not None else "N/A",
            "Max Load (%)": f"{pf_loading:.2f}" if pf_loading is not None else "N/A",
            "Time (s)": exec_time
        })

        # --- 2. Hardware & Complexity Table (QAOA Specific) ---
        if "Classical IP" not in solver_name:
            hw = opf_meta.get("hardware_metrics", {})
            alg = opf_meta.get("algorithmic_metrics", {}).get("qaoa", {})
            
            p_layer = alg.get("depths", ["N/A"])[-1]
            l_qub = hw.get("logical_qubits", "N/A")
            c_depth = hw.get("circuit_depth", "N/A")
            cnot_count = hw.get("physical_two_qubit_gates", hw.get("two_qubit_gates", "N/A"))
            routing = hw.get("routing_overhead_multiplier", "N/A")
            
            p_ground = alg.get("ideal_p_ground", alg.get("p_ground_exact", "N/A"))
            feas_frac = alg.get("feasible_fraction", "N/A")
            
            # Post-processing mitigation stats
            mit = alg.get("mitigated_readout", {})
            fixed_flips = mit.get("total_hardware_errors_fixed", "N/A")
            
            rows_res.append({
                "Solver": solver_name,
                "p-Layer": p_layer,
                "Qubits": l_qub,
                "Depth": c_depth,
                "CNOTs": cnot_count,
                "Overhead": f"{routing:.2f}x" if isinstance(routing, float) else "N/A",
                "P(Ground)": f"{p_ground*100:.1f}%" if isinstance(p_ground, float) else "N/A",
                "Feasible": f"{feas_frac*100:.1f}%" if isinstance(feas_frac, float) else "N/A",
                "Errors Fixed": fixed_flips
            })

    # Print First Table
    df_perf = pd.DataFrame(rows_perf)
    print(df_perf.to_string(index=False))

    # Print Second Table
    print("\n" + "-"*115)
    print(" GATE-BASED CIRCUIT COMPLEXITY & METRICS ".center(115, "-"))
    print("-" * 115)
    if rows_res:
        df_res = pd.DataFrame(rows_res)
        print(df_res.to_string(index=False))
    else:
        print("No QAOA solvers executed to report hardware metrics.")
    print("="*115 + "\n")


###############################################################################################################
#                                                Runner
###############################################################################################################

def run_pipeline(formulator_config, pristine_net, p_sweep, shots=2000, n_restarts=15, method="Powell", hw_preset="ibm_eagle"):
    # ------------------------------------------------------------------------------------------------- #
    # STEP 1: Initialize Problem
    # ------------------------------------------------------------------------------------------------- #
    original_problem_parameters = extract_network_info(pristine_net)
    grid_name = original_problem_parameters.get("network_name", "unknown_grid").replace(" ", "_").lower()

    output_dir = os.path.join("output_qaoa", grid_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(pristine_net, 'ext_grid') and not pristine_net.ext_grid.empty:
        pristine_net.ext_grid['min_p_mw'] = 0.0

    form_type = formulator_config["formulation"]
    prec = formulator_config["mw_precision"]
    target_p = p_sweep[-1] if isinstance(p_sweep, tuple) else p_sweep

    # ------------------------------------------------------------------------------------------------- #
    # STEP 2: Classical IP (Continuous Ground Truth)
    # ------------------------------------------------------------------------------------------------- #
    filename_ip = os.path.join(output_dir, f"classical_ip_dc.json")
    
    if os.path.exists(filename_ip):
        print(f"[Cache Hit] Classical IP solver: {filename_ip}")
        ground_truth_payload = load_payload(filename_ip)
    else:
        t_start = time.time()
        ip_solver = ClassicalIPSolver(formulation="dc")
        net_ip = copy.deepcopy(pristine_net)
        ip_sol, ip_meta = ip_solver.solve_opf(net_ip)
        
        ground_truth_payload = {
            "problem_parameters": original_problem_parameters,
            "opf_results": {"solution": ip_sol, "metadata": ip_meta}
        }
        save_payload(ground_truth_payload, filename_ip)
        print(f"Step 2 (Classical Exact) Time: {round(time.time() - t_start, 4)} seconds")

    # ------------------------------------------------------------------------------------------------- #
    # STEP 3: Ideal QAOA (Noiseless Statevector Training)
    # ------------------------------------------------------------------------------------------------- #
    filename_ideal = os.path.join(output_dir, f"ideal_qaoa_{form_type}_prec_{prec}_p{target_p}.json")

    if os.path.exists(filename_ideal):
        print(f"[Cache Hit] Ideal QAOA statevector: {filename_ideal}")
        ideal_payload = load_payload(filename_ideal)
    else:
        t_start = time.time()
        ideal_solver = IdealQAOA(
            **formulator_config, p_sweep=p_sweep, shots=shots, 
            method=method, n_restarts=n_restarts, maxiter=1000
        )
        net_ideal = copy.deepcopy(pristine_net)
        ideal_sol, ideal_meta = ideal_solver.solve_opf(net_ideal)
        
        ideal_payload = {
            "problem_parameters": original_problem_parameters,
            "opf_results": {"solution": ideal_sol, "metadata": ideal_meta}
        }
        save_payload(ideal_payload, filename_ideal)
        print(f"Step 3 (Ideal QAOA Training) Time: {round(time.time() - t_start, 4)} seconds")

    # ------------------------------------------------------------------------------------------------- #
    # STEP 4: Noisy Hardware QAOA (IBM Eagle Emulator + Mitigation)
    # ------------------------------------------------------------------------------------------------- #
    filename_emu = os.path.join(output_dir, f"emu_{hw_preset}_{form_type}_prec_{prec}_p{target_p}.json")

    if os.path.exists(filename_emu):
        print(f"[Cache Hit] Noisy Hardware Emulator: {filename_emu}")
        emu_payload = load_payload(filename_emu)
    else:
        t_start = time.time()
        emu_solver = EmuQAOA(
            **formulator_config, p_sweep=p_sweep, shots=shots, hardware_preset=hw_preset,
            method=method, n_restarts=n_restarts, maxiter=1000
        )
        net_emu = copy.deepcopy(pristine_net)
        emu_sol, emu_meta = emu_solver.solve_opf(net_emu)

        emu_payload = {
            "problem_parameters": original_problem_parameters,
            "opf_results": {"solution": emu_sol, "metadata": emu_meta}
        }
        save_payload(emu_payload, filename_emu)
        print(f"Step 4 (Noisy QAOA Emulation) Time: {round(time.time() - t_start, 4)} seconds")

    # ------------------------------------------------------------------------------------------------- #
    # STEP 5: Execution Summary
    # ------------------------------------------------------------------------------------------------- #
    all_payloads = {
        "Classical IP (dc)": ground_truth_payload,
        "Ideal Statevector QAOA": ideal_payload,
        f"IBM Noisy Emulator QAOA": emu_payload
    }
    
    print_execution_summary(all_payloads, formulator_config)


###############################################################################################################
#                                              Execution Block
###############################################################################################################
if __name__ == "__main__":
    from homemade_grids.small_grids import case3_low_gen
    
    default_config = {
        "formulation": "dc_ptdf",
        "encoding": "radix",
        "mw_precision": 50.0, 
    }

    # QAOA Specific Hyperparameters
    p_sweep = (1, 2, 3)  
    n_restarts = 15      
    shots = 2000         

    run_pipeline(
        default_config, 
        case3_low_gen(), 
        p_sweep=p_sweep, 
        shots=shots, 
        n_restarts=n_restarts
    )