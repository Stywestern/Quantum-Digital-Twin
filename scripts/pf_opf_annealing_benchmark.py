###############################################################################################################
#                                            SETUP
###############################################################################################################
# Library imports
import pandapower as pp
import pandapower.networks as pn
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
from solvers.simulated_quantum_annealing.run_emulator import EmulatedQPUSolver

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

        # --- 2. Hardware & Complexity Table ---
        if "Classical IP" not in solver_name:
            comp = opf_meta.get("problem_complexity", {})
            hw = opf_meta.get("hardware_metrics", {})
            alg = opf_meta.get("algorithmic_metrics", {})
            
            # SQA puts this at root metadata, SA doesn't have it
            feas = opf_meta.get("hardware_feasibility_report", {})

            form = comp.get("formulation", "N/A")
            enc = comp.get("encoding", "N/A")
            prec = config.get("mw_precision", "N/A") if config else "N/A"

            q_domain = comp.get("quantum_domain_qubo", {})
            l_qub = q_domain.get("total_logical_qubits", "N/A")
            dens = q_domain.get("density_percent", "N/A")
            
            # HW limits might be directly under problem_complexity depending on the solver
            hw_limits = comp.get("hardware_limits", {})
            dr = hw_limits.get("dynamic_range", "N/A")

            p_qub = hw.get("physical_qubits", "N/A")
            max_chain = hw.get("max_chain_length", "N/A")

            # Safely format fractional metrics (handling 0.0 correctly)
            sampler_stats = alg.get("sampler_stats", {})
            cb_frac = sampler_stats.get("chain_break_fraction_mean")
            cb = f"{cb_frac * 100:.3f}%" if cb_frac is not None else "N/A"

            risk_frac = feas.get("at_risk_fraction")
            risk = f"{risk_frac * 100:.2f}%" if risk_frac is not None else "N/A"
            
            # Formatting density cleanly
            dens_str = f"{dens}%" if isinstance(dens, (int, float)) else "N/A"
            
            # Formatting dynamic range cleanly
            dr_str = f"{dr:,.1f}" if isinstance(dr, (int, float)) else "N/A"

            rows_res.append({
                "Solver": solver_name,
                "Form": form,
                "Enc": enc,
                "Prec": prec,
                "L-Qubits": l_qub,
                "P-Qubits": p_qub,
                "Max-Chain": max_chain,
                "Density": dens_str,
                "Dyn-Range": dr_str,
                "At-Risk": risk,
                "Chain-Break": cb
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
    # STEP 5: Hardware-Aware Analog Simulation (SA vs SVMC vs SQA)
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    t_start = time.time()
    
    analog_payloads = {}
    samplers_to_test = ["sa", "svmc", "sqa"]

    for sampler_type in samplers_to_test:
        print(f"\n[+] Booting EmulatedQPU with backend: {sampler_type.upper()}")
        emulator_solver = EmulatedQPUSolver(
            **formulator_config, 
            num_reads=num_reads, 
            num_sweeps=num_sweeps,
            sampler=sampler_type,
            trotter_slices=trotter_slices
        )

        net_emu_opf = copy.deepcopy(pristine_net)
        emu_opf_solution, emu_opf_metadata = emulator_solver.solve_opf(net_emu_opf)

        if emu_opf_metadata["status"] != "Success":
            print(f"[{sampler_type.upper()} OPF] Solver failed.")

        payload = {
            "problem_parameters": original_problem_parameters,
            "opf_results": {"solution": emu_opf_solution, "metadata": emu_opf_metadata}
        }

        # Prepend 'sqa_' so the aggregation script groups all three under hardware-aware solvers
        filename_emu = os.path.join(output_dir, f"sqa_{sampler_type}_{form_type}_enc_{encoding_type}_prec_{prec}.json")
        save_payload(payload, filename_emu)
        
        analog_payloads[f"Emulated QPU ({sampler_type.upper()})"] = payload

    print(f"\nStep 5 (Hardware Emulators) Time: {round(time.time() - t_start, 4)} seconds")

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP X: Execution Summary
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    all_payloads = {
        f"Classical IP ({ip_solver.formulation})": ground_truth_payload,
        f"Classical SA ({form_type})": discrete_payload
    }
    # Merge the three hardware emulators into the final output
    all_payloads.update(analog_payloads)
    
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

    run_pipeline(default_config, pn.case5(), num_reads=num_reads, num_sweeps=num_sweeps)