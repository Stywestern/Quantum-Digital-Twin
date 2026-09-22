import sys
import json
import numpy as np
from pathlib import Path
import csv

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def get_stats(arr):
    """Returns mean and std safely."""
    valid = [x for x in arr if x is not None]
    if not valid: return {"mean": "N/A", "std": "N/A"}
    return {"mean": round(float(np.mean(valid)), 2), "std": round(float(np.std(valid)), 2)}

def format_row(label, stats, unit=""):
    """Formats a table row with mean ± std."""
    if stats["mean"] == "N/A": return f" {label:<28} | N/A"
    return f" {label:<28} | {stats['mean']:>10.2f} ± {stats['std']:<7.2f} {unit}"

def analyze_sa_results(files, grid_name, solver_name):
    costs, energies, times, imbalances, slack_usages, line_vios = [], [], [], [], [], []
    feasible_count = 0
    complexity, hw = {}, {}

    for f in files:
        with open(f, "r") as fp: data = json.load(fp)
        sol = data.get("solution", {})
        meta = data.get("metadata", {})
        stats = meta.get("algorithmic_metrics", {}).get("sampler_stats", {})
        feas = sol.get("grid_state", {}).get("feasibility", {})
        
        complexity = meta.get("problem_complexity", {}).get("quantum_domain_qubo", {})
        hw = stats

        costs.append(sol.get("cost_eur_per_hr"))
        energies.append(stats.get("energy_best"))
        times.append(meta.get("execution_time_seconds", {}).get("total"))
        imbalances.append(abs(feas.get("raw_imbalance_mw", 0.0)))
        line_vios.append(abs(feas.get("max_line_violation_mw", 0.0)))
        slack_usages.append(feas.get("raw_slack_dispatch_mw", 0.0))
        if feas.get("is_feasible"): feasible_count += 1

    print("\n" + "="*65)
    print(f" 📊 CLASSICAL SA ASSESSMENT: {grid_name.upper()}")
    print("="*65)
    print(" --- 1. PROBLEM & SOLVER COMPLEXITY ---")
    print(f" Logical Qubits : {complexity.get('total_logical_qubits')}    | Interactions : {complexity.get('num_interactions')}")
    print(f" Reads Requested: {hw.get('num_reads_requested')}   | Sweeps/Read  : {hw.get('num_sweeps_per_read')}")
    print("-" * 65)
    print(" --- 2. SOLUTION METRICS (Averages over all runs) ---")
    print(f" Feasible Runs                | {feasible_count}/{len(files)}")
    print(format_row("Financial Cost", get_stats(costs), "EUR/hr"))
    print(format_row("QUBO Energy (Best)", get_stats(energies), ""))
    print(format_row("Raw Imbalance", get_stats(imbalances), "MW"))
    print(format_row("Max Line Violation", get_stats(line_vios), "MW"))
    print(format_row("Slack Generator Usage", get_stats(slack_usages), "MW"))
    print(format_row("Execution Time", get_stats(times), "sec"))
    print("="*65 + "\n")

def analyze_sqa_results(files, grid_name, solver_name):
    costs, times, imbalances, slack_usages, line_vios = [], [], [], [], []
    raw_energies, pol_energies, polish_gains, spreads = [], [], [], []
    feasible_count = 0
    complexity, hw = {}, {}

    for f in files:
        with open(f, "r") as fp: data = json.load(fp)
        sol = data.get("solution", {})
        meta = data.get("metadata", {})
        stats = meta.get("algorithmic_metrics", {}).get("sampler_stats", {})
        feas = sol.get("grid_state", {}).get("feasibility", {})
        
        complexity = meta.get("problem_complexity", {}).get("quantum_domain_qubo", {})
        hw = stats

        costs.append(sol.get("cost_eur_per_hr"))
        times.append(meta.get("execution_time_seconds", {}).get("total"))
        imbalances.append(abs(feas.get("raw_imbalance_mw", 0.0)))
        line_vios.append(abs(feas.get("max_line_violation_mw", 0.0)))
        slack_usages.append(feas.get("raw_slack_dispatch_mw", 0.0))
        if feas.get("is_feasible"): feasible_count += 1
        
        raw_energies.append(stats.get("energy_best_raw"))
        pol_energies.append(stats.get("energy_best_after_greedy"))
        polish_gains.append(stats.get("greedy_polish_gain_eur"))
        spreads.append(stats.get("trotter_spread_best_read_normalized"))

    print("\n" + "="*65)
    print(f" 🔬 QUANTUM SQA ASSESSMENT: {grid_name.upper()}")
    print("="*65)
    print(" --- 1. PROBLEM & SOLVER COMPLEXITY ---")
    print(f" Logical Qubits : {complexity.get('total_logical_qubits')}    | Interactions : {complexity.get('num_interactions')}")
    print(f" Dynamic Range  : {hw.get('dynamic_range'):.2e} | Trotter Slics: {hw.get('trotter_slices')}")
    print(f" Reads Requested: {hw.get('num_reads_requested')}   | Sweeps/Read  : {hw.get('num_sweeps_per_read')}")
    print("-" * 65)
    print(" --- 2. PHYSICAL SOLUTION METRICS ---")
    print(f" Feasible Runs                | {feasible_count}/{len(files)}")
    print(format_row("Final Financial Cost", get_stats(costs), "EUR/hr"))
    print(format_row("Raw Imbalance", get_stats(imbalances), "MW"))
    print(format_row("Max Line Violation", get_stats(line_vios), "MW"))
    print(format_row("Slack Generator Usage", get_stats(slack_usages), "MW"))
    print(format_row("Execution Time", get_stats(times), "sec"))
    print("-" * 65)
    print(" --- 3. QUANTUM-CLASSICAL DYNAMICS (Hamming Cliff Analysis) ---")
    print(format_row("Energy Best (Raw Quantum)", get_stats(raw_energies), "EUR"))
    print(format_row("Energy Best (After Greedy)", get_stats(pol_energies), "EUR"))
    print(format_row("Greedy Polish Gain", get_stats(polish_gains), "EUR"))
    print(format_row("Trotter Spread (Normalized)", get_stats(spreads), ""))
    print("="*65 + "\n")

def examine_solver_results(grid_name, solver_name):
    target_dir = project_root / "output" / grid_name / solver_name
    
    if not target_dir.exists():
        print(f"Error: Directory {target_dir} does not exist.")
        return

    files = [f for f in target_dir.glob(f"{solver_name}_itnum_*.json") if "avg" not in f.name]
    if not files:
        print(f"No iteration files found in {target_dir}")
        return
        
    if "sqa" in solver_name.lower():
        analyze_sqa_results(files, grid_name, solver_name)
    else:
        analyze_sa_results(files, grid_name, solver_name)


def export_side_by_side_csv(grid_name, solvers):
    """
    Generates a wide-format CSV comparing multiple solvers side-by-side.
    Columns expand dynamically based on the solvers provided.
    """
    output_dir = project_root / "output" / grid_name
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{grid_name}_solver_comparison.csv"

    row_dict = {"Problem_Grid": grid_name}
    fieldnames = ["Problem_Grid"]

    for solver_name in solvers:
        target_dir = project_root / "output" / grid_name / solver_name
        if not target_dir.exists():
            print(f"Warning: Data for {solver_name} not found. Skipping in CSV.")
            continue

        files = [f for f in target_dir.glob(f"{solver_name}_itnum_*.json") if "avg" not in f.name]
        if not files:
            continue

        costs, times, imbs, vios, slacks = [], [], [], [], []
        feasible_count = 0

        for f in files:
            with open(f, "r") as fp:
                data = json.load(fp)
            
            sol = data.get("solution", {})
            meta = data.get("metadata", {})
            feas = sol.get("grid_state", {}).get("feasibility", {})
            
            costs.append(sol.get("cost_eur_per_hr"))
            times.append(meta.get("execution_time_seconds", {}).get("total"))
            imbs.append(abs(feas.get("raw_imbalance_mw", 0.0)))
            vios.append(abs(feas.get("max_line_violation_mw", 0.0)))
            slacks.append(feas.get("raw_slack_dispatch_mw", 0.0))
            if feas.get("is_feasible"): 
                feasible_count += 1

        def s_mean(arr):
            v = [x for x in arr if x is not None]
            return round(float(np.mean(v)), 2) if v else "N/A"
            
        def s_std(arr):
            v = [x for x in arr if x is not None]
            return round(float(np.std(v)), 2) if v else "N/A"

        # Use uppercase prefix for column headers (e.g., DC_SA_Cost_Mean)
        prefix = solver_name.upper()
        
        metrics = {
            f"{prefix}_Feasible_Ratio": f"{feasible_count}/{len(files)}",
            f"{prefix}_Cost_EUR_hr_Mean": s_mean(costs),
            f"{prefix}_Cost_EUR_hr_Std": s_std(costs),
            f"{prefix}_Execution_Time_sec": s_mean(times),
            f"{prefix}_Imbalance_MW_Mean": s_mean(imbs),
            f"{prefix}_Line_Violation_MW_Mean": s_mean(vios),
            f"{prefix}_Slack_MW_Mean": s_mean(slacks),
        }

        row_dict.update(metrics)
        fieldnames.extend(metrics.keys())

    if len(row_dict) > 1:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row_dict)
        print(f"\n 💾 Saved CSV comparison to: {csv_path}")
    else:
        print("\n ❌ Could not generate CSV: No valid solver data found.")

if __name__ == "__main__":
    grid = "case5"
    solvers_to_compare = ["dc_sa", "dc_qsa"]

    # 1. Solver exemination
    examine_solver_results(grid, "dc_sa")
    examine_solver_results(grid, "dc_qsa")

    # 2. Generate side-by-side CSV
    export_side_by_side_csv(grid, solvers_to_compare)
