import sys
import json
import numpy as np
from pathlib import Path
import csv
import os

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("Matplotlib is required for plotting. Run: pip install matplotlib")

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
        hw = stats if stats else {}

        costs.append(sol.get("cost_eur_per_hr"))
        energies.append(stats.get("energy_best") if stats else None)
        times.append(meta.get("execution_time_seconds", {}).get("total"))
        imbalances.append(abs(feas.get("raw_imbalance_mw", 0.0)))
        line_vios.append(abs(feas.get("max_line_violation_mw", 0.0)))
        slack_usages.append(feas.get("raw_slack_dispatch_mw", 0.0))
        if feas.get("is_feasible"): feasible_count += 1

    print("\n" + "="*65)
    print(f" 📊 CLASSICAL SA/IP ASSESSMENT: {solver_name.upper()}")
    print("="*65)
    print(" --- 1. PROBLEM & SOLVER COMPLEXITY ---")
    print(f" Logical Qubits : {complexity.get('total_logical_qubits', 'N/A')}    | Interactions : {complexity.get('num_interactions', 'N/A')}")
    print(f" Reads Requested: {hw.get('num_reads_requested', 'N/A')}   | Sweeps/Read  : {hw.get('num_sweeps_per_read', 'N/A')}")
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
        hw = stats if stats else {}

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
    print(f" 🔬 QUANTUM SQA ASSESSMENT: {solver_name.upper()}")
    print("="*65)
    print(" --- 1. PROBLEM & SOLVER COMPLEXITY ---")
    print(f" Logical Qubits : {complexity.get('total_logical_qubits')}    | Interactions : {complexity.get('num_interactions')}")
    if 'dynamic_range' in hw:
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
    if not target_dir.exists(): return
    files = [f for f in target_dir.glob(f"{solver_name}_itnum_*.json") if "avg" not in f.name]
    if not files: return
        
    if "sqa" in solver_name.lower(): analyze_sqa_results(files, grid_name, solver_name)
    else: analyze_sa_results(files, grid_name, solver_name)

def generate_comparative_report(grid_name):
    """
    Scans the grid output folder, generates a row-based pivot CSV, 
    and plots Execution Time vs. Financial Cost.
    """
    grid_dir = project_root / "output" / grid_name
    if not grid_dir.exists():
        print(f"Error: Output directory for {grid_name} not found.")
        return

    # Auto-discover all solver subdirectories
    solver_dirs = [d for d in grid_dir.iterdir() if d.is_dir()]
    if not solver_dirs:
        print("No solver data found.")
        return

    csv_path = grid_dir / f"{grid_name}_comparative_pivot.csv"
    plot_path = grid_dir / f"{grid_name}_time_vs_cost_plot.png"
    
    table_data = []

    for d in solver_dirs:
        solver_name = d.name
        files = [f for f in d.glob(f"{solver_name}_itnum_*.json") if "avg" not in f.name]
        if not files: continue

        costs, times, imbs, vios, slacks = [], [], [], [], []
        feasible_count = 0

        for f in files:
            with open(f, "r") as fp: data = json.load(fp)
            sol = data.get("solution", {})
            meta = data.get("metadata", {})
            feas = sol.get("grid_state", {}).get("feasibility", {})
            
            if sol.get("cost_eur_per_hr") is not None:
                costs.append(sol.get("cost_eur_per_hr"))
            times.append(meta.get("execution_time_seconds", {}).get("total"))
            imbs.append(abs(feas.get("raw_imbalance_mw", 0.0)))
            vios.append(abs(feas.get("max_line_violation_mw", 0.0)))
            slacks.append(feas.get("raw_slack_dispatch_mw", 0.0))
            if feas.get("is_feasible"): feasible_count += 1

        def s_mean(arr): return round(float(np.mean(arr)), 2) if arr else "N/A"
        def s_std(arr): return round(float(np.std(arr)), 2) if arr else "N/A"

        table_data.append({
            "Solver": solver_name,
            "Feasible_Runs": f"{feasible_count}/{len(files)}",
            "Cost_Mean": s_mean(costs),
            "Cost_Std": s_std(costs),
            "Time_Mean_sec": s_mean(times),
            "Imbalance_Mean": s_mean(imbs),
            "Line_Vio_Mean": s_mean(vios),
            "Slack_MW_Mean": s_mean(slacks)
        })

    # --- 1. Export Pivot CSV ---
    fieldnames = ["Solver", "Feasible_Runs", "Cost_Mean", "Cost_Std", "Time_Mean_sec", "Imbalance_Mean", "Line_Vio_Mean", "Slack_MW_Mean"]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_data)
    print(f"\n 💾 Saved Pivot CSV comparison to: {csv_path.name}")

    # --- 2. Generate Plot ---
    plt.figure(figsize=(12, 7))
    ip_cost = None

    for d in table_data:
        if d["Cost_Mean"] == "N/A" or d["Time_Mean_sec"] == "N/A": continue
        
        s_name = d["Solver"].lower()
        x, y = d["Time_Mean_sec"], d["Cost_Mean"]

        # Identify IP for the baseline
        if "ip" in s_name:
            ip_cost = y
            plt.scatter(x, y, color='red', marker='*', s=300, label='Classical IP (Optimal)', zorder=5)
            plt.annotate("Optimal IP", (x, y), xytext=(8, 8), textcoords='offset points', color='red', fontweight='bold')
        else:
            # Different markers for SA vs SQA
            marker = '^' if 'sqa' in s_name else 'o'
            color = 'blue' if 'sqa' in s_name else 'green'
            plt.scatter(x, y, color=color, marker=marker, s=120, alpha=0.7)
            plt.annotate(d["Solver"], (x, y), xytext=(8, -4), textcoords='offset points', fontsize=9, alpha=0.8)

    if ip_cost is not None:
        plt.axhline(y=ip_cost, color='red', linestyle='--', alpha=0.4)

    plt.xlabel("Execution Time (Seconds)", fontweight='bold')
    plt.ylabel("Average Financial Cost (EUR/hr)", fontweight='bold')
    plt.title(f"Solver Performance Analysis: Cost vs. Time ({grid_name.upper()})", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Simulated Annealing (SA)', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='Simulated Quantum Annealing (SQA)', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Interior Point (IP)', markerfacecolor='red', markersize=15)
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f" 📈 Saved Performance Plot to: {plot_path.name}\n")


if __name__ == "__main__":
    grid = "case5"
    
    grid_dir = project_root / "output" / grid
    if grid_dir.exists():
        # 1. Terminal assessment for every solver found
        solver_dirs = [d.name for d in grid_dir.iterdir() if d.is_dir()]
        for s in solver_dirs:
            examine_solver_results(grid, s)
            
        # 2. Generate Pivot CSV & Scatter Plot
        generate_comparative_report(grid)
    else:
        print(f"No output directory found for {grid}.")