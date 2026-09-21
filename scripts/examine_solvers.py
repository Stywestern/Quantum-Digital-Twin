import sys
import json
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def examine_solver_results(grid_name, solver_name):
    target_dir = project_root / "output" / grid_name / solver_name
    
    if not target_dir.exists():
        print(f"Error: Directory {target_dir} does not exist.")
        return

    # Find all iteration files (ignoring any existing avg files)
    files = list(target_dir.glob(f"{solver_name}_itnum_*.json"))
    if not files:
        print(f"No iteration files found in {target_dir}")
        return

    costs, energies, times, imbalances, slack_usages, line_vios = [], [], [], [], [], []
    feasible_count = 0

    for f in files:
        if "avg" in f.name:
            continue
            
        with open(f, "r") as fp:
            data = json.load(fp)

        # 1. Financial Cost
        cost = data.get("solution", {}).get("cost_eur_per_hr")
        if cost is not None: 
            costs.append(cost)

        # 2. QUBO Energy (Landscape)
        energy = data.get("metadata", {}).get("algorithmic_metrics", {}).get("sampler_stats", {}).get("energy_best")
        if energy is not None: 
            energies.append(energy)

        # 3. Execution Time
        t = data.get("metadata", {}).get("execution_time_seconds", {}).get("total")
        if t is not None: 
            times.append(t)

        # 4. Feasibility & Mismatch (Updated for PTDF)
        feas = data.get("solution", {}).get("grid_state", {}).get("feasibility", {})
        if feas:
            # In PTDF, the QUBO KCL error is the raw imbalance before the reference bus absorbs it
            imbalances.append(abs(feas.get("raw_imbalance_mw", 0.0)))
            line_vios.append(abs(feas.get("max_line_violation_mw", 0.0)))
            if feas.get("is_feasible"): 
                feasible_count += 1
                
        # 5. Slack Usage (Physical Ext_Grid dispatch)
        slack_dict = data.get("solution", {}).get("slack_dispatch_mw", {})
        total_slack = sum(d.get("p_mw", 0.0) for d in slack_dict.values())
        slack_usages.append(total_slack)

    # Calculate Aggregates safely
    def get_stats(arr):
        if not arr: return {"mean": None, "std": None}
        return {"mean": round(float(np.mean(arr)), 3), "std": round(float(np.std(arr)), 3)}

    cost_stats = get_stats(costs)
    energy_stats = get_stats(energies)
    time_stats = get_stats(times)
    imb_stats = get_stats(imbalances)
    line_stats = get_stats(line_vios)
    slack_stats = get_stats(slack_usages)

    # Print the Diagnostic Report
    print("\n" + "="*65)
    print(f" 📊 SOLVER EXAMINATION REPORT: {solver_name.upper()} on {grid_name.upper()}")
    print("="*65)
    print(f" Total Iterations Analyzed : {len(files)}")
    print(f" Strictly Feasible Runs    : {feasible_count}/{len(files)}")
    print("-" * 65)
    print(f" Financial Cost (EUR/hr)   : Mean = {cost_stats['mean']} | Std = ±{cost_stats['std']}")
    print(f" QUBO Energy (Landscape)   : Mean = {energy_stats['mean']} | Std = ±{energy_stats['std']}")
    print(f" Raw QUBO Imbalance (MW)   : Mean = {imb_stats['mean']} | Std = ±{imb_stats['std']}")
    print(f" Max Line Violation (MW)   : Mean = {line_stats['mean']} | Std = ±{line_stats['std']}")
    print(f" Ext_Grid Dispatch (MW)    : Mean = {slack_stats['mean']} | Std = ±{slack_stats['std']}")
    print(f" Execution Time (sec)      : Mean = {time_stats['mean']} | Std = ±{time_stats['std']}")
    print("="*65 + "\n")

    # Generate the Payload and Save
    agg_payload = {
        "grid": grid_name,
        "solver": solver_name,
        "total_runs_analyzed": len(files),
        "feasible_runs": feasible_count,
        "metrics": {
            "cost_eur_per_hr": cost_stats,
            "energy_best": energy_stats,
            "raw_qubo_imbalance_mw": imb_stats,
            "max_line_violation_mw": line_stats,
            "ext_grid_dispatch_mw": slack_stats,
            "execution_time_seconds": time_stats
        }
    }

    out_file = target_dir / f"{solver_name}_avg.json"
    with open(out_file, "w") as fp:
        json.dump(agg_payload, fp, indent=4)
    print(f"Saved aggregated statistics to: {out_file.name}\n")

if __name__ == "__main__":
    examine_solver_results("case5", "dc_sa")