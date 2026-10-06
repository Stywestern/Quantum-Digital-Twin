import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_solver_tag(file_name: str, meta: dict) -> str:
    """Identifies the solver cleanly across classical, SA, and emulated QPU variants."""
    solver_name = meta.get("solver_name", "").lower()
    
    if "classical_ip" in solver_name or file_name.startswith("classical_ip"):
        return "Classical IP"
    if "neal_simulated_annealing" in solver_name or (file_name.startswith("sa_") and not file_name.startswith("sqa_")):
        return "Classical SA"
    if "emulated_qpu_sa" in solver_name or file_name.startswith("sqa_sa_"):
        return "EMU-SA"
    if "emulated_qpu_svmc" in solver_name or file_name.startswith("sqa_svmc_"):
        return "EMU-SVMC"
    if "emulated_qpu_sqa" in solver_name or file_name.startswith("sqa_sqa_"):
        return "EMU-SQA"
    if file_name.startswith("sqa_"):
        return "EMU-SQA"
        
    return meta.get("solver_name", "Unknown")


def generate_wp2_plots(df: pd.DataFrame, out_dir: str):
    """Generates analytical visualizations for WP2 reporting."""
    sns.set_theme(style="whitegrid", context="paper")
    
    # 1. Hardware Overhead: Logical vs Physical Qubits (Encoding Comparison)
    qpu_df = df[df["Solver"].str.startswith("EMU")].copy()
    if not qpu_df.empty:
        overhead_df = qpu_df.groupby(["Grid", "Enc"])[["L-Qub", "P-Qub"]].mean().reset_index()
        
        # Get a sorted list of all unique grids to serve as the master X-axis
        grids = sorted(overhead_df["Grid"].unique())
        x = np.arange(len(grids))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, enc in enumerate(["radix", "unary"]):
            # Filter by encoding, set Grid as index, and reindex to the master grid list
            # This pads missing grid/encoding combinations with NaN so matplotlib doesn't crash
            enc_data = overhead_df[overhead_df["Enc"] == enc].set_index("Grid").reindex(grids)
            
            # Plot Logical Qubits
            ax.bar(x + (i*width) - width/4, enc_data["L-Qub"], width/2, 
                   label=f'{enc} (Logical)', color=sns.color_palette("muted")[i], alpha=0.9)
            # Plot Physical Qubits (Stacked alongside)
            ax.bar(x + (i*width) + width/4, enc_data["P-Qub"], width/2, 
                   label=f'{enc} (Physical)', color=sns.color_palette("dark")[i], alpha=0.9)

        ax.set_ylabel('Average Qubit Count')
        ax.set_title('Hardware Overhead: Logical vs. Embedded Physical Qubits')
        ax.set_xticks(x + width / 2)
        # Rotated labels so the 11 grid names don't overlap
        ax.set_xticklabels(grids, rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "wp2_hardware_overhead.png"), dpi=300)
        plt.close()

    # 2. Analog Noise Impact: Dynamic Range Vulnerability vs. Chain Breaks
    if not qpu_df.empty:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=qpu_df, 
            x="At-Risk (%)", 
            y="CB (%)", 
            hue="Grid", 
            style="Enc", 
            s=100, 
            alpha=0.8
        )
        plt.title("Analog Degradation: Matrix Squashing vs. Chain Failures")
        plt.xlabel("Variables At Risk of DAC Erasure (%)")
        plt.ylabel("Observed Chain Break Fraction (%)")
        plt.axhline(0, color='grey', lw=1, ls='--')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "wp2_analog_degradation.png"), dpi=300)
        plt.close()

    # 3. Solver Accuracy: Optimality Gap comparison
    compare_df = df[df["Opt-Gap (%)"].notna()].copy()
    if not compare_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=compare_df, 
            x="Grid", 
            y="Opt-Gap (%)", 
            hue="Solver", 
            palette="viridis"
        )
        plt.title("Solver Accuracy: Optimality Gap relative to Continuous IP Baseline")
        plt.ylabel("Cost Deviation from True Minimum (%)")
        plt.axhline(0, color='red', lw=2, ls='--', label='IP Baseline (0%)')
        plt.xticks(rotation=45, ha="right")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "wp2_optimality_gap.png"), dpi=300)
        plt.close()


def aggregate_all_results():
    """Crawls all directories in output/, compiles data, and generates WP2 deliverables."""
    json_files = glob.glob(os.path.join("output", "*", "*.json"))
    
    if not json_files:
        print("No JSON result files found in any subdirectories of 'output/'.")
        return

    rows = []
    ip_baselines = {}  # {grid_name: cost}

    # Pass 1: Extract data and store IP ground truths
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError:
                continue

        file_name = os.path.basename(file_path)
        grid = payload.get("problem_parameters", {}).get("network_name", "Unknown")
        opf_sol = payload.get("opf_results", {}).get("solution", {}) or {}
        opf_meta = payload.get("opf_results", {}).get("metadata", {}) or {}
        cost = opf_sol.get("cost_eur_per_hr")

        solver_tag = parse_solver_tag(file_name, opf_meta)
        if solver_tag == "Classical IP" and cost is not None:
            ip_baselines[grid] = float(cost)

    # Pass 2: Compile structured records
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError:
                continue

        file_name = os.path.basename(file_path)
        grid = payload.get("problem_parameters", {}).get("network_name", "Unknown")
        opf_sol = payload.get("opf_results", {}).get("solution", {}) or {}
        opf_meta = payload.get("opf_results", {}).get("metadata", {}) or {}
        comp = opf_meta.get("problem_complexity", {})
        hw = opf_meta.get("hardware_metrics", {})
        alg = opf_meta.get("algorithmic_metrics", {})
        feas = opf_meta.get("hardware_feasibility_report", {})
        q_domain = comp.get("quantum_domain_qubo", {})
        hw_limits = comp.get("hardware_limits", {})
        grid_state = opf_sol.get("grid_state", {})
        feasibility = grid_state.get("feasibility", {})

        solver = parse_solver_tag(file_name, opf_meta)
        
        # Formulation, Encoding, and Precision
        form = comp.get("formulation", "dc_ptdf").replace("dc_", "")
        enc = comp.get("encoding", "radix")
        
        prec = payload.get("opf_results", {}).get("metadata", {}).get(
            "qubo_parameters", {}
        ).get("mw_precision")
        if prec is None and "_prec_" in file_name:
            try:
                prec = float(file_name.split("_prec_")[1].replace(".json", ""))
            except ValueError:
                prec = np.nan

        # Metrics
        cost = opf_sol.get("cost_eur_per_hr")
        is_feasible = feasibility.get("is_feasible", True)
        
        # Calculate true optimality gap against classical IP baseline
        ip_cost = ip_baselines.get(grid)
        opt_gap = np.nan
        if cost is not None and ip_cost is not None and ip_cost > 0:
            opt_gap = ((cost - ip_cost) / ip_cost) * 100.0

        slack_mw = sum(d.get("p_mw", 0.0) for d in opf_sol.get("slack_dispatch_mw", {}).values()) if opf_sol.get("slack_dispatch_mw") else 0.0
        max_load = grid_state.get("max_line_loading_percent", np.nan)
        time_s = opf_meta.get("execution_time_seconds", {}).get("total", np.nan)

        l_qub = q_domain.get("total_logical_qubits", np.nan)
        p_qub = hw.get("physical_qubits", np.nan)
        max_ch = hw.get("max_chain_length", np.nan)
        dens = q_domain.get("density_percent", np.nan)
        dr = hw_limits.get("ising_dynamic_range", np.nan)

        # Chain break and risk stats
        sampler_stats = alg.get("sampler_stats", {})
        cb_frac = sampler_stats.get("sds_chain_break_fraction_mean", 
                  sampler_stats.get("mean_chain_break_fraction", 
                  sampler_stats.get("chain_break_fraction_mean", np.nan)))
        
        risk_frac = feas.get("at_risk_fraction", np.nan)

        rows.append({
            "Grid": grid,
            "Solver": solver,
            "Form": form,
            "Enc": enc,
            "Prec": float(prec) if prec is not None else 0,
            "Cost (€)": cost if cost is not None else 0,
            "Opt-Gap (%)": opt_gap,
            "Feasible": is_feasible,
            "Slack (MW)": slack_mw,
            "Max Load (%)": max_load,
            "L-Qub": l_qub,
            "P-Qub": p_qub,
            "Max-Ch": max_ch,
            "Dens (%)": dens,
            "DynRng": dr,
            "At-Risk (%)": risk_frac * 100 if pd.notna(risk_frac) else 0,
            "CB (%)": cb_frac * 100 if pd.notna(cb_frac) else 0,
            "Time (s)": time_s
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid data parsed.")
        return
        
    # Sorting and display ordering
    solver_order = ["Classical IP", "Classical SA", "EMU-SA", "EMU-SVMC", "EMU-SQA"]
    present_solvers = [s for s in solver_order if s in df["Solver"].unique()]
    other_solvers = [s for s in df["Solver"].unique() if s not in solver_order]
    df["Solver"] = pd.Categorical(df["Solver"], categories=present_solvers + other_solvers, ordered=True)
    df = df.sort_values(["Grid", "Solver", "Enc", "Prec"], ascending=[True, True, True, False])

    out_dir = os.path.join("output", "global_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Master Multi-Index Pivot Table Output ---
    df_display = df.copy()
    df_display["Solver"] = df_display["Solver"].astype(str)
    df_display = df_display.replace("nan", "-").fillna("-")
    
    for col in ["Cost (€)", "Opt-Gap (%)", "Slack (MW)", "Max Load (%)", "Dens (%)", "DynRng", "At-Risk (%)", "CB (%)", "Time (s)"]:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        
    df_display = df_display.set_index(["Grid", "Solver", "Form", "Enc", "Prec"])

    print("\n" + "="*160)
    print(" WP2 MASTER BENCHMARK AGGREGATION ".center(160, "="))
    print("="*160)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 2500):
        print(df_display)
    print("="*160 + "\n")
    
    df.to_csv(os.path.join(out_dir, "master_wp2_results.csv"), index=False)
    
    # --- 2. Visual Deliverables for WP2 Report ---
    generate_wp2_plots(df, out_dir)
    print(f"Generated analytical plots saved to: {out_dir}")

if __name__ == "__main__":
    aggregate_all_results()