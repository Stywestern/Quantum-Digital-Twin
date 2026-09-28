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
        dr = hw_limits.get("dynamic_range", np.nan)

        # Chain break and risk stats
        sampler_stats = alg.get("sampler_stats", {})
        cb_frac = sampler_stats.get("sds_chain_break_fraction_mean", 
                  sampler_stats.get("chain_break_fraction_mean", np.nan))
        risk_frac = feas.get("at_risk_fraction", np.nan)

        rows.append({
            "Grid": grid,
            "Solver": solver,
            "Form": form,
            "Enc": enc,
            "Prec": float(prec) if prec is not None else np.nan,
            "Cost (€)": cost if cost is not None else np.nan,
            "Opt-Gap (%)": opt_gap,
            "Feasible": is_feasible,
            "Slack (MW)": slack_mw,
            "Max Load (%)": max_load,
            "L-Qub": l_qub,
            "P-Qub": p_qub,
            "Max-Ch": max_ch,
            "Dens (%)": dens,
            "DynRng": dr,
            "At-Risk (%)": risk_frac * 100 if pd.notna(risk_frac) else np.nan,
            "CB (%)": cb_frac * 100 if pd.notna(cb_frac) else np.nan,
            "Time (s)": time_s
        })

    df = pd.DataFrame(rows)
    
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

    # =========================================================================
    # VISUAL NARRATIVE (5 Focused Publication Plots)
    # =========================================================================
    sns.set_theme(style="whitegrid", font_scale=1.1)
    df_plot = df.copy()

    # --- PLOT 1: Spatial Resource Explosion (Qubit Footprint) ---
    df_qubits = df_plot.dropna(subset=["P-Qub", "L-Qub"]).drop_duplicates(subset=["Grid", "Enc", "Prec"])
    if not df_qubits.empty:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        melted = pd.melt(df_qubits, id_vars=["Grid", "Enc"], value_vars=["L-Qub", "P-Qub"], 
                         var_name="Qubit Type", value_name="Count")
        sns.barplot(data=melted, x="Grid", y="Count", hue="Qubit Type", ax=ax1, palette=["#3498db", "#e74c3c"])
        ax1.set_title("Spatial Resource Scaling: Logical vs. Physical Qubits (Minor-Embedding Overhead)")
        ax1.set_ylabel("Qubit Count")
        ax1.set_xlabel("Grid Topology")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot_1_spatial_scaling.png"), dpi=300)
        plt.close()

    # --- PLOT 2: Graph Density & Dynamic Range ---
    df_density = df_plot.dropna(subset=["DynRng", "Dens (%)"]).drop_duplicates(subset=["Grid", "Enc", "Prec"])
    if not df_density.empty:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_density, x="Dens (%)", y="DynRng", hue="Enc", style="Grid", s=150, palette="viridis", ax=ax1)
        ax1.set_yscale("log")
        ax1.axhline(32, color='orange', linestyle='--', alpha=0.7, label='5-bit DAC Full Scale (32)')
        ax1.axhline(1000, color='red', linestyle='--', alpha=0.7, label='Critical Noise Floor (1000)')
        ax1.set_title("Analog Bottlenecks: Dynamic Range vs. Problem Density")
        ax1.set_xlabel("QUBO Interaction Density (%)")
        ax1.set_ylabel("Dynamic Range (Log Scale)")
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot_2_density_and_dynamic_range.png"), dpi=300)
        plt.close()

    # --- PLOT 3: The DAC Trap (At-Risk Fraction vs Optimality Gap) ---
    df_risk = df_plot.dropna(subset=["At-Risk (%)", "Opt-Gap (%)"])
    if not df_risk.empty:
        plt.figure(figsize=(9, 5))
        sns.scatterplot(data=df_risk, x="At-Risk (%)", y="Opt-Gap (%)", hue="Solver", style="Enc", s=120, palette="tab10")
        plt.title("The DAC Trap: Impact of Sub-LSB Truncation on Solution Quality")
        plt.xlabel("Matrix Coefficients Below Hardware Noise Floor (At-Risk %)")
        plt.ylabel("Optimality Gap vs Classical IP (%)")
        plt.axvline(50, color='gray', linestyle=':', label='50% Information Loss')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot_3_dac_trap_optimality.png"), dpi=300)
        plt.close()

    # --- PLOT 4: Multi-Solver Economic Performance Across Grids ---
    df_solvers = df_plot.dropna(subset=["Cost (€)"])
    if not df_solvers.empty:
        plt.figure(figsize=(11, 6))
        sns.barplot(data=df_solvers, x="Grid", y="Cost (€)", hue="Solver", palette="magma")
        plt.title("Economic Dispatch Cost by Solver Backend Across Grid Topologies")
        plt.ylabel("Generation Cost (EUR/hr)")
        plt.xlabel("Grid Topology")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot_4_solver_economic_comparison.png"), dpi=300)
        plt.close()

    # --- PLOT 5: Computational Time Complexity ---
    df_time = df_plot.dropna(subset=["Time (s)"])
    if not df_time.empty:
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df_time, x="Grid", y="Time (s)", hue="Solver", marker="o", linewidth=2.5, palette="Dark2")
        plt.yscale("log")
        plt.title("Solver Execution Scaling: Classical vs. Analog Physics Engines")
        plt.ylabel("Total Execution Time (Seconds, Log Scale)")
        plt.xlabel("Grid Topology")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot_5_runtime_scaling.png"), dpi=300)
        plt.close()

    print(f"[+] Aggregation complete. Master Pivot Table printed, CSV and 5 Thesis Plots saved to: {out_dir}/")


if __name__ == "__main__":
    aggregate_all_results()