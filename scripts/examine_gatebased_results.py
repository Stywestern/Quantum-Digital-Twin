import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_solver_tag(file_name: str, meta: dict) -> str:
    """Identifies the solver cleanly across Classical IP, Ideal QAOA, and Noisy QAOA."""
    solver_name = meta.get("solver_name", "").lower()
    
    if "classical_ip" in solver_name or file_name.startswith("classical_ip"):
        return "Classical IP"
    if "ideal_qaoa" in solver_name or file_name.startswith("ideal_"):
        return "Ideal QAOA"
    if "emu_qaoa" in solver_name or file_name.startswith("emu_"):
        if "ibm_eagle" in solver_name:
            return "Emu QAOA (Eagle)"
        return "Emu QAOA"
        
    return meta.get("solver_name", "Unknown")


def generate_qaoa_plots(df: pd.DataFrame, out_dir: str):
    """Generates analytical visualizations for QAOA reporting, sorted by true quantum complexity."""
    sns.set_theme(style="whitegrid", context="paper")
    
    # --- Grid Exclusion List ---
    # Add any grid names here (e.g., ["case5", "case6ww"]) to omit them from the plots
    exclude_grids = ["case5", "case6ww", "case9"]
    
    # Filter the dataframe
    plot_df = df[~df["Grid"].isin(exclude_grids)].copy()
    
    if "L-Qub" not in plot_df.columns or plot_df["L-Qub"].dropna().empty:
        print("No logical qubit data available for plotting.")
        return
        
    # Sort grids by their absolute logical qubit footprint
    grid_complexity = plot_df[plot_df["L-Qub"].notna()].groupby("Grid")["L-Qub"].min().sort_values()
    sorted_grids = grid_complexity.index.tolist()

    # We force 50.0 MW precision for the overhead and feasibility plots because 
    # tighter precisions (20.0, 10.0) trigger OOMs on larger grids, leaving the plots empty.
    target_prec = 50.0 

    # 1. Hardware Overhead: Logical vs Physical CNOTs (Routing Penalty)
    qaoa_df = plot_df[plot_df["Solver"].str.contains("QAOA")].copy()
    if not qaoa_df.empty:
        overhead_df = qaoa_df[(qaoa_df["Prec"] == target_prec) & (qaoa_df["Solver"] == "Emu QAOA (Eagle)")]
        
        if not overhead_df.empty:
            overhead_df = overhead_df.set_index("Grid").reindex(sorted_grids).dropna(subset=["L-CNOT", "P-CNOT"])
            x = np.arange(len(overhead_df))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width/2, overhead_df["L-CNOT"], width, label='Logical CNOTs (All-to-All)', color='skyblue', edgecolor='black')
            ax.bar(x + width/2, overhead_df["P-CNOT"], width, label='Physical CNOTs (Heavy-Hex SWAPs)', color='salmon', edgecolor='black')

            ax.set_ylabel('Total 2-Qubit Gates')
            ax.set_title(f'Heavy-Hex Routing Overhead (CNOT Inflation) at {target_prec} MW Precision')
            ax.set_xticks(x)
            ax.set_xticklabels(overhead_df.index, rotation=45, ha="right")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "qaoa_cnot_overhead.png"), dpi=300)
            plt.close()

    # 2. The Noise Wall: Ideal vs Raw vs Mitigated Feasibility
    if not qaoa_df.empty:
        feas_data = []
        for grid in sorted_grids:
            grid_data = qaoa_df[(qaoa_df["Grid"] == grid) & (qaoa_df["Prec"] == target_prec)]
            
            ideal_row = grid_data[grid_data["Solver"] == "Ideal QAOA"]
            emu_row = grid_data[grid_data["Solver"] == "Emu QAOA (Eagle)"]
            
            if not ideal_row.empty and pd.notna(ideal_row["Feas (%)"].values[0]):
                feas_data.append({"Grid": grid, "State": "1. Ideal Statevector", "Feas (%)": ideal_row["Feas (%)"].values[0]})
            if not emu_row.empty:
                if pd.notna(emu_row["Raw-Feas (%)"].values[0]):
                    feas_data.append({"Grid": grid, "State": "2. Raw Hardware Readout", "Feas (%)": emu_row["Raw-Feas (%)"].values[0]})
                if pd.notna(emu_row["Feas (%)"].values[0]):
                    feas_data.append({"Grid": grid, "State": "3. Mitigated (Steepest Descent)", "Feas (%)": emu_row["Feas (%)"].values[0]})
                
        if feas_data:
            feas_df = pd.DataFrame(feas_data)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feas_df, x="Grid", y="Feas (%)", hue="State", palette="magma", edgecolor='black')
            
            plt.title(f"Feasibility Collapse and Classical Mitigation (QAOA p=3, {target_prec} MW Precision)")
            plt.ylabel("Probability of Sampling a Feasible State (%)")
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="Execution Environment")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "qaoa_feasibility_degradation.png"), dpi=300)
            plt.close()

    # 3. Coherence Saturation vs Estimated Success Probability
    emu_df = plot_df[plot_df["Solver"] == "Emu QAOA (Eagle)"].copy()
    emu_df = emu_df.dropna(subset=["Coh-Sat (%)", "ESP"])
    if not emu_df.empty:
        plt.figure(figsize=(9, 6))
        
        valid_grids = [g for g in sorted_grids if g in emu_df["Grid"].values]
        emu_df["Grid"] = pd.Categorical(emu_df["Grid"], categories=valid_grids, ordered=True)
        
        scatter = sns.scatterplot(
            data=emu_df, x="Grid", y="Coh-Sat (%)", 
            hue="Prec", size="ESP", sizes=(50, 400), 
            palette="viridis", alpha=0.8, edgecolor='black'
        )
        
        plt.axhline(100, color='red', lw=1.5, ls='--', label="Absolute Coherence Limit (T2 = 100 µs)")
        plt.title("Physical Latency: Coherence Saturation and Success Probability")
        plt.ylabel("Coherence Window Consumed (%)")
        plt.xticks(rotation=45, ha="right")
        
        handles, labels = scatter.get_legend_handles_labels()
        plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "qaoa_coherence_saturation.png"), dpi=300)
        plt.close()

    # 4. Economic Accuracy: Optimality Gap across Precision Levels
    opt_df = plot_df[(plot_df["Solver"] == "Emu QAOA (Eagle)") & (plot_df["Opt-Gap (%)"].notna())].copy()
    if not opt_df.empty:
        opt_df["Opt-Gap (%)"] = opt_df["Opt-Gap (%)"].abs()
        opt_df["Precision (MW)"] = opt_df["Prec"].astype(str) + " MW"
        
        # Sort precisions for consistent legend order (50.0, 20.0, 10.0)
        prec_order = sorted(opt_df["Precision (MW)"].unique(), key=lambda x: float(x.split()[0]), reverse=True)
        
        plt.figure(figsize=(10, 6))
        # Add edge color so exact 0.0% gaps show up as a flat line rather than looking like missing data
        sns.barplot(
            data=opt_df, x="Grid", y="Opt-Gap (%)",
            hue="Precision (MW)", hue_order=prec_order, palette="Set2",
            order=[g for g in sorted_grids if g in opt_df["Grid"].values],
            edgecolor='black', linewidth=1
        )
        
        plt.title("Post-Mitigation Economic Accuracy: Impact of Discretization Precision")
        plt.ylabel("Absolute Cost Deviation from Continuous IP Baseline (%)")
        plt.axhline(0, color="red", lw=1.5, ls="--", label="IP Global Minimum (0%)")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Discretization Step", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "qaoa_optimality_gap.png"), dpi=300)
        plt.close()


def aggregate_all_results():
    """Crawls all directories in output_qaoa/, compiles data, and generates deliverables."""
    json_files = glob.glob(os.path.join("output_qaoa", "*", "*.json"))
    
    if not json_files:
        print("No JSON result files found in any subdirectories of 'output_qaoa/'.")
        return

    rows = []
    ip_baselines = {}  # {grid_name: {precision: cost}}

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
        adv_hw = hw.get("advanced_metrics", {})
        alg = opf_meta.get("algorithmic_metrics", {}).get("qaoa", {})
        q_domain = comp.get("quantum_domain_qubo", {})

        solver = parse_solver_tag(file_name, opf_meta)
        
        form = comp.get("formulation", "dc_ptdf").replace("dc_", "")
        enc = comp.get("encoding", "radix")
        
        prec = opf_meta.get("qubo_parameters", {}).get("mw_precision")
        if prec is None and "_prec_" in file_name:
            try:
                prec = float(file_name.split("_prec_")[1].split("_p")[0].replace(".json", ""))
            except ValueError:
                prec = np.nan

        cost = opf_sol.get("cost_eur_per_hr")
        ip_cost = ip_baselines.get(grid)
        opt_gap = np.nan
        if cost is not None and ip_cost is not None and ip_cost > 0:
            opt_gap = ((cost - ip_cost) / ip_cost) * 100.0

        if solver == "Ideal QAOA":
            feas_frac = alg.get("feasible_fraction", np.nan)
            raw_feas_frac = np.nan
        else:
            feas_frac = alg.get("mitigated_readout", {}).get("feasible_fraction", np.nan)
            raw_feas_frac = alg.get("raw_readout", {}).get("feasible_fraction", np.nan)
            
        p_layer = alg.get("trained_depth", alg.get("depths", [np.nan])[-1] if isinstance(alg.get("depths"), list) else np.nan)
        errors_fixed = alg.get("mitigated_readout", {}).get("total_hardware_errors_fixed", np.nan)

        res = hw.get("resources", {})
        
        l_qub = hw.get("logical_qubits", q_domain.get("total_logical_qubits", np.nan))
        p_qub = hw.get("physical_qubits", res.get("qubits", np.nan))
        c_depth = hw.get("circuit_depth", res.get("circuit_depth", np.nan))
        
        l_cnot = hw.get("logical_two_qubit_gates", res.get("logical_two_qubit_gates", hw.get("two_qubit_gates", np.nan)))
        p_cnot = hw.get("physical_two_qubit_gates", res.get("physical_two_qubit_gates", hw.get("two_qubit_gates", np.nan)))
        
        esp = adv_hw.get("estimated_success_probability", res.get("fidelity", {}).get("estimated_success_probability", np.nan))
        coh_sat = adv_hw.get("coherence_saturation_percent", res.get("timing", {}).get("coherence_saturation_pct", np.nan))
        
        time_s = opf_meta.get("execution_time_seconds", {}).get("total", np.nan)

        rows.append({
            "Grid": grid,
            "Solver": solver,
            "Form": form,
            "Enc": enc,
            "Prec": float(prec) if prec is not None else 0,
            "p-Layer": p_layer,
            "Cost (€)": cost if cost is not None else np.nan,
            "Opt-Gap (%)": opt_gap,
            "Feas (%)": feas_frac * 100 if pd.notna(feas_frac) else np.nan,
            "Raw-Feas (%)": raw_feas_frac * 100 if pd.notna(raw_feas_frac) else np.nan,
            "L-Qub": l_qub,
            "P-Qub": p_qub,
            "Depth": c_depth,
            "L-CNOT": l_cnot,
            "P-CNOT": p_cnot,
            "ESP": esp,
            "Coh-Sat (%)": coh_sat,
            "Errors Fix": errors_fixed,
            "Time (s)": time_s
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid QAOA data parsed.")
        return
        
    solver_order = ["Classical IP", "Ideal QAOA", "Emu QAOA (Eagle)"]
    present_solvers = [s for s in solver_order if s in df["Solver"].unique()]
    other_solvers = [s for s in df["Solver"].unique() if s not in solver_order]
    df["Solver"] = pd.Categorical(df["Solver"], categories=present_solvers + other_solvers, ordered=True)
    df = df.sort_values(["Grid", "Solver", "Enc", "Prec"], ascending=[True, True, True, False])

    out_dir = os.path.join("output_qaoa", "global_analysis")
    os.makedirs(out_dir, exist_ok=True)

    df_display = df.copy()
    df_display["Solver"] = df_display["Solver"].astype(str)
    
    formatters = {
        "Cost (€)": "{:.2f}",
        "Opt-Gap (%)": "{:.2f}",
        "Feas (%)": "{:.1f}",
        "Raw-Feas (%)": "{:.1f}",
        "Coh-Sat (%)": "{:.1f}",
        "Time (s)": "{:.2f}",
        "ESP": "{:.2e}"
    }
    
    for col, fmt in formatters.items():
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "-")
            
    for col in ["p-Layer", "L-Qub", "P-Qub", "Depth", "L-CNOT", "P-CNOT", "Errors Fix"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: str(int(x)) if pd.notna(x) else "-")

    df_display = df_display.set_index(["Grid", "Solver", "Prec"])
    cols_to_print = [c for c in df_display.columns if c not in ["Form", "Enc"]]

    print("\n" + "="*160)
    print(" QAOA MASTER BENCHMARK AGGREGATION ".center(160, "="))
    print("="*160)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 2500):
        print(df_display[cols_to_print])
    print("="*160 + "\n")
    
    df.to_csv(os.path.join(out_dir, "master_qaoa_results.csv"), index=False)
    
    generate_qaoa_plots(df, out_dir)
    print(f"Generated QAOA analytical plots saved to: {out_dir}")

if __name__ == "__main__":
    aggregate_all_results()