import os
import json
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_all_results():
    """Crawls all directories in output/, compiles data, and generates WP2 deliverables."""
    
    json_files = glob.glob(os.path.join("output", "*", "*.json"))
    
    if not json_files:
        print("No JSON result files found in any subdirectories of 'output/'.")
        return

    rows = []
    
    for file_path in json_files:
        with open(file_path, "r") as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError:
                continue
                
        # --- 1. Bulletproof Extraction from Filename ---
        file_name = os.path.basename(file_path)
        
        # IP Solver removed. Only parse SA and SQA files.
        if file_name.startswith("sqa_"):
            solver = "SQA"
            form = file_name.split("sqa_")[1].split("_enc_")[0]
            enc = file_name.split("_enc_")[1].split("_prec_")[0]
            prec = file_name.split("_prec_")[1].replace(".json", "")
        elif file_name.startswith("sa_"):
            solver = "SA"
            form = file_name.split("sa_")[1].split("_enc_")[0]
            enc = file_name.split("_enc_")[1].split("_prec_")[0]
            prec = file_name.split("_prec_")[1].replace(".json", "")
        else:
            continue
            
        form = form.replace("dc_", "")

        # --- Extract Structures ---
        grid = payload.get("problem_parameters", {}).get("network_name", "Unknown")
        opf_sol = payload.get("opf_results", {}).get("solution", {}) or {}
        opf_meta = payload.get("opf_results", {}).get("metadata", {})
        
        comp = opf_meta.get("problem_complexity", {})
        hw = opf_meta.get("hardware_metrics", {})
        alg = opf_meta.get("algorithmic_metrics", {})
        feas = opf_meta.get("hardware_feasibility_report", {})
        
        # --- Metrics ---
        cost = opf_sol.get("cost_eur_per_hr")
        gap = alg.get("optimality_gap_percent")
        opt_pct = (100.0 - gap) if gap is not None else np.nan  # Explicit Optimality Metric
        time_s = opf_meta.get("execution_time_seconds", {}).get("total", 0.0)
        
        # Max Load is now safely extracted directly from the OPF grid_state
        pf_load = opf_sol.get("grid_state", {}).get("max_line_loading_percent")
            
        q_domain = comp.get("quantum_domain_qubo", {})
        hw_limits = comp.get("hardware_limits", {})
        
        l_qub = q_domain.get("total_logical_qubits", np.nan)
        p_qub = hw.get("physical_qubits", np.nan)
        max_ch = hw.get("max_chain_length", np.nan)
        dens = q_domain.get("density_percent", np.nan)
        dr = hw_limits.get("dynamic_range", np.nan)
        
        cb_frac = alg.get("sampler_stats", {}).get("mean_chain_break_fraction", np.nan)
        risk_frac = feas.get("at_risk_fraction", np.nan)

        rows.append({
            "Grid": grid,
            "Solver": solver,
            "Form": form,
            "Enc": enc,
            "Prec": float(prec) if prec != "-" else np.nan,
            "Cost (€)": cost if cost is not None else np.nan,
            "Opt-Gap (%)": gap if gap is not None else np.nan,
            "Max Load (%)": pf_load if pf_load is not None else np.nan,
            "L-Qub": l_qub,
            "P-Qub": p_qub,
            "Max-Ch": max_ch,
            "Dens (%)": dens,
            "DynRng": dr,
            "At-Risk (%)": risk_frac * 100 if pd.notna(risk_frac) else np.nan,
            "CB (%)": cb_frac * 100 if pd.notna(cb_frac) else np.nan,
            "Time (s)": time_s
        })

    # --- Build and Format the DataFrame ---
    df = pd.DataFrame(rows)
    
    # Removed IP from solver order
    solver_order = ["SA", "SQA"]
    df['Solver'] = pd.Categorical(df['Solver'], categories=solver_order, ordered=True)
    df = df.sort_values(["Grid", "Solver", "Form", "Enc", "Prec"], ascending=[True, True, True, True, False])
    
    out_dir = os.path.join("output", "global_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Master Table Output
    df_display = df.copy()
    df_display['Solver'] = df_display['Solver'].astype(str)
    df_display = df_display.replace("nan", "-").fillna("-")
    
    for col in ["Cost (€)", "Opt-Gap (%)", "Max Load (%)", "Dens (%)", "DynRng", "At-Risk (%)", "CB (%)", "Time (s)"]:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        
    df_display = df_display.set_index(["Grid", "Solver", "Form", "Enc", "Prec"])

    print("\n" + "="*155)
    print(" WP2 GLOBAL AGGREGATION ".center(155, "="))
    print("="*155)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):
        print(df_display)
    print("="*155 + "\n")
    
    df.to_csv(os.path.join(out_dir, "master_wp2_results.csv"), index=False)

    # =========================================================================
    # STATISTICAL AGGREGATIONS (SQA ONLY)
    # =========================================================================
    df_sqa = df[df["Solver"] == "SQA"].dropna(subset=["P-Qub", "DynRng"])
    
    if df_sqa.empty:
        print("Not enough SQA data to generate plots/stats.")
        return

    print("--- SQA AGGREGATE STATISTICS (Averaged across all Grids) ---")
    
    print("\n1. IMPACT OF ENCODING (Radix vs Hybrid):")
    enc_stats = df_sqa.groupby("Enc")[["P-Qub", "DynRng", "CB (%)"]].mean().round(2)
    print(enc_stats)
    
    print("\n2. IMPACT OF FORMULATION (PTDF vs Theta):")
    form_stats = df_sqa.groupby("Form")[["At-Risk (%)", "DynRng", "Max-Ch"]].mean().round(2)
    print(form_stats)
    
    print("\n3. IMPACT OF PRECISION (10 MW vs 1 MW):")
    prec_stats = df_sqa.groupby("Prec")[["P-Qub", "DynRng", "Opt-Gap (%)"]].mean().round(2)
    print(prec_stats)

    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================
    sns.set_theme(style="whitegrid")

    # Plot 1: The Quantum Resource Trilemma
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_sqa, x="P-Qub", y="DynRng", hue="Enc", style="Grid", s=100, palette="Set1")
    plt.yscale("log")
    plt.title("The Quantum Resource Trilemma: Encoding Trade-offs")
    plt.xlabel("Physical Qubits Required (Hardware Footprint)")
    plt.ylabel("Dynamic Range (Log Scale, Hardware Noise Susceptibility)")
    plt.axhline(1000, color='r', linestyle='--', alpha=0.5, label='Danger Zone (>1000)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_1_resource_trilemma.png"), dpi=300)
    plt.close()

    # Plot 2: Noise Vulnerability by Formulation
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_sqa, x="Form", y="At-Risk (%)", hue="Grid", palette="muted")
    plt.title("Formulation Hardware Vulnerability (At-Risk Matrix Fraction)")
    plt.ylabel("Matrix Values Below Hardware Noise Floor (%)")
    plt.xlabel("Power Flow Formulation")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_2_formulation_risk.png"), dpi=300)
    plt.close()

    # Plot 3: Qubit Scalability across Grids
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_sqa, x="Grid", y="P-Qub", hue="Enc", palette="viridis")
    plt.title("Physical Qubit Scaling: Radix vs Hybrid")
    plt.ylabel("Physical Qubits (Minor-Embedded)")
    plt.xlabel("Grid Topology")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_3_qubit_scaling.png"), dpi=300)
    plt.close()
    
    print(f"\n[+] Statistical analysis complete. 3 Thesis Plots saved to: {out_dir}/")

if __name__ == "__main__":
    aggregate_all_results()