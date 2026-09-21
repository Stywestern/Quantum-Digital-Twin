import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import pandapower as pp
import pandapower.networks as nw
import numpy as np
from solvers.base_solver import BaseSolver

class ClassicalIPSolver(BaseSolver):
    def __init__(self, formulation="ac", max_time=1800, **kwargs):
        super().__init__(max_time=max_time) if hasattr(super(), '__init__') else None
        if formulation not in ["ac", "dc"]:
            raise ValueError("Formulation must be 'ac' or 'dc'")
        self.formulation = formulation
        self.max_time = max_time

    def _calculate_complexity(self, net):
        """Calculates mathematical dimensions based on the formulation."""
        num_gen = len(net.gen) + len(net.ext_grid)
        num_bus = len(net.bus)
        num_line = len(net.line)

        if self.formulation == "ac":
            return {
                "num_continuous_variables": (2 * num_gen) + (2 * num_bus),
                "num_binary_variables": 0,
                "num_equality_constraints": 2 * num_bus,
                "num_inequality_constraints": (4 * num_gen) + (2 * num_bus) + (2 * num_line)
            }
        else: # dc
            return {
                "num_continuous_variables": num_gen + num_bus,
                "num_binary_variables": 0,
                "num_equality_constraints": num_bus,
                "num_inequality_constraints": (2 * num_line) + (2 * num_gen)
            }

    def _print_problem_parameters(self, net):
        """Dumps the exact mathematical parameters the IP solver uses."""
        print("\n" + "="*80)
        print(f"   CLASSICAL {self.formulation.upper()}-OPF: EXACT PROBLEM PARAMETERS SEEN BY IP SOLVER")
        print("="*80)
        
        # 1. Base Grid Demand
        total_load = net.load.p_mw.sum() if not net.load.empty else 0.0
        print(f"Total Grid Demand (Load): {total_load} MW\n")
        
        # 2. Generator Limits in Pandapower
        print("--- PANDAPOWER INPUTS: Generator Limits ---")
        if not net.gen.empty:
            for idx, row in net.gen.iterrows():
                print(f" Gen {idx}: Min = {row.get('min_p_mw', 'NaN')} MW, Max = {row.get('max_p_mw', 'NaN')} MW")
        if not net.ext_grid.empty:
            for idx, row in net.ext_grid.iterrows():
                print(f" Ext_Grid {idx}: Min = {row.get('min_p_mw', 'NaN')} MW, Max = {row.get('max_p_mw', 'NaN')} MW")
                
        # 3. Transmission Line Limits in Pandapower
        print("\n--- PANDAPOWER INPUTS: Transmission Line Limits ---")
        if not net.line.empty:
            for idx, row in net.line.iterrows():
                i_ka = row.get('max_i_ka', float('nan'))
                loading = row.get('max_loading_percent', 100.0)
                vn_kv = net.bus.loc[row['from_bus'], 'vn_kv']
                if not np.isnan(i_ka):
                    p_max = round(np.sqrt(3) * vn_kv * i_ka * (loading / 100.0), 3)
                    print(f" Line {idx} ({int(row['from_bus'])} -> {int(row['to_bus'])}): I_max = {i_ka} kA -> Approx Thermal Limit = {p_max} MW")
                else:
                    print(f" Line {idx} ({int(row['from_bus'])} -> {int(row['to_bus'])}): Unconstrained (NaN)")

        # 4. Costs in Pandapower
        print("\n--- PANDAPOWER INPUTS: Objective Costs ---")
        if hasattr(net, 'poly_cost') and not net.poly_cost.empty:
            for _, row in net.poly_cost.iterrows():
                print(f" PolyCost [{row['et']} {int(row['element'])}]: c0={row.get('cp0_eur',0)}, c1={row.get('cp1_eur_per_mw',0)}, c2={row.get('cp2_eur_per_mw2',0)}")
        if hasattr(net, 'pwl_cost') and not net.pwl_cost.empty:
            for _, row in net.pwl_cost.iterrows():
                print(f" PwlCost [{row['et']} {int(row['element'])}]: Points = {row['points']}")

        # 5. Internal PYPOWER Matrix Limits
        print("\n--- TRANSLATED MATRICES: Exact Internal PYPOWER Bounds (net._ppc) ---")
        if hasattr(net, '_ppc'):
            print(" Translated Generator Limits (PMAX/PMIN):")
            # In PYPOWER format, PMAX is col 8, PMIN is col 9
            for i, gen_row in enumerate(net._ppc['gen']):
                print(f"  PPC_Gen_{i}: Min = {gen_row[9]} MW, Max = {gen_row[8]} MW")
            
            print("\n Translated Branch Thermal Limits (RATE_A):")
            # RATE_A is col 5
            for i, br_row in enumerate(net._ppc['branch']):
                print(f"  PPC_Branch_{i}: Limit = {br_row[5]} MW/MVA")
            
            print("\n Translated Generator Cost Coefficients:")
            for i, cost_row in enumerate(net._ppc['gencost']):
                print(f"  PPC_GenCost_{i}: {list(cost_row)}")
        print("="*80 + "\n")

    def solve(self, net):
        start_time = time.time()
        
        try:
            # 1. Execution 
            if self.formulation == "ac":
                pp.runopp(net, verbose=False)
                solver_name = "pandapower_ac_interior_point"
            else:
                pp.rundcopp(net, verbose=False)
                solver_name = "pandapower_dc_interior_point"
            
            status = "Success"
            cost = round(net.res_cost, 3)
            
            # 2. Result Extraction
            sgen_dispatch = {}
            if self.formulation == "ac":
                dispatch = net.res_gen[['p_mw', 'q_mvar']].round(3).to_dict(orient='index')
                slack_dispatch = net.res_ext_grid[['p_mw', 'q_mvar']].round(3).to_dict(orient='index')
                if hasattr(net, 'res_sgen') and not net.res_sgen.empty:
                    sgen_dispatch = net.res_sgen[['p_mw', 'q_mvar']].round(3).to_dict(orient='index')
                max_bus_voltage = round(net.res_bus['vm_pu'].max(), 3)
                min_bus_voltage = round(net.res_bus['vm_pu'].min(), 3)
            else:
                dispatch = net.res_gen[['p_mw']].round(3).to_dict(orient='index')
                slack_dispatch = net.res_ext_grid[['p_mw']].round(3).to_dict(orient='index')
                if hasattr(net, 'res_sgen') and not net.res_sgen.empty:
                    sgen_dispatch = net.res_sgen[['p_mw']].round(3).to_dict(orient='index')
                max_bus_voltage = round(net.res_bus['vm_pu'].max(), 3) if 'vm_pu' in net.res_bus.columns else 1.0
                min_bus_voltage = round(net.res_bus['vm_pu'].min(), 3) if 'vm_pu' in net.res_bus.columns else 1.0

            max_line_loading = round(net.res_line['loading_percent'].max(), 3)

        except pp.optimal_powerflow.OPFNotConverged:
            status = "Failed to Converge"
            cost = None
            dispatch = {}
            sgen_dispatch = {}
            slack_dispatch = {}
            max_line_loading = None
            max_bus_voltage = None
            min_bus_voltage = None
            solver_name = f"pandapower_{self.formulation}_interior_point"

        execution_time = round(time.time() - start_time, 4)

        # 3. Payload Construction
        solution = {
            "cost_eur_per_hr": cost,
            "generator_dispatch_mw": dispatch,
            "static_generator_dispatch_mw": sgen_dispatch,  # Added SGen mapping
            "slack_dispatch_mw": slack_dispatch,
            "grid_state": {
                "max_line_loading_percent": max_line_loading,
                "max_voltage_pu": max_bus_voltage,
                "min_voltage_pu": min_bus_voltage
            }
        }

        metadata = {
            "status": status,
            "solver_name": solver_name,
            "execution_time_seconds": {
                "total": execution_time,
                "qpu_master": 0.0,
                "cpu_subproblem": execution_time
            },
            "problem_complexity": self._calculate_complexity(net),
            "algorithmic_metrics": {
                "framework": "standard_interior_point",
                "num_iterations": 1,
                "optimality_gap_percent": 0.0
            },
            "hardware_metrics": {
                "qpu_target": "none",
                "logical_qubits": 0,
                "physical_qubits": 0,
                "max_chain_length": 0,
                "circuit_depth": 0
            }
        }

        return solution, metadata

# =============================================================================
# Execution Block to inspect Case5
# =============================================================================
if __name__ == "__main__":
    print("Loading case5...")
    net = nw.case5()
    
    print("Instantiating Classical IP Solver (DC)...")
    solver = ClassicalIPSolver(formulation="dc")
    
    print("Running solve() to extract internal problem parameters...")
    solution, metadata = solver.solve(net)
    
    print(f"\nFinal IP Solver Cost: {solution['cost_eur_per_hr']} EUR/hr")
    
    print("\nGenerator Dispatches:")
    for idx, d in solution['generator_dispatch_mw'].items():
        print(f" Gen {idx}: {d['p_mw']} MW")
    
    print("\nStatic Generator Dispatches:")
    for idx, d in solution.get('static_generator_dispatch_mw', {}).items():
        print(f" SGen {idx}: {d['p_mw']} MW")
        
    print("\nSlack Dispatches:")
    for idx, d in solution['slack_dispatch_mw'].items():
        print(f" Slack {idx}: {d['p_mw']} MW")