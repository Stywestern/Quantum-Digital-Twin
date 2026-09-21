# Add project root to sys.path to enable imports from solvers
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import libs
import json
import pandapower.networks as nw
from tqdm import tqdm

# Import solvers
from solvers.classical_ip_solver import ClassicalIPSolver
from solvers.simulated_annealing_solver import SimulatedAnnealingSolver
from solvers.quantum_annealing_sim_solver import SimulatedQuantumAnnealingSolver

SOLVER_REGISTRY = {
    "ac_ip": lambda **kwargs: ClassicalIPSolver(formulation="ac", **kwargs),
    "dc_ip": lambda **kwargs: ClassicalIPSolver(formulation="dc", **kwargs),
    "dc_sa": lambda **kwargs: SimulatedAnnealingSolver(**kwargs),
    "dc_qsa": lambda **kwargs: SimulatedQuantumAnnealingSolver(**kwargs)
}

GRID_REGISTRY = {
    "case4": nw.case4gs,
    "case5": nw.case5,
    "case9": nw.case9,
    "case14": nw.case14
}

# Fixed seeds to ensure mathematical variance is measurable, not due to luck
FIXED_SEEDS = [42, 1337, 2026, 9999, 5555, 12345, 98765, 10101, 24601, 8675309]

def run_benchmark(target_grid, target_solver, max_time, **solver_kwargs):
    # 1. Setup save directory
    save_dir = project_root / "output" / target_grid / target_solver
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Determine iterations based on solver type
    is_stochastic = "ip" not in target_solver
    iterations = len(FIXED_SEEDS) if is_stochastic else 1
    
    # 3. Load Problem
    if target_grid not in GRID_REGISTRY:
        raise ValueError(f"Grid {target_grid} not found.")
    net = GRID_REGISTRY[target_grid]()

    if target_solver not in SOLVER_REGISTRY:
        raise ValueError(f"Solver {target_solver} not found.")

    print(f"\nExecuting {target_solver} on {target_grid} ({iterations} iterations)...")

    # 4. Execution Loop (Now 1-indexed to match human reading)
    for itnum in tqdm(range(1, iterations + 1), desc=f"{target_solver} Progress", unit="run"):
        # Map the 1-indexed iteration back to the 0-indexed seed array
        current_seed = FIXED_SEEDS[itnum - 1] if is_stochastic else None
        
        # Files will now save exactly as printed: e.g., dc_sa_itnum_1.json
        file_path = save_dir / f"{target_solver}_itnum_{itnum}.json"

        if file_path.exists():
            tqdm.write(f"  [{itnum}/{iterations}] Skipping: {file_path.name} already exists.")
            continue

        tqdm.write(f"  [{itnum}/{iterations}] Running solver (Seed: {current_seed})...")
        
        # Inject the seed dynamically for stochastic solvers
        if is_stochastic:
            solver_kwargs["seed"] = current_seed

        solver = SOLVER_REGISTRY[target_solver](max_time=max_time, **solver_kwargs)
        solution, metadata = solver.solve(net)

        # 5. Validate and Save Output
        feasibility = solution["grid_state"]["feasibility"]
        if not feasibility["is_feasible"]:
            tqdm.write(f"  -> [DISCARDED] Run {itnum} is mathematically infeasible. Mismatch: {feasibility['raw_imbalance_mw']} MW raw or line limit {feasibility['max_line_violation_mw']}")

        payload = {
            "grid": target_grid,
            "solver": target_solver,
            "max_time_allowed": max_time,
            "solution": solution,
            "metadata": metadata
        }

        with open(file_path, "w") as f:
            json.dump(payload, f, indent=4)
            
        tqdm.write(f"  -> Saved: {file_path.name} | Cost: {solution.get('cost_eur_per_hr', 'N/A')}")

if __name__ == "__main__":
    test_kwargs = {
        "formulation": "dc_theta",  # Switch back to sparse B-Theta
        "enforce_line_limits": True,
        "mw_precision": 1.0,         
        
        # --- SOLVER TUNING ---
        "num_reads": 500,        
        "num_sweeps": 10000,
        "trotter_slices": 8
    }
    
    # Run the classical stochastic solver 10 times
    run_benchmark(target_grid="case5", target_solver="dc_qsa", max_time=1800, **test_kwargs)