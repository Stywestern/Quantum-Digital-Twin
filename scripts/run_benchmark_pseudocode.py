import json
from pathlib import Path
import pandapower.networks as nw

# Import solver classes
from solvers.classical_ac import ClassicalACSolver
from solvers.simulated_annealing import SASolver

# Dictionaries eliminate the need for massive if/elif chains
SOLVER_REGISTRY = {
    "ac_opf": ClassicalACSolver,
    "sa": SASolver
}

GRID_REGISTRY = {
    "case14": nw.case14,
    "case39": nw.case39,
    "case118": nw.case118
}

if __name__ == "__main__":
    # 1. Configuration (Eventually, you can pass these via command line arguments)
    TARGET_GRID = "case14"
    TARGET_SOLVER = "ac_opf"
    MAX_TIME = 1800 # seconds
    
    # 2. Load Problem
    if TARGET_GRID not in GRID_REGISTRY:
        raise ValueError(f"Grid {TARGET_GRID} not found.")
    
    # We pass the entire Pandapower object. The specific solver class 
    # handles extracting the vars/consts it specifically needs.
    net = GRID_REGISTRY[TARGET_GRID]()

    # 3. Solve the problem dynamically
    if TARGET_SOLVER not in SOLVER_REGISTRY:
        raise ValueError(f"Solver {TARGET_SOLVER} not found.")
        
    solver = SOLVER_REGISTRY[TARGET_SOLVER](max_time=MAX_TIME)
    
    # Every solver must guarantee it returns this exact tuple format
    solution, metadata = solver.solve(net)

    # 4. Save Output
    save_dir = Path(f"output/{TARGET_GRID}")
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"{TARGET_SOLVER}.json"

    payload = {
        "grid": TARGET_GRID,
        "solver": TARGET_SOLVER,
        "max_time_allowed": MAX_TIME,
        "solution": solution,
        "metadata": metadata
    }

    with open(file_path, "w") as f:
        json.dump(payload, f, indent=4)
        
    print(f"Benchmark saved to {file_path}")