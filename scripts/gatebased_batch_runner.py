import warnings
import logging
import os
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger().setLevel(logging.ERROR)

import itertools
from tqdm import tqdm

# Import the QAOA pipeline runner instead of the SQA one
# Adjust the import path based on your exact file structure
from scripts.opf_gatebased_benchmark import run_pipeline

# Import your grid generators
import pandapower.networks as pn
from homemade_grids.small_grids import case3_low_gen, case3_high_gen, case4_low_gen, case4_high_gen

if __name__ == "__main__":
    
    # 1. Define the Grids
    grid_generators = [
        # --- Micro / Fundamental Verification (Safe for QAOA Emulation) ---
        case3_low_gen,
        case4_low_gen,
        case3_high_gen,
        case4_high_gen,
        pn.case5,          # 5 buses
        
        # --- WARNING: The following grids will likely cause OOM errors ---
        # Statevector tracking is 2^N. Density matrix emulation is 4^N.
        # Unless mw_precision is set extremely high to minimize logical qubits, 
        # these will exceed the 12-qubit emulator limit.
        
        pn.case6ww,        # 6-bus standard case
        pn.case9,          # 9 buses
        pn.case14,         # 14 buses
    ]
    
    # 2. Define the parameter space
    formulations = ["dc_ptdf"] # , "dc_theta"
    encodings = ["radix"] # , "unary"
    precisions = [50.0, 20.0, 10.0]
    
    # 3. Define Tiered QAOA Hyperparameters based on grid size
    def get_qaoa_hyperparameters(net):
        n_buses = len(net.bus)
        if n_buses <= 4:
            # Micro grids: We can afford deeper circuits and more optimizer restarts
            return {"p_sweep": (1, 2, 3), "shots": 2000, "n_restarts": 15, "method": "Powell"}
        else:
            # Pushing the limit: Keep p-depth and restarts low to prevent execution time from exploding
            return {"p_sweep": (1, 2), "shots": 1000, "n_restarts": 5, "method": "Powell"}
    
    total_runs = len(grid_generators) * len(formulations) * len(encodings) * len(precisions)

    print("\n" + "="*80)
    print(f" INITIATING QAOA BATCH SWEEP: {total_runs} COMBINATIONS ".center(80, "="))
    print("="*80 + "\n")

    # Initialize tqdm progress bar
    with tqdm(total=total_runs, desc="Sweeping", unit="run", dynamic_ncols=True) as pbar:
        
        # 4. Outer loop for Grids
        for grid_func in grid_generators:
            
            # Generate a fresh copy of the grid to evaluate its size
            net = grid_func()
            grid_name = net.name if hasattr(net, 'name') and net.name else 'Unknown'
            
            # Fetch dynamic QAOA hyperparameters
            hyperparams = get_qaoa_hyperparameters(net)
            current_p_sweep = hyperparams["p_sweep"]
            current_shots = hyperparams["shots"]
            current_restarts = hyperparams["n_restarts"]
            current_method = hyperparams["method"]
            
            # 5. Inner loop for configurations
            for form, enc, prec in itertools.product(formulations, encodings, precisions):
                
                # Regenerate the grid to ensure state isolation between runs
                net = grid_func()
                
                # Enforce standard rule: prevent external grid from exporting power
                if hasattr(net, 'ext_grid') and not net.ext_grid.empty:
                    net.ext_grid['min_p_mw'] = 0.0
                
                # Update progress bar info text dynamically
                pbar.set_postfix(grid=grid_name[:10], form=form[-4:], enc=enc[:3], prec=prec)
                
                config = {
                    "formulation": form,
                    "encoding": enc,
                    "mw_precision": prec,
                    "penalty_safety": 1.1,
                    "ptdf_threshold": 0.05,
                }
                
                try:
                    # Pass the dynamic QAOA hyperparameters to the pipeline
                    run_pipeline(
                        formulator_config=config, 
                        pristine_net=net, 
                        p_sweep=current_p_sweep, 
                        shots=current_shots, 
                        n_restarts=current_restarts,
                        method=current_method,
                        hw_preset="ibm_eagle"
                    )
                except Exception as e:
                    # Use tqdm.write so error messages don't break the progress bar UI
                    tqdm.write(f"\n[!] RUN FAILED: Grid={grid_name}, Form={form}, Enc={enc}, Prec={prec}")
                    tqdm.write(f"    Error: {str(e)}")
                    
                # Advance the progress bar by 1
                pbar.update(1)
                
    print("\n" + "="*80)
    print(" BATCH SWEEP COMPLETE. ALL DATA SAVED TO JSON. ".center(80, "="))
    print("="*80 + "\n")