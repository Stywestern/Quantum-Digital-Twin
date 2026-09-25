import itertools
from tqdm import tqdm
from scripts.pf_opf_annealing_benchmark import run_pipeline

# Import your grid generators
import pandapower.networks as pn
from homemade_grids.small_grids import case3_low_gen, case4_high_gen

if __name__ == "__main__":
    
    # 1. Define the Grids
    grid_generators = [
        #case3_low_gen,
        #case4_high_gen,
        pn.case5,
        pn.case9,
        pn.case14
    ]
    
    # 2. Define the parameter space (Dropped theta, added 5.0)
    formulations = ["dc_ptdf"]
    encodings = ["radix", "hybrid"]
    precisions = [10.0, 5.0, 1.0]
    
    # 3. Define the SQA Hyperparameters
    num_reads = 500
    num_sweeps = 2000
    trotter_slices = 16
    
    total_runs = len(grid_generators) * len(formulations) * len(encodings) * len(precisions)

    print("\n" + "="*80)
    print(f" INITIATING BATCH SWEEP: {total_runs} COMBINATIONS ".center(80, "="))
    print("="*80 + "\n")

    # Initialize tqdm progress bar
    with tqdm(total=total_runs, desc="Sweeping", unit="run", dynamic_ncols=True) as pbar:
        
        # 4. Outer loop for Grids
        for grid_func in grid_generators:
            
            # 5. Inner loop for configurations
            for form, enc, prec in itertools.product(formulations, encodings, precisions):
                
                # Generate a fresh copy of the grid for each run
                net = grid_func()
                grid_name = net.name if hasattr(net, 'name') and net.name else 'Unknown'
                
                # Update progress bar info text dynamically
                pbar.set_postfix(grid=grid_name[:10], form=form[-4:], enc=enc[:3], prec=prec)
                
                config = {
                    "formulation": form,
                    "encoding": enc,
                    "mw_precision": prec,
                    "penalty_safety": 1.2,
                    "ptdf_threshold": 0.05,
                    "hybrid_chunk_size": 50.0 
                }
                
                try:
                    # Pass the hyperparameters to the pipeline
                    run_pipeline(config, net, num_reads, num_sweeps, trotter_slices)
                except Exception as e:
                    # Use tqdm.write so error messages don't break the progress bar UI
                    tqdm.write(f"\n[!] RUN FAILED: Grid={grid_name}, Form={form}, Enc={enc}, Prec={prec}")
                    tqdm.write(f"    Error: {str(e)}")
                    
                # Advance the progress bar by 1
                pbar.update(1)
                
    print("\n" + "="*80)
    print(" BATCH SWEEP COMPLETE. ALL DATA SAVED TO JSON. ".center(80, "="))
    print("="*80 + "\n")