import numpy as np
import os
import time
import multiprocessing
from itertools import product

# Import your meep functions
import meep_functions as mf

# -----------------------------------------------------------------
# 1. DEFINE A SINGLE "WORKER" FUNCTION
# -----------------------------------------------------------------
# This function is what each of your CPU cores will run.
# It simulates one geometry and returns the result.

def run_one_simulation(params):
    """
    Worker function for a single (diam, space) simulation.
    """
    diam, space = params
    
    # Check if spacing is valid (must be > diameter)
    if space <= diam:
        return (diam, space, 0) # Return 0 for absorption
        
    print(f"--- Starting: Diam={diam*1000:.0f}nm, Space={space*1000:.0f}nm")
    
    # --- PHASE 1A: Run Meep ---
    mf.run_meep_simulation(diameter=diam, spacing=space)

    # --- PHASE 1B: Process Data ---
    total_si_absorption = mf.process_field_data(diameter=diam, spacing=space)
    
    # --- Clean up ---
    try:
        # Generate the filename to delete it
        filename = f'field_data_diam_{diam:.4f}_space_{space:.4f}.npz'
        os.remove(filename)
    except OSError as e:
        print(f"    Warning: Could not remove file {filename}: {e}")
        
    print(f"--- Finished: Diam={diam*1000:.0f}nm, Absorp={total_si_absorption:.4e}")
    
    # Return the result so we can build the grid later
    return (diam, space, total_si_absorption)

# -----------------------------------------------------------------
# 2. MAIN SCRIPT
# -----------------------------------------------------------------
if __name__ == "__main__":
    
    print("--- Starting Optimization Sweep ---")
    start_time = time.time()
    
    # 1. DEFINE YOUR SEARCH SPACE (in micrometers)
    diams_um = np.linspace(0.05, 0.25, num=20)  # 50nm to 250nm, 20 steps
    spaces_um = np.linspace(0.1, 0.4, num=15)   # 100nm to 400nm, 15 steps
    
    # 2. CREATE A LIST OF ALL JOBS TO RUN
    # This creates a list of all (diam, space) tuples
    job_list = list(product(diams_um, spaces_um))
    print(f"Total simulations to run: {len(job_list)}")

    # 3. SET UP THE MULTIPROCESSING POOL
    # This will use all available cores. 
    # You can set a specific number, e.g., Pool(processes=8)
    num_cores = multiprocessing.cpu_count()
    print(f"Running on {num_cores} CPU cores...")
    pool = multiprocessing.Pool(processes=num_cores)

    # 4. RUN THE POOL
    # pool.map() runs the 'run_one_simulation' function on every
    # item in the 'job_list' in parallel.
    # 'results' will be a list of (diam, space, absorption) tuples
    results = pool.map(run_one_simulation, job_list)

    # 5. CLOSE THE POOL
    pool.close()
    pool.join()
    
    print("\n--- All simulations complete. Assembling results grid. ---")
    
    # 6. PROCESS THE RESULTS INTO A GRID
    # Create a dictionary for easy lookup
    results_map = { (r[0], r[1]): r[2] for r in results }
    
    results_grid = np.zeros((len(diams_um), len(spaces_um)))
    for i, diam in enumerate(diams_um):
        for j, space in enumerate(spaces_um):
            results_grid[i, j] = results_map.get((diam, space), 0)

    # 7. ANALYSIS
    end_time = time.time()
    print(f"--- Sweep Complete in {(end_time-start_time)/3600:.2f} hours ---")

    # Save your final results map
    np.savez('optimization_results.npz', 
             diams=diams_um, 
             spaces=spaces_um, 
             absorption_map=results_grid)

    # Find the best geometry
    max_val = np.max(results_grid)
    max_idx = np.unravel_index(np.argmax(results_grid), results_grid.shape)
    best_diam = diams_um[max_idx[0]]
    best_space = spaces_um[max_idx[1]]

    print(f"\n--- OPTICAL MAXIMUM FOUND ---")
    print(f"Best Diameter: {best_diam*1000:.1f} nm")
    print(f"Best Spacing:  {best_space*1000:.1f} nm")
    print(f"Absorption Value (arb. units): {max_val}")