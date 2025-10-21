import numpy as np
import os
import time

# Import the functions you just made
# (You might need to put them in a separate file, e.g., 'meep_functions.py')
from meep_functions import run_meep_simulation, process_field_data

# 1. DEFINE YOUR SEARCH SPACE (in micrometers)
diams_um = np.linspace(0.05, 0.25, num=20)  # 50nm to 250nm, 20 steps
spaces_um = np.linspace(0.1, 0.4, num=15)   # 100nm to 400nm, 15 steps

# 2. CREATE A GRID TO STORE RESULTS
results_grid = np.zeros((len(diams_um), len(spaces_um)))

# 3. RUN THE PARAMETER SWEEP
print("--- Starting Optimization Sweep ---")
start_time = time.time()

for i, diam in enumerate(diams_um):
    for j, space in enumerate(spaces_um):
        
        # Check if spacing is valid (must be > diameter)
        if space <= diam:
            results_grid[i, j] = 0 # Invalid geometry
            continue

        print(f"Running: Diam={diam*1000:.0f}nm, Space={space*1000:.0f}nm")
        
        # --- PHASE 1A: Run Meep ---
        # This function runs the Meep sim and saves the .npz file
        # You need to modify run_simulation.py for this
        run_meep_simulation(diameter=diam, spacing=space)

        # --- PHASE 1B: Process Data ---
        # This function loads the .npz file, calculates P_abs(z),
        # and returns the *total* absorption in the silicon.
        total_si_absorption = process_field_data(diameter=diam, spacing=space)
        
        # 4. STORE THE RESULT
        results_grid[i, j] = total_si_absorption
        
        # Clean up the large .npz file to save disk space
        os.remove(f'field_data_diam_{diam}_space_{space}.npz')

# 4. ANALYSIS
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