import meep_functions as mf
import time

# ----------------------------------------------------------------------
# 1. SET YOUR CHAMPION PARAMETERS
# ----------------------------------------------------------------------
#
# AFTER 'optimize.py' FINISHES:
# Manually enter the best geometry you found here.
#
# (These are just example values)
CHAMPION_DIAMETER = 0.130  # 130 nm
CHAMPION_SPACING = 0.210  # 210 nm

# ----------------------------------------------------------------------
# 2. (Optional) INCREASE SIMULATION QUALITY
# ----------------------------------------------------------------------
#
# You can optionally edit 'meep_functions.py' one last time
# to increase the quality for this final run.
#
# Open 'meep_functions.py' and find these variables:
#
# - 'resolution = 30'  -> Change to a high value like 80 or 100.
# - 'sim.run(until_after_sources=150)' -> Change to a high value like 300.
#
# This will make the simulation very slow, but very accurate.
#
print(f"--- Starting FINAL run for champion geometry ---")
print(f"    Diameter: {CHAMPION_DIAMETER*1000} nm")
print(f"    Spacing:  {CHAMPION_SPACING*1000} nm")
print("    (Remember to increase resolution in meep_functions.py for best results!)")

start_time = time.time()

# ----------------------------------------------------------------------
# 3. RUN THE SIMULATION & GENERATE FILES
# ----------------------------------------------------------------------

# Step 1: Run the high-resolution Meep simulation
# This creates the final 'field_data_...npz' file
mf.run_meep_simulation(diameter=CHAMPION_DIAMETER, spacing=CHAMPION_SPACING)

# Step 2: Process the .npz file and save the final .dat file for SCAPS
# This creates 'generation_profile_for_scaps.dat' and 'generation_profile_plot.png'
mf.save_champion_profile(diameter=CHAMPION_DIAMETER, spacing=CHAMPION_SPACING)

end_time = time.time()
print(f"\n--- Champion file generation complete! ---")
print(f"    Total time: {(end_time - start_time) / 60:.2f} minutes")
print("\nNext step: Load 'generation_profile_for_scaps.dat' into SCAPS-1D.")