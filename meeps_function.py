import meep as mp
import numpy as np

# ----------------------------------------------------------------------
# GLOBAL PARAMETERS & MATERIAL DEFINITIONS
# ----------------------------------------------------------------------
# These are defined globally so both functions can access them.
# All units are in micrometers (µm)

wl = 1.5      # 1500 nm wavelength
freq = 1 / wl

# --- Device Geometry
tio2_thick = 0.06  # 60 nm
si_thick = 2.0     # We only need to simulate a few µm of Si for FDTD
air_thick = 1.0    # Spacing above device
pml_thick = 1.0    # Absorbing boundary layer thickness

# --- Simulation Resolution
# Start with a lower value (e.g., 30) for faster optimization sweeps.
# Use a high value (e.g., 50-100) for your final "champion" run.
resolution = 30 # pixels/µm

# ----------------------------------------------------------------------
# --- MATERIAL DEFINITIONS at 1500 nm ---
# ----------------------------------------------------------------------

# 1. Gold (Au) (n=0.55, k=11.5)
n_au = 0.55
k_au = 11.5
eps_au_real = n_au**2 - k_au**2
eps_au_imag = 2 * n_au * k_au
Au_medium = mp.Medium(epsilon=eps_au_real,
                     D_conductivity=2 * np.pi * freq * eps_au_imag / eps_au_real)

# 2. Titanium Dioxide (TiO2) (e.g., n~2.43, k=0)
n_tio2 = 2.43
TiO2_medium = mp.Medium(index=n_tio2)

# 3. Silicon (n-Si)
# !! CRITICAL !! You must replace k_si with your researched value.
n_si = 3.48
k_si = 1e-4  # PLACEHOLDER for n-Si free-carrier absorption
# ----------------------------------------------------
eps_si_real = n_si**2 - k_si**2
eps_si_imag = 2 * n_si * k_si
Si_medium = mp.Medium(epsilon=eps_si_real,
                     D_conductivity=2 * np.pi * freq * eps_si_imag / eps_si_real)

# ----------------------------------------------------------------------
# FUNCTION 1: RUN MEEP SIMULATION
# ----------------------------------------------------------------------
def run_meep_simulation(diameter, spacing):
    """
    Runs a 2D Meep simulation for a given Au nanoparticle diameter and spacing.
    Saves the E-field data in the silicon layer to an .npz file.
    
    Args:
        diameter (float): Nanoparticle diameter in µm.
        spacing (float): Nanoparticle spacing (cell width) in µm.
    """
    
    # --- 1. Set up Cell & Geometry ---
    cell_x = spacing
    cell_z = pml_thick + air_thick + tio2_thick + si_thick + pml_thick
    
    # Calculate Z-positions
    z_si_center = 0.5 * cell_z - pml_thick - 0.5 * si_thick
    z_tio2_center = z_si_center + 0.5 * si_thick + 0.5 * tio2_thick
    z_au_center = z_tio2_center + 0.5 * tio2_thick

    geometry = [
        mp.Block(center=mp.Vector3(0, z_si_center),
                 size=mp.Vector3(mp.inf, si_thick), material=Si_medium),
        mp.Block(center=mp.Vector3(0, z_tio2_center),
                 size=mp.Vector3(mp.inf, tio2_thick), material=TiO2_medium),
        mp.Cylinder(center=mp.Vector3(0, z_au_center),
                    radius=0.5 * diameter, height=mp.inf, material=Au_medium)
    ]
    
    # --- 2. Set up Boundaries & Source ---
    boundary_layers = [
        mp.PeriodicBoundary(direction=mp.X),
        mp.PML(thickness=pml_thick, direction=mp.Z, side=mp.High),
        mp.PML(thickness=pml_thick, direction=mp.Z, side=mp.Low)
    ]

    src_z_pos = 0.5 * cell_z - pml_thick - 0.5 * air_thick
    sources = [
        mp.Source(
            mp.GaussianSource(frequency=freq, fwidth=0.1 * freq),
            component=mp.Ex,
            center=mp.Vector3(0, src_z_pos),
            size=mp.Vector3(cell_x, 0)
        )
    ]
    
    # --- 3. Set up Simulation Object ---
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_x, cell_z),
        geometry=geometry,
        sources=sources,
        boundary_layers=boundary_layers,
        resolution=resolution,
        default_material=mp.Medium(index=1.0) # Air
    )
    
    # --- 4. Define Field Monitor ---
    # Monitor only the silicon region
    monitor_vol_size = mp.Vector3(cell_x, si_thick)
    monitor_vol_center = mp.Vector3(0, z_si_center)
    
    dft_monitor = sim.add_dft_fields(
        [mp.Ex, mp.Ey, mp.Ez],
        freq, 0, 1,
        center=monitor_vol_center,
        size=monitor_vol_size
    )
    
    # --- 5. Run Simulation ---
    # A shorter runtime is OK for optimization sweeps.
    # Increase 'until_after_sources' for the final, high-res run.
    sim.run(until_after_sources=150) 
    
    # --- 6. Get & Save Field Data ---
    Ex = sim.get_dft_array(dft_monitor, mp.Ex)
    Ey = sim.get_dft_array(dft_monitor, mp.Ey)
    Ez = sim.get_dft_array(dft_monitor, mp.Ez)
    Eps = sim.get_epsilon_array(dft_monitor) # Real part of epsilon
    
    # Generate a unique filename
    output_filename = f'field_data_diam_{diameter:.4f}_space_{spacing:.4f}.npz'
    
    np.savez(output_filename,
             Ex=Ex, Ey=Ey, Ez=Ez, Eps=Eps,
             si_thick_um=si_thick,
             cell_x_um=cell_x,
             resolution=resolution)
    
    print(f"    ... Meep sim complete. Data saved to {output_filename}")
    
    return True

# ----------------------------------------------------------------------
# FUNCTION 2: PROCESS FIELD DATA
# ----------------------------------------------------------------------
def process_field_data(diameter, spacing):
    """
    Loads an .npz file (created by run_meep_simulation) and calculates
    the total integrated optical absorption within the silicon layer.
    
    Args:
        diameter (float): Nanoparticle diameter in µm.
        spacing (float): Nanoparticle spacing (cell width) in µm.
        
    Returns:
        float: Total absorption in the silicon (arbitrary units).
    """
    
    # --- 1. Load Data ---
    filename = f'field_data_diam_{diameter:.4f}_space_{spacing:.4f}.npz'
    try:
        data = np.load(filename)
    except FileNotFoundError:
        print(f"    ERROR: Could not find file {filename}")
        return 0.0

    Ex = data['Ex']
    Ey = data['Ey']
    Ez = data['Ez']
    Eps = data['Eps'] # Real part of epsilon
    
    # --- 2. Calculate 2D Absorption Map ---
    omega = 2 * np.pi * freq
    
    # Calculate |E|²
    E_sq = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
    
    # Create a 2D map of Im(ε) for silicon
    # We identify silicon by its *real* epsilon value
    is_silicon = np.isclose(Eps, eps_si_real, rtol=1e-2)
    Img_Eps = np.where(is_silicon, eps_si_imag, 0)
    
    # Calculate Power Absorption Density P_abs(x,z)
    # P_abs = 0.5 * ω * Im(ε) * |E|²
    P_abs_map = 0.5 * omega * Img_Eps * E_sq
    
    # --- 3. Get 1D Profile & Total Absorption ---
    
    # Average P_abs(x,z) along the x-axis (axis=0) to get P_abs(z)
    P_abs_z_profile = np.mean(P_abs_map, axis=0)
    
    # The metric for optimization is the *total* absorption in the Si
    total_si_absorption = np.sum(P_abs_z_profile)
    
    print(f"    ... Processing complete. Total Si Absorption: {total_si_absorption:.4e}")
    
    return total_si_absorption


# ----------------------------------------------------------------------
# FUNCTION 3: SAVE CHAMPION PROFILE (You will call this *after* optimization)
# ----------------------------------------------------------------------
def save_champion_profile(diameter, spacing):
    """
    This is a standalone function to be run *once* for the best geometry.
    It loads the .npz file, calculates the G(z) profile,
    and saves it to 'generation_profile_for_scaps.dat'.
    It also generates a plot.
    """
    
    print(f"--- Generating final profile for champion: D={diameter*1000}nm, S={spacing*1000}nm ---")
    
    # --- 1. Load Data ---
    filename = f'field_data_diam_{diameter:.4f}_space_{spacing:.4f}.npz'
    try:
        data = np.load(filename)
    except FileNotFoundError:
        print(f"    ERROR: Could not find file {filename}.")
        print("    Please run run_meep_simulation() again for this geometry (with high resolution).")
        return

    Ex = data['Ex']
    Ey = data['Ey']
    Ez = data['Ez']
    Eps = data['Eps']
    si_thick_um = data['si_thick_um']
    
    # --- 2. Calculate Absorption Profile ---
    omega = 2 * np.pi * freq
    E_sq = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
    is_silicon = np.isclose(Eps, eps_si_real, rtol=1e-2)
    Img_Eps = np.where(is_silicon, eps_si_imag, 0)
    
    P_abs_map = 0.5 * omega * Img_Eps * E_sq
    P_abs_z_profile = np.mean(P_abs_map, axis=0)
    
    # --- 3. Create Z-coordinates and Save File ---
    
    # Create z-axis coordinates (in nm)
    # The profile starts at the top of the silicon layer (z=0)
    num_pts = P_abs_z_profile.shape[0]
    z_coords_nm = np.linspace(0, si_thick_um * 1000, num_pts)

    # SCAPS format: [depth (nm)] [generation (arb. units)]
    output_filename = 'generation_profile_for_scaps.dat'
    output_data = np.stack((z_coords_nm, P_abs_z_profile), axis=1)
    
    np.savetxt(output_filename, output_data,
               fmt='%.6e',
               header='Depth(nm)  Generation(arb. units)',
               delimiter='  ')

    print(f"    Success! Champion profile saved to {output_filename}")
    
    # --- 4. Plot the Profile ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(z_coords_nm, P_abs_z_profile)
        plt.xlabel("Depth into Silicon (nm)")
        plt.ylabel("Optical Generation G(z) (Arbitrary Units)")
        plt.title(f"Champion Generation Profile (D={diameter*1000:.0f}nm, S={spacing*1000:.0f}nm)")
        plt.grid(True)
        plt.yscale('log') # Use log scale to see penetration
        plt.savefig('generation_profile_plot.png')
        print(f"    Plot saved to generation_profile_plot.png")
    except ImportError:
        print("    (Skipping plot: matplotlib not found. Install with 'pip install matplotlib')")

    return