import pandas as pd
import numpy as np
import agama  
import matplotlib.pyplot as plt
from pylab import *
import os
from pathlib import Path
from scipy.stats import kstest,ks_2samp,chi2

import scienceplots

def calculate_kinetic_energy_tensor(df):
  """
  Calculates the kinetic energy tensor from particle velocity data in a text file.

  Args:
    df: panda data frame that contain x y z vx vy vz.

  Returns:
    A 3x3 NumPy array representing the kinetic energy tensor.
  """
 
  # Read the data from the text file, skipping the first few rows if they are not data
  # Assuming columns 3, 4, and 5 are velocity components (vx, vy, vz)

  # Convert velocity columns to numeric, coercing errors
  for col in [3, 4, 5]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values that resulted from coercion in velocity columns
  df.dropna(subset=[3, 4, 5], inplace=True)

    # Extract velocity components
  vx = df[3].values
  vy = df[4].values
  vz = df[5].values

    # Assuming equal mass for all particles (M/N is constant)
    # We can calculate the tensor scaled by 1/2 * M/N later if needed,
    # or work with the unscaled tensor which represents the sum of (v_i * v_j) for all particles.
    # For now, we will calculate the unscaled tensor (sum of outer products).

  kinetic_tensor = np.zeros((3, 3))

  for i in range(len(vx)):
      velocity_vector = np.array([vx[i], vy[i], vz[i]])
      outer_product = np.outer(velocity_vector, velocity_vector)
      kinetic_tensor += outer_product

    # If you have the total mass (M) and number of particles (N),
    # you can scale the tensor:
    # M = your_total_mass
    # N = len(vx)
    # scaled_kinetic_tensor = (0.5 * M / N) * kinetic_tensor
    # return scaled_kinetic_tensor
    # For now, returning the unscaled tensor:
  print(f'K_tensor:{kinetic_tensor}')
  return kinetic_tensor

def diagonalize_kinetic_energy_tensor(K_tensor):
    """
    Diagonalize the kinetic energy tensor and construct a right-handed
    orthonormal eigenbasis such that the eigenvector corresponding to the
    largest eigenvalue is aligned with the z'-axis.

    Parameters
    ----------
    K_tensor : ndarray
        3x3 kinetic energy tensor.

    Returns
    -------
    eigenvalues : ndarray
        Sorted eigenvalues (ascending).
    eigenvectors : ndarray
        3x3 orthonormal rotation matrix whose columns correspond to the
        x', y', z' axes (right-handed coordinate system).
    """
    # Diagonalize (K is symmetric, so use eigh)
    eigenvalues, eigenvectors = np.linalg.eigh(K_tensor)

    # Sort eigenvalues and eigenvectors (ascending order)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Extract eigenvectors by ascending eigenvalue
    z_axis = eigenvectors[:, np.argmax(eigenvalues)]
    x_axis = eigenvectors[:, np.argmin(eigenvalues)]
    y_axis = np.cross(z_axis,x_axis)

    # Combine into rotation matrix
    R = np.column_stack((x_axis, y_axis, z_axis))

    #checking the determinant is positve
    if np.linalg.det(R) < 0:
        print("Error det (R) < 0 !.")

    print("Eigenvalues (ascending):", eigenvalues)
    print("Rotation matrix (columns = x', y', z'):")
    print(R)
    

    return eigenvalues, R

def transform_coordinates_to_eigenbasis(data_df, eigenvectors):
  """
  Transforms particle position and/or velocity coordinates into the basis formed by eigenvectors.

  Args:
    data_df: pandas DataFrame of particle coordinates (e.g., [x, y, z] or [vx, vy, vz] columns).
    eigenvectors: A 3x3 NumPy array where each column is an eigenvector.

  Returns:
    A pandas DataFrame or array containing the transformed coordinates.
  """
  if data_df is None or eigenvectors is None:
    print("Error: Input data or eigenvectors are None.")
    return None

  try:
    
    data = data_df.values
   
    # Perform the change of basis: P' = P * V, where P is the original data
    # and V is the matrix of eigenvectors (columns are eigenvectors).
    # The dot product of each vector with the eigenvector matrix
    # gives the coordinates in the new basis.
    transformed_data = np.dot(data, eigenvectors)

    # If input was a DataFrame, return a DataFrame with appropriate column names
  
    # Determine appropriate column names based on input column count
    if data_df.shape[1] == 3:
        col_names = [0, 1, 2]
    else:
        print('Wrong format of data frame it should [X_COL, Y_COL, Z_COL,] data_df.shape[1] = 3')

    transformed_df = pd.DataFrame(transformed_data, columns=col_names)
    print(f'transformed_coordinates{transformed_df}')
    return transformed_df

  except Exception as e:
    print(f"An error occurred during coordinate transformation: {e}")
    return None
 
def make_frame(file_path):
    """
    make a panda frame from the text file frame
    ----------
    file_path:path to frame text file, string type variable.

    Returns
    -------
      panda frame of the  coordinates 
    """
    df = pd.read_csv(file_path, sep='\\s+', header=None, skiprows=1)
    df.dropna(subset=[0, 1, 2, 3, 4, 5], inplace=True) # Ensure all relevant columns are not NaN
    positions_df = df[[0, 1, 2]]
    velocities_df = df[[3, 4, 5]] # Extract velocity columns

    return df, positions_df, velocities_df

def get_potential(transformed_positions_df):
    """
    Create a frozen AGAMA potential from a phase-space DataFrame or array.

    Parameters
    ----------
    transformed_positions_df, transformed_velocities_df : pandas.DataFrame  with columns:
        [x', y', z'] where x' is 1 by N  and similarly 

    Returns
    -------
    potential : agama.Potential
        Frozen potential constructed from particle positions and masses.
    """

    # --- Handle the pandas DataFrame ---
    
     #--- Combine into phase-space array ---
     #transform the dataframe into ndarray this format [[x1',y1',z1'],[x2',y2',z2'],[...],...]
    positions = transformed_positions_df[[0,1,2]].values
    print(positions)
    
    # --- Create equal-mass array ---
    num_particles = len(positions)
    if num_particles == 0:
        raise ValueError("No particles found for potential calculation.")

    mass_per_particle = 1.0 / num_particles
    mass_array = np.full(num_particles, mass_per_particle)

    print(f"N = {num_particles}")
    print("Mass array successfully created (equal-mass particles).")


    # --- Construct the frozen AGAMA potential ---
    potential = agama.Potential(
        type="Multipole",
        particles=(positions, mass_array),
        symmetry="axisymmetric"
    )

    # --- sanity_checks ---
    print(f"Potential successfully created with {num_particles} particles.")
    print(f"Potential value at origin: {potential.potential(0, 0, 0):.5e}")

    return potential

def runOrbit(pot,initial_conditions,N_orbit=100,trajsize=3000000,acc=1e-8):

  orbit=agama.orbit(potential=pot, ic=initial_conditions, lyapunov=True,time=N_orbit*pot.Tcirc(initial_conditions), trajsize=trajsize,accuracy=acc)


  return orbit


# -----------------------------------------------------------
# Surface of Section plotting function (modified to save files)
# -----------------------------------------------------------

def sos_plot(orb, lyp, outdir=None, prefix="", s=5, c='r'):
    """
    Create a 4-panel diagnostic plot:
        1. Surface of section (R, vR)
        2. X-Y projection
        3. X-Z projection
        4. R-Z projection

    Parameters
    ----------
    orb : ndarray
        Orbit array with columns [x, y, z, vx, vy, vz].
    lyp : float
        Lyapunov exponent for the orbit.
    outdir : str, optional
        Directory where the plot is saved.
    prefix : str, optional
        Filename prefix.
    """

    # ---------------------------------------
    # Precompute quantities
    # ---------------------------------------
    x  = orb[:, 0]
    y  = orb[:, 1]
    z  = orb[:, 2]
    vx = orb[:, 3]
    vy = orb[:, 4]
    vz = orb[:, 5]

    R = np.sqrt(x**2 + y**2)
    vR = (vx*x + vy*y) / np.where(R == 0, 1e-12, R)

    # ---------------------------------------
    # Surface of section detection
    # ---------------------------------------
    R_sos = []
    vR_sos = []

    # Detect z-crossings from negative to positive
    for i in range(1, len(orb)):
        if orb[i, 2] > 0 and orb[i-1, 2] < 0:
            # Linear interpolation to z=0
            f = -orb[i-1, 2] / (orb[i, 2] - orb[i-1, 2])
            rv = orb[i-1] + f * (orb[i] - orb[i-1])
            R0 = np.sqrt(rv[0]**2 + rv[1]**2)
            vR0 = (rv[3]*rv[0] + rv[4]*rv[1]) / R0
            R_sos.append(R0)
            vR_sos.append(vR0)

    # ---------------------------------------
    # Create 4-panel figure
    # ---------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    ax1, ax2, ax3, ax4 = axs.flatten()

    # ========= 1. Surface of Section (R vs vR) =========
    ax1.scatter(R_sos, vR_sos, s=s, c=c)
    ax1.set_xlabel("R")
    ax1.set_ylabel("vR")
    ax1.set_title("Surface of Section (z=0 crossing)")

    if lyp > 0:
        ax1.text(0.98, 0.98, rf"$\lambda = {lyp:.6e}$",
                 ha='right', va='top', transform=ax1.transAxes)

    # ========= 2. X-Y Projection =========
    ax2.plot(x, y, lw=0.6, color='steelblue')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Orbit Projection: X vs Y")
    ax2.set_aspect('equal', 'box')

    # ========= 3. X-Z Projection =========
    ax3.plot(x, z, lw=0.6, color='darkgreen')
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.set_title("Orbit Projection: X vs Z")
    ax3.set_aspect('equal', 'box')

    # ========= 4. R-Z Projection =========
    ax4.plot(R, z, lw=0.6, color='purple')
    ax4.set_xlabel("R")
    ax4.set_ylabel("Z")
    ax4.set_title("Orbit Projection: R vs Z")
    ax4.set_aspect('equal', 'box')

    plt.tight_layout()

    # ---------------------------------------
    # Save or show
    # ---------------------------------------
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        filename = f"{prefix}_combined.png"
        filepath = os.path.join(outdir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# -----------------------------------------------------------
# Particle sampling function (unchanged)
# -----------------------------------------------------------

def sample_particles(transformed_positions_df, transformed_velocities_df, step=1024,start=0):
    """
    Uniformly sample every 'step'-th particle from an already transformed
    phase-space array (output of make_a_transformed_frame).

    Parameters
    ----------
    transformed_positions_df, transformed_velocities_df: panda data frame of transformed velocity and positions 
    both of them following this format:

                        0           1           2
        0         -0.004510   -0.015569    0.015375
        1         -0.028965    0.025020   -0.013821
        2         -0.001520    0.008577    0.030919
        3          0.015330    0.032071    0.020039
        4         -0.017020    0.041001    0.015149
        ...             ...         ...         ...
        1048571  -17.659141 -550.851059 -136.503597
        1048572 -278.163166 -119.769727  191.221427
        1048573  445.606453 -231.538013 -132.865564
        1048574 -517.436863 -134.563575  170.388198
        1048575   58.134793  -58.161596   75.768366 

    Returns
    -------
    sampled : pandas.DataFrame
        Uniformly sampled phase-space data with columns:
        ['x_prime', 'y_prime', 'z_prime', 'vx_prime', 'vy_prime', 'vz_prime', 'orig_index']
        where 'orig_index' is the particle’s original row number in the full transformed array.
    """

    # --- Convert to DataFrame if needed ---
    transformed_phase_space= pd.concat([transformed_positions_df, transformed_velocities_df], axis=1)
    if isinstance(transformed_phase_space, pd.DataFrame):
        # Ensure proper column names if user passed a DataFrame directly
        df = transformed_phase_space.copy()
        if df.shape[1] == 6:
            df.columns = ['x_prime', 'y_prime', 'z_prime', 'vx_prime', 'vy_prime', 'vz_prime']
            print(f"transformed_phase_space: {df}")
    else:
        raise TypeError("Input must be a NumPy array or pandas DataFrame with 6 columns.")

    # --- Sample every `step`-th particle ---
    sampled = df.iloc[start::step].copy()

    # --- Preserve the original row index ---
    sampled['orig_index'] = sampled.index.to_numpy()

    # Reset the DataFrame’s index to 0..N_sampled-1
    #sampled.reset_index(drop=True, inplace=True)#keep original index 

    print(f"Uniformly sampled {len(sampled)} particles (every {step} rows).")
    print("Columns:", sampled.columns.tolist())

    return sampled


# -----------------------------------------------------------
# Main run function (modified to organize and name outputs)
# -----------------------------------------------------------

def run_sample(sample, potential, step,frame):
    """
    Runs orbit integrations for a uniformly sampled set of particles,
    saves SOS plots, and logs diagnostic quantities.

    For each sampled orbit, saves:
        - index in the original file
        - Lyapunov exponent
        - specific energy [J/kg] = 0.5*v^2 + Phi
        - angular momentum magnitude |r x v|

    Chaotic orbits (positive Lyapunov exponent) are saved in a 'chaotic' subfolder.
    """
    lyp_list = []

    #  Main and subdirectories
    outdir = Path(f"sos_output/step{step}_frame{frame}")
    chaotic_dir = outdir / "chaotic"
    regular_dir = outdir / "regular"
    chaotic_dir.mkdir(parents=True, exist_ok=True)
    regular_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(sample)} orbits (sampled every {step} rows)...")

    # Store diagnostics in a list to save at the end
    diagnostics = []

    for idx, row in sample.iterrows():
        # --- Orbit integration ---
        orbit = runOrbit(
            potential,
            [row.x_prime, row.y_prime, row.z_prime,
             row.vx_prime, row.vy_prime, row.vz_prime]
        )
        lyap_exp = orbit[1][0]
        lyp_list.append(lyap_exp)

        # --- Compute specific energy ---
        r_vec = np.array([row.x_prime, row.y_prime, row.z_prime])
        v_vec = np.array([row.vx_prime, row.vy_prime, row.vz_prime])
        v2 = np.dot(v_vec, v_vec)

        # potential returns potential value phi(r) (J/kg if consistent units)
        Phi = potential.potential(r_vec)
        specific_energy = 0.5 * v2 + Phi

        # Compute angular momentum magnitude |r x v| 
        L_vec = np.cross(r_vec, v_vec)
        #L_mag = np.linalg.norm(L_vec)
        L_z = L_vec[2]

        # --- Build filename prefix ---
        prefix = (
            f"{idx:05d}_"
            f"x{row.x_prime:.3f}_y{row.y_prime:.3f}_z{row.z_prime:.3f}_"
            f"vx{row.vx_prime:.3f}_vy{row.vy_prime:.3f}_vz{row.vz_prime:.3f}"
        ).replace('-', 'm')

        # --- Choose save directory based on Lyapunov exponent ---
        save_dir = chaotic_dir if lyap_exp > 0 else regular_dir

        # Save SOS plot
        sos_plot(orbit[0][1], orbit[1][0], outdir=str(save_dir), prefix=prefix)

        # --- Store diagnostics ---
        diagnostics.append([idx, lyap_exp, specific_energy, L_z])

    #  Save Lyapunov + diagnostics
    diagnostics = np.array(diagnostics)
    summary_path = outdir / "lyapunov_summary.txt"
    header = "index  lyapunov_exp  specific_energy[J/kg]  angular_momentum"
    np.savetxt(summary_path, diagnostics, fmt=["%d", "%.6e", "%.6e", "%.6e"], header=header)
    print(f"Saved Lyapunov + energy + angular momentum summary to:\n  {summary_path}")

    # Final summary
    n_chaotic = np.sum(diagnostics[:, 1] > 0)
    print(f"Completed {len(sample)} orbits: {n_chaotic} chaotic, {len(sample)-n_chaotic} regular.")
    print(f"Results saved in '{outdir}/'.")

    return diagnostics

def run_sample_no_plot(sample, potential, step,frame):
    """
    Runs orbit integrations for a uniformly sampled set of particles,
    saves SOS plots, and logs diagnostic quantities.

    For each sampled orbit, saves:
        - index in the original file
        - Lyapunov exponent
        - specific energy [J/kg] = 0.5*v^2 + Phi
        - angular momentum magnitude |r x v|

    Chaotic orbits (positive Lyapunov exponent) are saved in a 'chaotic' subfolder.
    """
    lyp_list = []

    #  Main and subdirectories
    outdir = Path(f"sos_output/step{step}_frame{frame}")
    chaotic_dir = outdir / "chaotic"
    regular_dir = outdir / "regular"
    chaotic_dir.mkdir(parents=True, exist_ok=True)
    regular_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(sample)} orbits (sampled every {step} rows)...")

    # Store diagnostics in a list to save at the end
    diagnostics = []

    for idx, row in sample.iterrows():
        # --- Orbit integration ---
        orbit = runOrbit(
            potential,
            [row.x_prime, row.y_prime, row.z_prime,
             row.vx_prime, row.vy_prime, row.vz_prime]
        )
        lyap_exp = orbit[1][0]
        lyp_list.append(lyap_exp)

        # --- Compute specific energy ---
        r_vec = np.array([row.x_prime, row.y_prime, row.z_prime])
        v_vec = np.array([row.vx_prime, row.vy_prime, row.vz_prime])
        v2 = np.dot(v_vec, v_vec)

        # potential returns potential value phi(r) (J/kg if consistent units)
        Phi = potential.potential(r_vec)
        specific_energy = 0.5 * v2 + Phi

        # Compute angular momentum magnitude |r x v| 
        L_vec = np.cross(r_vec, v_vec)
        #L_mag = np.linalg.norm(L_vec)
        L_z = L_vec[2]

        # --- Build filename prefix ---
        prefix = (
            f"{idx:05d}_"
            f"x{row.x_prime:.3f}_y{row.y_prime:.3f}_z{row.z_prime:.3f}_"
            f"vx{row.vx_prime:.3f}_vy{row.vy_prime:.3f}_vz{row.vz_prime:.3f}"
        ).replace('-', 'm')

        # --- Choose save directory based on Lyapunov exponent ---
        save_dir = chaotic_dir if lyap_exp > 0 else regular_dir

        # Save SOS plot
        #sos_plot(orbit[0][1], orbit[1][0], outdir=str(save_dir), prefix=prefix)

        # --- Store diagnostics ---
        diagnostics.append([idx, lyap_exp, specific_energy, L_z])

    #  Save Lyapunov + diagnostics
    diagnostics = np.array(diagnostics)
    summary_path = outdir / "lyapunov_summary.txt"
    header = "index  lyapunov_exp  specific_energy[J/kg]  angular_momentum"
    np.savetxt(summary_path, diagnostics, fmt=["%d", "%.6e", "%.6e", "%.6e"], header=header)
    print(f"Saved Lyapunov + energy + angular momentum summary to:\n  {summary_path}")

    # Final summary
    n_chaotic = np.sum(diagnostics[:, 1] > 0)
    print(f"Completed {len(sample)} orbits: {n_chaotic} chaotic, {len(sample)-n_chaotic} regular.")
    print(f"Results saved in '{outdir}/'.")

    return diagnostics

def plot_potential (potential):
  """
  # Calculate potential along each axis
  description: plot phi vs position
  args:
      potential object from agama
  return: none 
  """
  try:
    x_range = np.linspace(-5,5, 1000)
    y_range = np.linspace(-5 ,5, 1000)
    z_range = np.linspace(-5, 5, 1000)
    potential_x1 = [potential.potential(xi, 0, 0) for xi in x_range]
    potential_y1 = [potential.potential(0, yi, 0) for yi in y_range]
    potential_z1 = [potential.potential(0, 0, zi) for zi in z_range]



      
     # Plotting
    plt.plot(x_range, potential_x1,color="red",label='x-direction',ls='-')
    plt.xlabel("x'")
    #plt.xscale('log')
    plt.ylabel("potential in x' direction")
    
    plt.plot(y_range, potential_y1,color="green",label='y-direction',ls='-')
    plt.xlabel("y'")
    #plt.xscale('log')
    plt.ylabel("potential in y' direction")
     
    plt.plot(z_range, potential_z1,color='blue',label='z-direction',ls='-')
    plt.xlabel("z'")
    #plt.xscale('log')
    plt.ylabel("potential")
    plt.title("poential vs positions'")
    plt.grid(True)

  

    plt.tight_layout()
    plt.legend()
    plt.show()
  except NameError:
      print("Error: agama library not found. Please ensure it is installed and imported.")
  except Exception as e:
      print(f"An error occurred: {e}")
  return None




# -----------------------------------------------------------
# Example usage
# -----------------------------------------------------------
"""
file_path = 'r10A_10000.txt'
transformed_data_frame= make_a_transformed_frame(file_path)
potential2 = get_potential(transformed_data_frame)
plot_potential(potential2)
sampled = sample_particles( transformed_data_frame, step=1024)
lyp_list = run_sample(sampled, potential2, step=1024)
"""


# -----------------------------------------------------------
# usage for one frame 
# -----------------------------------------------------------


#--------------------------------------
#production  mode
#-------------------------------------
import re
from multiprocessing import Pool, cpu_count

# ---------- helpers ----------


def extract_frame_id(file_path: str) -> str:
    """
    For names like: r10A_0b00.txt -> "0b00" (hex string)
    Fallbacks to first digit group if pattern not found.
    """
    base = os.path.basename(file_path)
    m = re.search(r"_(?P<hex>[0-9a-fA-F]+)\.", base)
    if m:
        return m.group("hex").lower()
    m2 = re.search(r"(\d+)", base)
    return m2.group(1) if m2 else os.path.splitext(base)[0]


def extract_frame_decimal(file_path: str) -> int | None:
    """
    r10A_0b00.txt -> 2816
    Returns None if no hex suffix found.
    """
    base = os.path.basename(file_path)
    m = re.search(r"_([0-9a-fA-F]+)\.", base)
    return int(m.group(1), 16) if m else None


# ---------- worker ----------
def thread(file_path):
    """
    Worker function executed in parallel.
    Returns something small (e.g., frame id and a result summary)
    so the parent process can log/aggregate.
    """
    frame = extract_frame_id(file_path)

    # Build the frame data
    df, positions_df, velocities_df = make_frame(file_path)

    print(f"[{frame}] positions_df.shape = {positions_df.shape}")
    print(f"[{frame}] velocities_df.shape = {velocities_df.shape}")
    print(f"[{frame}] df.shape = {df.shape}")

    # Transform frame into the right coordinate system
    kinetic_tensor = calculate_kinetic_energy_tensor(df)  # (as you wrote: uses df)
    eigenvalues, eigenvectors = diagonalize_kinetic_energy_tensor(kinetic_tensor)

    if eigenvalues is None or eigenvectors is None:
        print(f"[{frame}] diagonalization failed; skipping.")
        return (frame, None)

    # Transform positions and velocities
    transformed_positions_df = transform_coordinates_to_eigenbasis(positions_df, eigenvectors)
    transformed_velocities_df = transform_coordinates_to_eigenbasis(velocities_df, eigenvectors)

    # Potential + sampling + Lyapunov
    potential2 = get_potential(transformed_positions_df)

    # IMPORTANT: plotting from multiple processes is usually a mess.
    # If you *must* plot, do it in the parent after collecting results.
    # plot_potential(potential2)

    sampled = sample_particles(transformed_positions_df, transformed_velocities_df, step=1024)
    frame_hex = extract_frame_id(file_path)          # e.g. "0b00"
    frame_dec = extract_frame_decimal(file_path)     # e.g. 2816

    print(f"[frame hex={frame_hex} dec={frame_dec}] processing {file_path}")

    lyp_list = run_sample_no_plot(
        sampled,
        potential2,
        step=1024,
        frame=f"frame_{frame_hex}",   # or use frame_dec if you prefer
    )


    # Return something the parent can store/inspect
    return (frame, lyp_list)


# ---------- main multiprocessing launcher ----------
def run_parallel(file_paths, n_workers=4, chunksize=1):
    """
    Runs `thread` over all file_paths with multiprocessing.
    """
    n_workers = min(n_workers, cpu_count(), len(file_paths))
    print(f"Using {n_workers} worker(s) on {len(file_paths)} frame(s).")

    results = []
    with Pool(processes=n_workers) as pool:
        # imap_unordered streams results as workers finish
        for out in pool.imap_unordered(thread, file_paths, chunksize=chunksize):
            results.append(out)

    # Sort results by numeric frame if possible
    def sort_key(x):
        frame, _ = x
        return int(frame) if str(frame).isdigit() else frame

    results.sort(key=sort_key)
    return results


#------------sampeling the frames unifrmly wtihin a time range-------------------
def select_frames(frames_path,t_start,t_stop,step,pattern="*"):
    """
    Select simulation frames within a time/index range.

    Parameters
    ----------
    frames_path : str or Path
        Directory containing simulation frames.
    t_start : int
        Start frame index (inclusive).
    t_stop : int
        Stop frame index (inclusive).
    step : int, optional
        Subsampling step (default: 1).
    pattern : str, optional
        Glob pattern to match frame files (default: "*").

    Returns
    -------
    list[str]
        Sorted list of frame file paths.
    """

_HEX_RE = re.compile(r"_([0-9a-fA-F]+)\.")

def _parse_frame_index(path):
    base = os.path.basename(path)
    m = _HEX_RE.search(base)
    if not m:
        raise ValueError(f"Cannot extract frame index from {base}")
    return int(m.group(1), 16)


def select_frames(frames_path,t_start,t_stop,step=1,pattern="*"):
    if step <= 0:
        raise ValueError("step must be >= 1")

    frames_path = os.fspath(frames_path)
    files = glob.glob(os.path.join(frames_path, pattern))

    indexed = []
    for f in files:
        try:
            idx = _parse_frame_index(f)
        except ValueError:
            continue
        if t_start <= idx <= t_stop:
            indexed.append((idx, f))

    indexed.sort(key=lambda x: x[0])

    return [f for i, f in indexed if (i - t_start) % step == 0]

def plot_agama_minus_hernquist(potential,rmax=10,npts=1025,M=1.0,a=1.0,science=True):
    """
    Plot:
        Phi_AGAMA - Phi_Hernquist

    along x, y, z axes.
    """

    if science:
        import scienceplots
        plt.style.use(["science", "ieee"])
    else:
        pass
        # plot
    fig, ax = plt.subplots(figsize=(8,6))
    x, y, z, phi = np.loadtxt('xyzpot.txt', unpack=True)

    phiH = - 1.0 / (1.0 + np.abs(x[0:1025]))

    ax.plot(x[0:1025], phi[0:1025] - phiH,ls=':',c='red', label=r"$x$ axis N-body potential")

    ax.plot(y[1025:2050], phi[1025:2050] - phiH,ls=':',c='green', label=r"$y$ axis N-body potential")

    ax.plot(z[2050:3075], phi[2050:3075] - phiH,ls=':',c='blue', label=r"$z$ axis N-body potential")

    # coordinate ranges
    x = np.linspace(-rmax, rmax, npts)
    y = np.linspace(-rmax, rmax, npts)
    z = np.linspace(-rmax, rmax, npts)

    # AGAMA potential along each axis
    phi_x = np.array([
        potential.potential(xi, 0, 0)
        for xi in x
    ])

    phi_y = np.array([
        potential.potential(0, yi, 0)
        for yi in y
    ])

    phi_z = np.array([
        potential.potential(0, 0, zi)
        for zi in z
    ])

    # Hernquist contribution
    phiH_x = -M / (a + np.abs(x))
    phiH_y = -M / (a + np.abs(y))
    phiH_z = -M / (a + np.abs(z))

    # residuals
    res_x = phi_x - phiH_x
    res_y = phi_y - phiH_y
    res_z = phi_z - phiH_z

    

    ax.plot(x, res_x, lw=2, label=r"$x$ axis AGAMA potential",ls='--',c='red')
    ax.plot(y, res_y, lw=2, label=r"$y$ axis AGAMA potential",ls='--',c='green')
    ax.plot(z, res_z, lw=2, label=r"$z$ axis AGAMA potential",ls='--',c='blue')

    ax.axhline(0, color='k', lw=0.8, alpha=0.4)

    ax.set_xlim(-rmax, rmax)

    ax.set_xlabel(r"coordinate position", fontsize=15)
    ax.set_ylabel(r"$\Phi_{\rm}-\Phi_{\rm Hernquist}$", fontsize=15)

    ax.legend(frameon=False)

    ax.tick_params(direction='in', top=True, right=True)

    plt.tight_layout()
    plt.show()

    return fig, ax


from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize




def plot_potential_time_evolution_xyz(potentials,times=None,rmax=10.0,npts=1025,M=1.0,a=1.0,cmap="viridis",figsize=(9, 6),alpha=0.75,lw=1.5):
    """
    Plot time evolution of

        Phi_AGAMA - Phi_Hernquist

    along x, y, z axes simultaneously.

    Different line styles distinguish axes:
        x : solid
        y : dashed
        z : dotted

    Color represents time evolution.
    """

    try:
        import scienceplots
        plt.style.use(["science", "ieee"])
    except:
        pass

    if times is None:
        times = np.arange(len(potentials))

    times = np.asarray(times)

    q = np.linspace(-rmax, rmax, npts)

    # Hernquist contribution
    phiH = -M / (a + np.abs(q))

    # colormap
    norm = Normalize(vmin=times.min(), vmax=times.max())
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize)

    # linestyle convention
    linestyles = {
        "x": "-",
        "y": "--",
        "z": ":",
    }

    for pot, t in zip(potentials, times):

        color = cmap_obj(norm(t))

        # x-axis
        phi_x = np.array([
            pot.potential(xi, 0, 0)
            for xi in q
        ])

        # y-axis
        phi_y = np.array([
            pot.potential(0, yi, 0)
            for yi in q
        ])

        # z-axis
        phi_z = np.array([
            pot.potential(0, 0, zi)
            for zi in q
        ])

        res_x = phi_x - phiH
        res_y = phi_y - phiH
        res_z = phi_z - phiH

        ax.plot(
            q,
            res_x,
            color=color,
            ls=linestyles["x"],
            lw=lw,
            alpha=alpha,
        )

        ax.plot(
            q,
            res_y,
            color=color,
            ls=linestyles["y"],
            lw=lw,
            alpha=alpha,
        )

        ax.plot(
            q,
            res_z,
            color=color,
            ls=linestyles["z"],
            lw=lw,
            alpha=alpha,
        )

    # time colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("time")

    # dummy handles for axis legend
    ax.plot([], [], color="k", ls="-",  label=r"$x$ axis")
    ax.plot([], [], color="k", ls="--", label=r"$y$ axis")
    ax.plot([], [], color="k", ls=":",  label=r"$z$ axis")

    ax.legend(frameon=False)

    ax.axhline(0.0, color='k', lw=0.8, alpha=0.4)

    ax.set_xlim(-rmax, rmax)

    ax.set_xlabel(r"coordinate")
    ax.set_ylabel(r"$\Phi_{\rm AGAMA}-\Phi_{\rm Hernquist}$")

    ax.tick_params(direction='in', top=True, right=True)

    fig.tight_layout()
    plt.show()

    return fig, ax


frame_c280,pos_c280,vel_c280=make_frame("r10A_c280.txt")
KE_c280=calculate_kinetic_energy_tensor(frame_c280)
eigenvalues,R=diagonalize_kinetic_energy_tensor(KE_c280)
trans_pos_c280=transform_coordinates_to_eigenbasis(pos_c280,R)
trans_vel_c280=transform_coordinates_to_eigenbasis(vel_c280,R)

potential=get_potential(trans_pos_c280)

plot_agama_minus_hernquist(potential,rmax=10,npts=1025,M=1.0,a=1.0)




def plot_potential_time_evolution_R_z(potentials,times=None,rmax=10.0,npts=1025,M=1.0,a=1.0,cmap="viridis",figsize=(9, 7),alpha=0.8,lw=1.6):
    """
    Plot time evolution of Phi_AGAMA - Phi_Hernquist
    in two panels:

        top: radial cylindrical coordinate R
        bottom: vertical coordinate z

    For an axisymmetric potential:
        R = sqrt(x^2 + y^2)

    The radial curve is evaluated at:
        (R, 0, 0)

    The vertical curve is evaluated at:
        (0, 0, z)
    """

    try:
        import scienceplots
        plt.style.use(["science"])
    except Exception:
        pass

    if times is None:
        times = np.arange(len(potentials))

    times = np.asarray(times)

    R = np.linspace(0.0, rmax, npts)
    z = np.linspace(-rmax, rmax, npts)

    phiH_R = -M / (a + np.abs(R))
    phiH_z = -M / (a + np.abs(z))

    norm = Normalize(vmin=times.min(), vmax=times.max())
    cmap_obj = plt.get_cmap(cmap)

    fig, axes = plt.subplots(
        2, 1,
        figsize=figsize,
        sharex=False,
        constrained_layout=True
    )

    axR, axz = axes

    for pot, t in zip(potentials, times):

        color = cmap_obj(norm(t))

        phi_R = np.array([
            pot.potential(Ri, 0.0, 0.0)
            for Ri in R
        ])

        phi_z = np.array([
            pot.potential(0.0, 0.0, zi)
            for zi in z
        ])

        res_R = phi_R - phiH_R
        res_z = phi_z - phiH_z

        axR.plot(R, res_R, color=color, lw=lw, alpha=alpha)
        axz.plot(z, res_z, color=color, lw=lw, alpha=alpha)

    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axes, pad=0.02)
    cbar.set_label("time")

    axR.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    axz.axhline(0.0, color="k", lw=0.8, alpha=0.4)

    axR.set_xlim(0.0, rmax)
    axz.set_xlim(-rmax, rmax)

    axR.set_ylabel(r"$\Phi_{\rm AGAMA}-\Phi_{\rm Hernquist}$")
    axz.set_ylabel(r"$\Phi_{\rm AGAMA}-\Phi_{\rm Hernquist}$")

    axR.set_xlabel(r"$R$")
    axz.set_xlabel(r"$z$")

    axR.set_title(r"Radial residual potential")
    axz.set_title(r"Vertical residual potential")

    for ax in axes:
        ax.tick_params(direction="in", top=True, right=True)

    plt.show()

    return fig, axes
def generate_hex_indices(start, stop, step):
    """
    Generate a list of hexadecimal frame indices.

    Parameters
    ----------
    start : str
        Starting hex index (e.g. "8000")
    stop : str
        Final hex index, inclusive (e.g. "10000")
    step : str or int
        Hex increment (e.g. "40" or 0x40)

    Returns
    -------
    indices : list[str]
        List of hex indices as lowercase strings without '0x'
    """
    # Convert inputs to integers
    start_i = int(start, 16)
    stop_i  = int(stop, 16)
    step_i  = int(step, 16) if isinstance(step, str) else int(step)

    if step_i <= 0:
        raise ValueError("step must be positive")
    if stop_i < start_i:
        raise ValueError("stop must be >= start")

    indices = []
    i = start_i
    while i <= stop_i:
        indices.append(f"{i:x}")  # hex string without 0x
        i += step_i

    return indices
indices =generate_hex_indices("3000", "10000", "1000")
def make_time_array(M, sample_dt, sim_dt, t0=0.0, dtype=np.float64, check_multiple=True, rtol=1e-12):
    """
    Construct a 1D time array for uniformly sampled frames.

    Parameters
    ----------
    M : int
        Number of frames (length of output array).
    sample_dt : float
        Time spacing between consecutive *saved frames* (sampling cadence).
        (This is the "constant sampling timesteps" you mentioned.)
    sim_dt : float
        Fixed simulation timestep of the integrator.
        Included mainly to validate/sample_dt consistency.
    t0 : float, optional
        Starting time for the first frame (default 0.0).
    dtype : numpy dtype, optional
        dtype of output array.
    check_multiple : bool, optional
        If True, checks that sample_dt is an integer multiple of sim_dt.
    rtol : float, optional
        Relative tolerance for the multiple check.

    Returns
    -------
    t : (M,) ndarray
        Time array: t[j] = t0 + j * sample_dt
    """
    if M <= 0:
        raise ValueError("M must be >= 1")
    if sample_dt <= 0:
        raise ValueError("sample_dt must be > 0")
    if sim_dt <= 0:
        raise ValueError("sim_dt must be > 0")


    # Uniform sampling grid
    t = t0 + sample_dt*sim_dt * np.arange(M, dtype=dtype)
    return t
time = make_time_array(M=len(indices), sample_dt=1, sim_dt=64, t0=192)
print(f'time:{time}')
list_agama_pot=[]
for index in indices:
    df,frame_pos,vel=make_frame(f"Sequence_frame/r10A{index}.txt")
    KE=calculate_kinetic_energy_tensor(df)
    eigenvalues,R=diagonalize_kinetic_energy_tensor(KE)
    trans_pos=transform_coordinates_to_eigenbasis(frame_pos,R)
    pot=get_potential(trans_pos)
    list_agama_pot.append(pot)


fig, axes = plot_potential_time_evolution_R_z(
    potentials=list_agama_pot,
    times=time,
    rmax=10,
)
plot_potential_time_evolution_xyz(list_agama_pot,times=None,rmax=10.0,npts=1025,M=1.0,a=1.0,cmap="viridis",figsize=(8, 6),alpha=0.9,lw=1.5)


def make_eigenbasis_snapshot(file_path):
    """
    Load one frame and transform x,y,z,vx,vy,vz into the kinetic-energy eigenbasis.

    Returns
    -------
    snapshot : pandas.DataFrame
        Columns:
        x_prime, y_prime, z_prime, vx_prime, vy_prime, vz_prime
    """

    # Original frame: columns 0,1,2 = x,y,z and 3,4,5 = vx,vy,vz
    df = pd.read_csv(file_path, sep=r"\s+", header=None, skiprows=1)
    df = df.dropna(subset=[0, 1, 2, 3, 4, 5]).copy()

    for col in [0, 1, 2, 3, 4, 5]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[0, 1, 2, 3, 4, 5]).copy()

    pos = df[[0, 1, 2]].to_numpy()
    vel = df[[3, 4, 5]].to_numpy()

    # Kinetic energy tensor, equal-mass particles so scale is irrelevant
    K = vel.T @ vel

    # Diagonalize
    evals, evecs = np.linalg.eigh(K)

    # Sort by eigenvalue
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Largest eigenvalue -> z' axis
    z_axis = evecs[:, -1]

    # Smallest eigenvalue -> x' axis
    x_axis = evecs[:, 0]

    # Complete right-handed basis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Recompute x to enforce exact orthogonality
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])

    if np.linalg.det(R) < 0:
        y_axis *= -1
        R = np.column_stack([x_axis, y_axis, z_axis])

    # Transform into eigenbasis
    pos_p = pos @ R
    vel_p = vel @ R

    snapshot = pd.DataFrame({
        "x_prime": pos_p[:, 0],
        "y_prime": pos_p[:, 1],
        "z_prime": pos_p[:, 2],
        "vx_prime": vel_p[:, 0],
        "vy_prime": vel_p[:, 1],
        "vz_prime": vel_p[:, 2],
    })

    return snapshot, evals, R

def build_eigenbasis_snapshots(frame_files):
    """
    Build the snapshots list used by the z-vz plotting function.
    """

    snapshots = []
    eigenvalues_list = []
    rotation_matrices = []

    for file_path in frame_files:
        snap, evals, R = make_eigenbasis_snapshot(file_path)

        snapshots.append(snap)
        eigenvalues_list.append(evals)
        rotation_matrices.append(R)

    return snapshots, eigenvalues_list, rotation_matrices


frame_files = []
for index in indices:
    frame_files.append(f"Sequence_frame/r10A_{index}.txt")
    print(f"Sequence_frame/r10A_{index}.txt")

snapshots, eigenvalues_list, rotation_matrices = build_eigenbasis_snapshots(frame_files)





def plot_z_vz_highlight_selected(snapshots,selected_csv,time_array=None,z_col="z_prime",vz_col="vz_prime",csv_index_col="particle_index",frame_indices=None,start_index=0,sampling_step=1,background_stride=20,ncols=3,regular_color="blue",selected_color="red",regular_alpha=0.15,selected_alpha=0.9,regular_size=2,selected_size=4,savefile=None,show=True):

    """
    Plot z-vz phase space and highlight selected particles.

    Parameters
    ----------
    background_stride : int
        Keep only every N-th regular particle.
        Example:
            1  -> plot all particles
            10 -> plot 10%
            20 -> plot 5%
            50 -> plot 2%
    """

    # ---------------------------
    # Load selected particle list
    # ---------------------------

    selected_df = pd.read_csv(selected_csv)

    actual_idx = selected_df[csv_index_col].astype(int).values

    selected_rows = (
        (actual_idx - start_index) // sampling_step
    ).astype(int)

    if frame_indices is None:
        frame_indices = np.arange(len(snapshots))

    nframes = len(frame_indices)

    nrows = int(np.ceil(nframes / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4*ncols, 3*nrows),
        sharex=True,
        sharey=True
    )

    axes = np.atleast_1d(axes).ravel()

    # ---------------------------------------------------------
    # Loop over frames
    # ---------------------------------------------------------

    for ax, frame_id in zip(axes, frame_indices):

        df = snapshots[frame_id]

        z = df[z_col].values
        vz = df[vz_col].values

        N = len(df)

        # valid selected rows
        sel = selected_rows[
            (selected_rows >= 0) &
            (selected_rows < N)
        ]

        # boolean mask
        selected_mask = np.zeros(N, dtype=bool)
        selected_mask[sel] = True

        # regular particles only
        regular_rows = np.where(~selected_mask)[0]

        # thin background
        regular_rows = regular_rows[::background_stride]

        # --------------------
        # Regular particles
        # --------------------

        ax.scatter(
            z[regular_rows],
            vz[regular_rows],
            s=regular_size,
            c=regular_color,
            alpha=regular_alpha,
            rasterized=True,
            label="Regular" if frame_id == frame_indices[0] else None,
        )

        # --------------------
        # Selected particles
        # --------------------

        ax.scatter(
            z[sel],
            vz[sel],
            s=selected_size,
            c=selected_color,
            alpha=selected_alpha,
            rasterized=True,
            label="Selected" if frame_id == frame_indices[0] else None,
        )

        # title

        if time_array is not None:
            ax.set_title(
                rf"$t={time_array[frame_id]:.0f}$",
                fontsize=11
            )
        else:
            ax.set_title(
                f"Frame {frame_id}",
                fontsize=11
            )

        ax.grid(alpha=0.25)

    # remove empty panels

    for ax in axes[nframes:]:
        ax.axis("off")

    # labels

    for ax in axes[-ncols:]:
        ax.set_xlabel(r"$z$")

    for ax in axes[::ncols]:
        ax.set_ylabel(r"$v_z$")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True
    )

    fig.tight_layout()

    if savefile is not None:
        fig.savefig(
            savefile,
            dpi=300,
            bbox_inches="tight"
        )
        print(f"Saved: {savefile}")

    if show:
        plt.show()

    return fig, axes
plot_z_vz_highlight_selected(
    snapshots=snapshots,
    selected_csv="selected_particles_mode-above_thr-0.400_step-128.csv",
    background_stride=50,
    time_array=time,
    z_col="z_prime",
    vz_col="vz_prime",
    start_index=600_001,
    sampling_step=128,
    savefile="z_vz_selected_particles.pdf"
)


plot_z_vz_highlight_selected(
    snapshots=snapshots,
    selected_csv="selected_chaotic_particles.csv",
    background_stride=50,
    time_array=time,
    z_col="z_prime",
    vz_col="vz_prime",
    csv_index_col="actual_index",
    start_index=600_001,
    sampling_step=128,
    savefile="z_vz_selected_particles.pdf"
)





def load_raw_frame(file_path):
    """
    Load one simulation frame as raw x,y,z,vx,vy,vz dataframe.
    """

    df = pd.read_csv(file_path, sep=r"\s+", header=None, skiprows=1)

    df = df.dropna(subset=[0, 1, 2, 3, 4, 5]).copy()

    for col in [0, 1, 2, 3, 4, 5]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[0, 1, 2, 3, 4, 5]).copy()

    df.columns = ["x", "y", "z", "vx", "vy", "vz"]

    return df


def kinetic_basis_from_frame(df):
    """
    Compute kinetic-energy eigenbasis from one raw frame.

    The largest kinetic eigenvector defines z'.
    The smallest kinetic eigenvector defines x'.
    y' completes a right-handed basis.
    """

    vel = df[["vx", "vy", "vz"]].to_numpy()

    K = vel.T @ vel

    evals, evecs = np.linalg.eigh(K)

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    z_axis = evecs[:, -1]
    x_axis = evecs[:, 0]

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])

    if np.linalg.det(R) < 0:
        y_axis *= -1
        R = np.column_stack([x_axis, y_axis, z_axis])

    return evals, R


def transform_positions_with_R(df, R):
    """
    Transform raw positions into a fixed eigenbasis R.
    """

    pos = df[["x", "y", "z"]].to_numpy()

    pos_prime = pos @ R

    out = pd.DataFrame({
        "x_prime": pos_prime[:, 0],
        "y_prime": pos_prime[:, 1],
        "z_prime": pos_prime[:, 2],
    })

    return out



def plot_selected_xyz_projections_fixed_basis(frame_files,selected_csv,plot_frame_id,basis_frame_id=0,csv_index_col="particle_index",start_index=0,sampling_step=1,time_array=None,bakground_stride=20,regular_size=2,selected_size=5,regular_alpha=0.15,selected_alpha=0.9,savefile=None,show=True):
    """
    Plot x'y', z'x', and z'y' projections.

    Red = selected chaotic particles.
    Blue = thinned regular/background particles.

    The eigenbasis R is computed from basis_frame_id and then applied
    to plot_frame_id.
    """

    selected_df = pd.read_csv(selected_csv)
    actual_idx = selected_df[csv_index_col].astype(int).values
    selected_rows = ((actual_idx - start_index) // sampling_step).astype(int)

    # Reference eigenbasis
    basis_df = load_raw_frame(frame_files[basis_frame_id])
    evals_ref, R_ref = kinetic_basis_from_frame(basis_df)

    # Frame to plot
    plot_df = load_raw_frame(frame_files[plot_frame_id])
    pos_prime = transform_positions_with_R(plot_df, R_ref)

    N = len(pos_prime)

    selected_rows = selected_rows[
        (selected_rows >= 0) &
        (selected_rows < N)
    ]

    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected_rows] = True

    regular_rows = np.where(~selected_mask)[0]
    regular_rows = regular_rows[::background_stride]

    x = pos_prime["x_prime"].values
    y = pos_prime["y_prime"].values
    z = pos_prime["z_prime"].values

    fig, axes = plt.subplots(
        1, 3,
        figsize=(13, 4),
        constrained_layout=True
    )

    projections = [
        (x, y, r"$x'$", r"$y'$", r"$x'-y'$"),
        (z, x, r"$z'$", r"$x'$", r"$z'-x'$"),
        (z, y, r"$z'$", r"$y'$", r"$z'-y'$"),
    ]

    for ax, (u, v, xlabel, ylabel, title) in zip(axes, projections):

        ax.scatter(
            u[regular_rows],
            v[regular_rows],
            s=regular_size,
            c="blue",
            alpha=regular_alpha,
            rasterized=True,
            label="Regular"
        )

        ax.scatter(
            u[selected_rows],
            v[selected_rows],
            s=selected_size,
            c="red",
            alpha=selected_alpha,
            rasterized=True,
            label="Selected"
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=True)

    if time_array is not None:
        fig.suptitle(
            rf"Frame $t={time_array[plot_frame_id]:.0f}$, "
            rf"basis from $t={time_array[basis_frame_id]:.0f}$",
            fontsize=13
        )
    else:
        fig.suptitle(
            f"Frame {plot_frame_id}, basis frame {basis_frame_id}",
            fontsize=13
        )

    if savefile is not None:
        fig.savefig(savefile, dpi=300, bbox_inches="tight")
        print(f"Saved: {savefile}")

    if show:
        plt.show()

    return fig, axes, R_ref
fig, axes, R_ref = plot_selected_xyz_projections_fixed_basis(
    frame_files=frame_files,
    selected_csv="selected_particles_mode-above_thr-0.600_step-128.csv",
    plot_frame_id=0,
    basis_frame_id=7,
    start_index=0,
    sampling_step=128,
    time_array=time,
    savefile="selected_particles_xyz_projection_fixed_basis.pdf",
)

fig, axes, R_ref = plot_selected_xyz_projections_fixed_basis(
    frame_files=frame_files,
    selected_csv="selected_chaotic_particles.csv",
    plot_frame_id=0,
    basis_frame_id=7,
    start_index=0,
    csv_index_col="actual_index",
    sampling_step=128,
    time_array=time,
    savefile="selected_particles_xyz_projection_fixed_basis.pdf",
)
print(R_ref)



def plot_Lnorm_components_fixed_basis(frame_files,selected_csv,plot_frame_id,basis_frame_id=0,csv_index_col="particle_index",start_index=0,sampling_step=1,time_array=None,background_stride=20,regular_size=2,selected_size=5,regular_alpha=0.15,selected_alpha=0.9,savefile=None,show=True):
    """
    Plot normalized angular momentum components in a fixed eigenbasis.

    Red  = selected particles
    Blue = thinned regular/background particles

    Panels:
        Lx_hat vs Ly_hat
        Lx_hat vs Lz_hat
        Ly_hat vs Lz_hat
    """

    selected_df = pd.read_csv(selected_csv)
    actual_idx = selected_df[csv_index_col].astype(int).values
    selected_rows = ((actual_idx - start_index) // sampling_step).astype(int)

    # Reference kinetic eigenbasis
    basis_df = load_raw_frame(frame_files[basis_frame_id])
    evals_ref, R_ref = kinetic_basis_from_frame(basis_df)

    # Frame to plot
    plot_df = load_raw_frame(frame_files[plot_frame_id])

    r = plot_df[["x", "y", "z"]].to_numpy()
    v = plot_df[["vx", "vy", "vz"]].to_numpy()

    # Transform positions and velocities into SAME fixed basis
    r_prime = r @ R_ref
    v_prime = v @ R_ref

    # Angular momentum in fixed eigenbasis
    L = np.cross(r_prime, v_prime)

    Lnorm = np.linalg.norm(L, axis=1)

    valid = Lnorm > 0

    Lhat = np.full_like(L, np.nan)
    Lhat[valid] = L[valid] / Lnorm[valid, None]

    Lx = Lhat[:, 0]
    Ly = Lhat[:, 1]
    Lz = Lhat[:, 2]

    N = len(plot_df)

    selected_rows = selected_rows[
        (selected_rows >= 0) &
        (selected_rows < N) &
        valid[selected_rows]
    ]

    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected_rows] = True

    regular_rows = np.where((~selected_mask) & valid)[0]
    regular_rows = regular_rows[::background_stride]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(13, 4),
        constrained_layout=True
    )

    panels = [
        (Lx, Ly, r"$\hat{L}_x$", r"$\hat{L}_y$", r"$\hat{L}_x$ vs $\hat{L}_y$"),
        (Lx, Lz, r"$\hat{L}_x$", r"$\hat{L}_z$", r"$\hat{L}_x$ vs $\hat{L}_z$"),
        (Ly, Lz, r"$\hat{L}_y$", r"$\hat{L}_z$", r"$\hat{L}_y$ vs $\hat{L}_z$"),
    ]

    for ax, (u, w, xlabel, ylabel, title) in zip(axes, panels):

        ax.scatter(
            u[regular_rows],
            w[regular_rows],
            s=regular_size,
            c="blue",
            alpha=regular_alpha,
            rasterized=True,
            label="Regular"
        )

        ax.scatter(
            u[selected_rows],
            w[selected_rows],
            s=selected_size,
            c="red",
            alpha=selected_alpha,
            rasterized=True,
            label="Selected"
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=True)

    if time_array is not None:
        fig.suptitle(
            rf"Normalized angular momentum at $t={time_array[plot_frame_id]:.0f}$, "
            rf"basis from $t={time_array[basis_frame_id]:.0f}$",
            fontsize=13
        )
    else:
        fig.suptitle(
            f"Normalized angular momentum: frame {plot_frame_id}, basis frame {basis_frame_id}",
            fontsize=13
        )

    if savefile is not None:
        fig.savefig(savefile, dpi=300, bbox_inches="tight")
        print(f"Saved: {savefile}")

    if show:
        plt.show()

    return fig, axes, R_ref

fig, axes, R_ref = plot_Lnorm_components_fixed_basis(
    frame_files=frame_files,
    selected_csv="selected_particles_mode-above_thr-0.600_step-128.csv",
    plot_frame_id=0,
    basis_frame_id=7,
    start_index=0,
    sampling_step=128,
    background_stride=30,
    time_array=time,
    savefile="Lnorm_selected_fixed_basis.pdf",
)

################################################################################################################################
#tracing back chaotic orbit to initial frame, apply coordinates transform of kinetic tensor using KE from  frame C280 at t=778:
################################################################################################################################
"""
frame_0000,pos_0,vel_0=make_frame("r10A_0000.txt")
frame_c280,pos_c280,vel_c280=make_frame("r10A_c280.txt")
KE_c280=calculate_kinetic_energy_tensor(frame_c280)
#print(KE_c280)
eigenvalues,R=diagonalize_kinetic_energy_tensor(KE_c280)
inv_R=np.linalg.inv(R)
#print("Eigenvalues:")
#print(eigenvalues)
#print("R matrix found using np.linalg:")
#print(R)
#print("R inverse:")
#print(inv_R)
#print('VERIFICATION R^tKR:')
diag=np.matmul(np.matmul(inv_R,KE_c280),R)
#print(diag)
trans_pos_0=transform_coordinates_to_eigenbasis(pos_0,R)
trans_vel_0=transform_coordinates_to_eigenbasis(vel_0,R)


L = np.cross(trans_pos_0.to_numpy(), trans_vel_0.to_numpy())  # ensure ndarray
Lmag = np.linalg.norm(L, axis=1, keepdims=True)               # (N,1)
L_norm = np.divide(L, Lmag, out=np.zeros_like(L), where=(Lmag > 0))
"""
#################trying Phase space plotting#################################























#print(Lmag)
#print(L_norm)
#print(Lmag)
#print(np.linalg.norm(L,axis=1))
#Recover index with abs(L_z) >=0.9 and in between 600k and 900k:

'''
def get_high_Lz_particles(L_norm, threshold=0.9, idx_min=600_000, idx_max=900_000):
    """
    Return indices of particles with |Lz| >= threshold
    and index within [idx_min, idx_max].

    Parameters
    ----------
    L_norm : ndarray of shape (N, 3)
    threshold : float
    idx_min : int
    idx_max : int

    Returns
    -------
    list
    """
    Lz = L_norm[:, 2]

    # condition on Lz
    cond_Lz = np.abs(Lz) >= threshold

    # condition on index range
    indices = np.arange(len(L_norm))
    cond_idx = (indices >= idx_min) & (indices <= idx_max)

    # combine both conditions
    final_indices = np.where(cond_Lz & cond_idx)[0]

    return final_indices.tolist()
def get_low_Lz_particles(L_norm, threshold=0.1, idx_min=600_000, idx_max=900_000):

    """
    Return indices of particles with |Lz| >= threshold
    and index within [idx_min, idx_max].

    Parameters
    ----------
    L_norm : ndarray of shape (N, 3)
    threshold : float
    idx_min : int
    idx_max : int

    Returns
    -------
    list
    """
    Lz = L_norm[:, 2]

    # condition on Lz
    cond_Lz = np.abs(Lz) <= threshold

    # condition on index range
    indices = np.arange(len(L_norm))
    cond_idx = (indices >= idx_min) & (indices <= idx_max)

    # combine both conditions
    final_indices = np.where(cond_Lz & cond_idx)[0]

    return final_indices.tolist()
high_Lz=get_high_Lz_particles(L_norm, threshold=0.9, idx_min=600_000, idx_max=900_000)
low_Lz=get_low_Lz_particles(L_norm)
print(len(high_Lz))
print(len(low_Lz))
'''
#print(trans_pos_0)
'''
def plot_particle_projections(trans_pos_0, particle_indices, figsize=(15, 5), marker='.', s=5):

    pos = np.asarray(trans_pos_0)[particle_indices]

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    pairs = [
        (x, y, "x", "y", "x-y projection"),
        (x, z, "x", "z", "x-z projection"),
        (y, z, "y", "z", "y-z projection"),
    ]

    for ax, (u, v, xlabel, ylabel, title) in zip(axes, pairs):
        ax.scatter(u, v, marker=marker, s=s)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')

        # optional symmetric padding
        umin, umax = np.min(u), np.max(u)
        vmin, vmax = np.min(v), np.max(v)
        du = umax - umin
        dv = vmax - vmin
        pad_u = 0.05 * du if du > 0 else 1.0
        pad_v = 0.05 * dv if dv > 0 else 1.0
        ax.set_xlim(umin - pad_u, umax + pad_u)
        ax.set_ylim(vmin - pad_v, vmax + pad_v)

    plt.tight_layout()
    plt.show()

    return fig, axes

plot_particle_projections(trans_pos_0,high_Lz, figsize=(15, 5), marker='.', s=5)
plot_particle_projections(trans_pos_0,low_Lz, figsize=(15, 5), marker='.', s=5)
'''

"""
trans_pos_c280=transform_coordinates_to_eigenbasis(pos_c280,R)
trans_vel_c280=transform_coordinates_to_eigenbasis(vel_c280,R)
pot=get_potential(trans_pos_c280)
high_Lz_pos_c280=np.asarray(trans_pos_c280)[high_Lz]
high_Lz_vel_c280=np.asarray(trans_vel_c280)[high_Lz]
high_Lz_pos_c280_df=pd.DataFrame(high_Lz_pos_c280)
high_Lz_vel_c280_df=pd.DataFrame(high_Lz_vel_c280)
print(high_Lz_pos_c280_df)
sample= pd.concat([high_Lz_pos_c280_df, high_Lz_vel_c280_df], axis=1)
sampled=sample.iloc[0::32].copy()
sampled.columns = ['x_prime', 'y_prime', 'z_prime', 'vx_prime', 'vy_prime', 'vz_prime']

print(sampled)

#out=run_sample(sampled, pot, 32,'c280')
"""

def get_positive_lyap_indices(filepath,eps=0):
    """
    Extract particle indices with positive Lyapunov exponent.

    Parameters
    ----------
    filepath : str
        Path to lyapunov_summary.txt

    Returns
    -------
    list
        List of particle indices with lyap > 0
    """

    # Load file (skip header)
    data = np.loadtxt(filepath, skiprows=1)

    indices =data[:, 0].astype(int)
    lyap = data[:, 1]

    # filter positive lyapunov exponents
    mask = lyap > eps
    positive_indices = indices[mask]

    return positive_indices.tolist()

#chaotic_list=get_positive_lyap_indices('sos_output_Lz_geq_0.9/step16_framec280/lyapunov_summary.txt')
#print(chaotic_list)
#print(len(chaotic_list))
#chaotic_list=get_positive_lyap_indices('sos_output_Lz_leq_0.1/step64_framec280/lyapunov_summary.txt')
#print(chaotic_list)
#print(len(chaotic_list))


def map_to_true_indices(lyap_indices, high_Lz):
    """
    Map indices stored in the Lyapunov file back to the true particle indices.

    Parameters
    ----------
    lyap_indices : list or array
        Indices read from lyapunov_summary.txt
    high_Lz : list or array
        True particle indices of the high_Lz subset

    Returns
    -------
    list
        True particle indices
    """
    high_Lz = np.asarray(high_Lz)
    lyap_indices = np.asarray(lyap_indices, dtype=int)

    return high_Lz[lyap_indices].tolist()

#true_idx=map_to_true_indices(chaotic_list,low_Lz)
#print(true_idx)
#print(len(true_idx))

#plot_particle_projections(trans_pos_0, true_idx, figsize=(15, 5), marker='.', s=5)


"""
# Create a figure and 3D axes
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
limit_min=-1
limit_max=1
ax.set_xlim(limit_min, limit_max)
ax.set_ylim(limit_min, limit_max)
ax.set_zlim(limit_min, limit_max)

#ax.set_aspect('equal')
# Plot the scatter points
for index in sus_chaos_pos:
    ax.scatter(L_norm[index,0], L_norm[index,1], L_norm[index,2], c='r', marker='.',s=5,alpha=0.5)
#ax.plot_surface(X,Y)
#ax.plot_surface()
#ax.plot_surface()
plt.show()
"""

def plot_L_components_pairwise(Lx, Ly, Lz, figsize=(10, 4), marker='.', s=None):
    """
    Plot pairwise projections of normalized angular momentum components.

    Parameters
    ----------
    Lx, Ly, Lz : array-like
        1D arrays of equal length.
    figsize : tuple
        Figure size.
    marker : str
        Marker style.
    s : float or None
        Marker size for scatter. If None, uses plot(..., marker=marker, linestyle='None').

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """
    plt.style.use(['default'])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=False)

    if s is None:
        ax1.plot(Lx, Ly, linestyle='None', marker=marker, color="red")
        ax2.plot(Lx, Lz, linestyle='None', marker=marker, color="blue")
        ax3.plot(Ly, Lz, linestyle='None', marker=marker, color="green")
    else:
        ax1.scatter(Lx, Ly, color="red", s=s)
        ax2.scatter(Lx, Lz, color="blue", s=s)
        ax3.scatter(Ly, Lz, color="green", s=s)

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel(r"$\hat{L}_x$")
    ax1.set_ylabel(r"$\hat{L}_y$")
    ax1.set_title(r"$\hat{L}_x$ vs $\hat{L}_y$")

    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel(r"$\hat{L}_x$")
    ax2.set_ylabel(r"$\hat{L}_z$")
    ax2.set_title(r"$\hat{L}_x$ vs $\hat{L}_z$")

    ax3.set_aspect('equal', adjustable='box')
    ax3.set_xlabel(r"$\hat{L}_y$")
    ax3.set_ylabel(r"$\hat{L}_z$")
    ax3.set_title(r"$\hat{L}_y$ vs $\hat{L}_z$")

    plt.tight_layout()
    plt.show()

    return fig, (ax1, ax2, ax3)

def ks_test_Lz_isotropy(L_norm, indices=None):
    """
    KS test for isotropy using Lz ~ Uniform(-1,1)

    Parameters
    ----------
    L_norm : (N,3) array
        Unit angular momentum vectors.
    indices : array-like or None
        Optional subset of particle indices.

    Returns
    -------
    statistic, p_value
    """

    if indices is not None:
        L = L_norm[indices]
    else:
        L = L_norm

    Lz = L[:,2]

    # KS test against Uniform(-1,1)
    statz, pz = kstest(Lz, 'uniform', args=(-1,1))

    print("KS test for isotropy (Lz uniformity)")
    print("Statistic:", statz)
    print("p-value:", pz)

    Lx = L[:,0]

    # KS test against Uniform(-1,1)
    statx, px = kstest(Lx, 'uniform', args=(-1,1))

    print("KS test for isotropy (Lx uniformity)")
    print("Statistic:", statx)
    print("p-value:", px)
    Ly = L[:,1]

    # KS test against Uniform(-1,1)
    staty, py = kstest(Ly, 'uniform', args=(-1,1))

    print("KS test for isotropy (Ly uniformity)")
    print("Statistic:", staty)
    print("p-value:", py)

    return statz, pz


def rayleigh_test_isotropy(L_norm, indices=None):
    """
    Rayleigh test for preferred direction.

    Parameters
    ----------
    L_norm : (N,3) array
        Unit angular momentum vectors
    indices : array-like or None
        Optional subset of particle indices

    Returns
    -------
    Z statistic and p-value
    """

    if indices is not None:
        L = L_norm[indices]
    else:
        L = L_norm

    N = L.shape[0]

    mean_vec = np.mean(L, axis=0)
    R = np.linalg.norm(mean_vec)

    Z = N * R**2

    # approximate p-value
    p = np.exp(-Z)

    print("Rayleigh test for preferred direction")
    print("N:", N)
    print("Mean direction vector:", mean_vec)
    print("R:", R)
    print("Z statistic:", Z)
    print("p-value:", p)

    return Z, p


def extract_chaotic_angular_momentum_components(L_norm, csv_filepath, index_col="particle_index",idx_min=600_000,idx_max=2_000_000):
    """
    Extract chaotic angular momentum components from L_norm using particle indices
    stored in a CSV file.

    Parameters
    ----------
    L_norm : array-like, shape (N, 3)
        Array of normalized angular momentum vectors.
        Columns are assumed to be [Lx, Ly, Lz].
    csv_filepath : str
        Path to CSV file containing a column of particle indices.
    index_col : str, optional
        Name of the CSV column containing indices. Default is 'particle_index'.

    Returns
    -------
    Lx : np.ndarray
        1D array of chaotic Lx values.
    Ly : np.ndarray
        1D array of chaotic Ly values.
    Lz : np.ndarray
        1D array of chaotic Lz values.
    L_norm_chaos : np.ndarray
        Array of shape (M, 3) containing the selected chaotic angular momentum vectors.

    Raises
    ------
    ValueError
        If L_norm does not have shape (N, 3).
    KeyError
        If index_col is not found in the CSV.
    IndexError
        If any particle index is outside the valid range for L_norm.
    """
    df = pd.read_csv(csv_filepath)
    indices = df[index_col].to_numpy()

    # cast to integer
    indices = indices.astype(int)

    mask = (indices >= idx_min) & (indices <= idx_max)
    indices = indices[mask]

    L_norm_chaos = L_norm[indices]



    Lx = L_norm_chaos[:, 0]
    Ly = L_norm_chaos[:, 1]
    Lz = L_norm_chaos[:, 2]

    return Lx, Ly, Lz, L_norm_chaos,indices


def extract_chaotic_pos_components(pos_0, csv_filepath, index_col="particle_index",idx_min=600_000,idx_max=800_000):

    df = pd.read_csv(csv_filepath)
    indices = df[index_col].to_numpy()

   

    mask = (indices >= idx_min) & (indices <= idx_max)
    indices = indices[mask]

    #norm= np.linalg.norm(np.asarray(pos_0),axis=1,keepdims=True)
    #pos_0=np.asarray(pos_0)/norm
    pos_0_chaos=np.asarray(pos_0)[indices]




    x = pos_0_chaos[:, 0]
    y = pos_0_chaos[:, 1]
    z = pos_0_chaos[:, 2]

    return x, y, z, pos_0_chaos,indices


################### Manually implement KS test and NUll hypothesis##################
def ks2_manual(x, y, tol=1e-300, max_terms=100):


    if len(x) == 0 or len(y) == 0:
        raise ValueError("Both samples must contain at least one finite value.")

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    z = np.sort(np.unique(np.concatenate([x_sorted, y_sorted])))
    cdf_x = np.searchsorted(x_sorted, z, side="right") / len(x_sorted)
    cdf_y = np.searchsorted(y_sorted, z, side="right") / len(y_sorted)

    diff = np.abs(cdf_x - cdf_y)
    idx = np.argmax(diff)

    D = diff[idx]
    x0 = z[idx]
    cdf_x0 = cdf_x[idx]
    cdf_y0 = cdf_y[idx]

    n = len(x_sorted)
    m = len(y_sorted)
    n_eff = n * m / (n + m)
    lam = np.sqrt(n_eff) * D

    if lam == 0:
        pvalue = 1.0
    else:
        s = 0.0
        for k in range(1, max_terms + 1):
            term = 2 * (-1)**(k - 1) * np.exp(-2 * (k**2) * (lam**2))
            s += term
            if abs(term) < tol:
                break
        pvalue = max(0.0, min(1.0, s))
    
    return {
        "D": D,
        "pvalue": pvalue,
        "x0": x0,
        "cdf_x_at_x0": cdf_x0,
        "cdf_y_at_x0": cdf_y0,
        "z": z,
        "cdf_x": cdf_x,
        "cdf_y": cdf_y,
    }


def custom_ks_test_plot_manual(x, y, ax=None, labels=("Sample 1", "Sample 2")):
    result = ks2_manual(x, y)

    created_fig = False
    if ax is None:
        plt.style.use(['science'])
        fig, ax = plt.subplots(figsize=(8, 5))
        created_fig = True

    ax.step(result["z"], result["cdf_x"], where="post", label=labels[0], linewidth=2)
    ax.step(result["z"], result["cdf_y"], where="post", label=labels[1], linewidth=2)

    ax.vlines(
        result["x0"],
        ymin=min(result["cdf_x_at_x0"], result["cdf_y_at_x0"]),
        ymax=max(result["cdf_x_at_x0"], result["cdf_y_at_x0"]),
        linewidth=2,
        linestyles="solid",
        label=f"Max diff = {result['D']:.4f}"
    )

    ax.plot(
        [result["x0"], result["x0"]],
        [result["cdf_x_at_x0"], result["cdf_y_at_x0"]],
        "o"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("CDF")
    ax.set_title(f"Manual two-sample KS test\nD = {result['D']:.4f}, p = {result['pvalue']}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if created_fig:
        plt.tight_layout()
        plt.show()

    return result


def diagnostics(Lx,Ly,Lz,L_norm_chaos,seed):
    rayleigh_test_isotropy(L_norm_chaos)
    ks_test_Lz_isotropy(L_norm_chaos)
    rng = np.random.default_rng(seed)
    uniform_x = rng.uniform(-1.0, 1.0, size=len(Lx))
    uniform_y = rng.uniform(-1.0, 1.0, size=len(Ly))
    uniform_z = rng.uniform(-1.0, 1.0, size=len(Lz))
    print("Manual KS test result: ")
    res_x = custom_ks_test_plot_manual(Lx, uniform_x, labels=(r"$L_x$", "Uniform(-1,1)"))
    scipy_resultx = ks_2samp(Lx, uniform_x)
    print("Manual KS test Lx")
    print("D =", res_x["D"])
    print("p-value =", res_x["pvalue"])
    print("\nSciPy ks_2samp sanity check")
    print(f"D       = {scipy_resultx.statistic:.8f}")
    print(f"p-value = {scipy_resultx.pvalue:.8e}")

    res_y = custom_ks_test_plot_manual(Ly, uniform_y, labels=(r"$L_y$", "Uniform(-1,1)"))
    scipy_resulty = ks_2samp(Ly, uniform_y)
    print("Manual KS test Ly")
    print("D =", res_y["D"])
    print("p-value =", res_y["pvalue"])
    print("\nSciPy ks_2samp sanity check")
    print(f"D       = {scipy_resulty.statistic:.8f}")
    print(f"p-value = {scipy_resulty.pvalue:.8e}")

    res_z = custom_ks_test_plot_manual(Lz, uniform_z, labels=(r"$L_z$", "Uniform(-1,1)"))
    scipy_resultz = ks_2samp(Lz, uniform_z)
    print("Manual KS test Lz")
    print("D =", res_z["D"])
    print("p-value =", res_z["pvalue"])
    print("\nSciPy ks_2samp sanity check")
    print(f"D       = {scipy_resultz.statistic:.8f}")
    print(f"p-value = {scipy_resultz.pvalue:.8e}")

    return res_x,res_y,res_z



def plot_spatial_components_pairwise(x, y, z, figsize=(10, 4), marker='.', s=None):
    """
    Plot pairwise projections of normalized angular momentum components.

    Parameters
    ----------
    Lx, Ly, Lz : array-like
        1D arrays of equal length.
    figsize : tuple
        Figure size.
    marker : str
        Marker style.
    s : float or None
        Marker size for scatter. If None, uses plot(..., marker=marker, linestyle='None').

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """
    plt.style.use(['default'])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=False)

    if s is None:
        ax1.plot(x, y, linestyle='None', marker=marker, color="red")
        ax2.plot(x, z, linestyle='None', marker=marker, color="blue")
        ax3.plot(y, z, linestyle='None', marker=marker, color="green")
    else:
        ax1.scatter(x, y, color="red", s=s)
        ax2.scatter(z, x, color="blue", s=s)
        ax3.scatter(y, z, color="green", s=s)

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_title(r"$x$ vs $y$")

    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel(r"$z$")
    ax2.set_ylabel(r"$x$")
    ax2.set_title(r"$x$ vs $z$")

    ax3.set_aspect('equal', adjustable='box')
    ax3.set_xlabel(r"$y$")
    ax3.set_ylabel(r"$z$")
    ax3.set_title(r"$y$ vs $z$")

    plt.tight_layout()
    plt.show()

    return fig, (ax1, ax2, ax3)





"""
time_fraction_tested=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#time_fraction_tested=[0.4,0.5,0.6]
for threshold in time_fraction_tested:
    print(f"====================={threshold}=====================")
    Lx,Ly,Lz,L_norm_chaos,indices=extract_chaotic_angular_momentum_components(L_norm, f"selected_particles_mode-above_thr-{threshold:.3f}_step-1024.csv")
    x,y,z,pos_0_chaos,_=extract_chaotic_pos_components(pos_0,f"selected_particles_mode-above_thr-{threshold:.3f}_step-128.csv")
    plot_L_components_pairwise(Lx, Ly, Lz, figsize=(10, 4), marker='.', s=None)
    diagnostics(Lx,Ly,Lz,L_norm_chaos,42)
    #plt.hist(Lz,bins=30)
    #plt.show()
    #plt.hist(Lx,bins=30)
    #plt.show()
    plot_spatial_components_pairwise(x, y, z, figsize=(10, 4), marker='.', s=None)
    #diagnostics(x,y,z,pos_0_chaos,42)


Lx,Ly,Lz,L_norm_chaos,indices=extract_chaotic_angular_momentum_components(L_norm,"early_chaotic_particles.csv")
x,y,z,pos_0_chaos,_=extract_chaotic_pos_components(pos_0,"early_chaotic_particles.csv")
plot_L_components_pairwise(Lx, Ly, Lz, figsize=(10, 4), marker='.', s=None)
diagnostics(Lx,Ly,Lz,L_norm_chaos,42)
plt.hist(Lz,bins=30)
plt.show()
plt.hist(Lx,bins=30)
plt.show()
plot_spatial_components_pairwise(x, y, z, figsize=(10, 4), marker='.', s=None)

"""















 
