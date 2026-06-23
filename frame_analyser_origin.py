import pandas as pd
import numpy as np
import agama  
import matplotlib.pyplot as plt
from pylab import *
import os
from pathlib import Path


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

    sampled = sample_particles(transformed_positions_df, transformed_velocities_df, step=128,start=600001)#600 001
    frame_hex = extract_frame_id(file_path)          # e.g. "0b00"
    frame_dec = extract_frame_decimal(file_path)     # e.g. 2816

    print(f"[frame hex={frame_hex} dec={frame_dec}] processing {file_path}")

    lyp_list = run_sample_no_plot(
        sampled,
        potential2,
        step=128,
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


def select_frames(
    frames_path,
    t_start,
    t_stop,
    step=1,
    pattern="*",
):
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

#usage:
"""
file_paths = select_frames(
    "Sequence_frame",
    t_start=0x0000,
    t_stop=0x0fc0,
    step=0x40,
    pattern="r10A_*.txt",
)

results = run_parallel(file_paths, n_workers=4, chunksize=1)
"""
# do it maually :
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
file_paths=[]
indices =generate_hex_indices("7000", "8000", "40")

print(len(indices))
for index in indices:
    file_paths.append(os.path.join('Sequence_frame',f"r10A_{index}.txt"))


start = time.time()
# run 4 threads (processes) in parallel
#file_paths=['Sequence_frame/r10A_0000.txt','Sequence_frame/r10A_0b00.txt','Sequence_frame/r10A_0c00.txt','Sequence_frame/r10A_0d40.txt','Sequence_frame/r10A_0e80.txt','Sequence_frame/r10A_0fc0.txt']
results = run_parallel(file_paths, n_workers=8, chunksize=1)
stop = time.time()
print(f"Running time: {stop - start:.3f}s")


