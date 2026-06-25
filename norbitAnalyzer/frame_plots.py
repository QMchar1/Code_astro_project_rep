import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.patches as mpatches


def plot_energy_vs_lyapunov(summary_file, savefig=True, cmap='plasma'):
    """
    Read a Lyapunov summary text file and scatter-plot Specific Energy vs Lyapunov Exponent,
    color-coded by Angular Momentum magnitude |L|.

    Parameters
    ----------
    summary_file : str or Path
        Path to the lyapunov_summary.txt file (from run_sample()).
    savefig : bool, optional
        If True, saves the plot as a PNG next to the summary file.
    cmap : str, optional
        Matplotlib colormap name (default: 'plasma').

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes._axes.Axes
    """
    # --- Load data ---
    data = np.loadtxt(summary_file, comments="#")

    # Columns:
    # orig_index  local_index  lyapunov_exp  specific_energy[J/kg]  angular_momentum
    lyap = data[:, 1]
    energy = data[:, 2]
    L = data[:, 3]

    # --- Normalize angular momentum for color mapping ---
    norm = plt.Normalize(vmin=np.min(L), vmax=np.max(L))

    # --- Create scatter plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        energy, lyap,
        c=L, cmap=cmap, norm=norm,
        s=40, alpha=0.8, edgecolor='k', linewidths=0.3
    )

    # --- Add colorbar for angular momentum ---
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("|L|  (Angular Momentum)", fontsize=12)

    # --- Axis labels and title ---
    ax.set_xlabel("Specific Energy  [J/kg]", fontsize=13)
    ax.set_ylabel("Lyapunov Exponent", fontsize=13)
    ax.set_title("Energy vs Lyapunov Exponent\n(color = Angular Momentum |L|)", fontsize=14, fontweight='bold')

    # --- Grid and style ---
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.tight_layout()

    # --- Save or show ---
    if savefig:
        outname = Path(summary_file).with_name("Energy_vs_Lyapunov_coloredByL.png")
        plt.savefig(outname, dpi=300)
        print(f"Gradient scatter plot saved: {outname}")
    else:
        plt.show()

    return fig, ax





def plot_angular_momentum_vs_lyapunov(summary_file, savefig=True):
    """
    Read a Lyapunov summary text file and scatter-plot Angular Momentum vs Lyapunov Exponent.

    Parameters
    ----------
    summary_file : str or Path
        Path to the lyapunov_summary.txt file (from run_sample()).
    savefig : bool, optional
        If True, saves the plot as a PNG next to the summary file.
        If False, shows the plot interactively.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes._axes.Axes
    """
    # --- Load data ---
    data = np.loadtxt(summary_file, comments="#")

    # Column structure:
    # orig_index  local_index  lyapunov_exp  specific_energy[J/kg]  angular_momentum
    lyap = data[:, 1]
    L = data[:, 3]  # angular momentum magnitude

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(L, lyap, s=25, alpha=0.7, edgecolor='k', linewidths=0.3)

    ax.set_xlabel("Angular Momentum  |L|", fontsize=13)
    ax.set_ylabel("Lyapunov Exponent", fontsize=13)
    ax.set_title("Angular Momentum vs. Lyapunov Exponent", fontsize=14, fontweight='bold')

    # Add horizontal line at Lyap=0
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- Save or show ---
    if savefig:
        outname = Path(summary_file).with_name("AngularMomentum_vs_Lyapunov.png")
        plt.savefig(outname, dpi=300)
        print(f"Saved scatter plot: {outname}")
    else:
        plt.show()

    return fig, ax




def plot_3d_energy_L_lyapunov(summary_file, color_by_lyap=True, savefig=True):
    """
    Generate a 3D scatter plot of Specific Energy vs Angular Momentum vs Lyapunov Exponent.

    Parameters
    ----------
    summary_file : str or Path
        Path to the 'lyapunov_summary.txt' file.
    color_by_lyap : bool, optional
        If True, color-code points by Lyapunov exponent.
    savefig : bool, optional
        If True, save the plot as a PNG next to the summary file.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes3D objects
    """
    # --- Load data ---
    data = np.loadtxt(summary_file, comments="#")

    # Expected columns:
    # orig_index, local_index, lyapunov_exp, specific_energy[J/kg], angular_momentum
    lyap = data[:, 1]
    energy = data[:, 2]
    L = data[:, 3]

    # --- Create 3D figure ---
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # --- Choose coloring ---
    if color_by_lyap:
        scatter = ax.scatter(
            energy, L, lyap,
            c=lyap, cmap='plasma', s=30, alpha=0.8, edgecolor='k', linewidths=0.3
        )
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label("Lyapunov Exponent", fontsize=12)
    else:
        ax.scatter(
            energy, L, lyap,
            color='royalblue', s=25, alpha=0.7, edgecolor='k', linewidths=0.3
        )

    # --- Labels and aesthetics ---
    ax.set_xlabel("Specific Energy [J/kg]", fontsize=12)
    ax.set_ylabel(r"Angular Momentum  z-axis L_z [m^2/s]", fontsize=12)
    ax.set_zlabel("Lyapunov Exponent", fontsize=12)
    ax.set_title("3D Phase-Space Stability Distribution", fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3)

    # --- Improve perspective ---
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.show()

    # --- Save or show ---

    if savefig:
        outname = Path(summary_file).with_name("3D_Energy_L_Lyapunov.png")
        plt.savefig(outname, dpi=300)
        print(f"3D plot saved to: {outname}")
    return fig, ax



def plot_lyapunov_histogram(summary_file, bins=40, log_scale=False, savefig=True, min_lyap=1e-12):
    """
    Plot a histogram of *non-zero* (chaotic) Lyapunov exponents from the summary text file.

    Parameters
    ----------
    summary_file : str or Path
        Path to the 'lyapunov_summary.txt' file produced by run_sample().
    bins : int, optional
        Number of histogram bins (default = 40).
    log_scale : bool, optional
        If True, use logarithmic scale on the x-axis.
    savefig : bool, optional
        If True, save the plot as a PNG next to the summary file.
    min_lyap : float, optional
        Minimum lambda value to include (to exclude regular orbits with lambda approx 0).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # --- Load Lyapunov exponents ---
    data = np.loadtxt(summary_file, comments="#")
    lyap = data[:, 1]

    # --- Filter out zero or negative Lyapunov exponents ---
    chaotic_lyap = lyap[lyap > min_lyap]

    if chaotic_lyap.size == 0:
        print(" No chaotic orbits found (lambda > 0). Nothing to plot.")
        return None, None

    print(f"Plotting histogram for {len(chaotic_lyap)} chaotic orbits (lambda > {min_lyap})")

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        np.log10(chaotic_lyap),
        bins=bins,
        color="firebrick",
        edgecolor="black",
        alpha=0.8
    )

    # --- Axis labeling ---
    ax.set_xlabel("log-Lyapunov Exponent (lambda)", fontsize=13)

    ax.set_ylabel("Number of Chaotic Orbits", fontsize=13)
    ax.set_title("Distribution of Chaotic Orbit Lyapunov Exponents", fontsize=14, fontweight="bold")

    # --- Log scale option ---
    if log_scale:
        ax.set_xscale('log')
        ax.set_xlabel("Lyapunov Exponent (lambda) [log scale]", fontsize=13)

    ax.grid(alpha=0.3)

    plt.tight_layout()

    # --- Save or show ---
    if savefig:
        outname = Path(summary_file).with_name("Histogram_Lyapunov_Chaotic_log.png")
        plt.savefig(outname, dpi=300)

        print(f"Lyapunov histogram (chaotic only) saved: {outname}")
    else:
        plt.show()

    return fig, ax






#=========================================================================
# having the posibility to retrieve data manually to play with in Ipython
#=========================================================================

def recover_data(summary_file):
    # --- Load data ---
    data = np.loadtxt(summary_file, comments="#")

    # Columns:
    # orig_index  local_index  lyapunov_exp  specific_energy[J/kg]  angular_momentum
    index=data[:,0]
    lyap = data[:, 1]
    energy = data[:, 2]
    L = data[:, 3]
    frame_data=np.array([index,lyap,energy,L])
    return frame_data









def plot_lyap_comparison(file1, file2, label1="Frame 1", label2="Frame 2"):
    """
    Compare Lyapunov exponents from two different frames by producing
    a scatter plot: Lyp_frame1  vs  Lyp_frame2.

    Each input file should have at least:
        column 0 = particle index (int)
        column 1 = Lyapunov exponent (float)

    Parameters
    ----------
    file1 : str
        Path to Lyapunov output file from run_sample (frame 1).
    file2 : str
        Path to Lyapunov output file from run_sample (frame 2).
    """
    # Load table: assume whitespace separated, ignore comments
    data1 = np.loadtxt(file1)
    data2 = np.loadtxt(file2)

    # Extract columns: col0=index, col1=lyapunov
    idx1 = data1[:,0].astype(int)
    lyp1 = data1[:,1]

    idx2 = data2[:,0].astype(int)
    lyp2 = data2[:,1]

    # Match indices: assume same list of indices sampled
    # Build dictionaries for alignment
    map1 = {idx1[i]: lyp1[i] for i in range(len(idx1))}
    map2 = {idx2[i]: lyp2[i] for i in range(len(idx2))}

    # Intersection of indices present in both
    common_indices = sorted(set(map1.keys()) & set(map2.keys()))

    if len(common_indices) == 0:
        print("ERROR: no common particle indices between files.")
        return

    ly1 = np.array([map1[i] for i in common_indices])
    ly2 = np.array([map2[i] for i in common_indices])

    # Remove zeros (regular orbits)
    keep = (ly1 > 0) & (ly2 > 0)
    ly1 = ly1[keep]
    ly2 = ly2[keep]

    # --- Plot ---
    plt.figure(figsize=(8,6))
    plt.scatter(ly1, ly2, s=25, alpha=0.8, color='dodgerblue')

    # add 1:1 line
    maxv = max(ly1.max(), ly2.max())
    plt.plot([0, maxv], [0, maxv], 'r--', linewidth=1, label="1:1 line")

    plt.xlabel(f"Lyapunov ({label1})")
    plt.ylabel(f"Lyapunov ({label2})")
    plt.title("Lyapunov Exponent Comparison: Frame 1 vs Frame 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Plotted {len(ly1)} particles with nonzero Lyapunov exponents.")






def acceleration_pot_vs_positions(potential):
  """
  # Calculate acceeration along each axis
  description: plot phi vs position
  args:
      potential object from agama
  return: none 
  """
  # Determine the range for plotting based on the particle positions
  x_range = np.linspace(-5,5, 1000)
  y_range = np.linspace(-5 ,5, 1000)
  z_range = np.linspace(-5, 5, 1000)

  # Calculate force along each axis
  force_x = [potential.force(xi, 0, 0)[0] for xi in x_range]
  force_y = [potential.force(0, yi, 0)[1] for yi in y_range]
  force_z = [potential.force(0, 0, zi)[2] for zi in z_range]

  # Plotting
  plt.figure(figsize=(12, 4))

  plt.subplot(1, 3, 1)
  plt.plot(x_range, force_x)
  plt.xlabel("x'")
  plt.xscale('log')
  plt.ylabel("Force in x' direction")
  plt.title("Force vs x'")
  plt.grid(True)

  plt.subplot(1, 3, 2)
  plt.plot(y_range, force_y)
  plt.xlabel("y'")
  plt.xscale('log')
  plt.ylabel("Force in y' direction")
  plt.title("Force vs y'")
  plt.grid(True)

  plt.subplot(1, 3, 3)
  plt.plot(z_range, force_z)
  plt.xlabel("z'")
  plt.xscale('log')
  plt.ylabel("Force in z' direction")
  plt.title("Force vs z'")
  plt.grid(True)

  plt.tight_layout()
  plt.show()
  return None


def compare_potential_vs_pos_plot(potential1,potential2):
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
    potential_x1 = [potential1.potential(xi, 0, 0) for xi in x_range]
    potential_y1 = [potential1.potential(0, yi, 0) for yi in y_range]
    potential_z1 = [potential1.potential(0, 0, zi) for zi in z_range]

    potential_x2 = [potential2.potential(xi, 0, 0) for xi in x_range]
    potential_y2 = [potential2.potential(0, yi, 0) for yi in y_range]
    potential_z2 = [potential2.potential(0, 0, zi) for zi in z_range]



      
     # Plotting
    plt.plot(x_range, potential_x1,color="red",label='x-direction',ls='--')
    plt.xlabel("x'")
    #plt.xscale('log')
    plt.ylabel("potential in x' direction")
    
    plt.plot(y_range, potential_y1,color="green",label='y-direction',ls='--')
    plt.xlabel("y'")
    #plt.xscale('log')
    plt.ylabel("potential in y' direction")
     
    plt.plot(z_range, potential_z1,color='blue',label='z-direction',ls='--')
    plt.xlabel("z'")
    #plt.xscale('log')
    plt.ylabel("potential")
    plt.title("poential vs positions'")
    plt.grid(True)

   

    plt.plot(x_range, potential_x2,color="red",label='x-direction')
    plt.xlabel("x'")
    #plt.xscale('log')
    plt.ylabel("potential in x' direction")
    
    plt.plot(y_range, potential_y2,color="green",label='y-direction')
    plt.xlabel("y'")
    #plt.xscale('log')
    plt.ylabel("potential in y' direction")
     
    plt.plot(z_range, potential_z2,color='blue',label='z-direction')
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



################################################################################

def comparison_with_lyap_color(file1, file2,xcol, ycol,xlabel, ylabel, title):
    """
    Generic comparison plot:
    - xcol, ycol are the column indices for the quantity to compare
      (E=2, L=3, etc.)
    - Colors determined from Lyap exponents in col1 of each file.
    """

    d1 = np.loadtxt(file1)
    d2 = np.loadtxt(file2)

    # Extract quantities and Lyapunov exponents
    X1 = d1[:, xcol]
    X2 = d2[:, ycol]
    ly1 = d1[:, 1]
    ly2 = d2[:, 1]

    n = min(len(X1), len(X2))
    X1, X2, ly1, ly2 = X1[:n], X2[:n], ly1[:n], ly2[:n]

    # Mask: valid values only
    valid = (~np.isnan(X1)) & (~np.isnan(X2))
    X1 = X1[valid]
    X2 = X2[valid]
    ly1 = ly1[valid]
    ly2 = ly2[valid]

    # Color classification
    colors = []
    for a, b in zip(ly1, ly2):
        if a > 0 and b > 0:
            colors.append("red")      # chaotic both
        elif a <= 0 and b > 0:
            colors.append("green")    # chaotic only in frame 2
        elif a > 0 and b <= 0:
            colors.append("purple")   # chaotic only in frame 1
        else:
            colors.append("gray")     # regular-regular (optional)

    # Scatter plot
    plt.figure(figsize=(8, 6))
    
    plt.scatter(X1, X2, c=colors, s=25, alpha=0.75)

    # 1:1 reference
    minv = min(np.min(X1), np.min(X2))
    maxv = max(np.max(X1), np.max(X2))
    plt.plot([minv, maxv], [minv, maxv], 'k--', lw=1)

    # Labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)

    # Legend
    
    patches = [
        mpatches.Patch(color='red', label=f'Chaotic in both {len(colors)}'),
        mpatches.Patch(color='green', label='Chaotic only in frame 2'),
        mpatches.Patch(color='purple', label='Chaotic only in frame 1'),
        mpatches.Patch(color='gray', label='Regular in both'),
    ]
    plt.legend(handles=patches)

    plt.tight_layout()
    plt.show()

    print(f"Plotted {len(X1)} points.")


def plot_delta_quantity(file1, file2, label1, label2,col, ylabel, title):
    """
    Generic Δ-plot:
    Computes delta = quantity(frame2) - quantity(frame1)
    col = column index in your format (E=2, L=3)
    """

    # Load files
    d1 = np.loadtxt(file1)
    d2 = np.loadtxt(file2)

    # Extract the quantity of interest
    Q1 = d1[:, col]
    Q2 = d2[:, col]

    # Equalize lengths
    n = min(len(Q1), len(Q2))
    Q1 = Q1[:n]
    Q2 = Q2[:n]

    # Compute delta
    dQ = Q2 - Q1

    # Mask invalid values
    valid = ~np.isnan(dQ)
    dQ = dQ[valid]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(np.arange(len(dQ)), dQ, s=8, alpha=0.7)

    plt.axhline(0, color="k", linestyle="--", linewidth=1)

    plt.xlabel("Particle index (matching order)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Plotted Δ for {len(dQ)} particles.")


def plot_energy_comparison(file1, file2,label1,label2):
    comparison_with_lyap_color(
        file1, file2,
        xcol=2, ycol=2,
        xlabel=f"Energy {label1}",
        ylabel=f"Energy {label2}",
        title="Energy Comparison Between Frames"
    )

def plot_L_comparison(file1, file2,label1,label2):
    comparison_with_lyap_color(
        file1, file2,
        xcol=3, ycol=3,
        xlabel=f"Angular Momentum Lz{label1}",
        ylabel=f"Angular Momentum Lz{label2}",
        title="Angular Momentum Comparison Between Frames"
    )

def plot_lyap_comparison(file1, file2,label1,label2):
    comparison_with_lyap_color(
        file1, file2,
        xcol=1, ycol=1,
        xlabel=f"Lyapunov exponent {label1}",
        ylabel=f"Lyapunov exponent {label2}",
        title="Lyapunov Comparison Between Frames"
    )



def plot_delta_energy(file1, file2, label1="frame1", label2="frame2"):
    plot_delta_quantity(
        file1, file2,
        label1, label2,
        col=2,
        ylabel=f"ΔE = E({label2}) - E({label1})",
        title="Energy Change Between Frames"
    )


def plot_delta_L(file1, file2, label1="frame1", label2="frame2"):
    plot_delta_quantity(
        file1, file2,
        label1, label2,
        col=3,
        ylabel=f"ΔL = L({label2}) - L({label1})",
        title="Angular Momentum Change Between Frames"
    )




def plot_delta_E_vs_delta_L(file1, file2, label1="frame1", label2="frame2",use_lyap_colors=True):
    """
    Scatter plot of ΔE vs ΔL between two Lyapunov summary files.
    ΔE = E2 - E1
    ΔL = L2 - L1

    If use_lyap_colors=True:
      - red   = chaotic in both frames
      - green = chaotic only in frame2
      - purple= chaotic only in frame1
      - gray  = regular in both
    """

    # Load files
    d1 = np.loadtxt(file1)
    d2 = np.loadtxt(file2)

    # Extract E, L, Lyap
    E1 = d1[:, 2]
    L1 = d1[:, 3]
    ly1 = d1[:, 1]

    E2 = d2[:, 2]
    L2 = d2[:, 3]
    ly2 = d2[:, 1]

    # Equalize lengths
    n = min(len(E1), len(E2))
    E1, L1, ly1 = E1[:n], L1[:n], ly1[:n]
    E2, L2, ly2 = E2[:n], L2[:n], ly2[:n]

    # Compute deltas
    dE = E2 - E1
    dL = L2 - L1

    # Filter valid values
    valid = (~np.isnan(dE)) & (~np.isnan(dL))
    dE = dE[valid]
    dL = dL[valid]
    ly1 = ly1[valid]
    ly2 = ly2[valid]

    # Set up colors
    if use_lyap_colors:
        colors = []
        for a, b in zip(ly1, ly2):
            if a > 0 and b > 0:
                colors.append("red")        # chaotic in both
            elif a <= 0 and b > 0:
                colors.append("green")      # chaotic only in frame 2
            elif a > 0 and b <= 0:
                colors.append("purple")     # chaotic only in frame 1
            else:
                colors.append("gray")       # regular both
    else:
        colors = "blue"

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(dE, dL, c=colors, s=20, alpha=0.8)

    # Zero reference crosshair
    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.axvline(0, color="k", linestyle="--", linewidth=1)

    plt.xlabel(f"ΔE = E({label2}) - E({label1})")
    plt.ylabel(f"ΔL = L({label2}) - L({label1})")
    plt.title(f"ΔE vs ΔL Between Frames {label1} and {label2}")

    plt.grid(alpha=0.3)

    # Legend if Lyap colors are active
    if use_lyap_colors:
        patches = [
            mpatches.Patch(color='red',    label='Chaotic in both'),
            mpatches.Patch(color='green',  label='Chaotic only frame 2'),
            mpatches.Patch(color='purple', label='Chaotic only frame 1'),
            mpatches.Patch(color='gray',   label='Regular in both'),
        ]
        plt.legend(handles=patches)

    plt.tight_layout()
    plt.show()

    print(f"Plotted {len(dE)} particles.")






def comparison_plot_only_green(file1, file2,xcol, ycol,xlabel, ylabel, title):
    """
    Same structure as comparison_with_lyap_color,
    but only plots points where:
       lyapunov1 <= 0  AND  lyapunov2 > 0
    (chaotic only in frame 2 → green points)
    """

    d1 = np.loadtxt(file1)
    d2 = np.loadtxt(file2)

    # Extract quantities + lyapunov
    X1 = d1[:, xcol]
    X2 = d2[:, ycol]
    ly1 = d1[:, 1]
    ly2 = d2[:, 1]

    # Equal length
    n = min(len(X1), len(X2))
    X1, X2, ly1, ly2 = X1[:n], X2[:n], ly1[:n], ly2[:n]

    # Valid numeric mask
    valid = (~np.isnan(X1)) & (~np.isnan(X2))
    X1, X2, ly1, ly2 = X1[valid], X2[valid], ly1[valid], ly2[valid]

    # Mask for GREEN points only
    green_mask = (ly1 <= 0) & (ly2 > 0)
    purple_mask=(ly1 > 0) & (ly2 <= 0)

    # Keep only those
    X1_green = X1[green_mask]
    X2_green = X2[green_mask]

    X1_purp=X1[purple_mask]
    X2_purp=X2[purple_mask]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X1_green, X2_green, c="green", s=25, alpha=0.8, label="Chaotic only in frame 2")
    plt.scatter(X1_purp, X2_purp, c="purple", s=25, alpha=0.8, label="Chaotic only in frame 1")

    # 1:1 reference line
    if len(X1) > 0:
        minv = min(np.min(X1), np.min(X2))
        maxv = max(np.max(X1), np.max(X2))
        plt.plot([minv, maxv], [minv, maxv], 'k--', lw=1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"[INFO] Plotted {len(X1)} green points.")



def plot_mean_vs_delta(file1, file2,col,xlabel, ylabel, title,use_lyap_colors=True):
    """
    Plots mean vs delta between two frames:

        mean = 0.5*(X1 + X2)
        delta = X2 - X1

    col: which column to compare (E=2, L=3, lyap=1, etc.)
    Colors optionally based on Lyap sign in col 1 of each file (same as your logic).
    """

    d1 = np.loadtxt(file1)
    d2 = np.loadtxt(file2)

    X1 = d1[:, col]
    X2 = d2[:, col]

    ly1 = d1[:, 1]
    ly2 = d2[:, 1]

    n = min(len(X1), len(X2))
    X1, X2, ly1, ly2 = X1[:n], X2[:n], ly1[:n], ly2[:n]

    valid = (~np.isnan(X1)) & (~np.isnan(X2))
    X1, X2, ly1, ly2 = X1[valid], X2[valid], ly1[valid], ly2[valid]

    mean = 0.5 * (X1 + X2)
    delta = (X2 - X1)

    # Colors using your Lyap rule
    if use_lyap_colors:
        colors = []
        for a, b in zip(ly1, ly2):
            if a > 0 and b > 0:
                colors.append("red")      # chaotic both
            elif a <= 0 and b > 0:
                colors.append("green")    # chaotic only frame 2
            elif a > 0 and b <= 0:
                colors.append("purple")   # chaotic only frame 1
            else:
                colors.append("gray")     # regular both
    else:
        colors = "blue"

    plt.figure(figsize=(8, 6))
    plt.scatter(mean, delta, c=colors, s=25, alpha=0.75)

    # Reference line Δ=0
    plt.axhline(0, color="k", linestyle="--", lw=1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)

    if use_lyap_colors:
        patches = [
            mpatches.Patch(color='red', label='Chaotic in both'),
            mpatches.Patch(color='green', label='Chaotic only in frame 2'),
            mpatches.Patch(color='purple', label='Chaotic only in frame 1'),
            mpatches.Patch(color='gray', label='Regular in both'),
        ]
        plt.legend(handles=patches)

    plt.tight_layout()
    plt.show()

    print(f"Plotted {len(mean)} points.")



def plot_mean_vs_delta_energy(file1, file2, label1, label2):
    plot_mean_vs_delta(
        file1, file2,
        col=2,
        xlabel=f"Mean Energy 0.5*(E{label1}+E{label2})",
        ylabel=f"ΔE = E{label2}-E{label1}",
        title="Mean Energy vs ΔE"
    )

def plot_mean_vs_delta_L(file1, file2, label1, label2):
    plot_mean_vs_delta(
        file1, file2,
        col=3,
        xlabel=f"Mean L 0.5*(L{label1}+L{label2})",
        ylabel=f"ΔL = L{label2}-L{label1}",
        title="Mean Angular Momentum vs ΔL"
    )



summary_file_10040="sos_output/step1024_frame10040/lyapunov_summary.txt"
summary_file_10000="sos_output/step1024_frame10000/lyapunov_summary.txt"
summary_file_00000="sos_output/step1024_frame0000/lyapunov_summary.txt"
summary_file_14000="sos_output/step1024_frame14000/lyapunov_summary.txt"

summary_file_0c00="sos_output/step1024_frame0c00/lyapunov_summary.txt"
summary_file_0a00="sos_output/step1024_frame0c00/lyapunov_summary.txt"



"""
plot_energy_comparison(summary_file_10000,summary_file_14000,label1=10000,label2=14000)
plot_L_comparison(summary_file_10000,summary_file_14000,label1=10000,label2=14000)
plot_lyap_comparison(summary_file_10000,summary_file_14000,label1=10000,label2=14000)
plot_energy_comparison(summary_file_10000,summary_file_10040,label1=10000,label2=10040)
plot_L_comparison(summary_file_10000,summary_file_10040,label1=10000,label2=10040)
plot_lyap_comparison(summary_file_10000,summary_file_10040,label1=10000,label2=10040)
plot_delta_energy(summary_file_10000,summary_file_14000, 10000, 10040)
plot_delta_L(summary_file_10000,summary_file_14000, 10000, 10040)
plot_delta_E_vs_delta_L(summary_file_10000, summary_file_14000, label1="frame 10000", label2="frame 14000",use_lyap_colors=True)



plot_energy_vs_lyapunov(summary_file_00000)
plot_energy_vs_lyapunov(summary_file_10000)
plot_energy_vs_lyapunov(summary_file_10040)
plot_energy_vs_lyapunov(summary_file_14000)


plot_angular_momentum_vs_lyapunov(summary_file_00000)
plot_angular_momentum_vs_lyapunov(summary_file_10000)
plot_angular_momentum_vs_lyapunov(summary_file_10040)
plot_angular_momentum_vs_lyapunov(summary_file_14000)

comparison_plot_only_green(summary_file_10000,summary_file_10040,xcol=3, ycol=3,xlabel="L (frame 10000)", ylabel="L (frame 14000)",title="Green-only Angular Momentum Comparison")


"""

#plot_3d_energy_L_lyapunov(summary_file, color_by_lyap=True, savefig=True)


#plot_angular_momentum_vs_lyapunov(summary_file)
#plot_3d_energy_L_lyapunov(summary_file, color_by_lyap=True, savefig=True)
#plot_lyapunov_histogram(summary_file, bins=40, log_scale=False, savefig=True)


"""
plot_energy_vs_lyapunov(summary_file_0c00)
plot_energy_vs_lyapunov(summary_file_0a00)

plot_angular_momentum_vs_lyapunov(summary_file_0c00)
plot_angular_momentum_vs_lyapunov(summary_file_0a00)
"""

plot_3d_energy_L_lyapunov(summary_file_0a00)
plot_3d_energy_L_lyapunov(summary_file_0c00)


plot_lyapunov_histogram(summary_file_0a00)
plot_lyapunov_histogram(summary_file_0c00)


plot_L_comparison(summary_file_0a00,summary_file_0c00,label1="frame 0a00",label2="frame 0c00")
plot_energy_comparison(summary_file_0a00,summary_file_0c00,label1="frame 0a00",label2="frame 0c00")

plot_delta_E_vs_delta_L(summary_file_0a00, summary_file_0c00, label1="frame 0a00", label2="frame 0c00",use_lyap_colors=True)

plot_mean_vs_delta_energy(summary_file_0a00, summary_file_0c00, "0a00", "0c00")
plot_mean_vs_delta_L(summary_file_0a00, summary_file_0c00,"0a00", "0c00")


plot_mean_vs_delta_energy(summary_file_10000, summary_file_14000, 10000, 14000)
plot_mean_vs_delta_L(summary_file_10000, summary_file_14000, 10000, 14000)

