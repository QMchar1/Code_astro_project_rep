import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os
import pandas as pd
import scienceplots
def _frame_path(step, frame_index, root = "sos_output", ext= ".txt"):
    """
    Build the frame filepath.

    You said files are stored in: sos_output/step{step}_frameframe_{index}
    If your files actually end with .txt, pass ext=".txt".
    """
    return os.path.join(root, f"step{step}_frameframe_{frame_index}",f"lyapunov_summary{ext}")

def load_time_sequence(indices, step, root="sos_output", ext="", dtype=np.float64):
    """
    Load a sequence of SOS frame outputs and stack into (N, M) arrays.

    Parameters
    ----------
    indices : list[str|int]
        Frame identifiers used in filenames, e.g. ["0b00","0c00"] or [0, 1, 2].
    step : int
        The step used in the output folder naming (step{step}).
    root : str
        Base directory containing outputs (default: "sos_output").
    ext : str
        Optional extension (e.g., ".txt") if your files have one.
    dtype : numpy dtype
        dtype for the output arrays.

    Returns
    -------
    particle_ids : (N,) int array
        The particle index column from the files (time-invariant IDs).
    E : (N, M) array
        specific_energy[J/kg] for each particle across frames.
    lyap : (N, M) array
        lyapunov_exp across frames.
    L : (N, M) array
        angular_momentum across frames.
    frame_paths : list[str]
        The resolved frame file paths in the same order as columns in the arrays.
    """

    # ---- read first frame to define particle ordering ----
    first_path = _frame_path(step, indices[0], root=root, ext=ext)
    if not os.path.exists(first_path):
        raise FileNotFoundError(f"Missing frame file: {first_path}")

    # Data format is whitespace-delimited with a commented header line like:
    # # index lyapunov_exp specific_energy[J/kg] angular_momentum  :contentReference[oaicite:2]{index=2}
    first = np.loadtxt(first_path, comments="#", dtype=dtype)
    if first.ndim != 2 or first.shape[1] < 4:
        raise ValueError(f"Unexpected format in {first_path}: got shape {first.shape}")

    particle_ids = first[:, 0].astype(np.int64)
    N = particle_ids.size
    M = len(indices)

    lyap = np.empty((N, M), dtype=dtype)
    E    = np.empty((N, M), dtype=dtype)
    L    = np.empty((N, M), dtype=dtype)

    # fill column 0
    lyap[:, 0] = first[:, 1]
    E[:, 0]    = first[:, 2]
    L[:, 0]    = first[:, 3]

    frame_paths = [first_path]

    # ---- read remaining frames and align by particle id ----
    for j, idx in enumerate(indices[1:], start=1):
        path = _frame_path(step, idx, root=root, ext=ext)
        frame_paths.append(path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing frame file: {path}")

        arr = np.loadtxt(path, comments="#", dtype=dtype)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"Unexpected format in {path}: got shape {arr.shape}")

        ids = arr[:, 0].astype(np.int64)

        # Fast path: identical ordering
        if ids.shape == particle_ids.shape and np.array_equal(ids, particle_ids):
            lyap[:, j] = arr[:, 1]
            E[:, j]    = arr[:, 2]
            L[:, j]    = arr[:, 3]
            continue
        else:
            print("Error di,ension are not matching")

    return E, lyap, L 

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

def chaotic_regular_fractions(lyap, eps=0.0):
    """
    Compute chaotic/regular fractions vs time index.

    Parameters
    ----------
    lyap : (N, M) array
        Lyapunov exponent values for each particle (rows) across frames (cols).
    eps : float
        Threshold offset. Chaotic if lyap > eps. (Default eps=0.0)

    Returns
    -------
    f_chaotic : (M,) array
    f_regular : (M,) array
    n_valid   : (M,) int array  (# non-NaN particles per frame)
    """
    lyap = np.asarray(lyap)
    if lyap.ndim != 2:
        raise ValueError(f"lyap must be 2D (N,M), got shape {lyap.shape}")

    valid = np.isfinite(lyap)
    n_valid = valid.sum(axis=0)

    # avoid divide-by-zero: set empty frames to NaN fraction
    f_chaotic = np.full(lyap.shape[1], np.nan, dtype=float)
    f_regular = np.full(lyap.shape[1], np.nan, dtype=float)

    chaotic = (lyap > eps) & valid
    n_chaotic = chaotic.sum(axis=0)

    mask = n_valid > 0
    f_chaotic[mask] = n_chaotic[mask] / n_valid[mask]
    f_regular[mask] = 1.0 - f_chaotic[mask]

    return f_chaotic, f_regular, n_valid

def plot_chaotic_regular_fraction(t, lyap, eps=0.0):
    """
    Plot chaotic and regular orbit fraction as a function of time.
    """
    t = np.asarray(t)
    fC, fR, n_valid = chaotic_regular_fractions(lyap, eps=eps)

    if t.shape[0] != fC.shape[0]:
        raise ValueError(f"t has length {t.shape[0]} but lyap has M={fC.shape[0]} frames")

    plt.figure()
    plt.plot(t, fC, label="Chaotic fraction (lyap > eps)")
    #plt.plot(t, fR, label="Regular fraction (lyap <= eps)")
    plt.xlabel("Time")
    plt.ylabel("Fraction")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return fC, fR, n_valid

def plot_time_L_E_chaos_binary(t, E, L, lyap,eps=0.0,mode="points",sample=1,s=10):
    """
    3D plot with axes (time, angular momentum, energy),
    coloring chaotic orbits in red and regular orbits in blue.

    Parameters
    ----------
    t : (M,) array
        Time array.
    E : (N, M) array
        Energy per particle per frame.
    L : (N, M) array
        Angular momentum per particle per frame.
    lyap : (N, M) array
        Lyapunov exponent per particle per frame.
    eps : float
        Chaos threshold (chaotic if lyap > eps).
    mode : {"points","mean","median"}
        - "points": scatter all particle-time points.
        - "mean"/"median": aggregate over particles at each time.
    sample : int
        Plot every `sample`-th particle (for "points" mode).
    s : float
        Marker size.

    Returns
    -------
    fig, ax
    """
    t = np.asarray(t)
    E = np.asarray(E)
    L = np.asarray(L)
    lyap = np.asarray(lyap)

    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if not (E.shape == L.shape == lyap.shape):
        raise ValueError("E, L, lyap must all have shape (N, M)")
    N, M = E.shape
    if t.size != M:
        raise ValueError("Length of t must match number of frames")

    # Time grid
    T = np.broadcast_to(t, (N, M))

    # Valid data mask
    valid = np.isfinite(T) & np.isfinite(E) & np.isfinite(L) & np.isfinite(lyap)

    chaotic = (lyap > eps) & valid
    regular = (~chaotic) & valid

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if mode in ("mean", "median"):
        # Aggregate per time
        if mode == "mean":
            y = np.nanmean(np.where(valid, L, np.nan), axis=0)
            z = np.nanmean(np.where(valid, E, np.nan), axis=0)
        else:
            y = np.nanmedian(np.where(valid, L, np.nan), axis=0)
            z = np.nanmedian(np.where(valid, E, np.nan), axis=0)

        # Decide chaotic/regular by majority vote at each time
        frac_chaotic = np.nanmean(chaotic, axis=0)
        colors = np.where(frac_chaotic > 0.5, "red", "blue")

        for j in range(M):
            ax.scatter(t[j], y[j], z[j], c=colors[j], s=20)

    elif mode == "points":
        if sample < 1:
            raise ValueError("sample must be >= 1")

        rows = np.arange(0, N, sample)

        # Regular orbits (blue)
        ax.scatter(
            T[rows, :][regular[rows, :]],
            L[rows, :][regular[rows, :]],
            E[rows, :][regular[rows, :]],
            c="blue",
            s=s,
            label="Regular",
            alpha=0.3
        )

        # Chaotic orbits (red)
        ax.scatter(
            T[rows, :][chaotic[rows, :]],
            L[rows, :][chaotic[rows, :]],
            E[rows, :][chaotic[rows, :]],
            c="red",
            s=s,
            label="Chaotic",
        )
    else:
        raise ValueError("mode must be 'points', 'mean', or 'median'")

    ax.set_xlabel("Time")
    ax.set_ylabel("Angular momentum")
    ax.set_zlabel("Energy")

    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_time_L_E_chaos_binary_subfigs(t, E, L, lyap, eps=0.0, sample=1, s=10, ncols=None):
    """
    Replace the 3D (time, L, E) plot by ONE figure containing subfigures (panels).
    Each panel corresponds to a time t[j] and shows E vs L for all particles at that frame:
      - regular (lyap <= eps): blue
      - chaotic (lyap >  eps): red

    The number of panels is dictated by len(t) (i.e., one panel per frame).

    Parameters
    ----------
    t : (M,) array
        Time array.
    E : (N, M) array
        Energy per particle per frame.
    L : (N, M) array
        Angular momentum per particle per frame.
    lyap : (N, M) array
        Lyapunov exponent per particle per frame.
    eps : float
        Chaos threshold (chaotic if lyap > eps).
    sample : int
        Plot every `sample`-th particle to reduce clutter (>=1).
    s : float
        Marker size.
    ncols : int or None
        Number of columns in the panel grid. If None, choose automatically.

    Returns
    -------
    fig, axes
    """
    t = np.asarray(t)
    E = np.asarray(E)
    L = np.asarray(L)
    lyap = np.asarray(lyap)

    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if not (E.shape == L.shape == lyap.shape):
        raise ValueError("E, L, lyap must all have shape (N, M)")
    N, M = E.shape
    if t.size != M:
        raise ValueError("Length of t must match number of frames")
    if sample < 1:
        raise ValueError("sample must be >= 1")

    # Choose grid layout (one panel per time step)
    if ncols is None:
        # sensible automatic choice; you can override with ncols=...
        ncols = int(np.ceil(np.sqrt(M)))
    nrows = int(np.ceil(M / ncols))

    # Shared axis limits for comparability across panels
    valid_all = np.isfinite(E) & np.isfinite(L) & np.isfinite(lyap)
    if np.any(valid_all):
        Lmin, Lmax = np.nanpercentile(L[valid_all], [1, 99])
        Emin, Emax = np.nanpercentile(E[valid_all], [1, 99])
    else:
        Lmin = Lmax = Emin = Emax = None

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.0 * ncols, 2.8 * nrows),
        sharex=True, sharey=True,
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    rows = np.arange(0, N, sample)

    for j in range(M):
        ax = axes.flat[j]

        valid = np.isfinite(E[:, j]) & np.isfinite(L[:, j]) & np.isfinite(lyap[:, j])
        chaotic = (lyap[:, j] > eps) & valid
        regular = (~chaotic) & valid

        # Apply sampling
        rr = rows[regular[rows]]
        cc = rows[chaotic[rows]]

        # Plot E vs L (x=L, y=E)
        if rr.size:
            ax.scatter(L[rr, j], E[rr, j], c="blue", s=s, alpha=0.25, linewidths=0)
        if cc.size:
            ax.scatter(L[cc, j], E[cc, j], c="red",  s=s, alpha=0.90, linewidths=0)

        frac_chaos = np.nanmean(chaotic[rows]) if np.any(valid[rows]) else np.nan
        ax.set_title(f"$t={t[j]:.3g}$  (frame {j})\n$ f_{{\\rm chaos}}={frac_chaos:.2f}$", fontsize=9)

        ax.grid(True, alpha=0.25)
        if Lmin is not None:
            ax.set_xlim(Lmin, Lmax)
        if Emin is not None:
            ax.set_ylim(Emin, Emax)

        # Label only outer axes to reduce clutter
        if (j // ncols) == (nrows - 1):
            ax.set_xlabel("$L$")
        if (j % ncols) == 0:
            ax.set_ylabel("$E$")

    # Hide unused panels (if grid bigger than M)
    for k in range(M, nrows * ncols):
        axes.flat[k].set_visible(False)

    # Single legend for the whole figure
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",
                   markersize=7, label="Regular"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=7, label="Chaotic"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=True)
    plt.show()

    return fig, axes


def make_L_bins_quantile(L_values, n_bins=2, use_abs=True, use_log=True, eps=1e-300):
    """
    Quantile bin edges for angular momentum values.
    L_values: 1D array (e.g., L[:,0] or L[:,j])
    """
    x = np.asarray(L_values)
    if use_abs:
        x = np.abs(x)
    if use_log:
        x = np.log10(x + eps)

    q = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(x, q)

    # handle duplicates (common if many identical values)
    edges = np.unique(edges)
    if edges.size < 2:
        raise ValueError("Not enough unique values to form bins. Try fewer bins or disable log/abs.")
    return edges

def assign_bins(values, edges):
    """Return bin index for each value (0..K-1), -1 for out-of-range/nonfinite."""
    v = np.asarray(values)
    out = np.full(v.shape, -1, dtype=int)
    mask = np.isfinite(v)
    out[mask] = np.digitize(v[mask], edges, right=False) - 1
    out[out == len(edges) - 1] = len(edges) - 2
    out[(out < 0) | (out >= len(edges) - 1)] = -1 
    #out is a N-list of id for the particle to tell who goes to which bin 
    return out


def chaos_fraction_by_L0_bin(lyap, bin_id, eps=0.0):
    """
    Chaotic fraction vs time for fixed L0-bins.

    Parameters
    ----------
    lyap : (N, M) array
        Lyapunov exponents.
    bin_id : (N,) int array
        Bin assignment from initial L0.
    eps : float
        Chaos threshold.

    Returns
    -------
    f : (K, M) array
        f[k, j] = chaotic fraction in bin k at time j
    """
    N, M = lyap.shape
    K = bin_id.max() + 1

    f = np.full((K, M), np.nan)

    for k in range(K):
        in_bin = bin_id == k
        if not np.any(in_bin):
            continue
        f[k, :] = np.mean(lyap[in_bin, :] > eps, axis=0)

    return f

def plot_time_L_E_trajectory_lines(t, E, L, lyap,eps=0.0,sample=1,alpha=0.7,lw=0.8,alpha_regular=0.15):
    """
    3D plot with axes (time, angular momentum, energy) and line segments connecting
    each particle between successive times.

    Segment coloring:
      - red   : chaotic -> chaotic   (lyap>eps at j AND j+1)
      - blue  : regular -> regular   (lyap<=eps at j AND j+1, faded)
      - green : state change         (one chaotic, one regular)
    """


    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if not (E.shape == L.shape == lyap.shape):
        raise ValueError("E, L, lyap must all have shape (N, M)")
    N, M = E.shape
    if t.size != M:
        raise ValueError("Length of t must match number of frames (M)")
    if M < 2:
        raise ValueError("Need at least 2 time points to draw trajectory segments")
    if sample < 1:
        raise ValueError("sample must be >= 1")

    rows = np.arange(0, N, sample)

    valid = np.isfinite(E) & np.isfinite(L) & np.isfinite(lyap)
    chaos = (lyap > eps)

    segs_rr, segs_bb, segs_gg = [], [], []

    for i in rows:
        for j in range(M - 1):
            if not (valid[i, j] and valid[i, j + 1]):
                continue

            p0 = (t[j],     L[i, j],     E[i, j])
            p1 = (t[j + 1], L[i, j + 1], E[i, j + 1])

            c0 = chaos[i, j]
            c1 = chaos[i, j + 1]

            if c0 and c1:
                segs_rr.append([p0, p1])
            elif (not c0) and (not c1):
                segs_bb.append([p0, p1])
            else:
                segs_gg.append([p0, p1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def add_lc_3d(segs, color, alpha_use, label):
        if not segs:
            return None
        lc = Line3DCollection(segs, colors=color, linewidths=lw, alpha=alpha_use)
        ax.add_collection3d(lc)
        ax.plot([], [], [], color=color, linewidth=2, label=label)
        return lc

    # Prominent chaotic segments
    add_lc_3d(segs_rr, "red",   alpha,          "Chaotic → Chaotic")

    # Faded regular segments  ← THIS IS THE KEY CHANGE
    add_lc_3d(segs_bb, "blue",  alpha_regular,  "Regular → Regular")

    # Prominent transitions
    add_lc_3d(segs_gg, "green", alpha,          "Transition")

    ax.set_xlabel("Time")
    ax.set_ylabel("Angular momentum")
    ax.set_zlabel("Energy")
    ax.legend(loc="best")

    ax.set_xlim(np.nanmin(t), np.nanmax(t))
    ax.set_ylim(np.nanmin(L[rows, :]), np.nanmax(L[rows, :]))
    ax.set_zlim(np.nanmin(E[rows, :]), np.nanmax(E[rows, :]))

    plt.tight_layout()
    plt.show()

    return fig, ax

def empirical_cdf(x):
    """
    Returns sorted values and cumulative probabilities.
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, None

    xs = np.sort(x)
    ys = np.linspace(0, 1, xs.size, endpoint=False)
    return xs, ys

def plot_cdf_chaotic_L_over_time(t, L, lyap,eps=0.0,use_abs=True,use_log=False,sample_frames=1):
  
    N, M = L.shape

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, M))

    fig, ax = plt.subplots()   # ← IMPORTANT

    for j in range(0, M, sample_frames):
        chaotic = (lyap[:, j] > eps) & np.isfinite(L[:, j])
        if not np.any(chaotic):
            continue

        x = L[chaotic, j]

        if use_abs:
            x = np.abs(x)

        if use_log:
            x = np.log10(x + 1e-300)

        xs = np.sort(x)
        ys = np.arange(xs.size) / xs.size

        ax.plot(xs, ys, color=colors[j], alpha=0.8)

    ax.set_xlabel("Angular momentum" + (" (log10|L|)" if use_log else ""))
    ax.set_ylabel("CDF")
    ax.set_title("CDF of L for chaotic particles over time")
   

    # Create a proper mappable tied to time
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=t.min(), vmax=t.max())
    )
    sm.set_array([])

    fig.colorbar(sm, ax=ax, label="Time")   #  THIS FIXES THE ERROR

    plt.tight_layout()
    plt.show()

def plot_cdf_chaotic_L_over_time_counts(t, L, lyap,eps=0.0,use_abs=True,use_log=False,sample_frames=1):
    """
    Plot cumulative counts (NOT normalized CDF) of L for chaotic particles over time.
    Y-axis = number of chaotic particles with L <= x
    """


    N, M = L.shape

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, M))

    fig, ax = plt.subplots()

    for j in range(0, M, sample_frames):

        chaotic = (lyap[:, j] > eps) & np.isfinite(L[:, j])
        if not np.any(chaotic):
            continue

        x = L[chaotic, j]

        if use_abs:
            x = np.abs(x)

        if use_log:
            x = np.log10(x + 1e-300)

        xs = np.sort(x)

        # CUMULATIVE COUNTS (not normalized)
        ys = np.arange(1, xs.size + 1)

        ax.plot(xs, ys, color=colors[j], alpha=0.85)

    ax.set_xlabel("Angular momentum" + (" (log10|L|)" if use_log else ""))
    ax.set_ylabel("Cumulative number of chaotic particles")
    ax.set_title("Cumulative distribution of L for chaotic population")

    # Time colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=t.min(), vmax=t.max())
    )
    sm.set_array([])

    fig.colorbar(sm, ax=ax, label="Time")

    plt.tight_layout()
    plt.show()

def count_transition_orbits(t, lyap, eps=0.0):
    """
    Count number of particles that change dynamical state
    between successive frames.

    Returns
    -------
    t_mid : (M-1,) array
        Time values between frames (midpoints).
    n_trans : (M-1,) array
        Number of transitions between t[j] and t[j+1].
    """

    # Boolean chaos state
    chaotic = lyap > eps

    # XOR between consecutive frames
    transitions = chaotic[:, :-1] ^ chaotic[:, 1:]

    # Count transitions at each step
    n_trans = np.sum(transitions, axis=0)

    # Associate time with midpoint of the interval
    t_mid = 0.5 * (t[:-1] + t[1:])

    return t_mid, n_trans

def plot_transition_histogram(t, lyap, eps=0.0):
    """
    Histogram: number of transitioning orbits vs time.
    """

    t_mid, n_trans = count_transition_orbits(t, lyap, eps=eps)

    plt.figure()
    plt.bar(t_mid, n_trans, width=(t_mid[1] - t_mid[0]), align='center')

    plt.xlabel("Time")
    plt.ylabel("Number of transitioning orbits")
    plt.title("Orbit transitions vs time")

    plt.tight_layout()
    plt.show()

def plot_chaos_fraction_L0_E0_bins(t, L, E, lyap,n_bins=2,eps=0.0,L_use_abs=True,L_use_log=False,E_use_abs=True,E_use_log=False):



    # -----------------------
    # Build L0 bins
    # -----------------------
    L0 = L[:, 0]
    L_edges = make_L_bins_quantile(L0, n_bins=n_bins, use_abs=L_use_abs, use_log=L_use_log)

    xL0 = np.abs(L0) if L_use_abs else L0
    if L_use_log:
        xL0 = np.log10(xL0 + 1e-300)

    bin_id_L = assign_bins(xL0, L_edges)
    f_L0 = chaos_fraction_by_L0_bin(lyap, bin_id_L, eps=eps)

    # -----------------------
    # Build E0 bins
    # -----------------------
    E0 = E[:, 0]
    E_edges = make_L_bins_quantile(E0, n_bins=n_bins, use_abs=E_use_abs, use_log=E_use_log)

    xE0 = np.abs(E0) if E_use_abs else E0
    if E_use_log:
        xE0 = np.log10(xE0 + 1e-300)

    bin_id_E = assign_bins(xE0, E_edges)
    f_E0 = chaos_fraction_by_L0_bin(lyap, bin_id_E, eps=eps)

    # -----------------------
    # Plot
    # -----------------------
    plt.figure()
    lin_sty=[":",'-']

    # ----- L bins (solid) -----
    for k in range(f_L0.shape[0]):

        if n_bins == 2:
            label = "Low |L₀|" if k == 0 else "High |L₀|"
        else:
            label = f"L₀ bin {k}"

        plt.plot(t, f_L0[k], linewidth=2, label=label,linestyle=lin_sty[k],color="blue")

    # ----- E bins (dashed) -----
    for k in range(f_E0.shape[0]):

        if n_bins == 2:
            # Energy is negative:
            #   bin 0 = most negative (deeply bound)
            #   bin 1 = less negative (weakly bound)
            label = "Deeply bound (high |E₀|)" if k == 0 else "Weakly bound (low |E₀|)"
        else:
            label = f"E₀ bin {k}"

        plt.plot(
            t,
            f_E0[k],
            linestyle=lin_sty[k],
            linewidth=2,
            label=label,
            color="red"
        )

    plt.xlabel("Time")
    plt.ylabel("Chaotic fraction")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

def plot_transition_histograms_split(t, lyap, eps=1e-1):
    """
    Two stacked histograms:
      Top    : Regular → Chaotic transitions
      Bottom : Chaotic → Regular transitions
    """

    t = np.asarray(t)
    lyap = np.asarray(lyap)

    if lyap.ndim != 2:
        raise ValueError("lyap must have shape (N, M)")

    N, M = lyap.shape
    if t.size != M:
        raise ValueError("t length must match number of frames")

    # Chaos state
    chaotic = lyap > eps

    # Transition masks
    reg_to_chaos = (~chaotic[:, :-1]) & chaotic[:, 1:]
    chaos_to_reg = chaotic[:, :-1] & (~chaotic[:, 1:])

    # Counts per timestep
    n_reg_to_chaos = np.sum(reg_to_chaos, axis=0)
    n_chaos_to_reg = np.sum(chaos_to_reg, axis=0)

    # Associate times with midpoints between frames
    t_mid = 0.5 * (t[:-1] + t[1:])

    # ---------------------------
    # Plot
    # ---------------------------
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    # Top panel: regular → chaotic
    axes[0].bar(
        t_mid,
        n_reg_to_chaos,
        width=(t_mid[1] - t_mid[0]),
        color="red",
        alpha=0.8
    )
    axes[0].set_ylabel("# transitions")
    axes[0].set_title("Regular → Chaotic")

    # Bottom panel: chaotic → regular
    axes[1].bar(
        t_mid,
        n_chaos_to_reg,
        width=(t_mid[1] - t_mid[0]),
        color="blue",
        alpha=0.8
    )
    axes[1].set_ylabel("# transitions")
    axes[1].set_title("Chaotic → Regular")
    axes[1].set_xlabel("Time")

    plt.tight_layout()
    plt.show()

    return t_mid, n_reg_to_chaos, n_chaos_to_reg

def _bin_edges_from_centers(x):
    """
    Given bin centers x (length n), return edges (length n+1).
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        w = 1.0
        return np.array([x[0] - 0.5*w, x[0] + 0.5*w])
    dx = np.diff(x)
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * dx[0]
    edges[-1] = x[-1] + 0.5 * dx[-1]
    return edges

def _rolling_smooth(y, window=5):
    """
    Simple moving-average smoother.
    """
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y
    w = int(window)
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")

def _gaussian_moment_fit(x, y):
    """
    Simple 'fit' for a Gaussian-shaped curve via weighted moments:
      mu  = weighted mean
      sig = weighted std
      A   = max(y) (or weighted peak)
    This is not a strict nonlinear least-squares fit, but it’s a very good
    quick overlay for unimodal peaks.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_pos = np.clip(y, 0.0, None)
    s = y_pos.sum()
    if s <= 0:
        return None

    mu = (x * y_pos).sum() / s
    sig2 = (y_pos * (x - mu) ** 2).sum() / s
    sig = np.sqrt(max(sig2, 1e-30))
    A = y_pos.max()

    g = A * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    return g

def plot_transition_histograms_split_pretty(t, lyap,eps=0.0,smooth_window=7,add_gaussian_fit=True):
    """
    Two stacked 'pretty' histograms:
      Top    : Regular → Chaotic transitions
      Bottom : Chaotic → Regular transitions

    Styling improvements:
      - crisp bar edges (black contour)
      - optional step outline
      - optional smooth trend + gaussian-shaped overlay
    """
    t = np.asarray(t)
    lyap = np.asarray(lyap)

    if lyap.ndim != 2:
        raise ValueError("lyap must have shape (N, M)")
    N, M = lyap.shape
    if t.size != M:
        raise ValueError("t length must match number of frames")
    if M < 2:
        raise ValueError("Need at least 2 frames to define transitions")

    chaotic = lyap > eps

    reg_to_chaos = (~chaotic[:, :-1]) & chaotic[:, 1:]
    chaos_to_reg = (chaotic[:, :-1]) & (~chaotic[:, 1:])

    n_rc = np.sum(reg_to_chaos, axis=0).astype(float)
    n_cr = np.sum(chaos_to_reg, axis=0).astype(float)

    t_mid = 0.5 * (t[:-1] + t[1:])
    edges = _bin_edges_from_centers(t_mid)
    width = np.diff(edges)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7), constrained_layout=True)

    def draw_panel(ax, x_centers, counts, color, title):
        # Bars with "contour" edges
        ax.bar(
            x_centers,
            counts,
            width=width,
            align="center",
            color=color,
            alpha=0.55,
            edgecolor="black",
            linewidth=0.8,
        )

        # Step-outline on top (reads very cleanly)
        # Build a step array from edges
        y_step = np.r_[counts, counts[-1]]
        ax.step(edges, y_step, where="post", linewidth=1.2, color="black", alpha=0.9)

        # Smooth trend
        y_smooth = _rolling_smooth(counts, window=smooth_window)
        ax.plot(x_centers, y_smooth, linewidth=2.0, color="black", alpha=0.8)

        # Optional gaussian-shaped overlay (moment-based)
        if add_gaussian_fit:
            g = _gaussian_moment_fit(x_centers, counts)
            if g is not None:
                ax.plot(x_centers, g, linewidth=2.0, linestyle="--", color="black", alpha=0.9)

        ax.set_title(title)
        ax.set_ylabel("# transitions")
        ax.grid(True, alpha=0.2)

    draw_panel(axes[0], t_mid, n_rc, "red",  "Regular → Chaotic")
    draw_panel(axes[1], t_mid, n_cr, "blue", "Chaotic → Regular")

    axes[1].set_xlabel("Time")

    plt.show()
    return t_mid, n_rc.astype(int), n_cr.astype(int)

def plot_chaotic_count_histogram_pretty(t, lyap,eps=0.0,smooth_window=7,add_gaussian_fit=True,title="Number of chaotic orbits vs time",bar_alpha=0.55,edge_lw=0.8,step_lw=1.2,trend_lw=2.0):
    """
    Pretty histogram-style plot of number of chaotic orbits over time,
    with crisp bin edges and optional smooth trend + gaussian-like overlay.

    Chaotic definition: lyap > eps

    Parameters
    ----------
    t : (M,) array
        Time array.
    lyap : (N, M) array
        Lyapunov exponents.
    eps : float
        Chaos threshold.
    smooth_window : int
        Moving-average window for the solid trend line.
    add_gaussian_fit : bool
        If True, overlay a dashed gaussian-shaped moment-fit curve.
    """
    t = np.asarray(t)
    lyap = np.asarray(lyap)

    if lyap.ndim != 2:
        raise ValueError("lyap must have shape (N, M)")
    N, M = lyap.shape
    if t.size != M:
        raise ValueError("t length must match number of frames (M)")

    # Count chaotic at each time
    chaotic = (lyap > eps) & np.isfinite(lyap)
    n_chaotic = chaotic.sum(axis=0).astype(float)

    # Build histogram-like bins from time centers
    edges = _bin_edges_from_centers(t)
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)

    # Bars with crisp edges
    ax.bar(
        t,
        n_chaotic,
        width=widths,
        align="center",
        alpha=bar_alpha,
        edgecolor="black",
        linewidth=edge_lw,
    )

    # Step outline
    y_step = np.r_[n_chaotic, n_chaotic[-1]]
    ax.step(edges, y_step, where="post", linewidth=step_lw, color="black", alpha=0.9)

    # Smooth trend
    y_smooth = _rolling_smooth(n_chaotic, window=smooth_window)
    ax.plot(t, y_smooth, linewidth=trend_lw, color="black", alpha=0.8)

    # Optional gaussian-like overlay
    if add_gaussian_fit:
        g = _gaussian_moment_fit(t, n_chaotic)
        if g is not None:
            ax.plot(t, g, linewidth=trend_lw, linestyle="--", color="black", alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("# chaotic orbits")
    ax.grid(True, alpha=0.2)

    plt.show()
    return n_chaotic.astype(int)

def plot_characteristic_X_lambda_scatter(lyap, eps=0.0, s=12):
    """
    Scatter plot of the characteristic function X_lambda.

    For each particle i and frame j:
        if lyap[i,j] > eps:
            plot point at (i, j+1)

    x-axis: particle index
    y-axis: time column index + 1
    """

    lyap = np.asarray(lyap)
    if lyap.ndim != 2:
        raise ValueError("lyap must be 2D")

    N, M = lyap.shape

    # Boolean chaotic mask
    chaotic = lyap > eps

    # Get indices of chaotic points
    i_idx, j_idx = np.where(chaotic)

    # j+1 = characteristic function value
    y_vals = j_idx + 1

    plt.figure(figsize=(10, 5))
    plt.scatter(i_idx, y_vals, s=s)

    plt.xlabel("Particle index")
    plt.ylabel("X_lambda (j + 1)")
    plt.title("Characteristic function scatter: chaotic events")

    plt.yticks(np.arange(1, M+1))

    plt.tight_layout()
    plt.show()

def plot_characteristic_X_lambda(lyap, eps=0.0,delta=1,dpi=100, title=None,save_path=None):
    """
    Characteristic-function plot:
      X_lambda[i, j] = (j+1) if lyap[i, j] > eps else 0

    x-axis: particle index i (row index)
    y-axis: time/frame column j
    color: X_lambda value (0 means regular; j+1 means chaotic at that frame)

    Parameters
    ----------
    lyap : (N, M) array
        Lyapunov exponent array.
    eps : float
        Chaos threshold.
    title : str or None
        Plot title.

    Returns
    -------
    X : (N, M) int array
        The characteristic matrix.
    """
    lyap = np.asarray(lyap)
    if lyap.ndim != 2:
        raise ValueError("lyap must be 2D with shape (N, M)")

    N, M = lyap.shape
    # --- Make figure exactly N x M pixels ---
    fig_width_in  = N / dpi
    fig_height_in = M / dpi

    # Create column labels 1..M, broadcast to (N,M)
    col_ids = np.arange(1, M + 1, dtype=int)[None, :]  # shape (1,M)

    # Build characteristic matrix
    X = np.where((lyap > eps) & (lyap < delta), col_ids, 0).astype(int)

    # Plot: x = particle index, y = time column index
    fig, ax = plt.subplots(figsize=(fig_width_in,fig_height_in ), constrained_layout=True)

    # imshow expects array indexed as [y, x], so we transpose:
    # rows become time (M), cols become particles (N)
    im = ax.imshow(
        X.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest"
    )

    ax.set_xlabel("Particle index (row in lyap)")
    ax.set_ylabel("Time/frame column index j")
    ax.set_title(title or r"Characteristic function $X_\lambda$: value=j+1 if chaotic else 0")

    # y ticks as actual column indices
    ax.set_yticks(np.arange(M))
    ax.set_yticklabels([str(j) for j in range(M)])

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("X_lambda value (0=regular, j+1=chaotic)")
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0)

    plt.show()
    return X

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


###########################################
#######Study of orbits near separatix######
###########################################


def transition_indices(lyap, eps=0.0):
    """
    Find indices (row, col) of *transition events* in a Lyapunov array.

    A transition event occurs between consecutive columns j -> j+1.

    Returns two sets of 1D arrays:
      - chaotic -> regular:  (rows_cr, cols_cr) where cols_cr = j (transition from j to j+1)
      - regular -> chaotic:  (rows_rc, cols_rc) where cols_rc = j

    Interpretation:
      (rows_rc[k], cols_rc[k]) means particle=rows_rc[k] transitions
      regular->chaotic between time columns cols_rc[k] and cols_rc[k]+1.
    """
    lyap = np.asarray(lyap)
    if lyap.ndim != 2:
        raise ValueError("lyap must be a 2D array with shape (N, M)")

    # Define state per entry
    chaotic = (lyap > eps) & np.isfinite(lyap)

    # Transitions between consecutive columns
    reg_to_chaos = (~chaotic[:, :-1]) & chaotic[:, 1:]
    chaos_to_reg = chaotic[:, :-1] & (~chaotic[:, 1:])

    # (row, col) indices where the transition happens (col = left index j)
    rows_rc, cols_rc = np.where(reg_to_chaos)
    rows_cr, cols_cr = np.where(chaos_to_reg)

    return (rows_cr, cols_cr), (rows_rc, cols_rc)


def build_transition_arrays(lyap, eps=0.0):
    lyap = np.asarray(lyap)
    chaotic = lyap > eps

    reg_to_chaos = (~chaotic[:, :-1]) & chaotic[:, 1:]
    chaos_to_reg = chaotic[:, :-1] & (~chaotic[:, 1:])

    # Get indices
    rc_rows, rc_cols = np.where(reg_to_chaos)
    cr_rows, cr_cols = np.where(chaos_to_reg)

    # Stack into Nx2 arrays
    RC = np.column_stack((rc_rows, rc_cols))  # regular -> chaotic
    CR = np.column_stack((cr_rows, cr_cols))  # chaotic -> regular

    return RC, CR


def _cdf_xy(x, normalize=True):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, None
    xs = np.sort(x)
    if normalize:
        ys = np.arange(1, xs.size + 1) / xs.size
    else:
        ys = np.arange(1, xs.size + 1)
    return xs, ys


def plot_transition_L_cdfs_over_time(t, L, RC, CR,use_abs_L=True,use_log_L=False,normalize=True,which_L="pre", sample_steps=1,alpha=0.85):
    """
    Overplot CDF(L) curves over time, restricted to transition populations.

    RC, CR: arrays of shape (n_events, 2) where each row is [particle_index, j],
            with j referring to the transition j -> j+1.

    Produces two subplots:
      top: RC (Regular -> Chaotic)
      bottom: CR (Chaotic -> Regular)
    """
    t = np.asarray(t)
    L = np.asarray(L)
    RC = np.asarray(RC)
    CR = np.asarray(CR)

    if L.ndim != 2:
        raise ValueError("L must be (N, M)")
    N, M = L.shape
    if t.shape != (M,):
        raise ValueError("t must have length M")
    if which_L not in ("pre", "post"):
        raise ValueError("which_L must be 'pre' or 'post'")
    if M < 2:
        raise ValueError("Need at least 2 frames")

    # Transition times j live in 0..M-2
    t_mid = 0.5 * (t[:-1] + t[1:])
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=np.nanmin(t_mid), vmax=np.nanmax(t_mid))

    def prep_L(x):
        if use_abs_L:
            x = np.abs(x)
        if use_log_L:
            x = np.log10(x + 1e-300)
        return x

    def panel(ax, events, title):
        if events.size == 0:
            ax.text(0.5, 0.5, "No events", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.grid(True, alpha=0.2)
            return

        # Ensure integer columns
        p_idx = events[:, 0].astype(int)
        j_idx = events[:, 1].astype(int)

        # Defensive filtering (prevents your crash)
        ok = (p_idx >= 0) & (p_idx < N) & (j_idx >= 0) & (j_idx < M-1)
        p_idx = p_idx[ok]
        j_idx = j_idx[ok]

        if p_idx.size == 0:
            ax.text(0.5, 0.5, "No valid events after filtering", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            ax.grid(True, alpha=0.2)
            return

        # Loop over unique transition steps
        for j in np.unique(j_idx)[::sample_steps]:
            parts = p_idx[j_idx == j]
            if parts.size == 0:
                continue

            jj = j if which_L == "pre" else (j + 1)
            x = prep_L(L[parts, jj])

            xs, ys = _cdf_xy(x, normalize=normalize)
            if xs is None:
                continue

            ax.plot(xs, ys, color=cmap(norm(t_mid[j])), alpha=alpha)

        ax.set_title(title + f" (L at {'j' if which_L=='pre' else 'j+1'})")
        ax.set_ylabel("CDF" if normalize else "Cumulative count")
        ax.grid(True, alpha=0.2)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    panel(axes[0], RC, "RC: Regular → Chaotic")
    panel(axes[1], CR, "CR: Chaotic → Regular")

    xlab = "Angular momentum"
    if use_abs_L and use_log_L:
        xlab = "log10(|L|)"
    elif use_abs_L:
        xlab = "|L|"
    elif use_log_L:
        xlab = "log10(L)"
    axes[1].set_xlabel(xlab)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Transition time (midpoint)")

    plt.show()
    return fig, axes



def compute_dtmax_and_frac_chaotic_exact(t, lyap,eps=0.0):
    """
    dt_max[i] = max duration (in time units) that particle i stays continuously
                in the same regime (chaotic or regular), using frame boundaries t.

    frac_chaotic[i] = fraction of frames where particle i is chaotic.

    Notes:
    - Uses state at frames: chaotic[i,j] = (lyap[i,j] > eps)
    - A flip at j means state differs between j and j+1; boundary index is j+1
    - Segment durations are computed as t[end] - t[start]
    """

    chaotic = (lyap > eps) & np.isfinite(lyap)
    frac_chaotic = chaotic.mean(axis=1)
    N, M = lyap.shape
    dt_max = np.zeros(N, dtype=float)

    for i in range(N):
        # flip positions j where state changes between j and j+1
        flip_j = np.where(chaotic[i, :-1] ^ chaotic[i, 1:])[0]  # in 0..M-2

        # boundaries are start indices of segments in frame-index space
        # include 0 and last frame index (M-1)
        boundaries = np.concatenate(([0], flip_j + 1, [M - 1]))
        boundaries = np.unique(boundaries)          # ensure sorted unique
        boundaries.sort()

        # durations between consecutive boundaries
        # segments: [boundaries[k] ... boundaries[k+1]] in time
        durs = np.diff(t[boundaries])
        # if there are no diffs (shouldn't happen if M>1), fallback
        dt_max[i] = durs.max() if durs.size else (t[-1] - t[0])

    return dt_max, frac_chaotic
def plot_X_lambda_scatter_colored_by_bin(lyap,bin_id,t,eps=0.0,delta=None,s=1,alpha=0.9,cmap="viridis",save_path=None,dpi=100):
    """
    Scatter characteristic plot of chaotic events:
      plot (particle_index i, simulation time t[j])
    for events meeting condition, colored by particle bin_id.

    Parameters
    ----------
    lyap : (N, M) array
        Lyapunov exponent array.
    bin_id : (N,) int array
        Bin label per particle. -1 allowed for invalid/unbinned particles.
    t : (M,) array
        Simulation time array corresponding to lyap columns.
    eps : float
        Chaos threshold, lyap > eps.
    delta : float or None
        If provided, only count events with eps < lyap < delta.
    t_marker : float
        Time at which to draw a dashed horizontal line.
    """

    lyap = np.asarray(lyap)
    bin_id = np.asarray(bin_id)
    t = np.asarray(t)

    if lyap.ndim != 2:
        raise ValueError("lyap must be shape (N, M)")

    N, M = lyap.shape

    if bin_id.shape != (N,):
        raise ValueError(f"bin_id must have shape ({N},)")

    if t.shape != (M,):
        raise ValueError(f"t must have shape ({M},), one time value per lyap column")

    if delta is None:
        mask = lyap > eps
    else:
        mask = (lyap > eps) & (lyap < delta)

    i_idx, j_idx = np.where(mask)

    valid_particles = bin_id[i_idx] >= 0
    i_idx = i_idx[valid_particles]
    j_idx = j_idx[valid_particles]

    # Now X_lambda is scaled to physical/simulation time
    y_vals = t[j_idx]

    bins_used = bin_id[i_idx]
    if bins_used.size:
        K = int(bins_used.max()) + 1
    else:
        K = int(bin_id[bin_id >= 0].max()) + 1

    cmap_obj = plt.cm.get_cmap(cmap, K)

    fig_width_in = N / dpi
    fig_height_in = M / dpi

    fig, ax = plt.subplots(
        figsize=(fig_width_in, fig_height_in),
        facecolor="white"
    )
    ax.set_facecolor("white")

    sc = ax.scatter(
        i_idx,
        y_vals,
        c=bins_used,
        cmap=cmap_obj,
        s=s,
        alpha=alpha,
        linewidths=0
    )

    
    # Dashed line at key time
    ax.axhline(
        792,
        linestyle="--",
        linewidth=2,
        color="black",
        alpha=0.8,
        label=rf"$t=792$ mximum chaotic fraction"
    )

     # Dashed line at key time
    ax.axhline(
        560,
        linestyle="--",
        linewidth=3,
        color="red",
        alpha=0.8,
        #label=rf"$t=530$"
    )
    # Dashed line at key time
    ax.axhline(
        1019,
        linestyle="--",
        linewidth=3,
        color="red",
        alpha=0.8,
        #label=rf"$t=530$"
    )
    ax.axvline(
        400,
        linestyle="--",
        linewidth=3,
        color="red",
        alpha=0.8,
        #label=rf"$t=530$"
    )
    ax.axvline(
        850,
        linestyle="--",
        linewidth=3,
        color="red",
        alpha=0.8,
        #label=rf"$t=530$"
    )


    ax.set_xlabel("Particle index (E epoch)")
    ax.set_ylabel(r"$X_{\lambda} \equiv t$")
    ax.set_title("Characteristic plot: chaotic events colored by L-bin")

    ax.set_ylim(t.min(), t.max())

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Angular momentum bin ID")
    cbar.set_ticks(np.arange(0, K))

    ax.legend(loc="best", frameon=False)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

    plt.show()
    return fig, ax


def compute_dtmax_chaos_and_frac_chaotic_exact(t, lyap, eps=0.0):
    """
    dt_max_chaos[i] = max duration (time units) that particle i stays continuously chaotic,
                      using frame boundaries t (no midpoints).

    frac_chaotic[i] = fraction of frames where particle i is chaotic.

    Notes:
    - chaotic[i,j] = (lyap[i,j] > eps)
    - boundaries = [0, flip+1, M-1] define constant-state segments
    - segment k spans [boundaries[k], boundaries[k+1]] with duration t[end]-t[start]
      and state chaotic[i, boundaries[k]].
    """


    chaotic = (lyap > eps) & np.isfinite(lyap)
    frac_chaotic = chaotic.mean(axis=1)

    N, M = lyap.shape
    dt_max_chaos = np.zeros(N, dtype=float)

    for i in range(N):
        flip_j = np.where(chaotic[i, :-1] ^ chaotic[i, 1:])[0]  # 0..M-2

        boundaries = np.concatenate(([0], flip_j + 1, [M - 1]))
        boundaries = np.unique(boundaries)
        boundaries.sort()

        # segment durations
        durs = np.diff(t[boundaries])  # length = n_segments
        if durs.size == 0:
            dt_max_chaos[i] = 0.0
            continue

        # segment states: state is constant over segment k and equals chaotic at its start
        seg_states = chaotic[i, boundaries[:-1]]  # length matches durs

        # keep only chaotic segments
        chaos_durs = durs[seg_states]

        dt_max_chaos[i] = chaos_durs.max() if chaos_durs.size else 0.0

    return dt_max_chaos, frac_chaotic
def plot_cdf_dtmax_and_frac(dt_max, frac_chaotic, log_dt=False):
    """
    Two-panel CDF plot:
      - dt_max CDF
      - frac_chaotic CDF
    """
    dt_max = np.asarray(dt_max, dtype=float)
    frac_chaotic = np.asarray(frac_chaotic, dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)

    # --- dt_max CDF ---
    x = dt_max
    if log_dt:
        x = np.log10(x[x > 0])  # safe log
        xs, ys = empirical_cdf(x)
        if xs is not None:
            axes[0].plot(xs, ys)
            axes[0].set_xlabel("log10(Δt_max)")
    else:
        xs, ys = empirical_cdf(x)
        if xs is not None:
            axes[0].plot(xs, ys)
            axes[0].set_xlabel("Δt_max")

    axes[0].set_ylabel("CDF")
    axes[0].set_title("CDF of longest inter-transition gap (Δt_max)")
    axes[0].grid(True, alpha=0.2)

    # --- frac_chaotic CDF ---
    xs, ys = empirical_cdf(frac_chaotic)
    if xs is not None:
        axes[1].plot(xs, ys)
    axes[1].set_xlabel("Fraction of frames chaotic")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("CDF of chaotic frame fraction")
    axes[1].grid(True, alpha=0.2)

    plt.show()
    return fig, axes

def plot_frac_vs_particle_index_colored_by_Lbin(frac, bin_id_L,title="Chaotic fraction vs particle index (colored by L0z bin)",s=10,alpha=0.9,cmap="viridis"):
    frac = np.asarray(frac, dtype=float)
    bin_id_L = np.asarray(bin_id_L)

    N = frac.size
    if bin_id_L.shape != (N,):
        raise ValueError("bin_id_L must have same length as frac")

    idx = np.arange(N)

    valid = bin_id_L >= 0
    idx_v = idx[valid]
    frac_v = frac[valid]
    bins_v = bin_id_L[valid].astype(int)

    K = int(bins_v.max()) + 1 if bins_v.size else 1
    cmap_obj = plt.cm.get_cmap(cmap, K)

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    sc = ax.scatter(idx_v, frac_v, c=bins_v, cmap=cmap_obj, s=s, alpha=alpha, linewidths=0)

    ax.set_xlabel("Particle index")
    ax.set_ylabel("Fraction of frames chaotic")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("L₀z bin ID")
    cbar.set_ticks(np.arange(0, K))

    plt.show()
    return fig, ax

def plot_frac_vs_particle_index_by_Lbin(frac, bin_id_L,title="Chaotic fraction vs particle index by L₀ bin",alpha=0.85,s=10):
    frac = np.asarray(frac, dtype=float)
    bin_id_L = np.asarray(bin_id_L)

    if frac.ndim != 1:
        raise ValueError("frac must be 1D")
    N = frac.size
    if bin_id_L.shape != (N,):
        raise ValueError("bin_id_L must have same length as frac")

    idx = np.arange(N)

    valid = bin_id_L >= 0
    bins = np.unique(bin_id_L[valid]).astype(int)
    K = bins.size
    if K == 0:
        raise ValueError("No valid bins in bin_id_L")

    fig, axes = plt.subplots(K, 1, sharex=True, figsize=(12, 2.2*K), constrained_layout=True)
    if K == 1:
        axes = [axes]

    for ax, b in zip(axes, bins):
        m = (bin_id_L == b)
        ax.scatter(idx[m], frac[m], s=s, alpha=alpha)
        ax.set_ylabel("frac")
        ax.set_title(f"L₀ bin {b}  (N={m.sum()})")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Particle index")
    fig.suptitle(title, y=1.02)
    plt.show()
    return fig, axes

def plot_dtmax_vs_particle_index_by_Lbin(dt_max, bin_id_L,title="Δt_max vs particle index by L₀ bin",alpha=0.85,s=10,logy=False):
    
    N = dt_max.size
    idx = np.arange(N)

    valid = (bin_id_L >= 0) & np.isfinite(dt_max)
    bins = np.unique(bin_id_L[valid]).astype(int)
    K = bins.size
    if K == 0:
        raise ValueError("No valid bins in bin_id_L (or dt_max not finite)")

    fig, axes = plt.subplots(K, 1, sharex=True, figsize=(12, 2.2*K), constrained_layout=True)
    if K == 1:
        axes = [axes]

    for ax, b in zip(axes, bins):
        m = valid & (bin_id_L == b)
        ax.scatter(idx[m], dt_max[m], s=s, alpha=alpha)
        ax.set_ylabel("Δt_max")
        ax.set_title(f"L₀ bin {b}  (N={m.sum()})")
        ax.grid(True, alpha=0.2)
        if logy:
            ax.set_yscale("log")

    axes[-1].set_xlabel("Particle index")
    fig.suptitle(title, y=1.02)
    plt.show()
    return fig, axes



def plot_dtmax_vs_particle_index_colored_by_Lbin(dt_max, bin_id_L,title="Δt_max vs particle index (colored by L₀ bin)",s=10,alpha=0.9,cmap="viridis",logy=False):
    dt_max = np.asarray(dt_max, dtype=float)
    bin_id_L = np.asarray(bin_id_L)

    N = dt_max.size
    if bin_id_L.shape != (N,):
        raise ValueError("bin_id_L must have same length as dt_max")

    idx = np.arange(N)

    valid = (bin_id_L >= 0) & np.isfinite(dt_max)
    idx_v = idx[valid]
    dt_v = dt_max[valid]
    bins_v = bin_id_L[valid].astype(int)

    if bins_v.size == 0:
        raise ValueError("No valid points to plot")

    K = int(bins_v.max()) + 1
    cmap_obj = plt.cm.get_cmap(cmap, K)

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    sc = ax.scatter(idx_v, dt_v, c=bins_v, cmap=cmap_obj, s=s, alpha=alpha, linewidths=0)

    ax.set_xlabel("Particle index")
    ax.set_ylabel("Δt_max")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    if logy:
        ax.set_yscale("log")

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("L₀ bin ID")
    cbar.set_ticks(np.arange(0, K))

    plt.show()
    return fig, ax


def plot_frac_vs_L(frac, L,use_L0=True,frame_idx=0,use_abs=True,use_log=False,s=10,alpha=0.8,title="Fraction of frames chaotic vs angular momentum"):
    """
    Scatter: frac_chaotic (per particle) vs L (per particle).
    """
 
    if L.ndim == 2:
        Lx = L[:, 0] if use_L0 else L[:, frame_idx]
    elif L.ndim == 1:
        Lx = L
    else:
        raise ValueError("L must be 1D or 2D")

    if frac.shape[0] != Lx.shape[0]:
        raise ValueError("frac and L must have same length in particle dimension")

    x = Lx.copy()
    if use_abs:
        x = np.abs(x)
    if use_log:
        x = np.log10(x + 1e-300)

    m = np.isfinite(x) & np.isfinite(frac)

    plt.figure(figsize=(8, 5))
    plt.scatter(x[m], frac[m], s=s, alpha=alpha)

    xlabel = "|L_z0|" if use_abs else "L_z0"
    if use_log:
        xlabel = f"log10({xlabel})"

    plt.xlabel(xlabel)
    plt.ylabel("Fraction of frames chaotic")
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    
def plot_dtmax_vs_L(dt_max, L,use_L0=True,frame_idx=0,use_abs=True,use_logL=False,log_dt=False,s=10,alpha=0.8,title="Δt_max vs angular momentum"):
    """
    Scatter: dt_max (per particle) vs L (per particle).
    """
    dt_max = np.asarray(dt_max, dtype=float)
    L = np.asarray(L)

    if L.ndim == 2:
        Lx = L[:, 0] if use_L0 else L[:, frame_idx]
    elif L.ndim == 1:
        Lx = L
    else:
        raise ValueError("L must be 1D or 2D")

    if dt_max.shape[0] != Lx.shape[0]:
        raise ValueError("dt_max and L must have same length in particle dimension")

    x = Lx.copy()
    y = dt_max.copy()

    if use_abs:
        x = np.abs(x)
    if use_logL:
        x = np.log10(x + 1e-300)

    m = np.isfinite(x) & np.isfinite(y)
    if log_dt:
        m = m & (y > 0)
        y = np.log10(y + 1e-300)

    plt.figure(figsize=(8, 5))
    plt.scatter(x[m], y[m], s=s, alpha=alpha)

    xlabel = "|L|" if use_abs else "L"
    if use_logL:
        xlabel = f"log10({xlabel})"
    ylabel = "Δt_max" if not log_dt else "log10(Δt_max)"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_frac_vs_dtmax_colored_by_Lbin(frac, dt_max, bin_id_L,log_dt=False,s=12,alpha=0.85,cmap="viridis",title="Fraction chaotic vs Δt_max (colored by L₀ bin)"):
    
    # Filter valid
    m = (bin_id_L >= 0) & np.isfinite(frac) & np.isfinite(dt_max)
    x = dt_max[m]
    y = frac[m]
    b = bin_id_L[m].astype(int)

    if log_dt:
        m2 = x > 0
        x = np.log10(x[m2] + 1e-300)
        y = y[m2]
        b = b[m2]

    if b.size == 0:
        raise ValueError("No valid points to plot after filtering")

    K = int(b.max()) + 1
    cmap_obj = plt.cm.get_cmap(cmap, K)

    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    sc = ax.scatter(x, y, c=b, cmap=cmap_obj, s=s, alpha=alpha, linewidths=0)

    ax.set_xlabel("log10(Δt_max)" if log_dt else "Δt_max")
    ax.set_ylabel("Fraction of frames chaotic")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("L₀ bin ID")
    cbar.set_ticks(np.arange(0, K))

    plt.show()
    return fig, ax

def plot_frac_vs_dtmax_by_Lbin_subplots(frac, dt_max, bin_id_L,log_dt=False,s=12,alpha=0.85,title="Fraction chaotic vs Δt_max by L₀ bin"):
    frac = np.asarray(frac, dtype=float)
    dt_max = np.asarray(dt_max, dtype=float)
    bin_id_L = np.asarray(bin_id_L)

    if not (frac.shape == dt_max.shape == bin_id_L.shape):
        raise ValueError("frac, dt_max, bin_id_L must have same shape")

    valid = (bin_id_L >= 0) & np.isfinite(frac) & np.isfinite(dt_max)
    bins = np.unique(bin_id_L[valid]).astype(int)
    if bins.size == 0:
        raise ValueError("No valid bins to plot")

    K = bins.size
    fig, axes = plt.subplots(K, 1, figsize=(8.5, 2.4*K), sharex=True, sharey=True, constrained_layout=True)
    if K == 1:
        axes = [axes]

    # Precompute global x for consistent axes
    x_all = dt_max[valid]
    if log_dt:
        x_all = x_all[x_all > 0]
        x_all = np.log10(x_all + 1e-300)

    for ax, b in zip(axes, bins):
        m = valid & (bin_id_L == b)

        x = dt_max[m]
        y = frac[m]

        if log_dt:
            keep = x > 0
            x = np.log10(x[keep] + 1e-300)
            y = y[keep]

        ax.scatter(x, y, s=s, alpha=alpha, linewidths=0)
        ax.set_title(f"L₀ bin {b}  (N={np.sum(m)})")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("log10(Δt_max)" if log_dt else "Δt_max")
    axes[0].set_ylabel("Fraction of frames chaotic")
    for ax in axes[1:]:
        ax.set_ylabel("")

    fig.suptitle(title, y=1.01)
    plt.show()
    return fig, axes


def plot_frac_vs_lyap0(frac, lyap,use_log=False,eps_floor=1e-300,s=12,alpha=0.85,title="Fraction of frames chaotic vs initial Lyapunov exponent"):
    """
    Scatter plot: frac_chaotic (per particle) vs lyap[:,0] (initial frame).

    Parameters
    ----------
    frac : (N,) array
        Fraction of frames chaotic per particle.
    lyap : (N, M) array
        Lyapunov exponent array.
    use_abs : bool
        Plot against |lyap0|.
    use_log : bool
        Plot against log10(lyap0) (or log10(|lyap0|) if use_abs=True).
        Values <= 0 are masked unless use_abs=True.
    eps_floor : float
        Small floor to avoid log(0).
    """

    if frac.ndim != 1:
        raise ValueError("frac must be 1D (N,)")
    if lyap.ndim != 2 or lyap.shape[0] != frac.size:
        raise ValueError("lyap must be (N,M) with same N as frac")

    lyap0 = lyap[:, 0].copy()
    x = lyap0

    m = np.isfinite(x) & np.isfinite(frac)

    if use_log:
        m = m & (x > 0)  # can't log negative/zero
        x = np.log10(x + eps_floor)
        xlabel = "log10(lyap0)"
    else:
        xlabel =  "lyap0"

    plt.figure(figsize=(8, 5))
    plt.scatter(x[m], frac[m], s=s, alpha=alpha, linewidths=0)
    plt.xlabel(xlabel)
    plt.ylabel("Fraction of frames chaotic")
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_chaotic_fraction_with_axial_ratio(lyap,t,shape_csv_path,shell=6,eps=0.0,axial_col="b/a",time_col="#time",shell_col="n",t_marker=728,save_path=None,dpi=150):
    """
    Plot chaotic fraction vs time and overplot axial ratio b/a
    for a chosen density shell.

    Parameters
    ----------
    lyap : (N, M) array
        Lyapunov exponent array.
    t : (M,) array
        Simulation time corresponding to lyap columns.
    shape_csv_path : str
        Path to shape/axial-ratio CSV file.
    shell : int
        Density shell to use. Default is shell 6.
    eps : float
        Chaos threshold.
    axial_col : str
        Column to plot as axial ratio. Usually "b/a".
    t_marker : float
        Time where maximum elongation is expected.
    """
    plt.style.use(['science'])
    lyap = np.asarray(lyap)
    t = np.asarray(t)

    if lyap.ndim != 2:
        raise ValueError("lyap must have shape (N, M)")

    N, M = lyap.shape

    if t.shape != (M,):
        raise ValueError(f"t must have shape ({M},), got {t.shape}")

    f_chaotic, f_regular, n_valid = chaotic_regular_fractions(
        lyap, eps=eps
    )

    shape_df = pd.read_csv(shape_csv_path)

    shell_df = shape_df[shape_df[shell_col] == shell].copy()

    if shell_df.empty:
        raise ValueError(f"No data found for shell={shell}")

    shell_df = shell_df.sort_values(time_col)

    t_shape = shell_df[time_col].to_numpy()
    axial_ratio = shell_df[axial_col].to_numpy()

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(
        t,
        f_chaotic,
        linewidth=2,
        label="Chaotic fraction"
    )

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Chaotic fraction")
    ax1.set_ylim(0, 1)

    ax1.axvline(
        t_marker,
        linestyle="--",
        linewidth=1.2,
        color="red",
        alpha=0.8,
        label=rf"$t={t_marker}$ maximum chaotic fraction"
    )
    ax1.axvline(
        560,
        linestyle="--",
        linewidth=1.2,
        color="red",
        alpha=0.8,
        label=rf"$t=560$ earlier sign of chaos spreading"
    )

    ax2 = ax1.twinx()

    ax2.plot(
        t_shape,
        axial_ratio,
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
        label=rf"Shell {shell}: {axial_col}"
    )

    ax2.set_ylabel(rf"Axial ratio ${axial_col}$")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
        frameon=False
    )

    ax1.set_title(
        rf"Chaotic fraction vs time overplotted with shell {shell} axial ratio"
    )

    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()

    return fig, ax1, ax2


indices =generate_hex_indices("3000", "10000", "40")#generate_hex_indices("8000", "10000", "1000")
print(indices) #["8000", "9000", "a000", "b000", "c000
step=128
E, lyap, L = load_time_sequence(indices, step,root="sos_output_128",ext=".txt")
print(E.shape, lyap.shape, L.shape)   # (N, M) each
M = len(indices)
sim_dt = 1         # integrator dt
sample_dt =1  #number of fame you skiped         # time between saved frames (e.g., every 128 sim steps if sim_dt=0.001)
t = make_time_array(M, sample_dt, sim_dt, t0=192)
print(t)
fC, fR, n_valid = plot_chaotic_regular_fraction(t, lyap, eps=0.0)
#plot_characteristic_X_lambda(lyap, eps=0.0,delta=1,dpi=1)######
RC,CR=build_transition_arrays(lyap, eps=0.0)
#print(RC)
#print(CR)
#plot_transition_histogram(t, lyap, eps=0.0)
#plot_cdf_chaotic_L_over_time(t, L, lyap, eps=0.0, use_abs=False, use_log=False)
#plot_cdf_chaotic_L_over_time_counts(t, L,lyap,eps=0.0, use_abs=True, use_log=False,sample_frames=1)
#plot_characteristic_X_lambda_scatter(lyap, eps=0.0)

#plot_transition_L_cdfs_over_time(t, L,RC,CR,use_abs_L=False,use_log_L=False,normalize=False,which_L="pre",sample_steps=1)


L0 = L[:, 0]
L_edges = make_L_bins_quantile(L0, n_bins=4, use_abs=True, use_log=False)
bin_id_L = assign_bins(np.abs(L0), L_edges)

E0 = E[:, 0]
E_edges = make_L_bins_quantile(E0, n_bins=2, use_abs=True, use_log=False)
bin_id_E = assign_bins(np.abs(E0), E_edges)



#plot_time_L_E_chaos_binary(t, E, L, lyap,mode="points",sample=4,s=2)
#plot_time_L_E_chaos_binary_subfigs(t, E, L, lyap, eps=0.0, sample=1, s=10, ncols=None)
#plot_time_L_E_trajectory_lines(t, E, L, lyap,eps=0.0,sample=8,alpha=0.7,lw=0.8)
#res = plot_chaos_fraction_L0_E0_bins(t, L, E, lyap,n_bins=2,eps=0.0,L_use_abs=True,  L_use_log=False,E_use_abs=False, E_use_log=False)
#plot_transition_histograms_split(t, lyap)

#t_mid, n_rc, n_cr = plot_transition_histograms_split_pretty(t, lyap,eps=1e-1,smooth_window=7,add_gaussian_fit=False)

#n_chaotic = plot_chaotic_count_histogram_pretty(t, lyap,eps=0.0,smooth_window=7,add_gaussian_fit=True)

plot_X_lambda_scatter_colored_by_bin(lyap,bin_id_L,t,eps=9e-2,s=10,cmap="viridis",save_path="X_lambda_by_L-bin.png")
#plot_X_lambda_scatter_colored_by_bin(lyap,bin_id_E,eps=0.0,s=10,cmap="viridis",save_path="X_lambda_by_E-bin.png")




#dt_max, frac =compute_dtmax_chaos_and_frac_chaotic_exact(t, lyap, eps=0.0)
dt_max, frac =compute_dtmax_and_frac_chaotic_exact(t, lyap, eps=0.0)
#print(dt_max.min(), dt_max.max())

"""
print(len(dt_max),len(frac))
plot_cdf_dtmax_and_frac(dt_max, frac, log_dt=False)
plot_frac_vs_particle_index_colored_by_Lbin(frac, bin_id_L)
plot_frac_vs_particle_index_by_Lbin(frac, bin_id_L,title="Chaotic fraction vs particle index by L₀ bin",alpha=0.85,s=10)

plot_dtmax_vs_particle_index_by_Lbin(dt_max, bin_id_L,title="Δt_max vs particle index by L₀ bin",alpha=0.85,s=10,logy=False)
plot_dtmax_vs_particle_index_colored_by_Lbin(dt_max, bin_id_L, logy=False)

plot_dtmax_vs_L(dt_max, L, use_L0=True, use_abs=False, log_dt=False)
plot_frac_vs_L(frac, L, use_L0=True, use_abs=True, use_log=False)
plot_frac_vs_dtmax_colored_by_Lbin(frac, dt_max, bin_id_L,log_dt=False,s=12,alpha=0.85,cmap="viridis",title="Fraction chaotic vs Δt_max (colored by L₀ bin)")
plot_frac_vs_dtmax_by_Lbin_subplots(frac, dt_max, bin_id_L,log_dt=False,s=12,alpha=0.85,title="Fraction chaotic vs Δt_max by L₀ bin")
plot_frac_vs_lyap0(frac, lyap,use_log=False,eps_floor=1e-300,s=12,alpha=0.85,title="Fraction of frames chaotic vs initial Lyapunov exponent")
plot_frac_vs_lyap0(frac, lyap,use_log=True,eps_floor=1e-300,s=12,alpha=0.85,title="Fraction of frames chaotic vs initial Lyapunov exponent")
"""
fig, ax1, ax2 = plot_chaotic_fraction_with_axial_ratio(lyap=lyap,t=t,shape_csv_path="r10A_shape_sph.csv",shell=6,eps=0.0,t_marker=792,save_path="chaotic_fraction_with_axial_ratio_shell6.png")


def select_particles_by_frac(frac,threshold,step,start_index,mode="above",return_mask=False,save_csv=True,filename=None):
    """
    Select particle indices based on fraction of chaotic time
    and optionally save them to a CSV file.

    Parameters
    ----------
    frac : (N,) array
        Fraction of frames chaotic per particle.
    threshold : float
        Threshold value between 0 and 1.
    step : int
        Sampling step during frame analysis (normally 1024 or 512).
    mode : str
        Selection rule:
            "above"  -> frac >= threshold
            "below"  -> frac <= threshold
            "equal"  -> frac == threshold
    return_mask : bool
        If True, also return the boolean mask.
    save_csv : bool
        If True, save selected indices to CSV.
    filename : str or None
        Custom filename. If None, auto-generated.

    Returns
    -------
    indices : (K,) array of int
        Particle indices (scaled by step) satisfying the condition.
    mask : (N,) bool, optional
        Boolean mask (only if return_mask=True).
    """

    frac = np.asarray(frac, dtype=float)

    if frac.ndim != 1:
        raise ValueError("frac must be a 1D array")

    if mode == "above":
        mask = frac >= threshold
    elif mode == "below":
        mask = frac <= threshold
    elif mode == "equal":
        mask = frac == threshold
    else:
        raise ValueError("mode must be 'above', 'below', or 'equal'")

    indices = np.where(mask)[0]
    scaled_indices = start_index+(step * indices)

    # -------- Save to CSV --------
    if save_csv:
        if filename is None:
            filename = f"selected_particles_mode-{mode}_thr-{threshold:.3f}_step-{step}.csv"

        np.savetxt(
            filename,
            scaled_indices,
            fmt="%d",
            delimiter=",",
            header="particle_index",
            comments=""
        )

    if return_mask:
        return scaled_indices, mask

    return scaled_indices


def select_particles_by_early_chaos(lyap,t,step,start_index,t_min=192,t_max=512,threshold=0.0,save_csv=True,filename=None,return_mask=False,return_first_time=False,return_first_col=False,sort_by_first_time=True):
    """
    Select particles that show any positive Lyapunov exponent
    in the time interval [t_min, t_max].

    Parameters
    ----------
    lyap : (N, M) array
        Lyapunov exponent array. Rows = particles, columns = times.
    t : (M,) array
        Time array corresponding to the columns of lyap.
    step : int
        Sampling step used when mapping back to original particle indices.
    start_index : int
        Offset for original particle indexing.
    t_min : float
        Start of time window.
    t_max : float
        End of time window.
    threshold : float
        Chaos threshold. Usually 0.0, or slightly above 0 if you want to
        avoid numerical noise.
    save_csv : bool
        If True, save selected indices to CSV.
    filename : str or None
        Output filename. If None, auto-generated.
    return_mask : bool
        If True, also return the boolean particle-selection mask.
    return_first_time : bool
        If True, also return first chaotic time for each selected particle.
    return_first_col : bool
        If True, also return first chaotic column index in the full lyap array.
    sort_by_first_time : bool
        If True, sort selected particles by earliest chaos onset.

    Returns
    -------
    scaled_indices : (K,) array of int
        Original particle indices satisfying the condition.
    mask : (N,) bool, optional
        Boolean mask over all particles.
    first_times : (K,) array, optional
        Earliest time at which each selected particle becomes chaotic.
    first_cols : (K,) array of int, optional
        Earliest column index in the full lyap array where chaos appears.
    """

    lyap = np.asarray(lyap, dtype=float)
    t = np.asarray(t, dtype=float)

    if lyap.ndim != 2:
        raise ValueError("lyap must be a 2D array of shape (N, M)")
    if t.ndim != 1:
        raise ValueError("t must be a 1D array")
    if lyap.shape[1] != t.size:
        raise ValueError("lyap.shape[1] must match len(t)")

    # columns inside requested time window
    time_mask = (t >= t_min) & (t <= t_max)
    if not np.any(time_mask):
        raise ValueError("No time values fall inside the requested interval")

    window_cols = np.where(time_mask)[0]
    lyap_window = lyap[:, time_mask]

    # True where particle is chaotic at least once in window
    chaotic_mask = np.any(lyap_window > threshold, axis=1)

    # selected row indices in the sampled array
    row_indices = np.where(chaotic_mask)[0]
    scaled_indices = start_index + step * row_indices

    # first positive occurrence inside the window
    first_times = None
    first_cols = None
    if row_indices.size > 0:
        first_pos_in_window = np.argmax(lyap_window[row_indices] > threshold, axis=1)
        first_cols = window_cols[first_pos_in_window]
        first_times = t[first_cols]

        if sort_by_first_time:
            order = np.argsort(first_times)
            row_indices = row_indices[order]
            scaled_indices = scaled_indices[order]
            first_times = first_times[order]
            first_cols = first_cols[order]

    # save CSV
    if save_csv:
        if filename is None:
            filename = f"selected_particles_early_chaos_t-{t_min:g}-{t_max:g}_thr-{threshold:.3e}_step-{step}.csv"

        if row_indices.size > 0:
            out = np.column_stack((scaled_indices, first_times, first_cols))
            header = "particle_index,first_chaotic_time,first_chaotic_col"
            np.savetxt(filename, out, fmt=["%d", "%.8g", "%d"], delimiter=",",
                       header=header, comments="")
        else:
            np.savetxt(filename, np.empty((0, 3)),
                       fmt=["%d", "%.8g", "%d"], delimiter=",",
                       header="particle_index,first_chaotic_time,first_chaotic_col",
                       comments="")

    outputs = [scaled_indices]

    if return_mask:
        outputs.append(chaotic_mask)
    if return_first_time:
        outputs.append(first_times)
    if return_first_col:
        outputs.append(first_cols)

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


selected_idx, first_times = select_particles_by_early_chaos(lyap=lyap,t=t,step=128,start_index=600001,t_min=192,t_max=512,threshold=0.0,return_first_time=True,save_csv=True,filename="early_chaotic_particles.csv")

print(selected_idx)
print(first_times)



import numpy as np
import pandas as pd


def select_particles_by_chaotic_fraction(
    lyap,
    output_csv,
    energy_epoch_min=0,
    energy_epoch_max=2000,
    chaotic_fraction_min=0.4,
    chaotic_fraction_max=0.7,
    eps=9e-2,
    start_index=600_001,
    sampling_step=128,
    time=None,
    time_min=192,
    time_max=1024,
):
    """
    Select particles whose chaotic time fraction lies within a chosen range.

    Parameters
    ----------
    lyap : array, shape (N_particles, N_times)
        Lyapunov exponent array. Particle axis is assumed to be sorted by
        energy-epoch index.

    output_csv : str
        Path where the selected particle table will be saved.

    energy_epoch_min, energy_epoch_max : int
        Range of energy-epoch indices to search, e.g. 0 to 2000.

    chaotic_fraction_min, chaotic_fraction_max : float
        Keep particles with chaotic fraction between these limits.

    eps : float
        Chaos threshold. A particle is chaotic when lyap > eps.

    start_index : int
        Actual particle index corresponding to energy-epoch index 0.

    sampling_step : int
        Step converting energy-epoch index to actual particle index.

    time : array, optional
        Time array of length N_times.

    time_min, time_max : float, optional
        Restrict chaotic fraction calculation to a time window.

    Returns
    -------
    selected_df : pandas.DataFrame
        Table of selected particles.
    """

    lyap = np.asarray(lyap)

    if lyap.ndim != 2:
        raise ValueError(f"lyap must have shape (N_particles, N_times), got {lyap.shape}")

    N, M = lyap.shape

    i0 = max(0, int(energy_epoch_min))
    i1 = min(N - 1, int(energy_epoch_max))

    if i0 > i1:
        raise ValueError("Invalid energy-epoch index range.")

    # Time mask
    if time is None:
        tmask = np.ones(M, dtype=bool)
    else:
        time = np.asarray(time)
        if len(time) != M:
            raise ValueError("time must have length equal to lyap.shape[1]")

        tmask = np.ones(M, dtype=bool)

        if time_min is not None:
            tmask &= time >= time_min

        if time_max is not None:
            tmask &= time <= time_max

    if not np.any(tmask):
        raise ValueError("No time samples selected by time_min/time_max.")

    rows = []

    for eidx in range(i0, i1 + 1):
        y = lyap[eidx, tmask]

        valid = np.isfinite(y)
        n_valid = valid.sum()

        if n_valid == 0:
            continue

        chaotic = y[valid] > eps
        chaotic_fraction = chaotic.sum() / n_valid

        if chaotic_fraction_min <= chaotic_fraction <= chaotic_fraction_max:
            actual_index = start_index + eidx * sampling_step

            rows.append({
                "energy_epoch_index": eidx,
                "actual_index": actual_index,
                "chaotic_fraction": chaotic_fraction,
                "n_chaotic": int(chaotic.sum()),
                "n_valid": int(n_valid),
                "eps": eps,
            })

    selected_df = pd.DataFrame(rows)

    selected_df.to_csv(output_csv, index=False)

    print(f"Selected {len(selected_df)} particles")
    print(f"Saved to: {output_csv}")

    return selected_df

selected = select_particles_by_chaotic_fraction(
    lyap=lyap,
    output_csv="selected_chaotic_particles.csv",
    energy_epoch_min=0,
    energy_epoch_max=2000,
    chaotic_fraction_min=0.4,
    chaotic_fraction_max=0.9,
    eps=0,
    start_index=600001,
    sampling_step=128,
    time=t,
)

print(selected)

print("10 %:")
chaos_id=select_particles_by_frac(frac,0.1,128,600001)
print(list(chaos_id))
print(len(chaos_id))
print("20 %:")
chaos_id=select_particles_by_frac(frac,0.2,128,600001)
print(list(chaos_id))
print(len(chaos_id))
print("30 %:")
chaos_id=select_particles_by_frac(frac,0.3,128,600001)
print(list(chaos_id))
print(len(chaos_id))
print("40 %:")
chaos_id=select_particles_by_frac(frac,0.4,128,600001)
print(list(chaos_id))
print(len(chaos_id))
print("50 %:")
chaos_id=select_particles_by_frac(frac,0.5,128,600001)
print(list(chaos_id))
print(len(chaos_id))
print("60 %:")
chaos_id=select_particles_by_frac(frac,0.6,128,600001)
print(list(chaos_id))
print(len(chaos_id))
print("70 %:")
chaos_id=select_particles_by_frac(frac,0.7,128,600001)
print(list(chaos_id))
print(len(chaos_id))
print("80 %:")
chaos_id=select_particles_by_frac(frac,0.8,128,600001)
print(list(chaos_id))
print(len(chaos_id))
print("90 %:")
chaos_id=select_particles_by_frac(frac,0.9,128,600001)
print(list(chaos_id))
print(len(chaos_id))

