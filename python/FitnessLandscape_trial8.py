import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Fitness landscape sensitivity analysis (actual-p1 sensitivity model)
# ============================================================
# Transition-probability model:
#   P(Δ=0) = P0_FIXED.
#   The actual (unconditional) adjacent-arm probability p1 = P(Δ=1)
#   is varied across the x-axis.
#   For each p1, p2..p4 are chosen to decrease linearly with distance
#   over the active support and all probabilities p0..p4 sum to 1.
#   If the implied p4 would be negative, set p4=0 and refit linearly
#   over {1,2,3}; if p3 would be negative, use {1,2}.
#
# X-axis: p1 = P(Δ=1), grid from 0.25 to 1-P0_FIXED
# Y-axis: p_flip = P(change direction), grid from 0.0 to 1.0
#
# Metrics:
#  - mean_visits: mean visits to complete (only among completed runs)
#  - prop_perfect: proportion of perfect runs (complete with no revisits)
#  - mean_first_mistake_or_9: first mistake visit, or 9 if no mistake before completion
#  - log10_mean_visits: log10(mean_visits) for full-range visualization
#
# Outputs:
#  - CSV with all grid results
#  - Full plots:
#      * log10_mean_visits (with tick labels showing back-transformed values)
#      * prop_perfect
#      * mean_first_mistake_or_9
#  - Zoom plot:
#      * mean_visits (linear scale, colorbar rescaled to zoom region)
#      * empirical points labeled with simulated mean_visits (nearest grid)
# ============================================================

# ----------------------------
# User settings
# ----------------------------
CSV_TRL8 = "Transition_probabilities_trial8.csv"  # columns: Treatment, Probability, DeltaN
P0_FIXED = 0.0335

N_SIMS_PER_GRIDPOINT = 8000    # 8000 makes smoother heatmap than 4000
MAX_VISITS = 500
SEED = 1

# Grid
N_GRID_X = 101  # 101 makes smoother heatmap than 51
N_GRID_Y = 101
P1_GRID = np.linspace(0.25, 1.0 - P0_FIXED, N_GRID_X)  # actual p1; max leaves p0 fixed
PFLIP_GRID = np.linspace(0.0, 1.0, N_GRID_Y)              # 0..1

# Zoom window for mean_visits plot
ZOOM_XLIM = (0.25, 0.55)
ZOOM_YLIM = (0.0, 0.5)


# ----------------------------
# Read empirical trial-8 points from your CSV
# ----------------------------
def load_empirical_points(csv_path: str, p0_fixed: float):
    """
    File format:
      Treatment, Probability, DeltaN
    DeltaN values include: 0,1,2,3,4,ChangeDir

    Empirical x-axis point is the observed actual p1 = P(Δ=1).
    Empirical y-axis point is pflip = P(ChangeDir).
    """
    df = pd.read_csv(csv_path)
    df["Treatment"] = df["Treatment"].astype(str).str.strip()
    df["DeltaN"] = df["DeltaN"].astype(str).str.strip()
    df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")

    points = {}
    for trt in ["Ratio1", "Ratio5"]:
        sub = df[df["Treatment"] == trt]
        if sub.empty:
            continue

        p1_series = sub.loc[sub["DeltaN"] == "1", "Probability"]
        pf_series = sub.loc[sub["DeltaN"] == "ChangeDir", "Probability"]

        if len(p1_series) == 0 or p1_series.isna().all():
            continue
        if len(pf_series) == 0 or pf_series.isna().all():
            continue

        p1 = float(p1_series.iloc[0])
        pflip = float(pf_series.iloc[0])

        points[trt] = {"p1": p1, "pflip": pflip}

    return points


# ----------------------------
# Piecewise linear actual transition probabilities p_k
# ----------------------------
def delta_probs_from_p1(p1: float, p0_fixed: float):
    """
    Construct the actual (unconditional) transition probabilities
    P(Δ=0)..P(Δ=4).

    p0 is fixed. p1 is supplied by the grid. The remaining probability
    mass R = 1 - p0 - p1 is allocated to p2..p4 so that p1,p2,... decrease
    linearly with distance over the active support.

    With m active positive-distance categories (1..m), write
        p_k = p1 - (k-1)*b.
    The slope b is determined by
        sum_{k=1}^m p_k = 1 - p0.

    Start with m=4. If that would make p4 negative, use m=3 with p4=0.
    If that would make p3 negative, use m=2 with p3=p4=0.
    At p1 = 1-p0, p2=p3=p4=0.
    """
    total_positive = 1.0 - p0_fixed
    tol = 1e-12

    if p1 < 0 or p1 > total_positive + tol:
        raise ValueError(
            f"p1 must be between 0 and 1-p0 ({total_positive:.6f}); got {p1}"
        )

    # Try active support Δ=1..4, then 1..3, then 1..2.
    for m in (4, 3, 2):
        # Sum of p1, p1-b, ..., p1-(m-1)b equals total_positive.
        b = 2.0 * (m * p1 - total_positive) / (m * (m - 1))
        vals = [p1 - j * b for j in range(m)]

        # Require a non-increasing, non-negative sequence.
        if b >= -tol and vals[-1] >= -tol:
            probs = {0: p0_fixed, 1: p1, 2: 0.0, 3: 0.0, 4: 0.0}
            for k, val in enumerate(vals, start=1):
                probs[k] = max(0.0, val)

            # Remove tiny floating-point discrepancy while preserving p0 and p1:
            # adjust the last active category only.
            discrepancy = 1.0 - sum(probs.values())
            probs[m] += discrepancy

            if min(probs.values()) < -1e-10:
                raise ValueError(f"Negative probability generated for p1={p1}: {probs}")
            return probs

    # Endpoint: all non-reentry mass is on the adjacent arm.
    if np.isclose(p1, total_positive):
        return {0: p0_fixed, 1: total_positive, 2: 0.0, 3: 0.0, 4: 0.0}

    raise ValueError(f"No valid piecewise-linear distribution for p1={p1}")


# ----------------------------
# Simulator
# ----------------------------
def simulate_one_trial(p_flip: float, delta_probs: dict, max_visits: int, rng: np.random.Generator):
    """
    8-arm circular maze.
    Start in arm 1, initial direction random.
    Each step:
      - flip direction with probability p_flip
      - sample delta from delta_probs (0..4)
      - move accordingly on 8-arm ring
    Stop when all 8 arms visited or max_visits reached.

    Returns:
      visits_to_all8 (float) or np.nan if not completed
      first_mistake_or_9 (float): first mistake visit, or 9 if no mistake before completion
      perfect (bool): completed with no revisits before completion
    """
    current = 1
    direction = -1 if rng.random() < 0.5 else 1
    visited = {current}

    deltas = np.array([0, 1, 2, 3, 4], dtype=int)
    p = np.array([delta_probs[d] for d in deltas], dtype=float)
    cdf = np.cumsum(p / p.sum())

    first_mistake = None
    visits_to_all8 = None

    for visit_num in range(2, max_visits + 1):
        if rng.random() < p_flip:
            direction *= -1

        u = rng.random()
        delta = int(deltas[np.searchsorted(cdf, u, side="right")])

        current = ((current - 1 + direction * delta) % 8) + 1

        if first_mistake is None and current in visited:
            first_mistake = visit_num

        visited.add(current)
        if len(visited) == 8:
            visits_to_all8 = visit_num
            break

    perfect = (visits_to_all8 is not None) and (first_mistake is None)
    first_mistake_or_9 = 9.0 if first_mistake is None else float(first_mistake)

    return (np.nan if visits_to_all8 is None else float(visits_to_all8),
            first_mistake_or_9,
            perfect)


# ----------------------------
# Landscape run
# ----------------------------
def run_landscape(p1_grid, pflip_grid, n_sims, seed):
    rng = np.random.default_rng(seed)
    rows = []

    for i, p1 in enumerate(p1_grid, start=1):
        print(f"p1 row {i}/{len(p1_grid)} (p1={p1:.3f})")

        delta_probs = delta_probs_from_p1(p1, P0_FIXED)

        for pflip in pflip_grid:
            visits = np.empty(n_sims, dtype=float)
            firstm9 = np.empty(n_sims, dtype=float)
            perfects = 0
            completes = 0

            for s in range(n_sims):
                v, fm9, perf = simulate_one_trial(pflip, delta_probs, MAX_VISITS, rng)
                visits[s] = v
                firstm9[s] = fm9
                if np.isfinite(v):
                    completes += 1
                if perf:
                    perfects += 1

            v_fin = visits[np.isfinite(visits)]
            mean_vis = v_fin.mean() if len(v_fin) else np.nan
            sem_vis = v_fin.std(ddof=1) / np.sqrt(len(v_fin)) if len(v_fin) > 1 else np.nan

            mean_fm9 = float(np.mean(firstm9))
            sem_fm9 = float(np.std(firstm9, ddof=1) / np.sqrt(len(firstm9))) if len(firstm9) > 1 else np.nan

            rows.append({
                "p1": p1,
                "p_flip": pflip,
                "p0_same": P0_FIXED,
                "p1_uncond": delta_probs[1], "p2_uncond": delta_probs[2],
                "p3_uncond": delta_probs[3], "p4_uncond": delta_probs[4],
                "mean_visits": mean_vis,
                "sem_visits": sem_vis,
                "mean_first_mistake_or_9": mean_fm9,
                "sem_first_mistake_or_9": sem_fm9,
                "prop_complete": completes / n_sims,
                "prop_perfect": perfects / n_sims,
                "n_sims": n_sims
            })

    return pd.DataFrame(rows)


# ----------------------------
# Plot helpers
# ----------------------------
def nearest_sim_value(df, p1, pflip, col):
    """Return df[col] at nearest gridpoint to (p1, p_flip)."""
    idx = ((df["p1"] - p1) ** 2 + (df["p_flip"] - pflip) ** 2).idxmin()
    return df.loc[idx, col]


def plot_heatmap_contours(df_land: pd.DataFrame, value_col: str, title: str, out_png: str,
                          EMP_POINTS=None, xlim=None, ylim=None, vmin=None, vmax=None,
                          annotate_empirical_mean_visits=False, contours=True):
    x = np.sort(df_land["p1"].unique())
    y = np.sort(df_land["p_flip"].unique())
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    Z = (df_land.pivot(index="p_flip", columns="p1", values=value_col)
         .reindex(index=y, columns=x)
         .values)

    X, Y = np.meshgrid(x, y)

    plt.figure()
    hm = plt.pcolormesh(X, Y, Z, shading="auto", vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(hm)
    if value_col == "log10_mean_visits":
        ticks = cbar.get_ticks()
        cbar.set_ticklabels([f"{t:.2f} ({10**t:.0f})" for t in ticks])
        cbar.set_label("log10(mean visits) (mean visits)")
    else:
        cbar.set_label(value_col)

    if contours and np.isfinite(Z).any():
        zmin = np.nanmin(Z)
        zmax = np.nanmax(Z)
        if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
            levels = np.linspace(zmin, zmax, 10)
            cs = plt.contour(X, Y, Z, levels=levels, linewidths=0.7)
            plt.clabel(cs, inline=True, fontsize=7, fmt="%.2g")

    # Empirical points
    if EMP_POINTS:
        if "Ratio1" in EMP_POINTS:
            ex, ey = EMP_POINTS["Ratio1"]["p1"], EMP_POINTS["Ratio1"]["pflip"]
            plt.plot(ex, ey, marker="o", markersize=7, linestyle="None", label="Balanced (trial 8)")
            if annotate_empirical_mean_visits:
                mv = nearest_sim_value(df_land, ex, ey, "mean_visits")
                plt.text(ex + 0.01, ey - 0.015, f"{mv:.1f}", fontsize=8)

        if "Ratio5" in EMP_POINTS:
            ex, ey = EMP_POINTS["Ratio5"]["p1"], EMP_POINTS["Ratio5"]["pflip"]
            plt.plot(ex, ey, marker="^", markersize=7, linestyle="None", label="Unbalanced (trial 8)")
            if annotate_empirical_mean_visits:
                mv = nearest_sim_value(df_land, ex, ey, "mean_visits")
                plt.text(ex + 0.01, ey - 0.015, f"{mv:.1f}", fontsize=8)

    plt.xlabel("p1 = P(Δ=1)")
    plt.ylabel("P(flip direction)")
    plt.title(title)

    if xlim is None:
        plt.xlim(float(x.min()), float(x.max()))
    else:
        plt.xlim(xlim)

    if ylim is None:
        plt.ylim(0, 1)
    else:
        plt.ylim(ylim)

    if EMP_POINTS and (("Ratio1" in EMP_POINTS) or ("Ratio5" in EMP_POINTS)):
        plt.legend(loc="best", fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("Saved:", out_png)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    EMP_POINTS = load_empirical_points(CSV_TRL8, P0_FIXED)
    print("Empirical points (actual p1):", EMP_POINTS)

    df_land = run_landscape(P1_GRID, PFLIP_GRID, N_SIMS_PER_GRIDPOINT, seed=SEED)

    # Add log10 column for the full plot
    df_land["log10_mean_visits"] = np.log10(df_land["mean_visits"])

    out_csv = "fitness_landscape_actual_p1_pflip_piecewiseLinear.csv"
    df_land.to_csv(out_csv, index=False)
    print("Saved CSV:", out_csv)

    # ==================================================
    # FULL LANDSCAPES
    # ==================================================

    plot_heatmap_contours(
        df_land,
        value_col="log10_mean_visits",
        title="log10(Mean visits to complete) (fixed p0; piecewise-linear p1-p4)",
        out_png="fitness_landscape_log10_mean_visits_actual_p1.png",
        EMP_POINTS=EMP_POINTS
    )

    plot_heatmap_contours(
        df_land,
        value_col="prop_perfect",
        title="Proportion perfect (finish with no revisits; fixed p0; piecewise-linear p1-p4)",
        out_png="fitness_landscape_prop_perfect_actual_p1.png",
        EMP_POINTS=EMP_POINTS
    )

    plot_heatmap_contours(
        df_land,
        value_col="mean_first_mistake_or_9",
        title="Mean first-mistake visit (9 if no mistakes before completion)",
        out_png="fitness_landscape_mean_first_mistake_or_9_actual_p1.png",
        EMP_POINTS=EMP_POINTS
    )

    # ==================================================
    # ZOOMED mean_visits (linear scale, local colorbar)
    # ==================================================

    zoom_mask = (
        (df_land["p1"] >= ZOOM_XLIM[0]) & (df_land["p1"] <= ZOOM_XLIM[1]) &
        (df_land["p_flip"] >= ZOOM_YLIM[0]) & (df_land["p_flip"] <= ZOOM_YLIM[1])
    )

    zvals = df_land.loc[zoom_mask, "mean_visits"]
    vmin_zoom = float(zvals.min())
    vmax_zoom = float(np.percentile(zvals, 95))
    print("Zoom color range (vmin, vmax95):", vmin_zoom, vmax_zoom)

    plot_heatmap_contours(
        df_land,
        value_col="mean_visits",
        title="Mean visits to complete (zoom near empirical region; linear, local scale)",
        out_png="fitness_landscape_mean_visits_actual_p1_ZOOM.png",
        EMP_POINTS=EMP_POINTS,
        xlim=ZOOM_XLIM,
        ylim=ZOOM_YLIM,
        vmin=vmin_zoom,
        vmax=vmax_zoom,
        annotate_empirical_mean_visits=True
    )
