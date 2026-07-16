"""
MonteCarlo_trial8_CIs.py

Simulate many experiments of 30 bees to obtain 95% confidence intervals for:
  (1) mean total visits required to visit all 8 arms at least once
  (2) mean visit number of the first mistake, defined as the first revisit
      of a previously visited arm, including immediate same-arm reentry
      (DeltaN = 0). A value of 9 is assigned if no mistake occurs before
      all 8 arms have been visited.

Runs 3 models per treatment:
  A) Empirical trial-8 null model using the observed transition
     probabilities and probability of changing direction (pFlip)
  B) Random-8 model: each visit is chosen uniformly from all 8 arms,
     including the current arm (p0 = 1/8)
  C) Random-7 model with low reentry: with probability p0 = 0.0335,
     the bee immediately reenters the current arm; otherwise it chooses
     uniformly among the other 7 arms

Input CSV format:
  Treatment, Probability, DeltaN

DeltaN contains 0, 1, 2, 3, or 4 for transition-distance probabilities,
and "ChangeDir" for pFlip.
"""

import numpy as np
import pandas as pd


# -------------------------
# Utilities
# -------------------------

def normalize(probs):
    probs = np.asarray(probs, dtype=float)
    s = probs.sum()
    if s <= 0:
        raise ValueError("Probabilities sum to 0; check inputs.")
    return probs / s


def ci_percentile(x, alpha=0.05):
    x = np.asarray(x, dtype=float)
    lo = np.nanpercentile(x, 100 * (alpha / 2))
    hi = np.nanpercentile(x, 100 * (1 - alpha / 2))
    return lo, hi


# -------------------------
# Load empirical trial-8 probabilities
# -------------------------

def load_empirical_tables(csv_path):
    """
    Returns:
      tables[treatment] = {
        "deltas": array([0,1,2,3,4]),
        "p_delta": array([...]),
        "pFlip": float
      }
    """
    df = pd.read_csv(csv_path)
    df["Treatment"] = df["Treatment"].astype(str).str.strip()
    df["DeltaN"] = df["DeltaN"].astype(str).str.strip()

    tables = {}
    for trt, sub in df.groupby("Treatment"):
        # delta probs
        sub_d = sub[sub["DeltaN"].isin(["0", "1", "2", "3", "4"])].copy()
        if sub_d.empty:
            raise ValueError(f"No DeltaN 0..4 rows found for treatment {trt}")

        sub_d["DeltaN_int"] = sub_d["DeltaN"].astype(int)
        sub_d = sub_d.set_index("DeltaN_int")["Probability"].astype(float)

        # ensure all 0..4 present
        for k in [0, 1, 2, 3, 4]:
            if k not in sub_d.index:
                raise ValueError(f"Treatment {trt} missing DeltaN={k}")

        p_delta = normalize([sub_d.loc[0], sub_d.loc[1], sub_d.loc[2], sub_d.loc[3], sub_d.loc[4]])

        # pFlip (ChangeDir)
        sub_cd = sub[sub["DeltaN"].str.lower() == "changedir"]
        if sub_cd.empty:
            raise ValueError(f"Treatment {trt} missing ChangeDir row")
        pFlip = float(sub_cd["Probability"].iloc[0])

        tables[trt] = {
            "deltas": np.array([0, 1, 2, 3, 4], dtype=int),
            "p_delta": p_delta,
            "pFlip": pFlip
        }

    return tables


# -------------------------
# Core metrics per bee
# -------------------------

def compute_metrics_from_visits(visit_sequence, n_arms=8):
    """
    visit_sequence: list/array of arm IDs visited in order (1..n_arms).
    Returns:
      total_visits_to_finish: int (length up to and including first time all arms visited)
      visits_till_first_mistake: int (index in 1-based visit count when first re-visit occurs)
                                9 if no mistake occurred before all 8 arms were visited 
    """
    visited = set()
    first_mistake = np.nan

    for i, arm in enumerate(visit_sequence, start=1):  # i is 1-based visit number
        if np.isnan(first_mistake) and (arm in visited):
            first_mistake = i
        visited.add(arm)
        if len(visited) == n_arms:
            if np.isnan(first_mistake):
                first_mistake = 9
            return i, first_mistake

    # if didn't finish in provided sequence
    if np.isnan(first_mistake):
        first_mistake = 9
    return len(visit_sequence), first_mistake


# -------------------------
# Simulation models
# -------------------------

def simulate_bee_empirical(prob_table, n_arms=8, start_arm=1, max_steps=500):
    """
    Empirical model:
      - start at arm 1
      - maintain a left/right direction (+/-1) that flips with probability pFlip each step
      - choose delta from {0..4} with empirical probabilities
      - delta=0 means remain in same arm (same-arm reentry)
    """
    cur = int(start_arm)
    direction = 1 if np.random.rand() < 0.5 else -1  # initial direction
    visits = [cur]

    deltas = prob_table["deltas"]
    p_delta = prob_table["p_delta"]
    pFlip = prob_table["pFlip"]

    for _ in range(max_steps):
        if np.random.rand() < pFlip:
            direction *= -1

        delta = int(np.random.choice(deltas, p=p_delta))
        if delta == 0:
            nxt = cur
        else:
            nxt = ((cur - 1 + direction * delta) % n_arms) + 1

        visits.append(nxt)
        cur = nxt

        # early stop once finished (saves time)
        total_visits, _ = compute_metrics_from_visits(visits, n_arms=n_arms)
        if len(set(visits)) == n_arms:
            # visits already includes completion point; compute final metrics precisely
            return compute_metrics_from_visits(visits, n_arms=n_arms)

    # if not finished within max_steps, treat as censored: total_visits = max_steps+1
    # compute first mistake from what we have
    total_visits, first_mistake = compute_metrics_from_visits(visits, n_arms=n_arms)
    return total_visits, first_mistake


def simulate_bee_chance8(n_arms=8, start_arm=1, max_steps=500):
    """
    Random-chance model (includes p0):
      Each step, pick ANY of the 8 arms uniformly, including current arm.
      This implies p0 = 1/8 on any step (reenter same arm immediately).
    """
    cur = int(start_arm)
    visits = [cur]

    for _ in range(max_steps):
        nxt = np.random.randint(1, n_arms + 1)  # uniform 1..8
        visits.append(nxt)
        cur = nxt

        if len(set(visits)) == n_arms:
            return compute_metrics_from_visits(visits, n_arms=n_arms)

    total_visits, first_mistake = compute_metrics_from_visits(visits, n_arms=n_arms)
    return total_visits, first_mistake

def simulate_bee_low_reentry7(n_arms=8, start_arm=1, max_steps=500, p0=0.0335):
    """
    Low-reentry uniform null:
      - with prob p0, stay in current arm (Δ=0)  -> counts as mistake in your definition
      - otherwise choose uniformly among the other 7 arms
    """
    cur = int(start_arm)
    visits = [cur]
    arms = np.arange(1, n_arms + 1)

    for _ in range(max_steps):
        if np.random.rand() < p0:
            nxt = cur
        else:
            choices = arms[arms != cur]
            nxt = int(np.random.choice(choices))

        visits.append(nxt)
        cur = nxt

        if len(set(visits)) == n_arms:
            return compute_metrics_from_visits(visits, n_arms=n_arms)

    return compute_metrics_from_visits(visits, n_arms=n_arms)


# -------------------------
# Experiment-level simulation (30 bees)
# -------------------------

def simulate_experiment(model_fn, n_bees=30, **kwargs):
    totals = np.empty(n_bees, dtype=float)
    firsts = np.empty(n_bees, dtype=float)

    for i in range(n_bees):
        t, f = model_fn(**kwargs)
        totals[i] = t
        firsts[i] = f

    return np.nanmean(totals), np.nanmean(firsts)


def run_many_experiments(
    model_fn,
    n_experiments=10000,
    n_bees=30,
    seed=1,
    progress_every=500,
    label="",
    **kwargs
):
    np.random.seed(seed)

    mean_totals = np.empty(n_experiments, dtype=float)
    mean_firsts = np.empty(n_experiments, dtype=float)

    for k in range(n_experiments):
        mt, mf = simulate_experiment(model_fn, n_bees=n_bees, **kwargs)
        mean_totals[k] = mt
        mean_firsts[k] = mf

        if (k + 1) % progress_every == 0 or (k + 1) == n_experiments:
            print(
                f"  {label}: {k + 1}/{n_experiments} "
                f"({100 * (k + 1) / n_experiments:.1f}%)"
            )

    return {
        "mean_of_means_total": float(np.nanmean(mean_totals)),
        "CI95_mean_total": ci_percentile(mean_totals, alpha=0.05),
        "mean_of_means_first": float(np.nanmean(mean_firsts)),
        "CI95_mean_first": ci_percentile(mean_firsts, alpha=0.05),
        "dist_mean_total": mean_totals,
        "dist_mean_first": mean_firsts
    }


# -------------------------
# Main
# -------------------------

def main():
    from pathlib import Path

    SCRIPT_DIR = Path(__file__).resolve().parent
    csv_path = SCRIPT_DIR / "Transition_probabilities_trial8.csv"

    n_experiments = 100000
    n_bees = 30
    seed = 1
    max_steps = 500

    tables = load_empirical_tables(csv_path)

    all_results = []
    dist_rows = []

    def add_dist(trt, model, res):
        dist_rows.append(pd.DataFrame({
            "Treatment": trt,
            "Model": model,
            "MeanTotalVisits": res["dist_mean_total"],
            "MeanVisitsFirstMistake": res["dist_mean_first"]
        }))

    for trt, prob_table in tables.items():
        print(f"\n=== Treatment: {trt} ===")

        res_emp = run_many_experiments(
            model_fn=simulate_bee_empirical,
            n_experiments=n_experiments,
            n_bees=n_bees,
            seed=seed,
            label="Empirical",
            prob_table=prob_table,
            start_arm=1,
            max_steps=max_steps
        )

        res_ch8 = run_many_experiments(
            model_fn=simulate_bee_chance8,
            n_experiments=n_experiments,
            n_bees=n_bees,
            seed=seed,
            label="Chance-8",
            start_arm=1,
            max_steps=max_steps
        )

        p0_small = 0.0335
        res_lr7 = run_many_experiments(
            model_fn=simulate_bee_low_reentry7,
            n_experiments=n_experiments,
            n_bees=n_bees,
            seed=seed,
            label=f"Low-reentry-7 (p0={p0_small})",
            start_arm=1,
            max_steps=max_steps,
            p0=p0_small
        )


        add_dist(trt, "Empirical null", res_emp)
        add_dist(trt, "Random-8", res_ch8)
        add_dist(trt, f"Random-7 (p0={p0_small})", res_lr7)


        all_results.append({
            "Treatment": trt,
            "Model": "Empirical null",
            "n_experiments": n_experiments,
            "n_bees": n_bees,
            "mean_total_visits": res_emp["mean_of_means_total"],
            "CI95_total_lo": res_emp["CI95_mean_total"][0],
            "CI95_total_hi": res_emp["CI95_mean_total"][1],
            "mean_visits_first_mistake": res_emp["mean_of_means_first"],
            "CI95_first_lo": res_emp["CI95_mean_first"][0],
            "CI95_first_hi": res_emp["CI95_mean_first"][1],
        })
        all_results.append({
            "Treatment": trt,
            "Model": "Random-8",
            "n_experiments": n_experiments,
            "n_bees": n_bees,
            "mean_total_visits": res_ch8["mean_of_means_total"],
            "CI95_total_lo": res_ch8["CI95_mean_total"][0],
            "CI95_total_hi": res_ch8["CI95_mean_total"][1],
            "mean_visits_first_mistake": res_ch8["mean_of_means_first"],
            "CI95_first_lo": res_ch8["CI95_mean_first"][0],
            "CI95_first_hi": res_ch8["CI95_mean_first"][1],
        })


        all_results.append({
            "Treatment": trt,
            "Model": f"Random-7 (p0={p0_small})",
            "n_experiments": n_experiments,
            "n_bees": n_bees,
            "mean_total_visits": res_lr7["mean_of_means_total"],
            "CI95_total_lo": res_lr7["CI95_mean_total"][0],
            "CI95_total_hi": res_lr7["CI95_mean_total"][1],
            "mean_visits_first_mistake": res_lr7["mean_of_means_first"],
            "CI95_first_lo": res_lr7["CI95_mean_first"][0],
            "CI95_first_hi": res_lr7["CI95_mean_first"][1],
        })

    dist_df = pd.concat(dist_rows, ignore_index=True)

    dist_path = SCRIPT_DIR / "MonteCarlo_distributions.csv"
    summary_path = SCRIPT_DIR / "MonteCarlo_summary.csv"

    dist_df.to_csv(dist_path, index=False)
    print(f"Wrote: {dist_path}")

    out = pd.DataFrame(all_results)
    out.to_csv(summary_path, index=False)
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()

