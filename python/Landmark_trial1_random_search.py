"""
Landmark_trial1_random_search.py

Exact one-sided test of Trial-1 feeder visits against random search with
replacement in the landmark-learning experiment.

Under the null model, each feeder visit independently encounters the rewarded
feeder with probability p = 1/8. For each treatment, the total number of visits
required by n bees is therefore the sum of n geometric random variables. Equivalently,
total_visits - n follows a negative-binomial distribution with parameters n and p.

The reported P value is P(Total visits <= observed total), i.e. the probability
under random search of obtaining a total number of visits equal to or lower than
the observed total.

Input:
    ../data/LandmarkLearning.csv

Output:
    Printed table with n, observed total visits, mean visits, and exact one-sided P.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import nbinom


DATA = Path(__file__).resolve().parent.parent / "data" / "LandmarkLearning.csv"
P_SUCCESS = 1 / 8

df = pd.read_csv(DATA)

trial1 = df.loc[df["Trial number"] == 1].copy()

summary = (
    trial1.groupby("Treatment")["Number of visits feeders"]
    .agg(n="count", total_visits="sum", mean_visits="mean")
    .reset_index()
)

summary["total_visits"] = summary["total_visits"].astype(int)
summary["failures_before_successes"] = summary["total_visits"] - summary["n"]

summary["p_one_sided"] = summary.apply(
    lambda row: nbinom.cdf(
        row["failures_before_successes"],
        int(row["n"]),
        P_SUCCESS,
    ),
    axis=1,
)

print(summary[["Treatment", "n", "total_visits", "mean_visits", "p_one_sided"]].to_string(index=False))
