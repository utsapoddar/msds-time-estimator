"""Tempered Bayesian course-hour updates.

The model operates in *hours*, not calendar days. Calendar-day conversion is
left to the pipeline so it can happen only at the edges:
- input: user anchor calendar days -> actual study hours
- output: predicted study hours -> target calendar days
"""

import numpy as np
import pandas as pd
from pathlib import Path

def load_data_points(data_points_csv='data_points.csv'):
    """Load the append-able log of per-student course completion observations."""
    return pd.read_csv(data_points_csv)


def prepare_data_points_for_likelihood(data_points_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with usable per-student hour observations.

    Rules:
    - if `hours` is missing but both `days` and `hours_per_week` exist, compute
      `hours = days * hours_per_week / 7`
    - rows still missing `hours` are dropped before they reach the update step
    - `effective_n = min(weight, 5.0)` tempers aggregate rows so review counts
      increase confidence without acting like unbounded independent samples
    """
    if data_points_df.empty:
        return data_points_df.copy()

    # Normalize optional columns so later numeric operations are predictable.
    out = data_points_df.copy()
    for col in ("hours", "days", "hours_per_week", "weight"):
        if col not in out.columns:
            out[col] = np.nan

    out["hours"] = pd.to_numeric(out["hours"], errors="coerce")
    out["days"] = pd.to_numeric(out["days"], errors="coerce")
    out["hours_per_week"] = pd.to_numeric(out["hours_per_week"], errors="coerce")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(1.0)

    # If only calendar duration and weekly effort are known, convert to hours.
    missing_hours = out["hours"].isna()
    imputable = missing_hours & out["days"].notna() & out["hours_per_week"].notna()
    out.loc[imputable, "hours"] = (
        out.loc[imputable, "days"] * out.loc[imputable, "hours_per_week"] / 7.0
    )

    # Drop rows that cannot contribute a positive hour observation.
    out = out[out["course_id"].notna()].copy()
    out = out[out["hours"].notna()].copy()
    out = out[out["hours"] > 0].copy()
    out["effective_n"] = out["weight"].clip(lower=0.0, upper=5.0)
    out = out[out["effective_n"] > 0].copy()
    return out

def update_params_with_posterior(params_df, data_points_df):
    """Add posterior mean hours with a tempered effective-sample-size update."""
    # Use course_params as the complete modeled course universe.
    course_ids = params_df['course_id'].tolist()
    m0 = params_df['expected_hours'].values.astype(float)
    prior_sd = np.full_like(m0, 0.20 * float(np.nanmedian(m0)))
    data_points_df = prepare_data_points_for_likelihood(data_points_df)
    data_points_df = data_points_df[data_points_df["course_id"].isin(course_ids)].copy()
    
    out = params_df.copy()

    # n_obs is the literal number of usable rows per course. `review_weight`
    # preserves aggregate review counts, while `effective_n` is the tempered
    # weight used by the posterior update.
    ns = []
    review_weights = []
    effective_ns = []
    for cid in course_ids:
        course_rows = data_points_df[data_points_df['course_id'] == cid]
        ns.append(float(len(course_rows)))
        review_weights.append(float(course_rows['weight'].sum()) if len(course_rows) else 0.0)
        effective_ns.append(float(course_rows['effective_n'].sum()) if len(course_rows) else 0.0)
    out['n_obs'] = ns
    out['review_weight'] = review_weights
    out['effective_n'] = effective_ns

    day_per_hour = (
        params_df['expected_days'].values.astype(float) /
        params_df['expected_hours'].values.astype(float)
    )

    if data_points_df.empty:
        # Fallback if no valid observations
        out['posterior_mean_hours'] = m0
        out['posterior_sd_of_mean_hours'] = prior_sd
        out['posterior_mean_days'] = out['posterior_mean_hours'] * day_per_hour
        out['posterior_sd_of_mean'] = out['posterior_sd_of_mean_hours'] * day_per_hour
        return out

    # Each course gets a tempered weighted average of prior and observations.
    posterior_means = []
    posterior_sds = []
    for i, cid in enumerate(course_ids):
        course_rows = data_points_df[data_points_df['course_id'] == cid]
        effective_n = float(course_rows['effective_n'].sum())
        if effective_n <= 0:
            posterior_means.append(float(m0[i]))
            posterior_sds.append(float(prior_sd[i]))
            continue

        weighted_hours_sum = float((course_rows['hours'] * course_rows['effective_n']).sum())
        posterior_means.append(float((m0[i] + weighted_hours_sum) / (1.0 + effective_n)))
        posterior_sds.append(float(prior_sd[i] / np.sqrt(1.0 + effective_n)))

    out['posterior_mean_hours'] = posterior_means
    out['posterior_sd_of_mean_hours'] = posterior_sds
    out['posterior_mean_days'] = out['posterior_mean_hours'] * day_per_hour
    out['posterior_sd_of_mean'] = out['posterior_sd_of_mean_hours'] * day_per_hour
    return out

if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    params = pd.read_csv(here / 'course_params.csv')
    dps = load_data_points(here / 'data_points.csv')
    updated = update_params_with_posterior(params, dps)

    cols = ['course_id', 'expected_hours', 'posterior_mean_hours', 'posterior_sd_of_mean_hours', 'n_obs', 'review_weight', 'effective_n']
    disp = updated[cols].copy()
    disp['posterior_mean_hours'] = disp['posterior_mean_hours'].round(2)
    disp['posterior_sd_of_mean_hours'] = disp['posterior_sd_of_mean_hours'].round(2)
    print("Prior vs Posterior (hours) per course [tempered update]")
    print("-" * 72)
    print(disp.to_string(index=False))
