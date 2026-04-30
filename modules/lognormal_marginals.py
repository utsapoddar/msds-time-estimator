"""Per-course marginal distributions: lognormal for project-heavy courses, normal otherwise.

Calendar-day completion times for project-heavy courses are right-skewed (debugging,
compute blow-ups, life). Normal under-models the right tail and allows nonsensical
left tails. Lognormal matches the same (mean, sd) target but skews right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Courses where calendar-day completion is materially right-skewed.
PROJECT_HEAVY = {
    "DTSA5506",  # Data Mining Project
    "DTSA5509",  # Supervised ML
    "DTSA5510",  # Unsupervised ML
    "DTSA5511",  # Deep Learning
    "DTSA5733",  # Relational DB Design
    "DTSA5304",  # Viz project
}


def is_project_heavy(course_id: str) -> bool:
    """True if course should be modeled with a right-skewed (lognormal) marginal."""
    return course_id in PROJECT_HEAVY


def _lognorm_from_moments(mean_days: float, sd_days: float):
    """Frozen scipy lognorm with matching mean/sd in calendar-day space."""
    if mean_days <= 0 or sd_days <= 0:
        raise ValueError("mean_days and sd_days must be positive")
    cv2 = (sd_days / mean_days) ** 2
    sigma_ln_sq = np.log1p(cv2)
    sigma_ln = np.sqrt(sigma_ln_sq)
    mu_ln = np.log(mean_days) - sigma_ln_sq / 2.0
    return stats.lognorm(s=sigma_ln, scale=np.exp(mu_ln))


def _truncnorm_from_moments(mean_days: float, sd_days: float):
    """Frozen scipy truncnorm bounded at 0."""
    if mean_days <= 0 or sd_days <= 0:
        raise ValueError("mean_days and sd_days must be positive")
    # a, b are defined relative to the standard normal distribution
    a, b = (0 - mean_days) / sd_days, np.inf
    return stats.truncnorm(a, b, loc=mean_days, scale=sd_days)


def get_distribution(course_id: str, mean_days: float, sd_days: float):
    """Return frozen scipy distribution (lognorm if project-heavy, else truncnorm)."""
    if is_project_heavy(course_id):
        return _lognorm_from_moments(mean_days, sd_days)
    return _truncnorm_from_moments(mean_days, sd_days)


def predict_intervals(
    predictions_df: pd.DataFrame,
    quantiles: tuple = (0.10, 0.50, 0.90),
) -> pd.DataFrame:
    """Add p10/p50/p90 (or given quantiles) and dist_type columns to predictions.

    Input columns required: course_id, predicted_days, sd.
    Quantile column names are f"p{int(q*100)}".
    """
    qcols = [f"p{int(q * 100)}" for q in quantiles]
    out = predictions_df.copy()
    rows = []
    types = []
    for _, r in out.iterrows():
        dist = get_distribution(r["course_id"], r["predicted_days"], r["sd"])
        rows.append([float(dist.ppf(q)) for q in quantiles])
        types.append("lognormal" if is_project_heavy(r["course_id"]) else "truncnorm")
    q_df = pd.DataFrame(rows, columns=qcols, index=out.index)
    out[qcols] = q_df
    out["dist_type"] = types
    return out


def prob_finish_by(
    course_id: str, mean_days: float, sd_days: float, deadline_days: float
) -> float:
    """P(completion_days <= deadline_days) under the appropriate marginal."""
    return float(get_distribution(course_id, mean_days, sd_days).cdf(deadline_days))


if __name__ == "__main__":
    mean_days, sd_days = 100.0, 20.0
    qs = (0.10, 0.50, 0.90)

    demo = pd.DataFrame(
        [
            {"course_id": "DTSA5511", "predicted_days": mean_days, "sd": sd_days},
            {"course_id": "DTSA5002", "predicted_days": mean_days, "sd": sd_days},
        ]
    )
    result = predict_intervals(demo, quantiles=qs)
    print(f"Target mean={mean_days}, sd={sd_days}\n")
    print(result.to_string(index=False))
    print()

    for cid in ["DTSA5511", "DTSA5002"]:
        d = get_distribution(cid, mean_days, sd_days)
        p10, p50, p90 = d.ppf(0.10), d.ppf(0.50), d.ppf(0.90)
        label = "lognormal" if is_project_heavy(cid) else "truncnorm"
        print(
            f"{cid} ({label:9s}): "
            f"p10={p10:6.2f}  p50={p50:6.2f}  p90={p90:6.2f}  "
            f"left_tail_width={mean_days - p10:5.2f}  "
            f"right_tail_width={p90 - mean_days:5.2f}"
        )
