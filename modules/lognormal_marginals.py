"""Per-course marginal distributions: log-t for project-heavy courses, t otherwise.

Calendar-day completion times for project-heavy courses are right-skewed (debugging,
compute blow-ups, life). Non-project-heavy courses use a Student-t truncated at
zero. Project-heavy courses use a log-t with median at the predicted days.
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


class TruncatedT:
    """Student-t with location/scale, truncated at zero."""

    def __init__(self, center_days: float, frac: float, df: float):
        if center_days <= 0 or frac <= 0 or df <= 0:
            raise ValueError("center_days, frac, and df must be positive")
        self.center = float(center_days)
        self.scale = self.center * float(frac)
        self.df = float(df)
        self.f0 = stats.t.cdf((0.0 - self.center) / self.scale, self.df)

    def ppf(self, q: float) -> float:
        return float(self.center + self.scale * stats.t.ppf(self.f0 + q * (1.0 - self.f0), self.df))

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return float((stats.t.cdf((x - self.center) / self.scale, self.df) - self.f0) / (1.0 - self.f0))

    def pdf(self, x):
        x_arr = np.asarray(x)
        y = stats.t.pdf((x_arr - self.center) / self.scale, self.df) / self.scale / (1.0 - self.f0)
        return np.where(x_arr >= 0, y, 0.0)


class LogT:
    """Log Student-t with median at center_days."""

    def __init__(self, center_days: float, frac: float, df: float):
        if center_days <= 0 or frac <= 0 or df <= 0:
            raise ValueError("center_days, frac, and df must be positive")
        self.loc_ln = float(np.log(center_days))
        self.sigma_ln = float(frac)
        self.df = float(df)

    def ppf(self, q: float) -> float:
        return float(np.exp(self.loc_ln + self.sigma_ln * stats.t.ppf(q, self.df)))

    def cdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        return float(stats.t.cdf((np.log(x) - self.loc_ln) / self.sigma_ln, self.df))

    def pdf(self, x):
        x_arr = np.asarray(x)
        positive = x_arr > 0
        y = np.zeros_like(x_arr, dtype=float)
        y[positive] = (
            stats.t.pdf((np.log(x_arr[positive]) - self.loc_ln) / self.sigma_ln, self.df)
            / (self.sigma_ln * x_arr[positive])
        )
        return y


def get_distribution(course_id: str, center_days: float, frac: float, df: float):
    """Return log-t if project-heavy, otherwise a Student-t truncated at zero."""
    if is_project_heavy(course_id):
        return LogT(center_days, frac, df)
    return TruncatedT(center_days, frac, df)


def predict_intervals(
    predictions_df: pd.DataFrame,
    quantiles: tuple = (0.10, 0.50, 0.90),
) -> pd.DataFrame:
    """Add p10/p50/p90 (or given quantiles) and dist_type columns to predictions.

    Input columns required: course_id, predicted_days, frac, df.
    Quantile column names are f"p{int(q*100)}".
    """
    qcols = [f"p{int(q * 100)}" for q in quantiles]
    out = predictions_df.copy()
    rows = []
    types = []
    for _, r in out.iterrows():
        dist = get_distribution(r["course_id"], r["predicted_days"], r["frac"], r["df"])
        rows.append([float(dist.ppf(q)) for q in quantiles])
        types.append("log-t" if is_project_heavy(r["course_id"]) else "truncated-t")
    q_df = pd.DataFrame(rows, columns=qcols, index=out.index)
    out[qcols] = q_df
    out["dist_type"] = types
    return out


def prob_finish_by(
    course_id: str, center_days: float, frac: float, df: float, deadline_days: float
) -> float:
    """P(completion_days <= deadline_days) under the appropriate marginal."""
    return float(get_distribution(course_id, center_days, frac, df).cdf(deadline_days))


if __name__ == "__main__":
    mean_days, frac, df = 100.0, 0.20, 10.0
    qs = (0.10, 0.50, 0.90)

    demo = pd.DataFrame(
        [
            {"course_id": "DTSA5511", "predicted_days": mean_days, "frac": frac, "df": df},
            {"course_id": "DTSA5002", "predicted_days": mean_days, "frac": frac, "df": df},
        ]
    )
    result = predict_intervals(demo, quantiles=qs)
    print(f"Target center={mean_days}, frac={frac}, df={df}\n")
    print(result.to_string(index=False))
    print()

    for cid in ["DTSA5511", "DTSA5002"]:
        d = get_distribution(cid, mean_days, frac, df)
        p10, p50, p90 = d.ppf(0.10), d.ppf(0.50), d.ppf(0.90)
        label = "log-t" if is_project_heavy(cid) else "truncated-t"
        print(
            f"{cid} ({label:9s}): "
            f"p10={p10:6.2f}  p50={p50:6.2f}  p90={p90:6.2f}  "
            f"left_tail_width={mean_days - p10:5.2f}  "
            f"right_tail_width={p90 - mean_days:5.2f}"
        )
