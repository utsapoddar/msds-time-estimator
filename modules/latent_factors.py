"""Latent-factor decomposition of student speed across math / coding / writing axes.

Replaces the single scalar speed multiplier S with a 3-vector. Each course has a
hand-assigned nonnegative loading row (sums to 1) giving its effort fractions
across (math, coding, writing). Per-course predicted speed = loadings . skill_vector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# These three factors capture broad student skill shape, not course-topic content.
FACTORS = ["math", "coding", "writing"]

# Hand-assigned loadings: (math, coding, writing), each row sums to 1.0.
_LOADINGS: dict[str, tuple[float, float, float]] = {
    # Statistical Inference — heavy math
    "DTSA5001": (0.75, 0.15, 0.10),
    "DTSA5002": (0.75, 0.15, 0.10),
    "DTSA5003": (0.75, 0.15, 0.10),
    # DSA — moderate math, heavy coding
    "DTSA5501": (0.30, 0.60, 0.10),
    "DTSA5502": (0.30, 0.60, 0.10),
    "DTSA5503": (0.30, 0.60, 0.10),
    # Statistical Modeling in R — heavy math, moderate coding
    "DTSA5011": (0.55, 0.35, 0.10),
    "DTSA5012": (0.55, 0.35, 0.10),
    "DTSA5013": (0.55, 0.35, 0.10),
    # Data Mining — moderate math, heavy coding, some writing
    "DTSA5504": (0.25, 0.55, 0.20),
    "DTSA5505": (0.25, 0.55, 0.20),
    "DTSA5506": (0.25, 0.55, 0.20),
    # ML — moderate-high math, heavy coding
    "DTSA5509": (0.35, 0.55, 0.10),
    "DTSA5510": (0.35, 0.55, 0.10),
    "DTSA5511": (0.25, 0.65, 0.10),  # Deep Learning: heavier coding
    # Databases — low math, heavy coding, moderate writing
    "DTSA5733": (0.15, 0.65, 0.20),
    "DTSA5734": (0.15, 0.65, 0.20),
    "DTSA5735": (0.15, 0.65, 0.20),
    # Vital Skills — varies
    "DTSA5301": (0.10, 0.20, 0.70),  # Data Science as a Field
    "DTSA5302": (0.20, 0.40, 0.40),  # Cybersecurity
    "DTSA5303": (0.05, 0.05, 0.90),  # Ethics
    "DTSA5304": (0.20, 0.50, 0.30),  # Viz

    # Statistical Learning electives — stats + coding
    "DTSA5020": (0.55, 0.35, 0.10),
    "DTSA5021": (0.60, 0.30, 0.10),
    "DTSA5022": (0.45, 0.45, 0.10),
    # High Performance Computing — coding-heavy systems courses
    "DTSA5701": (0.20, 0.70, 0.10),
    "DTSA5702": (0.15, 0.75, 0.10),
    "DTSA5703": (0.20, 0.70, 0.10),
    # Quality improvement — applied stats/process analysis
    "DTSA5704": (0.40, 0.20, 0.40),
    "DTSA5705": (0.45, 0.20, 0.35),
    "DTSA5706": (0.45, 0.20, 0.35),
    # Marketing analytics text/network electives
    "DTSA5798": (0.25, 0.55, 0.20),
    "DTSA5799": (0.25, 0.55, 0.20),
    "DTSA5800": (0.35, 0.45, 0.20),
    # Communication electives — writing-heavy
    "DTSA5842": (0.05, 0.15, 0.80),
    "DTSA5843": (0.05, 0.15, 0.80),
}

for _cid, _row in _LOADINGS.items():
    assert abs(sum(_row) - 1.0) < 0.01, f"{_cid} loading row sums to {sum(_row)}"


def get_loading_matrix() -> pd.DataFrame:
    """Return course loadings indexed by course_id with columns math/coding/writing."""
    return pd.DataFrame.from_dict(_LOADINGS, orient="index", columns=FACTORS)


def decompose_speed(
    anchor_course_ids: list[str],
    observed_S_vec: list[float] | np.ndarray,
    user_skill_priors: tuple[float, float, float] | None = None,
    prior_weight: float = 1.0,
) -> np.ndarray:
    """Decompose anchor courses' observed speeds into per-factor speeds.

    Single anchor (len == 1): underdetermined (1 equation, 3 unknowns). Resolved
    by uniform alpha-scaling of `user_skill_priors` so L . (alpha * priors) = S.
    Preserves prior-shape; relies on self-ratings for shape, anchor for scale.

    Multiple anchors: solve Tikhonov-regularized least squares
        minimize ||L @ skill_vec - observed_S_vec||^2 + λ * ||skill_vec - priors||^2
    Closed form: skill_vec = (LᵀL + λI)⁻¹ (Lᵀ observed_S_vec + λ priors), λ = prior_weight.
    With diverse loadings the anchor system becomes overdetermined and the priors
    act as a soft regularizer toward the user's self-ratings.

    Returned vector is clipped to [0.25, 4.0] as a safety belt: extreme observed
    speeds (e.g. mis-entered dates) shouldn't cascade into absurd per-course
    predictions across all modeled courses.
    """
    # Priors preserve the user's self-rated shape when anchors are sparse.
    if user_skill_priors is None:
        user_skill_priors = (1.0, 1.0, 1.0)
    priors = np.asarray(user_skill_priors, dtype=float)
    observed_S_vec = np.asarray(observed_S_vec, dtype=float)
    if len(anchor_course_ids) != len(observed_S_vec):
        raise ValueError("anchor_course_ids and observed_S_vec length mismatch")
    if len(anchor_course_ids) == 0:
        raise ValueError("at least one anchor course is required")

    L_full = get_loading_matrix()

    # Single-anchor case scales the self-rating shape to match observed pace.
    if len(anchor_course_ids) == 1:
        # Preserve single-anchor behavior exactly.
        L = L_full.loc[anchor_course_ids[0]].to_numpy()
        denom = float(L @ priors)
        if denom <= 0:
            raise ValueError("anchor loading dot priors must be positive")
        alpha = float(observed_S_vec[0]) / denom
        skill_vec = alpha * priors
    else:
        L = L_full.loc[anchor_course_ids].to_numpy()  # (n_anchors, 3)
        lam = float(prior_weight)
        A = L.T @ L + lam * np.eye(3)
        b = L.T @ observed_S_vec + lam * priors
        skill_vec = np.linalg.solve(A, b)

    return np.clip(skill_vec, 0.25, 4.0)


def predict_per_course(params_df: pd.DataFrame, skill_vector: np.ndarray) -> pd.DataFrame:
    """Predict per-course speed multiplier = loadings_row . skill_vector.

    `params_df` must have a `course_id` column (extra columns ignored). Returns
    a DataFrame with columns [course_id, predicted_S], preserving input order.
    """
    # Matrix multiply converts one skill vector into one speed per course.
    L = get_loading_matrix()
    sv = np.asarray(skill_vector, dtype=float)
    ids = params_df["course_id"].tolist()
    speeds = L.loc[ids].to_numpy() @ sv
    return pd.DataFrame({"course_id": ids, "predicted_S": speeds})


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    params = pd.read_csv(os.path.join(here, "..", "course_params.csv"))

    skill_vec = decompose_speed(
        anchor_course_ids=["DTSA5001"],
        observed_S_vec=[1.3],
        user_skill_priors=(1.5, 0.8, 1.0),
    )
    print(f"decomposed skill vector (math, coding, writing) = {skill_vec}")

    preds = predict_per_course(params, skill_vec)
    preds = preds.merge(params[["course_id", "name"]], on="course_id")
    preds = preds.sort_values("predicted_S", ascending=True).reset_index(drop=True)
    print(preds.to_string(index=False))
