"""Course-topic loadings and content-similarity correlations.

Topic axes are derived from official CU Boulder/Coursera course descriptions.
Until same-student multi-course histories exist, these transparent topic vectors
power both per-course anchor transfer and degree-total correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Axes are broad enough to cover every course without making a 36x36 table by hand.
TOPICS = [
    "probability_statistics",
    "statistical_modeling",
    "machine_learning",
    "nlp_text",
    "data_mining",
    "programming_algorithms",
    "databases_sql",
    "visualization_hci",
    "communication_writing",
    "ethics_security_policy",
    "systems_hpc",
    "quality_measurement",
    "project_delivery",
]

# Values are hand-scored from official descriptions on a 0.0 to 1.0 relevance scale.
_TOPIC_LOADINGS: dict[str, tuple[float, ...]] = {
    # Statistical inference / modeling / statistical learning
    "DTSA5001": (1.00, 0.20, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00),
    "DTSA5002": (0.90, 0.45, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00),
    "DTSA5003": (0.90, 0.35, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.05, 0.00, 0.00, 0.10, 0.00),
    "DTSA5011": (0.35, 1.00, 0.10, 0.00, 0.00, 0.20, 0.00, 0.10, 0.05, 0.10, 0.00, 0.00, 0.00),
    "DTSA5012": (0.45, 0.90, 0.00, 0.00, 0.00, 0.15, 0.00, 0.05, 0.05, 0.10, 0.00, 0.35, 0.00),
    "DTSA5013": (0.35, 1.00, 0.20, 0.00, 0.00, 0.20, 0.00, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00),
    "DTSA5020": (0.25, 0.75, 0.65, 0.00, 0.00, 0.35, 0.00, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00),
    "DTSA5021": (0.35, 0.85, 0.45, 0.00, 0.00, 0.30, 0.00, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00),
    "DTSA5022": (0.15, 0.40, 0.95, 0.00, 0.10, 0.45, 0.00, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00),
    # Algorithms / data mining / machine learning
    "DTSA5501": (0.05, 0.00, 0.05, 0.00, 0.00, 1.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00),
    "DTSA5502": (0.05, 0.00, 0.05, 0.00, 0.00, 1.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00),
    "DTSA5503": (0.10, 0.00, 0.05, 0.00, 0.00, 1.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00),
    "DTSA5504": (0.15, 0.20, 0.20, 0.00, 1.00, 0.25, 0.45, 0.15, 0.15, 0.00, 0.00, 0.10, 0.10),
    "DTSA5505": (0.20, 0.25, 0.55, 0.00, 1.00, 0.35, 0.05, 0.10, 0.10, 0.00, 0.00, 0.00, 0.00),
    "DTSA5506": (0.15, 0.20, 0.45, 0.00, 1.00, 0.30, 0.10, 0.15, 0.45, 0.00, 0.00, 0.00, 1.00),
    "DTSA5509": (0.25, 0.45, 1.00, 0.00, 0.00, 0.45, 0.00, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00),
    "DTSA5510": (0.20, 0.30, 1.00, 0.00, 0.05, 0.45, 0.00, 0.15, 0.05, 0.00, 0.00, 0.00, 0.00),
    "DTSA5511": (0.10, 0.15, 1.00, 0.20, 0.00, 0.65, 0.00, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00),
    # Databases / vital skills / HPC / quality / marketing / communication
    "DTSA5733": (0.00, 0.00, 0.00, 0.00, 0.05, 0.25, 1.00, 0.00, 0.20, 0.00, 0.00, 0.00, 0.20),
    "DTSA5734": (0.00, 0.00, 0.00, 0.00, 0.05, 0.45, 1.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00),
    "DTSA5735": (0.00, 0.00, 0.05, 0.00, 0.20, 0.35, 1.00, 0.00, 0.20, 0.15, 0.10, 0.00, 0.00),
    "DTSA5301": (0.05, 0.00, 0.05, 0.00, 0.10, 0.05, 0.00, 0.10, 0.50, 0.55, 0.00, 0.00, 0.10),
    "DTSA5302": (0.00, 0.00, 0.05, 0.00, 0.00, 0.30, 0.10, 0.00, 0.20, 1.00, 0.25, 0.00, 0.00),
    "DTSA5303": (0.00, 0.00, 0.05, 0.00, 0.00, 0.05, 0.00, 0.00, 0.45, 1.00, 0.00, 0.00, 0.00),
    "DTSA5304": (0.05, 0.10, 0.10, 0.00, 0.00, 0.30, 0.05, 1.00, 0.45, 0.05, 0.00, 0.00, 0.20),
    "DTSA5701": (0.00, 0.00, 0.00, 0.00, 0.00, 0.45, 0.00, 0.00, 0.05, 0.00, 1.00, 0.00, 0.00),
    "DTSA5702": (0.00, 0.00, 0.00, 0.00, 0.00, 0.75, 0.00, 0.00, 0.05, 0.00, 0.85, 0.00, 0.00),
    "DTSA5703": (0.00, 0.00, 0.00, 0.00, 0.00, 0.65, 0.00, 0.00, 0.05, 0.00, 1.00, 0.00, 0.00),
    "DTSA5704": (0.80, 0.25, 0.00, 0.00, 0.00, 0.10, 0.00, 0.25, 0.15, 0.00, 0.00, 0.75, 0.00),
    "DTSA5705": (0.50, 0.20, 0.00, 0.00, 0.00, 0.10, 0.00, 0.15, 0.10, 0.00, 0.00, 1.00, 0.00),
    "DTSA5706": (0.45, 0.20, 0.00, 0.00, 0.00, 0.10, 0.00, 0.10, 0.10, 0.00, 0.00, 1.00, 0.00),
    "DTSA5798": (0.10, 0.20, 0.85, 1.00, 0.10, 0.45, 0.00, 0.05, 0.25, 0.00, 0.00, 0.00, 0.40),
    "DTSA5799": (0.10, 0.20, 0.80, 1.00, 0.10, 0.40, 0.00, 0.05, 0.25, 0.00, 0.00, 0.00, 0.40),
    "DTSA5800": (0.25, 0.25, 0.55, 0.55, 0.15, 0.40, 0.10, 0.25, 0.25, 0.00, 0.00, 0.00, 0.30),
    "DTSA5842": (0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.45, 1.00, 0.05, 0.00, 0.00, 0.35),
    "DTSA5843": (0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.35, 1.00, 0.05, 0.00, 0.00, 1.00),
}


def get_topic_matrix() -> pd.DataFrame:
    # DataFrame form makes inspection and coverage tests simple.
    return pd.DataFrame.from_dict(_TOPIC_LOADINGS, orient="index", columns=TOPICS)


def course_similarity_matrix(course_ids: list[str]) -> np.ndarray:
    # Cosine similarity compares topic shape, not total loading magnitude.
    topics = get_topic_matrix().loc[course_ids].to_numpy(dtype=float)
    norms = np.linalg.norm(topics, axis=1)
    if np.any(norms == 0):
        raise ValueError("Course topic vectors must be non-zero")
    normalized = topics / norms[:, None]
    return normalized @ normalized.T


def course_correlation_matrix(
    course_ids: list[str],
    base_corr: float = 0.10,
    max_corr: float = 0.65,
) -> np.ndarray:
    # Map 0..1 similarity into the chosen positive correlation range.
    sim = course_similarity_matrix(course_ids)
    corr = base_corr + (max_corr - base_corr) * sim
    np.fill_diagonal(corr, 1.0)
    return corr


def topic_speed_transfer(
    target_course_ids: list[str],
    anchor_course_ids: list[str],
    anchor_speeds: list[float] | np.ndarray,
    baseline_speeds: list[float] | np.ndarray,
    min_similarity: float = 0.15,
) -> pd.DataFrame:
    """Transfer observed anchor pace to target courses by topic similarity.

    `baseline_speeds` is the fallback speed from self-ratings / latent factors.
    Courses very similar to completed anchors move toward the observed anchor
    pace. Distant courses stay close to baseline, so a slow probability anchor
    does not fully imply the same pace for communication or database courses.
    """
    # Validate shapes before doing any matrix indexing.
    if len(anchor_course_ids) != len(anchor_speeds):
        raise ValueError("anchor_course_ids and anchor_speeds must have same length")
    if len(target_course_ids) != len(baseline_speeds):
        raise ValueError("target_course_ids and baseline_speeds must have same length")
    if not anchor_course_ids:
        raise ValueError("at least one anchor course is required")

    # Pull vectors for target and anchor courses in the requested order.
    topic_matrix = get_topic_matrix()
    target_vecs = topic_matrix.loc[target_course_ids].to_numpy(dtype=float)
    anchor_vecs = topic_matrix.loc[anchor_course_ids].to_numpy(dtype=float)
    target_norms = np.linalg.norm(target_vecs, axis=1)
    anchor_norms = np.linalg.norm(anchor_vecs, axis=1)
    if np.any(target_norms == 0) or np.any(anchor_norms == 0):
        raise ValueError("Course topic vectors must be non-zero")

    # Similar anchors get more vote; distant anchors are ignored below threshold.
    sims = (target_vecs / target_norms[:, None]) @ (anchor_vecs / anchor_norms[:, None]).T
    weights = np.where(sims >= min_similarity, sims, 0.0)
    anchor_speeds_arr = np.asarray(anchor_speeds, dtype=float)
    baseline_arr = np.asarray(baseline_speeds, dtype=float)

    # Blend topic-inferred pace with baseline so unrelated courses do not overreact.
    topic_speeds = []
    confidences = []
    for i, row in enumerate(weights):
        max_sim = float(np.max(sims[i]))
        if row.sum() <= 0:
            topic_speed = float(baseline_arr[i])
        else:
            topic_speed = float(np.average(anchor_speeds_arr, weights=row))
        confidence = max(0.0, min(1.0, max_sim))
        blended = (confidence * topic_speed) + ((1.0 - confidence) * float(baseline_arr[i]))
        topic_speeds.append(blended)
        confidences.append(confidence)

    return pd.DataFrame(
        {
            "course_id": target_course_ids,
            "topic_predicted_S": topic_speeds,
            "topic_anchor_similarity": confidences,
        }
    )
