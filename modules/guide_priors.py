"""Helpers for extracting course-level priors from the CU Boulder MSDS guide workbook.

The current simplified model in this repo copy uses:
1. weighted OLS on guide difficulty -> baseline course hours
2. the user's anchor/completed-course history -> personal multiplier
3. schedule conversion -> calendar days

We still keep the raw guide aggregates (student-estimated hours, Coursera hours,
review count) on each course row for inspection, but the primary baseline prior
is the weighted difficulty->hours fit unless that course has no guide difficulty.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GUIDE_XLSX = ROOT / "data_sources" / "cu_boulder_online_ms_curriculum_guides.xlsx"
DEFAULT_SHEET = "MSDS Reviews"
DEFAULT_HOURS_PER_WEEK = 5.0

# Verified on 2026-04-23 from the copied workbook's "MSDS Reviews" sheet using
# the 22 DTSA rows with numeric values for both:
# - Content Difficulty
# - Students' Estimated Time Commitment (hours)
#
# Unweighted OLS:
#   estimated_hours ~= 11.368350 * difficulty - 7.303371
#   R^2 ~= 0.628857
#
# Review-count-weighted least squares (weight = number of course reviews):
#   estimated_hours ~= 12.671204 * difficulty - 9.417626
#   weighted R^2 ~= 0.829265
#
# Keep these here as a reusable reference for prior imputation work.
DIFFICULTY_HOURS_FIT_UNWEIGHTED = {
    "slope": 11.368350331825372,
    "intercept": -7.303371467144469,
    "r_squared": 0.6288568160407905,
    "n_courses": 22,
}

DIFFICULTY_HOURS_FIT_WEIGHTED = {
    "slope": 12.671203533443347,
    "intercept": -9.417626403703878,
    "r_squared": 0.82926451231327,
    "n_courses": 22,
    "total_review_weight": 128.0,
}


def _normalize_course_id(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"(DTSA)\s*(\d{4})", value)
    if not match:
        return None
    return f"{match.group(1)}{match.group(2)}"


def _parse_review_count(value) -> float | pd.NA:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"(\d+)", str(value))
    return float(match.group(1)) if match else pd.NA


def _to_float(value) -> float | pd.NA:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str) and value.strip().upper() == "NA":
        return pd.NA
    try:
        return float(value)
    except (TypeError, ValueError):
        return pd.NA


def estimate_hours_from_difficulty(difficulty: float, weighted: bool = True) -> float:
    """Estimate course hours from guide difficulty using the verified linear fit."""
    fit = DIFFICULTY_HOURS_FIT_WEIGHTED if weighted else DIFFICULTY_HOURS_FIT_UNWEIGHTED
    return fit["slope"] * float(difficulty) + fit["intercept"]


def load_msds_review_priors(
    xlsx_path: str | Path = DEFAULT_GUIDE_XLSX,
    sheet_name: str = DEFAULT_SHEET,
) -> pd.DataFrame:
    """Return one clean row per DTSA course from the guide workbook."""
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=0, usecols="A:H")
    raw = raw.rename(
        columns={
            raw.columns[0]: "guide_course_title",
            raw.columns[1]: "guide_category",
            raw.columns[2]: "guide_num_reviews_raw",
            raw.columns[3]: "guide_overall_experience",
            raw.columns[4]: "guide_content_difficulty",
            raw.columns[5]: "guide_instructor_effectiveness",
            raw.columns[6]: "guide_student_estimated_hours",
            raw.columns[7]: "guide_coursera_published_hours",
        }
    )

    raw["course_id"] = raw["guide_course_title"].map(_normalize_course_id)
    raw = raw[raw["course_id"].notna()].copy()
    raw["guide_num_reviews"] = raw["guide_num_reviews_raw"].map(_parse_review_count)

    numeric_cols = [
        "guide_overall_experience",
        "guide_content_difficulty",
        "guide_instructor_effectiveness",
        "guide_student_estimated_hours",
        "guide_coursera_published_hours",
        "guide_num_reviews",
    ]
    for col in numeric_cols:
        raw[col] = raw[col].map(_to_float)

    raw["guide_hours_gap_vs_coursera"] = (
        raw["guide_student_estimated_hours"] - raw["guide_coursera_published_hours"]
    )
    raw["guide_hours_ratio_vs_coursera"] = (
        raw["guide_student_estimated_hours"] / raw["guide_coursera_published_hours"]
    )
    raw["guide_weighted_ols_hours"] = raw["guide_content_difficulty"].map(
        lambda v: estimate_hours_from_difficulty(v, weighted=True) if pd.notna(v) else pd.NA
    )
    raw["guide_unweighted_ols_hours"] = raw["guide_content_difficulty"].map(
        lambda v: estimate_hours_from_difficulty(v, weighted=False) if pd.notna(v) else pd.NA
    )

    cols = [
        "course_id",
        "guide_course_title",
        "guide_category",
        "guide_num_reviews",
        "guide_overall_experience",
        "guide_content_difficulty",
        "guide_instructor_effectiveness",
        "guide_student_estimated_hours",
        "guide_coursera_published_hours",
        "guide_hours_gap_vs_coursera",
        "guide_hours_ratio_vs_coursera",
        "guide_weighted_ols_hours",
        "guide_unweighted_ols_hours",
    ]
    return raw[cols].sort_values("course_id").reset_index(drop=True)


def apply_guide_priors(
    course_params_df: pd.DataFrame,
    guide_df: pd.DataFrame,
    hours_per_week: float = DEFAULT_HOURS_PER_WEEK,
) -> pd.DataFrame:
    """Merge guide aggregates into course params and choose the simplified prior.

    Priority in this repo copy:
    1. weighted OLS hours from content difficulty
    2. legacy expected_hours already present in course_params.csv

    Raw guide student-estimated hours and Coursera hours are preserved as
    reference columns, but they are not the main prior source right now.
    """
    out = course_params_df.copy()

    if "legacy_expected_hours" not in out.columns:
        out["legacy_expected_hours"] = out["expected_hours"]
    if "legacy_expected_days" not in out.columns:
        out["legacy_expected_days"] = out["expected_days"]
    if "legacy_source" not in out.columns:
        out["legacy_source"] = out["source"]

    extra_cols = [c for c in guide_df.columns if c != "course_id"]
    out = out.drop(columns=[c for c in extra_cols if c in out.columns], errors="ignore")
    out = out.merge(guide_df, on="course_id", how="left")

    difficulty_hours = out["guide_weighted_ols_hours"]

    out["expected_hours"] = difficulty_hours.combine_first(out["legacy_expected_hours"])

    out["source"] = out["legacy_source"]
    out.loc[difficulty_hours.notna(), "source"] = "guide_difficulty_weighted_ols"

    out["expected_days"] = out["expected_hours"] * 7.0 / hours_per_week
    return out
