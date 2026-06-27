"""Simplified predictor built around difficulty OLS, personal ratios, and schedule conversion.

Agreed model:
1. baseline course hours come from the weighted difficulty->hours OLS fit
2. your personalization comes from actual_hours / baseline_hours on completed courses
3. use one anchor now, then median completed-course ratio as your history grows
4. convert predicted hours to calendar days only after asking for study schedule
"""

from __future__ import annotations

from pathlib import Path
from statistics import median

import pandas as pd
from .guide_priors import estimate_hours_from_difficulty


ROOT = Path(__file__).resolve().parents[1]
COURSE_PARAMS_CSV = ROOT / "course_params.csv"


def load_course_params(path: str | Path = COURSE_PARAMS_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def get_course_row(course_id: str, course_params: pd.DataFrame) -> pd.Series:
    indexed = course_params.set_index("course_id")
    if course_id not in indexed.index:
        raise KeyError(f"unknown course_id: {course_id}")
    return indexed.loc[course_id]


def baseline_hours(course_id: str, course_params: pd.DataFrame) -> float:
    row = get_course_row(course_id, course_params)
    difficulty = row.get("difficulty")
    if pd.notna(difficulty):
        return float(estimate_hours_from_difficulty(float(difficulty), weighted=True))
    return float(row["expected_hours"])


def actual_hours(
    *,
    total_hours: float | None = None,
    calendar_days: float | None = None,
    hours_per_week: float | None = None,
    days_per_week: float | None = None,
    hours_per_day: float | None = None,
) -> float:
    """Convert user input into total study hours.

    Preferred input is total_hours directly.
    Otherwise provide calendar_days plus either:
    - hours_per_week
    - or days_per_week and hours_per_day
    """
    if total_hours is not None:
        return float(total_hours)

    if calendar_days is None:
        raise ValueError("provide total_hours or calendar_days")

    if hours_per_week is None:
        if days_per_week is None or hours_per_day is None:
            raise ValueError(
                "when total_hours is not provided, supply calendar_days plus either "
                "hours_per_week or days_per_week and hours_per_day"
            )
        hours_per_week = float(days_per_week) * float(hours_per_day)

    return float(calendar_days) * float(hours_per_week) / 7.0


def ratio_for_course(course_id: str, actual_course_hours: float, course_params: pd.DataFrame) -> float:
    return float(actual_course_hours) / baseline_hours(course_id, course_params)


def ratio_from_completed_courses(
    completed_courses: list[dict],
    course_params: pd.DataFrame,
    aggregate: str = "median",
) -> dict:
    """Build a personal multiplier from completed course history.

    Each entry needs:
    - course_id
    - actual_hours
    """
    if not completed_courses:
        raise ValueError("at least one completed course is required")

    ratios = []
    details = []
    for item in completed_courses:
        course_id = item["course_id"]
        actual_course_hours = float(item["actual_hours"])
        baseline = baseline_hours(course_id, course_params)
        ratio = actual_course_hours / baseline
        ratios.append(ratio)
        details.append(
            {
                "course_id": course_id,
                "actual_hours": actual_course_hours,
                "baseline_hours": baseline,
                "ratio": ratio,
            }
        )

    if aggregate == "mean":
        personal_ratio = sum(ratios) / len(ratios)
    elif aggregate == "median":
        personal_ratio = float(median(ratios))
    else:
        raise ValueError("aggregate must be 'median' or 'mean'")

    return {"personal_ratio": personal_ratio, "details": details, "aggregate": aggregate}


def predict_hours(course_id: str, personal_ratio: float, course_params: pd.DataFrame) -> float:
    return float(personal_ratio) * baseline_hours(course_id, course_params)


def schedule_hours_per_week(
    *, hours_per_week: float | None = None, days_per_week: float | None = None, hours_per_day: float | None = None
) -> float:
    if hours_per_week is not None:
        return float(hours_per_week)
    if days_per_week is None or hours_per_day is None:
        raise ValueError("provide hours_per_week or days_per_week and hours_per_day")
    return float(days_per_week) * float(hours_per_day)


def convert_hours_to_schedule(
    predicted_hours: float,
    *,
    hours_per_week: float | None = None,
    days_per_week: float | None = None,
    hours_per_day: float | None = None,
) -> dict:
    hpw = schedule_hours_per_week(
        hours_per_week=hours_per_week,
        days_per_week=days_per_week,
        hours_per_day=hours_per_day,
    )
    result = {
        "predicted_hours": float(predicted_hours),
        "hours_per_week": hpw,
        "calendar_days": float(predicted_hours) * 7.0 / hpw,
    }
    if hours_per_day is not None:
        result["study_days_needed"] = float(predicted_hours) / float(hours_per_day)
    return result


def predict_course_from_history(
    course_id: str,
    completed_courses: list[dict],
    course_params: pd.DataFrame,
    *,
    aggregate: str = "median",
    hours_per_week: float | None = None,
    days_per_week: float | None = None,
    hours_per_day: float | None = None,
) -> dict:
    history = ratio_from_completed_courses(completed_courses, course_params, aggregate=aggregate)
    hours = predict_hours(course_id, history["personal_ratio"], course_params)
    schedule = convert_hours_to_schedule(
        hours,
        hours_per_week=hours_per_week,
        days_per_week=days_per_week,
        hours_per_day=hours_per_day,
    )
    return {
        "course_id": course_id,
        "personal_ratio": history["personal_ratio"],
        "history": history,
        "baseline_hours": baseline_hours(course_id, course_params),
        **schedule,
    }
