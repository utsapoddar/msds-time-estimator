"""Refresh raw MSDS review observations and manage manual form-review promotion.

Main workflow:
1. Pull the raw `MSDS Reviews` tab from the published Google Sheet.
2. Normalize each row into one `data_points.csv` observation with
   `source='msds_reviews'`.
3. Replace any prior `msds_reviews` / `msds_reviews_xlsx` rows so the refresh is
   idempotent.

The older manual-review helpers are still kept for `data_points_review.csv`
promotion, but the default `python -m modules.form_intake` entrypoint now
refreshes the raw MSDS review source.
"""

from __future__ import annotations

import hashlib
import io
import pathlib
import re

import pandas as pd
import requests


SHEET_ID = "1lplPW_5DI-wgB_q6qgxmr9WTP12-JVV_yDXsKxLFMiM"
MSDS_REVIEWS_GID = "158871878"

# Preferred public export endpoint for the raw `MSDS Reviews` tab.
PUBLISHED_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={MSDS_REVIEWS_GID}"
)
ALT_PUBLISHED_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={MSDS_REVIEWS_GID}"
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_CSV = ROOT / "data_points.csv"
REVIEW_CSV = ROOT / "data_points_review.csv"
COURSE_PARAMS_CSV = ROOT / "course_params.csv"
LOCAL_WORKBOOK_FALLBACK = ROOT / "data_sources" / "cu_boulder_online_ms_curriculum_guides.xlsx"
LOCAL_WORKBOOK_FALLBACK_SHEET = "MSDS Reviews"

CANONICAL_COLUMNS = [
    "course_id", "student_id", "days", "hours", "source", "notes",
    "submission_id", "submitted_at", "semester_taken", "hours_per_week",
    "concurrent_courses", "grade", "has_adhd", "medicated",
    "math_skill", "coding_skill", "writing_skill", "prior_background",
    "review_needed", "include", "weight",
]

FORM_COLUMN_MAP = {
    "Which course?": "course_id",
    "When did you take it?": "semester_taken",
    "Start date": "_start_date",
    "End date (last assignment submitted)": "_end_date",
    "Hours per week available during the course": "hours_per_week",
    "Concurrent courses taken alongside this one": "concurrent_courses",
    "Grade received": "grade",
    "Do you have ADHD (diagnosed or strongly suspected)?": "_adhd_raw",
    "Self-rated math strength": "math_skill",
    "Self-rated coding strength": "coding_skill",
    "Self-rated writing strength": "writing_skill",
    "Prior background most relevant to this course": "prior_background",
    "Notes or anything unusual about your experience": "notes",
    "Contact (optional)": "_contact_raw",
    "Timestamp": "submitted_at",
    "review_needed": "review_needed",
}


def _normalize_header_string(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _find_column_containing(columns: list[str], *phrases: str) -> str | None:
    phrases_lc = tuple(phrase.lower() for phrase in phrases)
    for col in columns:
        normalized = str(col).lower()
        if any(phrase in normalized for phrase in phrases_lc):
            return col
    return None


def _find_course_title_column(df: pd.DataFrame) -> str | None:
    columns = list(df.columns)
    by_header = _find_column_containing(columns, "course title")
    if by_header is not None:
        return by_header

    for col in columns:
        values = df[col].dropna().astype(str)
        if values.str.contains(r"DTSA\s*\d{4}", case=False, regex=True).any():
            return col
    return None


def _coerce_hours(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("hours", "days", "hours_per_week"):
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    missing_hours = out["hours"].isna()
    imputable = missing_hours & out["days"].notna() & out["hours_per_week"].notna()
    out.loc[imputable, "hours"] = (
        out.loc[imputable, "days"] * out.loc[imputable, "hours_per_week"] / 7.0
    )

    return out


def _extract_course_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    match = re.search(r"(DTSA)\s*(\d{4})", str(value))
    if not match:
        return None
    return f"{match.group(1)}{match.group(2)}"


def _stable_submission_id(prefix: str, *parts: object) -> str:
    raw = "||".join("" if pd.isna(p) else str(p) for p in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _fetch_csv_from_url(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), header=0)


def _load_local_workbook_reviews() -> pd.DataFrame:
    if not LOCAL_WORKBOOK_FALLBACK.exists():
        raise FileNotFoundError(
            f"Review workbook fallback not found: {LOCAL_WORKBOOK_FALLBACK}"
        )
    raw = pd.read_excel(
        LOCAL_WORKBOOK_FALLBACK,
        sheet_name=LOCAL_WORKBOOK_FALLBACK_SHEET,
        engine="openpyxl",
        header=None,
    )
    header = raw.iloc[0].copy()
    header.iloc[0] = raw.iloc[1, 0]
    df = raw.iloc[2:].copy()
    df.columns = header
    return df.reset_index(drop=True)


def fetch_msds_reviews_raw(
    url: str = PUBLISHED_CSV_URL,
    alt_url: str = ALT_PUBLISHED_CSV_URL,
) -> tuple[pd.DataFrame, str]:
    """Fetch the raw MSDS review rows.

    Tries the public CSV export first, then the GViz CSV endpoint. If both fail,
    raise an error instead of silently falling back to any local workbook copy.
    """
    last_error: Exception | None = None
    for candidate in (url, alt_url):
        try:
            return _fetch_csv_from_url(candidate), candidate
        except Exception as exc:  # pragma: no cover - network varies by environment
            last_error = exc

    if last_error is not None:
        raise RuntimeError(f"Failed to fetch MSDS reviews CSV: {last_error}") from last_error
    raise RuntimeError("Failed to fetch MSDS reviews CSV.")


def normalize_msds_reviews(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize course-level aggregate rows into canonical data points."""
    df = raw_df.copy()
    if len(df):
        first_row = [_normalize_header_string(value).lower() for value in df.iloc[0].tolist()]
        if "course title" in first_row:
            df = df.iloc[1:].copy()

    df = df.reset_index(drop=True)
    df.columns = [_normalize_header_string(col) for col in df.columns]
    columns = list(df.columns)
    course_col = _find_course_title_column(df)
    review_count_col = _find_column_containing(columns, "number of course reviews")
    overall_col = _find_column_containing(columns, "overall experience")
    difficulty_col = _find_column_containing(columns, "content difficulty")
    instructor_col = _find_column_containing(columns, "instructor effectiveness")
    hours_col = _find_column_containing(
        columns,
        "students' estimated time commitment",
        "estimated time commitment",
    )
    coursera_hours_col = _find_column_containing(columns, "coursera published hours")

    required = {
        "course title": course_col,
        "review count": review_count_col,
        "hours": hours_col,
    }
    missing = [label for label, col in required.items() if col is None]
    if missing:
        raise ValueError(
            f"Could not find required MSDS reviews aggregate columns: {', '.join(missing)}."
        )

    df["course_title"] = df[course_col]
    df["course_id"] = df["course_title"].map(_extract_course_id)
    df["review_count"] = (
        df[review_count_col]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df["hours"] = pd.to_numeric(df[hours_col], errors="coerce")
    df["overall_experience"] = (
        pd.to_numeric(df[overall_col], errors="coerce") if overall_col is not None else pd.NA
    )
    df["difficulty_value"] = (
        pd.to_numeric(df[difficulty_col], errors="coerce") if difficulty_col is not None else pd.NA
    )
    df["instructor_effectiveness"] = (
        pd.to_numeric(df[instructor_col], errors="coerce") if instructor_col is not None else pd.NA
    )
    df["coursera_published_hours"] = (
        pd.to_numeric(df[coursera_hours_col], errors="coerce")
        if coursera_hours_col is not None
        else pd.NA
    )

    df = df[df["course_id"].notna()].copy()

    invalid_student_hours = df["hours"].isna() | (df["hours"] <= 0)
    fallback_rows = df[invalid_student_hours].copy()
    fallback_rows = fallback_rows[fallback_rows["coursera_published_hours"].notna()].copy()
    fallback_rows = fallback_rows[fallback_rows["coursera_published_hours"] > 0].copy()
    fallback_rows["hours"] = fallback_rows["coursera_published_hours"]
    fallback_rows["source"] = "coursera_published"
    fallback_rows["weight"] = 1.0

    review_df = df[df["review_count"].notna()].copy()
    review_df = review_df[review_df["review_count"] > 0].copy()

    aggregate_df = review_df.copy()

    canonical_rows = []

    review_rows = review_df[review_df["hours"].notna()].copy()
    review_rows = review_rows[review_rows["hours"] > 0].copy()
    review_rows["source"] = "msds_reviews"
    review_rows["weight"] = review_rows["review_count"].astype(float)
    canonical_rows.append(review_rows)
    canonical_rows.append(fallback_rows)

    df = pd.concat(canonical_rows, ignore_index=True) if canonical_rows else df.iloc[0:0].copy()
    df["submitted_at"] = ""
    df["notes"] = ""
    df["days"] = pd.NA
    df["hours_per_week"] = pd.NA
    df["review_needed"] = "auto"
    df["include"] = "true"
    df = _coerce_hours(df)

    df["submission_id"] = [
        _stable_submission_id(source, course, review_count, hours)
        for source, course, review_count, hours in zip(
            df["source"], df["course_id"], df["review_count"], df["hours"]
        )
    ]
    df["student_id"] = df["submission_id"]

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    canonical = df[CANONICAL_COLUMNS].copy()
    canonical["hours"] = canonical["hours"].map(lambda v: "" if pd.isna(v) else float(v))
    canonical["days"] = canonical["days"].map(lambda v: "" if pd.isna(v) else float(v))
    canonical["hours_per_week"] = canonical["hours_per_week"].map(
        lambda v: "" if pd.isna(v) else float(v)
    )
    canonical["weight"] = canonical["weight"].map(lambda v: "" if pd.isna(v) else float(v))

    aggregates = aggregate_df[
        [
            "course_id",
            "course_title",
            "review_count",
            "overall_experience",
            "difficulty_value",
            "instructor_effectiveness",
            "hours",
            "coursera_published_hours",
        ]
    ].rename(
        columns={
            "difficulty_value": "difficulty",
            "hours": "avg_hours",
        }
    )
    return canonical, aggregates


def refresh_msds_reviews_data(
    main_csv: pathlib.Path = MAIN_CSV,
    course_params_csv: pathlib.Path = COURSE_PARAMS_CSV,
) -> dict:
    """Replace the raw MSDS review rows in `data_points.csv` and refresh aggregates."""
    raw_df, source_url = fetch_msds_reviews_raw()
    reviews_df, aggregates = normalize_msds_reviews(raw_df)

    main = pd.read_csv(main_csv, dtype=str).fillna("")
    keep = ~main["source"].isin(["msds_reviews", "msds_reviews_xlsx", "coursera_published"])
    refreshed = pd.concat([main[keep], reviews_df.astype(str)], ignore_index=True)
    refreshed = refreshed.drop_duplicates(subset=["submission_id"], keep="first")
    refreshed.to_csv(main_csv, index=False)

    params = pd.read_csv(course_params_csv)
    agg_idx = aggregates.set_index("course_id")
    if "review_count" in params.columns:
        params["review_count"] = params["course_id"].map(agg_idx["review_count"]).combine_first(params["review_count"])
    else:
        params["review_count"] = params["course_id"].map(agg_idx["review_count"])
    if "difficulty" in params.columns:
        params["difficulty"] = params["course_id"].map(agg_idx["difficulty"]).combine_first(params["difficulty"])
    else:
        params["difficulty"] = params["course_id"].map(agg_idx["difficulty"])
    params.to_csv(course_params_csv, index=False)

    return {
        "source_url": source_url,
        "review_rows_added": int(len(reviews_df)),
        "courses_touched": int(reviews_df["course_id"].nunique()),
    }


def fetch_form_submissions(url: str) -> pd.DataFrame:
    """Read the published Google Form responses CSV into a DataFrame."""
    if not url:
        raise ValueError("Provide a Google Form responses CSV URL.")
    return pd.read_csv(url)


def _normalize_form_submissions(df: pd.DataFrame) -> pd.DataFrame:
    """Map Google Form columns into the canonical data-point schema."""
    df = df.rename(columns=FORM_COLUMN_MAP)
    if "course_id" in df.columns:
        df["course_id"] = df["course_id"].astype(str).str.extract(r"(DTSA\d+)", expand=False)
    if "_start_date" in df.columns and "_end_date" in df.columns:
        d1 = pd.to_datetime(df["_start_date"], errors="coerce")
        d2 = pd.to_datetime(df["_end_date"], errors="coerce")
        df["days"] = (d2 - d1).dt.days
        df = df.drop(columns=["_start_date", "_end_date"])
    if "_adhd_raw" in df.columns:
        raw = df["_adhd_raw"].fillna("").astype(str).str.lower()
        df["has_adhd"] = raw.str.contains("yes")
        df["medicated"] = raw.str.contains("medicated") & ~raw.str.contains("unmedicated")
        df = df.drop(columns=["_adhd_raw"])

    submitted = df.get("submitted_at", pd.Series([""] * len(df)))
    df["student_id"] = "form_" + submitted.astype(str).str.replace(r"\W", "", regex=True)
    df["submission_id"] = df["student_id"]
    df["source"] = "google_form"
    df["include"] = df["review_needed"].map(
        lambda r: "true" if str(r).strip().lower() == "auto" else "pending"
    )
    df["weight"] = 1.0
    df = _coerce_hours(df)
    df = df[~(df["hours"].isna() & df["hours_per_week"].isna())].copy()

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[CANONICAL_COLUMNS]


def sync_form_submissions(
    url: str,
    main_csv: pathlib.Path = MAIN_CSV,
    review_csv: pathlib.Path = REVIEW_CSV,
) -> dict:
    """Fetch form submissions, split by review_needed, append only new rows."""
    submissions = _normalize_form_submissions(fetch_form_submissions(url))
    main = pd.read_csv(main_csv, dtype=str).fillna("")
    review = pd.read_csv(review_csv, dtype=str).fillna("")

    known = set(main["submission_id"]) | set(review["submission_id"])
    new = submissions[~submissions["submission_id"].isin(known)]

    new_auto = new[new["review_needed"].astype(str).str.lower() == "auto"]
    new_review = new[new["review_needed"].astype(str).str.lower() != "auto"]

    if len(new_auto):
        pd.concat([main, new_auto], ignore_index=True).to_csv(main_csv, index=False)
    if len(new_review):
        pd.concat([review, new_review], ignore_index=True).to_csv(review_csv, index=False)

    return {
        "added_auto": int(len(new_auto)),
        "added_review": int(len(new_review)),
        "total_seen": int(len(submissions)),
    }


def promote_reviewed(
    include_decisions: dict[str, str],
    main_csv: pathlib.Path = MAIN_CSV,
    review_csv: pathlib.Path = REVIEW_CSV,
) -> int:
    """Apply notebook review decisions and dedupe before promotion to main CSV."""
    review = pd.read_csv(review_csv, dtype=str).fillna("")
    main = pd.read_csv(main_csv, dtype=str).fillna("")
    existing_submission_ids = set(main["submission_id"])
    to_main = []
    for sid, decision in include_decisions.items():
        mask = review["submission_id"] == sid
        if not mask.any():
            continue
        review.loc[mask, "include"] = decision
        if decision == "true" and sid not in existing_submission_ids:
            promoted = _coerce_hours(review[mask])
            promoted = promoted[~(promoted["hours"].isna() & promoted["hours_per_week"].isna())]
            if len(promoted):
                to_main.append(promoted.astype(str))
                existing_submission_ids.add(sid)
    if to_main:
        main = pd.concat([main] + to_main, ignore_index=True)
        main.to_csv(main_csv, index=False)
    review.to_csv(review_csv, index=False)
    return len(to_main)


if __name__ == "__main__":
    print(refresh_msds_reviews_data())
