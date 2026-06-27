"""End-to-end pipeline for the MSDS time estimator.

Wraps the four model modules (bayesian_priors, latent_factors, focus_ratio,
lognormal_marginals) into a small, cohesive API that the notebook calls.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from . import bayesian_priors, latent_factors, focus_ratio, lognormal_marginals, course_topics


def load_posterior_params(params_csv: str = "course_params.csv",
                          data_points_csv: str = "data_points.csv") -> pd.DataFrame:
    """Load course params and apply Bayesian update using only include='true' rows."""
    # Load priors and student observations before applying the posterior update.
    params = pd.read_csv(params_csv)
    data_points = bayesian_priors.load_data_points(data_points_csv)
    data_points = data_points[data_points["include"].astype(str).str.lower() == "true"]
    return bayesian_priors.update_params_with_posterior(params, data_points)


def anchors_from_days_dict(anchor_days_by_course: dict[str, float], course_order: list[str]) -> tuple[list[str], list[float]]:
    """Convert vertical course->days input into anchor lists.

    Values <= 0 mean "not completed / do not use as anchor". Positive values,
    including fractions, are used as completed-course calendar days. Output order
    follows `course_order` so notebook dictionaries can list all courses neatly.
    """
    # Build the legacy list inputs from the easier vertical dictionary format.
    anchor_courses: list[str] = []
    anchor_days_list: list[float] = []
    for course_id in course_order:
        days = float(anchor_days_by_course.get(course_id, 0.0) or 0.0)
        if days > 0:
            anchor_courses.append(course_id)
            anchor_days_list.append(days)

    if not anchor_courses:
        raise ValueError("Set at least one anchor course to a positive day count.")
    return anchor_courses, anchor_days_list


def build_profile(post_params: pd.DataFrame,
                  anchor_courses: list[str],
                  anchor_days_list: list[float],
                  user_focus_ratio: float,
                  has_adhd: bool,
                  medicated: bool,
                  skill_priors: tuple,
                  base_cv: float = 0.20,
                  concurrent_courses: int = 1,
                  courses_completed: int = 0,
                  anchor_concurrent_courses: int = 1,
                  anchor_courses_completed: int = 0,
                  anchor_hours_per_week: float = 10.0,
                  target_hours_per_week: float = 10.0) -> dict:
    """Derive the per-user state used downstream: S, CV, skill vector.

    Supports one or more anchor courses. With one anchor the skill vector is
    underdetermined (resolved via priors); with two+ diverse anchors the system
    becomes overdetermined and is solved via Tikhonov-regularized LS.
    """
    if len(anchor_courses) != len(anchor_days_list):
        raise ValueError("anchor_courses and anchor_days_list must be same length")
    if len(anchor_courses) == 0:
        raise ValueError("at least one anchor course is required")

    # Pull posterior means for the completed courses used as anchors.
    indexed = post_params.set_index("course_id")
    anchor_rows = [indexed.loc[c] for c in anchor_courses]

    def _lc(n):  # 2% speedup per completed course, cap 20%
        return max(0.80, 1.0 - (0.02 * n))

    def _cs(n):  # 15% time penalty per extra concurrent course
        return 1.0 + (0.15 * max(0, n - 1))

    # Anchor pace already includes the user's state at anchor time.
    # To project forward, apply only target-vs-anchor learning/load deltas.
    learning_curve_factor = _lc(courses_completed) / _lc(anchor_courses_completed)
    context_switch_factor = _cs(concurrent_courses) / _cs(anchor_concurrent_courses)

    # Convert anchor calendar time to actual study hours at the input boundary.
    anchor_actual_hours_list = [
        _hours_from_days(days, anchor_hours_per_week) for days in anchor_days_list
    ]

    # Per-anchor pace ratio in schedule-independent study hours.
    S_per_anchor = [h / r.posterior_mean_hours for h, r in zip(anchor_actual_hours_list, anchor_rows)]
    pace_per_anchor = S_per_anchor
    anchor_pace = float(np.mean(pace_per_anchor))
    S = anchor_pace * learning_curve_factor * context_switch_factor

    # Per-anchor speeds passed to decompose_speed get the same lc/cs adjustments.
    adjusted_S_per_anchor = [
        p * learning_curve_factor * context_switch_factor for p in pace_per_anchor
    ]

    # Convert 1-10 self-ratings to internal speed multipliers
    math, coding, writing = (_rating_to_multiplier(r) for r in skill_priors)
    skill_multipliers = (math, coding, writing)

    # CV controls uncertainty width; skill_vec is retained for inspection/compatibility.
    cv = base_cv * focus_ratio.adhd_variance_multiplier(1.0, has_adhd, medicated)
    skill_vec = latent_factors.decompose_speed(anchor_courses, adjusted_S_per_anchor, skill_multipliers)
    return {
        "anchor_courses": anchor_courses, "anchor_days_list": anchor_days_list,
        "anchor_actual_hours_list": anchor_actual_hours_list,
        "anchor_names": [r["name"] for r in anchor_rows],
        "anchor_posterior_means": [float(r.posterior_mean_hours) for r in anchor_rows],
        "pace_per_anchor": pace_per_anchor,
        "adjusted_S_per_anchor": adjusted_S_per_anchor,
        "anchor_pace": anchor_pace, "S": S, "cv": cv, "skill_vec": skill_vec,
        "skill_multipliers": skill_multipliers,
        "user_focus_ratio": user_focus_ratio, "has_adhd": has_adhd, "medicated": medicated,
        "learning_curve_factor": learning_curve_factor, "context_switch_factor": context_switch_factor,
        "concurrent_courses": concurrent_courses, "courses_completed": courses_completed,
        "anchor_concurrent_courses": anchor_concurrent_courses,
        "anchor_courses_completed": anchor_courses_completed,
        "anchor_hours_per_week": anchor_hours_per_week, "target_hours_per_week": target_hours_per_week,
    }


def print_profile(profile: dict) -> None:
    # Keep printed diagnostics close to the quantities used in prediction.
    p = profile
    print(f"Anchors ({len(p['anchor_courses'])}):")
    for cid, name, post, days, hours, pace in zip(
        p["anchor_courses"], p["anchor_names"], p["anchor_posterior_means"],
        p["anchor_days_list"], p["anchor_actual_hours_list"], p["pace_per_anchor"],
    ):
        print(f"  {cid} ({name}): posterior {post:.1f} h, you {hours:.1f} h ({days} d) -> pace {pace:.2f}")
    direction = "faster" if p["S"] < 1 else "slower"
    print(f"  anchor_pace = {p['anchor_pace']:.2f}  Adjusted speed = {p['S']:.2f}  ({direction} than typical by {abs(1-p['S'])*100:.0f}%)")
    print(f"  Adjustments (target vs anchor delta): learning curve "
          f"({p['anchor_courses_completed']}->{p['courses_completed']} done) = {p['learning_curve_factor']:.2f}x, "
          f"concurrent load ({p['anchor_concurrent_courses']}->{p['concurrent_courses']}) = {p['context_switch_factor']:.2f}x")
    print(f"  Schedule conversion: anchor {p['anchor_hours_per_week']:.1f} hr/wk -> target {p['target_hours_per_week']:.1f} hr/wk")
    print(f"  Effective CV = {p['cv']:.3f}  (ADHD mult {p['cv']/0.20:.2f}x)")
    print(f"  Focus ratio = {p['user_focus_ratio']:.2f}  (applied to predicted calendar days)")
    m, c, w = p["skill_vec"]
    print(f"  Skill vector (math, coding, writing) = ({m:.2f}, {c:.2f}, {w:.2f})")


def build_predictions(post_params: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Full per-course prediction table."""
    pace_per_anchor = profile["pace_per_anchor"]
    n_pace = len(pace_per_anchor)
    if n_pace >= 2:
        cv_pace = (np.std(pace_per_anchor, ddof=1) / np.sqrt(n_pace)) / np.mean(pace_per_anchor)
    else:
        cv_pace = 0.0

    # Baseline comes from self-ratings; topic transfer adjusts it using anchors.
    baseline_S = latent_factors.predict_per_course(post_params, profile["skill_multipliers"]).rename(
        columns={"predicted_S": "baseline_S"}
    )
    topic_S = course_topics.topic_speed_transfer(
        target_course_ids=post_params["course_id"].tolist(),
        anchor_course_ids=profile["anchor_courses"],
        anchor_speeds=profile["adjusted_S_per_anchor"],
        baseline_speeds=baseline_S["baseline_S"].tolist(),
    )
    # Merge posterior course means with per-course user speed multipliers.
    merged = post_params.merge(topic_S, on="course_id")
    rows = []
    for _, r in merged.iterrows():
        effective_n = r.effective_n if "effective_n" in merged.columns else 0.0
        predicted_hours = r.posterior_mean_hours * r.topic_predicted_S
        predicted_days = _days_from_hours(predicted_hours, profile["target_hours_per_week"])
        predicted_sd_days = _days_from_hours(
            profile["cv"] * predicted_hours,
            profile["target_hours_per_week"],
        )
        rows.append({
            "course_id": r.course_id,
            "name": r["name"],
            "prior_hours": r.expected_hours,
            "predicted_hours": predicted_hours,
            "topic_predicted_S": r.topic_predicted_S,
            "topic_anchor_similarity": r.topic_anchor_similarity,
            "prior_days": r.expected_days,
            "predicted_days": predicted_days,
            # sd scales with predicted_hours so the effective CV stays constant at profile['cv'].
            "sd": predicted_sd_days,
            "cv_est": (
                r.posterior_sd_of_mean_hours / r.posterior_mean_hours
                if r.posterior_mean_hours > 0
                else 0.0
            ),
            "df": max(1.0, 1.0 + effective_n),
        })
    # Focus ratio converts focused-hour estimates into realistic calendar days.
    predictions = pd.DataFrame(rows)
    predictions = focus_ratio.apply_focus_ratio(predictions, profile["user_focus_ratio"])
    predictions["frac"] = np.sqrt(predictions["cv_est"] ** 2 + cv_pace ** 2 + profile["cv"] ** 2)
    predictions = lognormal_marginals.predict_intervals(predictions)
    predictions["predicted_hours"] = predictions["predicted_hours"].round(1)
    predictions["predicted_days"] = predictions["predicted_days"].round(1)
    predictions["sd"] = predictions["sd"].round(1)
    predictions["80%_interval"] = predictions.apply(lambda r: f"{r.p10:.0f}-{r.p90:.0f}", axis=1)
    return predictions


def plot_course(predictions: pd.DataFrame, course_id: str, profile: dict):
    # Plot the final marginal distribution for one course after personalization.
    row = predictions.set_index("course_id").loc[course_id]
    mu, sig = row.predicted_days, row.predicted_days * row.frac
    dist = lognormal_marginals.get_distribution(course_id, row.predicted_days, row.frac, row.df)
    x = np.linspace(max(0, mu - 4 * sig), mu + 4 * sig, 400)
    y = dist.pdf(x)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, color="C0")
    ax.fill_between(x, y, alpha=0.2)
    ax.axvline(row.prior_days, color="gray", linestyle="--", label=f"Prior: {row.prior_days} days")
    ax.axvline(mu, color="C1", linestyle=":", label=f"Your mean: {mu:.1f} days")
    ax.axvspan(row.p10, row.p90, alpha=0.1, color="C1", label=f"80% interval: {row.p10:.0f}-{row.p90:.0f} days")
    ax.set_xlabel("Calendar days to complete")
    ax.set_ylabel("Probability density")
    anchors = profile["anchor_courses"]
    days_list = profile["anchor_days_list"]
    extra = f" +{len(anchors)-1} more" if len(anchors) > 1 else ""
    ax.set_title(
        f"{course_id}: {row['name']}  ({row.dist_type})\n"
        f"Anchor {anchors[0]}={days_list[0]}d{extra}  |  "
        f"focus={profile['user_focus_ratio']}  |  ADHD={profile['has_adhd']}"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


def prob_finish(predictions: pd.DataFrame, course_id: str, deadline_days: float, risk_tolerance: str = "med") -> tuple:
    """Returns (P(finish by deadline), safe_buffer_days)."""
    # Convert a course distribution into a deadline probability and safe buffer.
    row = predictions.set_index("course_id").loc[course_id]
    p = lognormal_marginals.prob_finish_by(course_id, row.predicted_days, row.frac, row.df, deadline_days)
    dist = lognormal_marginals.get_distribution(course_id, row.predicted_days, row.frac, row.df)
    
    risk_map = {"high": 0.80, "med": 0.90, "low": 0.95}
    target_p = risk_map.get(risk_tolerance, 0.90)
    
    return float(p), float(dist.ppf(target_p))


def filter_degree_plan(predictions: pd.DataFrame, course_ids: list[str], min_courses: int = 30) -> pd.DataFrame:
    """Return predictions for exactly the selected degree-plan courses.

    Preserves the order in `course_ids`, validates the minimum credit count, and raises if any requested course is
    missing from `predictions`. Use this before `predict_degree_total()` so the
    CLT aggregate sums the user's actual degree plan, not every modeled course.
    """
    # Validate the user's selected degree plan before summing with CLT.
    if not course_ids:
        raise ValueError("course_ids must contain at least one course")

    duplicates = sorted({cid for cid in course_ids if course_ids.count(cid) > 1})
    if duplicates:
        raise ValueError(f"Duplicate course IDs in degree plan: {duplicates}")

    available_count = int(predictions["course_id"].nunique())
    if len(course_ids) > available_count:
        raise ValueError(
            f"You selected {len(course_ids)} courses, but only {available_count} "
            "modeled courses are available. Check for typos or remove extra courses."
        )

    if len(course_ids) < min_courses:
        needed = min_courses - len(course_ids)
        credit_word = "credit" if needed == 1 else "credits"
        raise ValueError(
            f"To graduate you need at least {min_courses} credits; "
            f"you only chose {len(course_ids)} credits. "
            f"Please choose {needed} more {credit_word} and run it again."
        )

    indexed = predictions.set_index("course_id", drop=False)
    missing = sorted(set(course_ids) - set(indexed.index.astype(str)))
    if missing:
        raise ValueError(f"Unknown course IDs: {missing}")

    return indexed.loc[course_ids].reset_index(drop=True).copy()


def predict_degree_total(
    predictions: pd.DataFrame,
    post_params: pd.DataFrame,
    risk_tolerance: str = "med",
    topic_base_corr: float = 0.10,
    topic_max_corr: float = 0.65,
) -> dict:
    """Aggregate per-course predictions into an overall degree completion estimate.

    Uses the Central Limit Theorem with correlated course times. Correlations
    come from course-topic cosine similarity in `modules.course_topics`.

    Returns a dict with total_mu, total_sd, p10, p90, and safe_days.
    """
    from scipy import stats as sp_stats

    # Work on a copy so degree-total calculations never mutate predictions.
    pred = predictions.copy()

    n = len(pred)
    mus = pred["predicted_days"].values.astype(float)
    if {"frac", "df"}.issubset(pred.columns):
        dfs = pred["df"].values.astype(float)
        scales = pred["predicted_days"].values.astype(float) * pred["frac"].values.astype(float)
        var_multipliers = np.full_like(dfs, 3.0, dtype=float)
        mask = dfs > 2.0
        var_multipliers[mask] = dfs[mask] / (dfs[mask] - 2.0)
        sds = scales * np.sqrt(var_multipliers)
    else:
        sds = pred["sd"].values.astype(float)
    corr = course_topics.course_correlation_matrix(
        pred["course_id"].tolist(),
        base_corr=topic_base_corr,
        max_corr=topic_max_corr,
    )

    # Total variance: σᵀ Σ σ  where Σ_ij = corr_ij * sd_i * sd_j
    # Build covariance from topic correlations and per-course standard deviations.
    cov = corr * np.outer(sds, sds)
    total_mu = float(np.sum(mus))
    total_var = float(np.sum(cov))
    total_sd = float(np.sqrt(total_var))

    # CLT: model the aggregate as Normal
    # CLT approximates the many-course total as one normal distribution.
    dist = sp_stats.norm(loc=total_mu, scale=total_sd)

    risk_map = {"high": 0.80, "med": 0.90, "low": 0.95}
    target_p = risk_map.get(risk_tolerance, 0.90)

    return {
        "total_mu": total_mu,
        "total_sd": total_sd,
        "p10": float(dist.ppf(0.10)),
        "p90": float(dist.ppf(0.90)),
        "safe_days": float(dist.ppf(target_p)),
        "risk_tolerance": risk_tolerance,
        "n_courses": n,
        "topic_base_corr": topic_base_corr,
        "topic_max_corr": topic_max_corr,
    }


# ---------- interactive helpers used by the notebook ----------

def _ask(prompt: str, default, cast=str):
    # Small wrapper keeps notebook prompts readable and consistently typed.
    raw = input(f"{prompt} [{default}]: ").strip()
    return cast(raw) if raw else default


def _days_between(start: str, end: str) -> int:
    d1 = pd.to_datetime(start, errors="coerce")
    d2 = pd.to_datetime(end, errors="coerce")
    if pd.isna(d1) or pd.isna(d2):
        raise ValueError(f"couldn't parse dates: start={start!r}, end={end!r}. Try YYYY-MM-DD.")
    if d2 < d1:
        raise ValueError(f"end date {end} is before start date {start}")
    return int((d2 - d1).days)


def _hours_from_days(days: float, hours_per_week: float) -> float:
    if hours_per_week <= 0:
        raise ValueError("hours_per_week must be positive")
    return float(days) * float(hours_per_week) / 7.0


def _days_from_hours(hours: float, hours_per_week: float) -> float:
    if hours_per_week <= 0:
        raise ValueError("hours_per_week must be positive")
    return float(hours) * 7.0 / float(hours_per_week)


def _rating_to_multiplier(rating: float) -> float:
    """Map 1-10 self-rating to a speed multiplier for the latent-factor model.

    Linear: rating 1 -> 1.30 (30% slower), 5.5 -> 1.00 (average),
            rating 10 -> 0.70 (30% faster). Clipped to [0.7, 1.3].
    """
    mult = 1.3 - (rating - 1) * (0.6 / 9.0)
    return max(0.7, min(1.3, mult))


def prompt_profile() -> dict:
    """Collect profile inputs interactively. Returns kwargs for build_profile
    plus 'target_course' and 'deadline_days' for the viz/probability cells.
    """
    print("Leave blank to accept the default in brackets. Dates in YYYY-MM-DD.\n")
    # Build the legacy list inputs from the easier vertical dictionary format.
    anchor_courses: list[str] = []
    anchor_days_list: list[float] = []

    cid = _ask("Anchor course id (one you've already finished)", "DTSA5002")
    start = _ask("Anchor course start date", "2025-01-01")
    end   = _ask("Anchor course end date (last assignment, not final exam)", "2025-02-20")
    anchor_courses.append(cid)
    anchor_days_list.append(_days_between(start, end))
    print(f"  -> {anchor_days_list[-1]} calendar days\n")

    while True:
        more = _ask("Add another anchor course? (y/N)", "n").strip().lower()
        if more not in ("y", "yes"):
            break
        cid = _ask("  Anchor course id", "DTSA5511")
        start = _ask("  Start date", "2025-03-01")
        end   = _ask("  End date (last assignment)", "2025-04-20")
        anchor_courses.append(cid)
        anchor_days_list.append(_days_between(start, end))
        print(f"  -> {anchor_days_list[-1]} calendar days\n")

    anchor_hpw = _ask("Hours per week you actually put into the anchor course(s)", 10, float)

    focus_label = _ask(
        "Focus profile (neurotypical / adhd_unmed / adhd_med / <custom float>)",
        "adhd_unmed",
    )
    focus_map = {
        "neurotypical": focus_ratio.NEUROTYPICAL,
        "adhd_unmed": focus_ratio.ADHD_UNMEDICATED,
        "adhd_med": focus_ratio.ADHD_MEDICATED,
    }
    if focus_label in focus_map:
        user_focus_ratio = focus_map[focus_label]
    else:
        user_focus_ratio = float(focus_label)
    has_adhd = focus_label.startswith("adhd")
    medicated = focus_label == "adhd_med"

    print("\nSelf-rate your skill in each area, 1 (weak) to 10 (strong). 5-6 is average.")
    math_r    = _ask("Math (calculus, probability, linear algebra)", 5, float)
    coding_r  = _ask("Coding (Python, debugging, tooling)", 5, float)
    writing_r = _ask("Writing (essays, explanations, communication)", 5, float)
    print(f"  -> recorded raw ratings: math={math_r}, coding={coding_r}, writing={writing_r}\n")

    target_course = _ask("Course to plan / visualize", "DTSA5511")
    target_dpw = _ask("For the target course, days per week you'll commit", 5, float)
    target_hpd = _ask("Hours per day on those days", 2, float)
    target_hpw = target_dpw * target_hpd
    print(f"  -> {target_hpw:.1f} hr/wk on the target course\n")
    deadline_days = _ask("Deadline in days to finish this course?", 56, float)
    concurrent = _ask("How many courses will you take simultaneously now? (1, 2, 3)", 1, int)
    completed = _ask("How many courses have you already completed in the program?", 1, int)
    print("\nAnchor-time context (so target-vs-anchor adjustments aren't double-counted):")
    anchor_completed = _ask("  How many courses had you completed BEFORE the anchor course?", 0, int)
    anchor_concurrent = _ask("  How many courses were you taking alongside the anchor? (1 = alone)", 1, int)
    risk = _ask("\nRisk tolerance for deadline? (high=80% conf, med=90%, low=95%)", "med")

    # Return kwargs that can be passed directly to build_profile.
    return {
        "anchor_courses": anchor_courses,
        "anchor_days_list": anchor_days_list,
        "user_focus_ratio": user_focus_ratio,
        "has_adhd": has_adhd,
        "medicated": medicated,
        "skill_priors": (math_r, coding_r, writing_r),
        "target_course": target_course,
        "deadline_days": deadline_days,
        "concurrent_courses": concurrent,
        "courses_completed": completed,
        "anchor_concurrent_courses": anchor_concurrent,
        "anchor_courses_completed": anchor_completed,
        "anchor_hours_per_week": anchor_hpw,
        "target_hours_per_week": target_hpw,
        "risk_tolerance": risk,
    }
