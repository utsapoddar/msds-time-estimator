# How the current model works

This file describes the current code in `modules/`.

## Data sources

### Course priors (`course_params.csv`)

`course_params.csv` carries weakly informative prior means in hours:

- courses with guide difficulty use the weighted difficulty fit
  - `expected_hours = 12.671203533443347 * difficulty - 9.417626403703878`
  - `source = guide_difficulty_weighted_ols`
- courses with MSDS review aggregates but no guide-derived row use the aggregate
  review hours directly
  - `source = msds_reviews`
- courses with no student-review average but published Coursera hours use those
  hours directly
  - `source = coursera_published`
- courses with neither review averages nor published-hour evidence use the
  program-median weak prior
  - `expected_hours = 22.9`
  - `source = generic_program_median`

`expected_days` is a compatibility/display column derived from hours using a
baseline of 5 hours/week:

```text
expected_days = expected_hours * 7 / 5
```

### Observations (`data_points.csv`)

`data_points.csv` contains canonical usable observations with these current
sources:

- `msds_reviews` for course-level student-review aggregates from the MSDS review
  sheet
- `coursera_published` for fallback published-hour rows when student-review
  hours are missing
- `google_form` for future individual student submissions if synced later

Each row has `weight`. For aggregate rows, this preserves review count. For
published-hour fallback rows and individual form rows, it is normally 1.

## Intake rules (`modules/form_intake.py`)

`python -m modules.form_intake` refreshes the raw MSDS review rows.

Fetch order:

1. `https://docs.google.com/spreadsheets/d/1lplPW_5DI-wgB_q6qgxmr9WTP12-JVV_yDXsKxLFMiM/export?format=csv&gid=158871878`
2. `https://docs.google.com/spreadsheets/d/1lplPW_5DI-wgB_q6qgxmr9WTP12-JVV_yDXsKxLFMiM/gviz/tq?tqx=out:csv&gid=158871878`
3. if both fail, raise `RuntimeError`

Normalization rules:

- extract `course_id` from the review row
- convert the student time-commitment field into `hours`
- if student hours are missing but Coursera-published hours exist, create a
  `coursera_published` fallback row
- keep `weight` as the review count for aggregate review rows
- use stable hashed `submission_id` values so repeated refreshes dedupe cleanly
- replace prior `msds_reviews`, `msds_reviews_xlsx`, and `coursera_published`
  rows during refresh

## Posterior update (`modules/bayesian_priors.py`)

The posterior layer works in hours, not calendar days.

Cleaning before update:

- keep only rows with valid `course_id`
- compute `hours = days * hours_per_week / 7` if only days and weekly hours are
  available
- drop rows still missing `hours`
- drop non-positive hours
- drop observations for course IDs not present in `course_params.csv`

Tempered update:

```text
effective_n = min(weight, 5)
posterior_mean_hours = (prior_hours + sum(hours * effective_n)) / (1 + sum(effective_n))
posterior_sd_of_mean_hours = prior_sd / sqrt(1 + sum(effective_n))
```

The cap keeps high-review aggregate rows influential without letting one
aggregate completely swamp the prior.

Outputs added in memory:

- `posterior_mean_hours`
- `posterior_sd_of_mean_hours`
- `n_obs`: count of usable rows
- `review_weight`: sum of raw weights/review counts
- `effective_n`: sum of tempered weights used by the update
- compatibility day columns derived from hours

## Personalization (`modules/pipeline.py`)

### Anchor conversion

Users enter completed courses as a vertical dictionary:

```python
anchor_days_by_course = {
    "DTSA5001": 45,
    "DTSA5002": 0,
    ...
}
```

Values less than or equal to zero are ignored. Positive values become anchors.
For each anchor:

```text
anchor_actual_hours = anchor_days * anchor_hours_per_week / 7
anchor_speed = anchor_actual_hours / posterior_mean_hours(anchor_course)
```

Higher speed values mean the course took longer than the current posterior mean.
Lower values mean the student finished faster than the current posterior mean.

### Learning and concurrent-load deltas

The profile applies only target-vs-anchor deltas:

```text
lc(n) = max(0.80, 1 - 0.02 * n)
cs(n) = 1 + 0.15 * max(0, n - 1)
learning_curve_factor = lc(courses_completed) / lc(anchor_courses_completed)
context_switch_factor = cs(concurrent_courses) / cs(anchor_concurrent_courses)
adjusted_anchor_speed = anchor_speed * learning_curve_factor * context_switch_factor
```

This avoids double-counting the student's state at anchor time.

### Skill baseline (`modules/latent_factors.py`)

Self-ratings provide a broad fallback speed shape across math, coding, and
writing:

- rating `1 -> 1.3`
- rating `10 -> 0.7`
- clipped to `[0.7, 1.3]`

This baseline matters most for courses that are not topic-similar to any
completed anchor.

### Topic-based speed transfer (`modules/course_topics.py`)

Official CU Boulder/Coursera course descriptions are hand-scored into topic
vectors. Current axes:

- probability/statistics
- statistical modeling
- machine learning
- NLP/text
- data mining
- programming/algorithms
- databases/SQL
- visualization/HCI
- communication/writing
- ethics/security/policy
- systems/HPC
- quality/measurement
- project delivery

For each target course:

1. compute cosine similarity to completed anchor courses
2. weight anchor speeds by topic similarity
3. blend topic-inferred speed with the self-rating baseline
4. keep distant courses closer to baseline

Prediction output includes:

- `topic_predicted_S`
- `topic_anchor_similarity`

## Prediction step

For each course:

```text
predicted_hours = posterior_mean_hours * topic_predicted_S
predicted_days = predicted_hours * 7 / target_hours_per_week
predicted_days *= 0.70 / user_focus_ratio
```

So focus ratio affects the mean calendar-day estimate, not just variance.

## Variance and intervals

Base CV logic:

```text
cv = base_cv * adhd_variance_multiplier(1.0, has_adhd, medicated)
```

With default `base_cv = 0.20`:

- `0.20` neurotypical
- `0.28` ADHD unmedicated
- `0.23` ADHD medicated

Then:

1. `sd_hours = cv * predicted_hours`
2. convert `sd_hours` to target-schedule days
3. apply the same focus-ratio scale

`lognormal_marginals.py` builds final marginal distributions:

- project-heavy courses use lognormal marginals
- all other courses use truncated normal marginals at 0

## Degree-total estimate

The user selects the exact degree-plan course IDs. The helper validates:

- at least 30 courses/credits
- no duplicate course IDs
- no unknown course IDs
- no more selected courses than modeled courses

`predict_degree_total()` then uses CLT on the selected courses:

```text
total_mean = sum(course_means)
total_variance = sum(topic_correlation_ij * sd_i * sd_j)
total_duration ~ Normal(total_mean, sqrt(total_variance))
```

The covariance matrix uses the same course-topic cosine similarity matrix:

```text
correlation = 0.10 + 0.55 * topic_similarity
```

Diagonal entries are fixed at 1.0.

## Important current limitations

- some courses still rely on Coursera-published hours rather than student-review
  hour averages
- six courses still use the generic program-median weak prior
- the math/coding/writing skill matrix is hand-authored
- the topic matrix is hand-scored from official descriptions
- true empirical calibration needs same-student multi-course completion histories
