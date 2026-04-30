# How the current model works

This file describes the code as it exists now in `modules/`.

## Data sources

### 1. Course priors (`course_params.csv`)
Ryan and the synthetic rows are no longer part of the model.

`course_params.csv` now carries weakly-informative prior means in **hours**:

- if a course has guide difficulty, use the workbook-derived weighted fit
  - `expected_hours = 12.671203533443347 * difficulty - 9.417626403703878`
  - `source = guide_difficulty_weighted_ols`
- if guide difficulty is missing, fall back to the program-median weak prior
  - `expected_hours = 22.9`
  - `source = generic_program_median`

`expected_days` is only a compatibility column. It is derived from the hour prior
using a generic baseline of **5 hours/week**:

- `expected_days = expected_hours * 7 / 5`

### 2. Raw observations (`data_points.csv`)
`data_points.csv` now contains one row per usable student review with:

- `source = msds_reviews` for workbook / published-sheet review rows
- `source = google_form` for manual form submissions if those are later synced

Removed sources:

- `github_ryan`
- `artificial`
- `msds_reviews_xlsx`

## Intake rules (`modules/form_intake.py`)

`python -m modules.form_intake` refreshes the raw MSDS review rows.

Fetch order:

1. `https://docs.google.com/spreadsheets/d/1lplPW_5DI-wgB_q6qgxmr9WTP12-JVV_yDXsKxLFMiM/export?format=csv&gid=158871878`
2. `https://docs.google.com/spreadsheets/d/1lplPW_5DI-wgB_q6qgxmr9WTP12-JVV_yDXsKxLFMiM/gviz/tq?tqx=out:csv&gid=158871878`
3. if both fail, raise `RuntimeError` instead of silently using a local workbook snapshot

Normalization rules:

- extract `course_id` from the review row
- keep one `data_points.csv` row per review
- `hours` comes from the review's time-commitment field
- `submission_id` is a stable hash, so repeated refreshes dedupe cleanly
- existing `msds_reviews` / `msds_reviews_xlsx` rows are replaced on refresh

Hours handling:

- if `hours` is missing but `days` and `hours_per_week` exist, compute
  - `hours = days * hours_per_week / 7`
- rows still missing hours are dropped before they reach the likelihood
- rows with non-positive hours are dropped
- `promote_reviewed()` dedupes against existing `submission_id` values before appending to `data_points.csv`

## Posterior update (`modules/bayesian_priors.py`)

The Bayesian layer works in **hours**, not calendar days.

For each course:

- prior mean = `expected_hours` from `course_params.csv`
- all usable rows are treated as **one observation each**
- there is no aggregate-row weighting anymore

Cleaning before PyMC:

- keep only rows with valid `course_id`
- compute `hours` from `days` + `hours_per_week` when possible
- drop rows still missing `hours`
- drop non-positive `hours`
- drop observations for course IDs that are not in `course_params.csv`

Model:

- `sigma_global ~ HalfNormal(10)`
- `offset[i] ~ Normal(0, 1)`
- `mu_course[i] = expected_hours[i] + offset[i] * sigma_global`
- `sigma_student ~ HalfNormal(10)`
- observation likelihood: `y ~ Normal(mu_course[course_idx], sigma_student)`
- fit via `pm.find_MAP()`

Outputs added to `course_params` in memory:

- `posterior_mean_hours`
- `posterior_sd_of_mean_hours`
- compatibility day columns derived from `expected_days / expected_hours`

## Personalization (`modules/pipeline.py`)

### Anchor conversion
Anchor inputs are converted to actual study hours first:

- `anchor_actual_hours = anchor_days * anchor_hours_per_week / 7`

For each anchor:

- `S_anchor = anchor_actual_hours / posterior_mean_hours(anchor_course)`

The profile stores the mean anchor pace and then applies target-vs-anchor deltas.

### Learning-curve and concurrent-load deltas

- `lc(n) = max(0.80, 1 - 0.02 * n)`
- `cs(n) = 1 + 0.15 * max(0, n - 1)`
- `learning_curve_factor = lc(courses_completed) / lc(anchor_courses_completed)`
- `context_switch_factor = cs(concurrent_courses) / cs(anchor_concurrent_courses)`
- `S = mean(S_anchor) * learning_curve_factor * context_switch_factor`

### Skill decomposition (`modules/latent_factors.py`)
The course loadings are still the hand-authored `(math, coding, writing)` matrix.

Self-ratings map to speed multipliers with:

- rating `1 -> 1.3`
- rating `10 -> 0.7`
- clipped to `[0.7, 1.3]`

Single anchor:

- preserve the self-rating shape
- scale that vector so the anchor constraint matches

Multiple anchors:

- solve Tikhonov-regularized least squares toward the self-rating priors

Returned `skill_vec` is clipped to **`[0.25, 4.0]`**.

Per-course speed transfer now uses the topic matrix in `modules/course_topics.py`:

- compute a baseline speed from self-rated math/coding/writing skills
- compare each target course to completed anchor courses by topic cosine similarity
- blend toward the observed anchor pace when the target is topic-similar
- keep distant courses closer to the baseline instead of assuming one anchor applies equally to every course

The output columns include `topic_predicted_S` and `topic_anchor_similarity` for inspection.

## Prediction step (`modules/pipeline.py`)

The mean prediction is built in this order:

1. `predicted_hours = posterior_mean_hours * predicted_S`
2. convert those hours directly to target-schedule days
   - `predicted_days = predicted_hours * 7 / target_hours_per_week`
3. apply focus-ratio scaling
   - `predicted_days *= 0.70 / user_focus_ratio`

So `focus_ratio` now affects the **mean**, not just the variance.

## Variance and intervals

Base CV logic:

- `cv = base_cv * adhd_variance_multiplier(1.0, has_adhd, medicated)`
- with default `base_cv = 0.20`, this becomes
  - `0.20` neurotypical
  - `0.28` ADHD unmedicated
  - `0.23` ADHD medicated

Then:

1. `sd_hours = cv * predicted_hours`
2. convert `sd_hours` to target-schedule days
3. apply the same focus-ratio scale

`lognormal_marginals.py` then builds the final marginal distributions:

- project-heavy courses use lognormal marginals
- the rest use truncnorm marginals at 0

## Degree-total correlation

`predict_degree_total()` uses CLT on the selected degree-plan courses. By default, the covariance matrix uses transparent course-topic similarity from `modules/course_topics.py`:

- topic axes come from official course descriptions: probability/statistics, statistical modeling, machine learning, NLP/text, data mining, programming/algorithms, databases/SQL, visualization/HCI, communication/writing, ethics/security/policy, systems/HPC, quality/measurement, and project delivery
- pairwise similarity is cosine similarity between course topic vectors
- correlation is mapped as `0.10 + 0.55 * similarity`, with diagonal entries fixed at 1.0

This is still a prior correlation model, not an empirical one. Related courses share more covariance even when they sit under different catalog labels. When same-student multi-course completion histories exist, this can be replaced or blended with empirical pairwise correlations.

## Important current limitations

- some courses still rely on Coursera-published hours rather than student-reviewed hour averages
- the skill matrix is hand-authored; the topic matrix is hand-scored from official course descriptions
- the model is still MAP-based, not MCMC-based
