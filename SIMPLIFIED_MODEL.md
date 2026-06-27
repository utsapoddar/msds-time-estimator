# Simplified model for this repo copy

This file stores the agreed simplified predictor so we can reference it later.

## Core idea

Keep the model in **hours**, not days.

1. Use the guide workbook's **weighted OLS fit** to map course difficulty to baseline hours.
2. Use the user's **actual completed-course hours** to compute a personal multiplier.
3. Apply that multiplier to the target course baseline hours.
4. Only then convert hours into calendar days using the user's study schedule.

## Verified weighted OLS

From `MSDS Reviews` in the copied workbook:

- `estimated_hours ~= 12.671204 * difficulty - 9.417626`
- weighted `R^2 ~= 0.829265`
- weights = number of course reviews
- `n = 22` DTSA courses with numeric difficulty and student-estimated hours

Unweighted reference fit:

- `estimated_hours ~= 11.368350 * difficulty - 7.303371`
- `R^2 ~= 0.628857`

## Why weighted OLS is the main prior right now

Current agreement:

- low-review course-level average hours are noisy
- weighted OLS is more dependable as the baseline prior right now
- direct course-level student-estimated hours are still stored, but are not the primary baseline

## Baseline formula

For a course with known difficulty:

`baseline_hours(course) = 12.671204 * difficulty(course) - 9.417626`

If difficulty is missing, fall back to legacy priors already in `course_params.csv`.

## Personalization

### Single anchor

If only one completed course is known:

`personal_ratio = your_actual_anchor_hours / baseline_hours(anchor_course)`

`predicted_hours(target_course) = personal_ratio * baseline_hours(target_course)`

### Multiple completed courses

As more courses are completed:

`course_ratio_i = your_actual_hours_i / baseline_hours(course_i)`

Use the **median** of completed-course ratios as the default personal multiplier.

That is more robust than relying forever on one anchor.

## Schedule conversion

After predicting hours, convert to calendar time using either:

- `hours_per_week`
- or `days_per_week * hours_per_day`

Then:

- `calendar_days = predicted_hours * 7 / hours_per_week`
- `study_days_needed = predicted_hours / hours_per_day`

## User inputs

Preferred:

1. completed course(s)
2. actual total hours per completed course
3. target study schedule

If actual total hours are not known, infer them from:

- total calendar days
- anchor days per week
- anchor hours per day

with:

`actual_hours = calendar_days * hours_per_week / 7`
