# MSDS Time Estimator

Bayesian time estimator for CU Boulder MSDS courses, calibrated against student reviews.

## What it does

- Predicts calendar days for a target course from one or more anchor courses you have already finished.
- Uses crowdsourced CU Boulder MSDS review data, then updates course estimates with a tempered Bayesian weight.
- Adjusts for focus ratio, ADHD profile, and self-rated math/coding/writing strengths.
- Returns uncertainty-aware outputs, including per-course standard deviations and 80% intervals.

## Quick start

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd msds-time-estimator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m modules.form_intake
python example_run.py
```

What those commands do:

1. install the dependencies
2. refresh `data_points.csv` from the live sheet
3. run an example profile and print predictions

If you prefer the notebook flow instead, open `time_estimator.ipynb` after install and run the profile/prediction cells there.

## How it works (short)

The model anchors on hours, not just calendar days: your anchor-course days and hours/week are converted into study hours, then compared against each course's posterior mean hours to estimate your pace. Course posteriors are updated from student review data with a tempered effective weight of `min(N, 5)` so high-review courses tighten confidence without completely drowning the prior. For courses that still do not have student review rows, the intake step can fall back to Coursera-published hours from the same MSDS review sheet. Course predictions transfer anchor pace by topic similarity, and degree-total uncertainty uses the same course-topic matrix for CLT correlations. See [MODEL.md](MODEL.md) for the full derivation and implementation details.

## Data sources

Live MSDS Reviews sheet:

- https://docs.google.com/spreadsheets/d/1lplPW_5DI-wgB_q6qgxmr9WTP12-JVV_yDXsKxLFMiM/edit?gid=158871878#gid=158871878

`python -m modules.form_intake` pulls from that sheet's published `MSDS Reviews` tab and refreshes the local review observations automatically.

## Customizing for your own use

The fastest way to customize the example CLI run is to edit the `common = dict(...)` block in `example_run.py`.

Key knobs:

- anchor course and anchor duration: `anchor_courses`, `anchor_days_list`
- anchor and target schedule: `anchor_hours_per_week`, `target_hours_per_week`
- focus ratio and ADHD flags: `user_focus_ratio`, `has_adhd`, `medicated`
- self-ratings: `skill_priors=(math, coding, writing)` on a 1-10 scale

If you want to call the model directly, use `modules.pipeline.build_profile(...)` and `modules.pipeline.build_predictions(...)`.

If you prefer the notebook, edit the profile input cell or switch back to `pipeline.prompt_profile()` for an interactive prompt.

## Contributing

Add your own course review here:

- http://tinyurl.com/cu-elective-review

More reviews make the posterior estimates better.

## Limitations

- Some courses still rely on Coursera-published hours rather than student-reviewed hour averages.
- Focus-ratio constants are heuristic, not individually fitted.
- No cross-validation or held-out calibration study yet.

## License

MIT. See [LICENSE](LICENSE).
