# Nontechnical Guide: Using the MSDS Time Estimator

This guide is for people who want to use the estimator without understanding the code.

## What you need to know first

The estimator predicts calendar days for CU Boulder MSDS courses. It works best after you have finished at least one course, because it uses your completed-course time as an anchor.

You will mainly edit two sections in `example_run.py`:

1. completed courses you want to use as anchors
2. courses you plan to count toward the degree total

## Step 1: Install and run once

Open Terminal and run:

```bash
git clone https://github.com/utsapoddar/msds-time-estimator.git
cd msds-time-estimator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python example_run.py
```

If it prints course estimates, the setup worked.

## Step 2: Add your completed courses

Open `example_run.py` in any text editor.

Find this section:

```python
anchor_days_by_course = {
    "DTSA5001": 45.0,
    "DTSA5002": 0.0,
    "DTSA5003": 0.0,
    ...
}
```

Use this rule:

- put the number of calendar days for courses you completed
- leave `0.0` for courses you have not completed
- `45` and `45.0` both work
- decimals work too, for example `45.5`

Example:

```python
anchor_days_by_course = {
    "DTSA5001": 45,
    "DTSA5002": 52,
    "DTSA5003": 0.0,
    ...
}
```

This means the model will use DTSA5001 and DTSA5002 as anchors, and ignore DTSA5003.

## Step 3: Set your weekly study schedule

In the same file, find:

```python
anchor_hours_per_week=10.0,
target_hours_per_week=10.0,
```

Use:

- `anchor_hours_per_week`: how many hours per week you were putting in during the completed anchor course or courses
- `target_hours_per_week`: how many hours per week you expect to put in for future courses

Example:

```python
anchor_hours_per_week=12.0,
target_hours_per_week=8.0,
```

## Step 4: Set focus and ADHD options

Find:

```python
user_focus_ratio=0.70,
has_adhd=False,
medicated=False,
```

Simple defaults:

- use `0.70` if you usually study with decent focus
- use `0.50` if study hours often include a lot of distraction or attention loss
- set `has_adhd=True` if relevant
- set `medicated=True` only if relevant

Examples:

```python
user_focus_ratio=0.50,
has_adhd=True,
medicated=False,
```

or:

```python
user_focus_ratio=0.65,
has_adhd=True,
medicated=True,
```

## Step 5: Set your self-ratings

Find:

```python
skill_priors=(5, 5, 5),  # math, coding, writing; 1=stronger, 10=weaker
```

The order is:

```text
(math, coding, writing)
```

Important: lower number means stronger.

Examples:

```python
skill_priors=(3, 6, 5)
```

This means strong math, weaker coding, average writing.

```python
skill_priors=(7, 4, 3)
```

This means weaker math, stronger coding, strong writing.

## Step 6: Choose the courses in your degree plan

Find:

```python
degree_course_ids = [
    "DTSA5001", "DTSA5002", "DTSA5003",
    ...
]
```

This list controls the total degree-duration estimate.

Rules:

- include the courses you plan to take
- remove courses you do not plan to take
- you need at least 30 courses/credits
- do not list the same course twice

You can swap electives to compare plans. For example, run once with one set of electives, write down the total, then swap courses and run again.

## Step 7: Run the estimator again

After saving `example_run.py`, run:

```bash
python example_run.py
```

## How to read the output

For each course, look at:

- `predicted_days`: estimated average calendar days
- `80%_interval`: likely range for that course

For the full degree, look at:

- `total_mu`: average total calendar days
- `p10` and `p90`: rough 80% range
- `safe_days`: conservative planning number
- `n_courses`: how many courses are included in the degree plan

## Common errors

### Fewer than 30 courses

If you see an error saying you selected fewer than 30 credits, add more course IDs to `degree_course_ids`.

### Unknown course ID

Check spelling. Course IDs should look like:

```text
DTSA5001
DTSA5509
DTSA5842
```

### Duplicate course IDs

Remove the repeated course from `degree_course_ids`.

## Best way to improve accuracy over time

After you finish another course, update `anchor_days_by_course` with that course's actual calendar days and rerun the estimator. The model becomes more personalized as you add more completed courses.
