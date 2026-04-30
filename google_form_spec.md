# Google Form: MSDS Course Time Survey

Paste this spec into Google Forms (one-time). All fields except justification + contact are required.

**Form title:** CU Boulder MSDS — how long did your course actually take?
**Description:** Helping future students estimate course workload. Anonymous by default; contact field is optional. Takes ~90 seconds per course.

---

## Section 1 — Course

1. **Which course?** (Dropdown — required)
   - List every `course_id` from `course_params.csv` with the course name, e.g.:
     - `DTSA5001 — Probability Foundations for Data Science and AI`
     - `DTSA5002 — Statistical Estimation for Data Science and AI`
     - ... (22 total)
     - `Other (please specify in notes)`

2. **When did you take it?** (Short answer — required) — format hint: `Fall 2024`, `Summer 2025`, etc.

## Section 2 — Time

3. **Start date** (Date field — required)
   Help text: "The day you first opened the course."

4. **End date (last assignment submitted)** (Date field — required)
   Help text: "The day you submitted the last assignment. Exclude the final exam."

5. **Hours per week available during the course** (Short answer — number, required)
   Help text: "Realistic average, not aspirational. 5 = evenings only. 20+ = full-time student."

5. **Concurrent courses taken alongside this one** (Multiple choice — required)
   - None (took this course alone)
   - 1 other course
   - 2 other courses
   - 3+ other courses

6. **Grade received** (Multiple choice — required)
   - A / A-
   - B+ / B / B-
   - C or below
   - Pass (no letter grade)
   - Prefer not to say

## Section 3 — Personal context

7. **Do you have ADHD (diagnosed or strongly suspected)?** (Multiple choice — required)
   - No
   - Yes, unmedicated
   - Yes, medicated
   - Prefer not to say

8. **Self-rated math strength** (Linear scale 1–5 — required)
   Labels: 1 = weak, 5 = strong

9. **Self-rated coding strength** (Linear scale 1–5 — required)

10. **Self-rated writing strength** (Linear scale 1–5 — required)

11. **Prior background most relevant to this course** (Multiple choice — required)
    - None — this was entirely new material
    - Undergraduate coursework in a related area
    - Professional experience in a related area
    - Both undergrad and professional

## Section 4 — Optional

12. **Notes or anything unusual about your experience** (Paragraph — optional)
    Help text: "If your time was unusually short or long, please explain. E.g. 'I skipped lectures because I already knew the material' or 'I had a family emergency mid-course.'"

13. **Contact (optional)** (Short answer — optional)
    Help text: "Email or reddit handle, in case we have a follow-up question. Never published."

---

## Outlier logic (handled by Apps Script, not form branching)

After submission, Apps Script inspects the row and flags it for review if **any** of:
- Calendar days `< 30` **or** `> 90`
- Hours/week `< 2` or `> 60` (clearly non-serious or inhuman)
- Notes field has non-empty text (any justification, optional or not, triggers manual review)

Flagged rows are:
- Tagged `review_needed = "review"` in the linked sheet (otherwise `"auto"`)
- Trigger an email to the form owner

The form owner later sets `include = true/false/duplicate` via the notebook review cell; only `include == true` rows flow into the Bayesian posterior.
