"""Example MSDS time-estimator run. Edit the profile values below for your case."""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from modules import pipeline

post = pipeline.load_posterior_params("course_params.csv", "data_points.csv")

# Put completed courses here. Values are calendar days to finish.
# Keep 0.0 for courses that should not be used as anchors.
anchor_days_by_course = {
    "DTSA5001": 45.0,
    "DTSA5002": 0.0,
    "DTSA5003": 0.0,
    "DTSA5501": 0.0,
    "DTSA5502": 0.0,
    "DTSA5503": 0.0,
    "DTSA5011": 0.0,
    "DTSA5012": 0.0,
    "DTSA5013": 0.0,
    "DTSA5504": 0.0,
    "DTSA5505": 0.0,
    "DTSA5506": 0.0,
    "DTSA5509": 0.0,
    "DTSA5510": 0.0,
    "DTSA5511": 0.0,
    "DTSA5733": 0.0,
    "DTSA5734": 0.0,
    "DTSA5735": 0.0,
    "DTSA5301": 0.0,
    "DTSA5302": 0.0,
    "DTSA5303": 0.0,
    "DTSA5304": 0.0,
    "DTSA5020": 0.0,
    "DTSA5021": 0.0,
    "DTSA5022": 0.0,
    "DTSA5701": 0.0,
    "DTSA5702": 0.0,
    "DTSA5703": 0.0,
    "DTSA5704": 0.0,
    "DTSA5705": 0.0,
    "DTSA5706": 0.0,
    "DTSA5798": 0.0,
    "DTSA5799": 0.0,
    "DTSA5800": 0.0,
    "DTSA5842": 0.0,
    "DTSA5843": 0.0,
}
anchor_courses, anchor_days_list = pipeline.anchors_from_days_dict(anchor_days_by_course, post["course_id"].tolist())

common = dict(
    post_params=post,
    anchor_courses=anchor_courses,
    anchor_days_list=anchor_days_list,
    user_focus_ratio=0.70,
    has_adhd=False,
    medicated=False,
    skill_priors=(5, 5, 5),  # math, coding, writing; 1=weaker, 10=stronger
    anchor_hours_per_week=10.0,
    target_hours_per_week=10.0,
    anchor_concurrent_courses=1,
    anchor_courses_completed=0,
)

for conc in [1, 2, 3]:
    prof = pipeline.build_profile(concurrent_courses=conc, courses_completed=0, **common)
    preds = pipeline.build_predictions(post, prof)
    preds_sorted = preds.sort_values("course_id")
    print(f"\n========== concurrent_courses = {conc} ==========")
    pipeline.print_profile(prof)
    print()
    print(preds_sorted[["course_id", "name", "predicted_days", "sd", "80%_interval"]].to_string(index=False))

# Degree-plan total at conc=2 as a middle estimate. Edit to match your actual plan.
degree_course_ids = [
    "DTSA5001", "DTSA5002", "DTSA5003",
    "DTSA5011", "DTSA5012", "DTSA5013",
    "DTSA5501", "DTSA5502", "DTSA5503",
    "DTSA5504", "DTSA5505", "DTSA5506",
    "DTSA5509", "DTSA5510", "DTSA5511",
    "DTSA5733", "DTSA5734", "DTSA5735",
    "DTSA5301", "DTSA5302", "DTSA5303", "DTSA5304",
    "DTSA5020", "DTSA5021", "DTSA5022",
    "DTSA5701", "DTSA5702", "DTSA5703",
    "DTSA5704", "DTSA5705", "DTSA5706",
]

print("\n========== Degree-plan total (conc=2) ==========")
prof2 = pipeline.build_profile(concurrent_courses=2, courses_completed=0, **common)
preds2 = pipeline.build_predictions(post, prof2)
degree_preds2 = pipeline.filter_degree_plan(preds2, degree_course_ids)
tot = pipeline.predict_degree_total(degree_preds2, post, risk_tolerance="med")
for k, v in tot.items():
    print(f"  {k}: {v}")
