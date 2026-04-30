import unittest
from pathlib import Path

import pandas as pd

from modules import pipeline

ROOT = Path(__file__).resolve().parents[1]


class PublishTodoRegressionTests(unittest.TestCase):
    def test_all_included_data_point_courses_have_course_params(self):
        params = pd.read_csv(ROOT / "course_params.csv")
        data = pd.read_csv(ROOT / "data_points.csv")
        included = data[data["include"].astype(str).str.lower() == "true"]

        missing = sorted(set(included["course_id"].astype(str)) - set(params["course_id"].astype(str)))

        self.assertEqual(missing, [])

    def test_target_hours_per_week_directly_controls_calendar_days(self):
        post = pd.DataFrame(
            {
                "course_id": ["DTSA5001"],
                "name": ["Probability"],
                "specialization": ["Statistical Inference"],
                "expected_hours": [40.0],
                "expected_days": [56.0],
                "posterior_mean_hours": [40.0],
                "posterior_sd_of_mean_hours": [5.0],
            }
        )

        profile = pipeline.build_profile(
            post_params=post,
            anchor_courses=["DTSA5001"],
            anchor_days_list=[28.0],
            user_focus_ratio=0.70,
            has_adhd=False,
            medicated=False,
            skill_priors=(5, 5, 5),
            anchor_hours_per_week=10.0,
            target_hours_per_week=100.0,
        )
        preds = pipeline.build_predictions(post, profile).set_index("course_id")

        self.assertAlmostEqual(float(preds.loc["DTSA5001", "predicted_hours"]), 40.0, places=1)
        self.assertAlmostEqual(float(preds.loc["DTSA5001", "predicted_days"]), 2.8, places=1)

    def test_filter_degree_plan_preserves_order_and_rejects_unknown_courses(self):
        predictions = pd.DataFrame(
            {
                "course_id": ["DTSA5001", "DTSA5002", "DTSA5003"],
                "predicted_days": [10.0, 20.0, 30.0],
                "sd": [1.0, 2.0, 3.0],
            }
        )

        filtered = pipeline.filter_degree_plan(predictions, ["DTSA5003", "DTSA5001"], min_courses=2)

        self.assertEqual(filtered["course_id"].tolist(), ["DTSA5003", "DTSA5001"])
        self.assertEqual(filtered["predicted_days"].tolist(), [30.0, 10.0])

        with self.assertRaisesRegex(ValueError, "Unknown course IDs"):
            pipeline.filter_degree_plan(predictions, ["DTSA9999"], min_courses=1)

    def test_filter_degree_plan_requires_minimum_credits(self):
        predictions = pd.DataFrame(
            {
                "course_id": ["DTSA5001", "DTSA5002"],
                "predicted_days": [10.0, 20.0],
                "sd": [1.0, 2.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "To graduate you need at least 30 credits"):
            pipeline.filter_degree_plan(predictions, ["DTSA5001", "DTSA5002"])

    def test_filter_degree_plan_rejects_duplicate_courses(self):
        predictions = pd.DataFrame(
            {
                "course_id": ["DTSA5001", "DTSA5002"],
                "predicted_days": [10.0, 20.0],
                "sd": [1.0, 2.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "Duplicate course IDs"):
            pipeline.filter_degree_plan(
                predictions,
                ["DTSA5001", "DTSA5001"],
                min_courses=1,
            )

    def test_filter_degree_plan_rejects_more_courses_than_available(self):
        predictions = pd.DataFrame(
            {
                "course_id": ["DTSA5001", "DTSA5002"],
                "predicted_days": [10.0, 20.0],
                "sd": [1.0, 2.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "You selected 3 courses, but only 2 modeled courses are available"):
            pipeline.filter_degree_plan(
                predictions,
                ["DTSA5001", "DTSA5002", "DTSA5003"],
                min_courses=1,
            )

    def test_anchor_days_dict_keeps_only_positive_values_in_course_order(self):
        anchor_days = {"DTSA5002": 0, "DTSA5001": 45.5, "DTSA5003": -2, "DTSA5011": 12}

        courses, days = pipeline.anchors_from_days_dict(anchor_days, ["DTSA5001", "DTSA5002", "DTSA5003", "DTSA5011"])

        self.assertEqual(courses, ["DTSA5001", "DTSA5011"])
        self.assertEqual(days, [45.5, 12.0])

    def test_anchor_days_dict_requires_at_least_one_positive_anchor(self):
        with self.assertRaisesRegex(ValueError, "Set at least one anchor course"):
            pipeline.anchors_from_days_dict({"DTSA5001": 0.0}, ["DTSA5001"])

    def test_posterior_exposes_review_weight_not_just_row_count(self):
        from modules import bayesian_priors

        params = pd.DataFrame(
            {
                "course_id": ["DTSA5001"],
                "name": ["Probability"],
                "specialization": ["Statistical Inference"],
                "expected_hours": [40.0],
                "expected_days": [56.0],
                "source": ["prior"],
            }
        )
        data = pd.DataFrame({"course_id": ["DTSA5001"], "hours": [50.0], "weight": [14.0]})

        out = bayesian_priors.update_params_with_posterior(params, data).iloc[0]

        self.assertEqual(float(out["n_obs"]), 1.0)
        self.assertEqual(float(out["review_weight"]), 14.0)
        self.assertEqual(float(out["effective_n"]), 5.0)


if __name__ == "__main__":
    unittest.main()
