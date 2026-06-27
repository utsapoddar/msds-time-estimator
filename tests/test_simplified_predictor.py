import unittest
from pathlib import Path

import pandas as pd

from modules import simplified_predictor
from modules.guide_priors import estimate_hours_from_difficulty

ROOT = Path(__file__).resolve().parents[1]


class SimplifiedPredictorTests(unittest.TestCase):
    def test_baseline_hours_uses_weighted_ols_when_difficulty_is_present(self):
        params = pd.read_csv(ROOT / "course_params.csv")
        row = params.set_index("course_id").loc["DTSA5001"]
        difficulty = float(row["difficulty"])

        baseline = simplified_predictor.baseline_hours("DTSA5001", params)

        self.assertAlmostEqual(
            baseline,
            estimate_hours_from_difficulty(difficulty, weighted=True),
            places=7,
        )

    def test_baseline_hours_falls_back_to_expected_hours_when_difficulty_is_nan(self):
        params = pd.read_csv(ROOT / "course_params.csv")
        row = params.set_index("course_id").loc["DTSA5501"]
        self.assertTrue(pd.isna(row["difficulty"]))

        baseline = simplified_predictor.baseline_hours("DTSA5501", params)

        self.assertEqual(baseline, float(row["expected_hours"]))

    def test_ratio_from_completed_courses_uses_median_across_three_or_more_courses(self):
        params = pd.DataFrame(
            {
                "course_id": ["A", "B", "C"],
                "expected_hours": [10.0, 10.0, 10.0],
                "difficulty": [pd.NA, pd.NA, pd.NA],
            }
        )
        completed = [
            {"course_id": "A", "actual_hours": 10.0},
            {"course_id": "B", "actual_hours": 20.0},
            {"course_id": "C", "actual_hours": 100.0},
        ]

        ratio = simplified_predictor.ratio_from_completed_courses(completed, params)

        self.assertEqual(ratio["aggregate"], "median")
        self.assertEqual(ratio["personal_ratio"], 2.0)

    def test_convert_hours_to_schedule_converts_hours_to_calendar_days(self):
        schedule = simplified_predictor.convert_hours_to_schedule(30.0, hours_per_week=10.0)

        self.assertEqual(schedule["calendar_days"], 30.0 * 7.0 / 10.0)

    def test_predict_course_from_history_returns_prediction_fields(self):
        params = pd.read_csv(ROOT / "course_params.csv")
        completed = [
            {"course_id": "DTSA5001", "actual_hours": 40.0},
            {"course_id": "DTSA5002", "actual_hours": 50.0},
            {"course_id": "DTSA5011", "actual_hours": 35.0},
        ]

        prediction = simplified_predictor.predict_course_from_history(
            "DTSA5501",
            completed,
            params,
            hours_per_week=10.0,
        )

        self.assertIn("predicted_hours", prediction)
        self.assertIn("calendar_days", prediction)
        self.assertIn("personal_ratio", prediction)
        self.assertGreater(prediction["predicted_hours"], 0)


if __name__ == "__main__":
    unittest.main()
