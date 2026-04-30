import unittest
from unittest.mock import patch

import pandas as pd

from modules import bayesian_priors, form_intake


def _real_msds_reviews_fixture() -> pd.DataFrame:
    columns = [
        "CU MSDS on Coursera\nFind this helpful? Contribute to the reviews on \nhttp://tinyurl.com/cu-elective-review",
        "Pathway, Breadth, Elective",
        "Number of \nCourse Reviews",
        "Overall Experience\n1 - Very Poor\n5 - Very Good",
        "Content Difficulty\n1 - Very Easy\n5 - Very Difficult",
        "Instructor Effectiveness\n1 - Very Ineffective\n5 - Very Effective",
        "Students' Estimated Time Commitment (hours)",
        "Coursera\nPublished\nHours",
    ]
    rows = [["Course Title", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA]]
    rows.append(
        [
            "DTSA 5001 - Probability Theory: Applications for Data Science",
            "Pathway - Statistical Inference",
            "14 reviews",
            3.5,
            3.928571,
            3.428571,
            40.846154,
            48.0,
        ]
    )
    rows.append(
        [
            "DTSA 5012 - ANOVA and Experimental Design",
            "Pathway - Statistical Modeling",
            "1 review",
            4.0,
            2.0,
            5.0,
            15.9,
            22.0,
        ]
    )
    for i in range(28):
        course_number = 5100 + i
        rows.append(
            [
                f"DTSA {course_number} - Synthetic Course {i + 1}",
                "Elective",
                f"{(i % 9) + 2} reviews",
                3.0 + (i % 3) * 0.5,
                2.5 + (i % 4) * 0.4,
                3.2 + (i % 5) * 0.3,
                20.0 + i,
                15.0 + i,
            ]
        )
    return pd.DataFrame(rows, columns=columns)


class FormIntakeAggregateTests(unittest.TestCase):
    def test_normalize_msds_reviews_uses_real_live_sheet_headers(self):
        raw = _real_msds_reviews_fixture()

        canonical, aggregates = form_intake.normalize_msds_reviews(raw)

        self.assertEqual(len(canonical), 30)
        self.assertEqual(canonical["course_id"].tolist()[:2], ["DTSA5001", "DTSA5012"])
        self.assertTrue((canonical["source"] == "msds_reviews").all())
        self.assertEqual(canonical["days"].fillna("").tolist()[:2], ["", ""])
        self.assertEqual(canonical["weight"].astype(float).tolist()[:2], [14.0, 1.0])
        self.assertAlmostEqual(float(canonical.loc[0, "hours"]), 40.846154, places=6)

        agg = aggregates.set_index("course_id")
        self.assertEqual(float(agg.loc["DTSA5001", "review_count"]), 14.0)
        self.assertAlmostEqual(float(agg.loc["DTSA5001", "difficulty"]), 3.928571, places=6)

    def test_normalize_msds_reviews_drops_subheader_row_zero(self):
        raw = _real_msds_reviews_fixture()

        canonical, aggregates = form_intake.normalize_msds_reviews(raw)

        self.assertEqual(len(canonical), 30)
        self.assertNotIn("Course Title", canonical["course_id"].tolist())
        self.assertEqual(len(aggregates), 30)

    def test_normalize_msds_reviews_falls_back_to_coursera_published_hours(self):
        raw = _real_msds_reviews_fixture()
        raw.iloc[1, 6] = pd.NA
        raw.iloc[1, 7] = 48.0

        canonical, _ = form_intake.normalize_msds_reviews(raw)

        course_rows = canonical[canonical["course_id"] == "DTSA5001"]
        self.assertEqual(len(course_rows), 1)
        row = course_rows.iloc[0]
        self.assertEqual(row["source"], "coursera_published")
        self.assertAlmostEqual(float(row["hours"]), 48.0, places=6)
        self.assertEqual(float(row["weight"]), 1.0)

    def test_normalize_msds_reviews_falls_back_when_student_hours_is_zero(self):
        raw = _real_msds_reviews_fixture()
        raw.iloc[1, 6] = 0
        raw.iloc[1, 7] = 48.0

        canonical, _ = form_intake.normalize_msds_reviews(raw)

        course_rows = canonical[canonical["course_id"] == "DTSA5001"]
        self.assertEqual(len(course_rows), 1)
        row = course_rows.iloc[0]
        self.assertEqual(row["source"], "coursera_published")
        self.assertAlmostEqual(float(row["hours"]), 48.0, places=6)
        self.assertEqual(float(row["weight"]), 1.0)

    def test_normalize_msds_reviews_emits_expected_workbook_fallback_rows(self):
        raw = form_intake._load_local_workbook_reviews()

        canonical, _ = form_intake.normalize_msds_reviews(raw)

        source_counts = canonical["source"].value_counts().to_dict()
        self.assertEqual(source_counts.get("msds_reviews"), 22)
        self.assertEqual(source_counts.get("coursera_published"), 8)
        self.assertEqual(
            sorted(canonical.loc[canonical["source"] == "coursera_published", "course_id"].tolist()),
            [
                "DTSA5704",
                "DTSA5705",
                "DTSA5706",
                "DTSA5798",
                "DTSA5799",
                "DTSA5800",
                "DTSA5842",
                "DTSA5843",
            ],
        )

    def test_fetch_msds_reviews_raw_raises_instead_of_falling_back(self):
        with patch.object(form_intake, "_fetch_csv_from_url", side_effect=Exception("boom")):
            with patch.object(form_intake, "_load_local_workbook_reviews", return_value=pd.DataFrame()):
                with self.assertRaises(RuntimeError):
                    form_intake.fetch_msds_reviews_raw(url="https://example.com/a.csv", alt_url="https://example.com/b.csv")

    def test_fetch_csv_from_url_reads_live_sheet_with_header_zero(self):
        csv_text = _real_msds_reviews_fixture().to_csv(index=False)

        class _Response:
            def __init__(self, text: str):
                self.text = text

            def raise_for_status(self):
                return None

        with patch.object(form_intake.requests, "get", return_value=_Response(csv_text)):
            raw = form_intake._fetch_csv_from_url("https://example.com/reviews.csv")

        self.assertEqual(
            raw.columns[0],
            "CU MSDS on Coursera\nFind this helpful? Contribute to the reviews on \nhttp://tinyurl.com/cu-elective-review",
        )
        self.assertEqual(raw.iloc[0, 0], "Course Title")


class BayesianPriorWeightingTests(unittest.TestCase):
    def test_prepare_data_points_caps_effective_n(self):
        raw = pd.DataFrame(
            {
                "course_id": ["DTSA5001", "DTSA5012", "DTSA5002"],
                "hours": [40.846154, 15.9, 49.375],
                "weight": [14, 1, None],
            }
        )

        prepared = bayesian_priors.prepare_data_points_for_likelihood(raw)
        self.assertEqual(prepared["effective_n"].tolist(), [5.0, 1.0, 1.0])

    def test_higher_review_weight_reduces_posterior_sd_without_collapsing_to_observation(self):
        params = pd.DataFrame(
            {
                "course_id": ["DTSA5001", "DTSA5012"],
                "name": ["A", "B"],
                "specialization": ["Spec", "Spec"],
                "expected_hours": [30.0, 30.0],
                "expected_days": [42.0, 42.0],
                "source": ["prior", "prior"],
            }
        )
        data_points = pd.DataFrame(
            {
                "course_id": ["DTSA5001", "DTSA5012"],
                "hours": [45.0, 45.0],
                "weight": [14.0, 1.0],
            }
        )

        out = bayesian_priors.update_params_with_posterior(params, data_points).set_index("course_id")

        hi = out.loc["DTSA5001"]
        lo = out.loc["DTSA5012"]
        self.assertLess(float(hi["posterior_sd_of_mean_hours"]), float(lo["posterior_sd_of_mean_hours"]))
        self.assertGreater(float(hi["posterior_mean_hours"]), float(lo["posterior_mean_hours"]))
        self.assertLess(float(hi["posterior_mean_hours"]), 45.0)


if __name__ == "__main__":
    unittest.main()
