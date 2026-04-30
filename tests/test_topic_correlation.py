import unittest

import numpy as np
import pandas as pd

from modules import course_topics, pipeline


class TopicCorrelationTests(unittest.TestCase):
    def test_topic_similarity_matrix_is_symmetric_with_unit_diagonal(self):
        matrix = course_topics.course_correlation_matrix(
            ["DTSA5001", "DTSA5002", "DTSA5842"],
            base_corr=0.10,
            max_corr=0.65,
        )

        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.allclose(np.diag(matrix), 1.0))
        self.assertGreater(matrix[0, 1], matrix[0, 2])

    def test_degree_total_uses_topic_correlation(self):
        predictions = pd.DataFrame(
            {
                "course_id": ["DTSA5001", "DTSA5002", "DTSA5842"],
                "predicted_days": [10.0, 20.0, 30.0],
                "sd": [1.0, 2.0, 3.0],
            }
        )
        post = pd.DataFrame(
            {
                "course_id": ["DTSA5001", "DTSA5002", "DTSA5842"],
                "specialization": ["Stats", "Stats", "Comms"],
            }
        )

        topic_total = pipeline.predict_degree_total(predictions, post)

        self.assertIn("topic_base_corr", topic_total)
        self.assertIn("topic_max_corr", topic_total)
        self.assertGreater(topic_total["total_sd"], 0)

    def test_topic_speed_transfer_keeps_similar_courses_closer_to_anchor(self):
        speeds = course_topics.topic_speed_transfer(
            target_course_ids=["DTSA5002", "DTSA5842"],
            anchor_course_ids=["DTSA5001"],
            anchor_speeds=[2.0],
            baseline_speeds=[1.0, 1.0],
        )

        by_course = dict(zip(speeds["course_id"], speeds["topic_predicted_S"]))
        self.assertGreater(by_course["DTSA5002"], by_course["DTSA5842"])
        self.assertLess(by_course["DTSA5842"], 1.5)

    def test_build_predictions_uses_topic_speed_transfer(self):
        post = pd.DataFrame(
            {
                "course_id": ["DTSA5001", "DTSA5002", "DTSA5842"],
                "name": ["Probability", "Estimation", "Communication"],
                "specialization": ["Stats", "Stats", "Comms"],
                "expected_hours": [40.0, 40.0, 40.0],
                "expected_days": [56.0, 56.0, 56.0],
                "posterior_mean_hours": [40.0, 40.0, 40.0],
                "posterior_sd_of_mean_hours": [5.0, 5.0, 5.0],
            }
        )
        profile = pipeline.build_profile(
            post_params=post,
            anchor_courses=["DTSA5001"],
            anchor_days_list=[56.0],
            user_focus_ratio=0.70,
            has_adhd=False,
            medicated=False,
            skill_priors=(5, 5, 5),
            anchor_hours_per_week=10.0,
            target_hours_per_week=10.0,
        )

        preds = pipeline.build_predictions(post, profile).set_index("course_id")

        self.assertGreater(float(preds.loc["DTSA5002", "predicted_days"]), float(preds.loc["DTSA5842", "predicted_days"]))

    def test_topic_matrix_covers_every_course_param(self):
        params = pd.read_csv("course_params.csv")
        topic_ids = set(course_topics.get_topic_matrix().index.astype(str))

        self.assertEqual(sorted(set(params["course_id"].astype(str)) - topic_ids), [])


if __name__ == "__main__":
    unittest.main()
