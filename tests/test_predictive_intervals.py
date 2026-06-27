import statistics
import unittest

import numpy as np
from scipy import stats

from modules import lognormal_marginals


class PredictiveIntervalTests(unittest.TestCase):
    def test_non_skewed_width_decreases_as_df_increases_with_frac_fixed(self):
        center = 100.0
        frac = 0.20

        low_df = lognormal_marginals.get_distribution("DTSA5002", center, frac, 2.0)
        high_df = lognormal_marginals.get_distribution("DTSA5002", center, frac, 100.0)

        low_width = low_df.ppf(0.90) - low_df.ppf(0.10)
        high_width = high_df.ppf(0.90) - high_df.ppf(0.10)

        self.assertGreater(low_width, high_width)

    def test_large_df_with_cv_floor_matches_normal_width_and_does_not_collapse(self):
        center = 100.0
        cv = 0.20
        dist = lognormal_marginals.get_distribution("DTSA5002", center, cv, 1_000_000.0)

        width = dist.ppf(0.90) - dist.ppf(0.10)
        expected = center * cv * (stats.norm.ppf(0.90) - stats.norm.ppf(0.10))

        self.assertGreater(width, 0.0)
        self.assertAlmostEqual(width, expected, delta=0.05)

    def test_small_df_has_larger_p90_than_large_df_at_equal_center_and_frac(self):
        center = 100.0
        frac = 0.20

        low_df = lognormal_marginals.get_distribution("DTSA5002", center, frac, 2.0)
        high_df = lognormal_marginals.get_distribution("DTSA5002", center, frac, 1_000_000.0)

        low_std_units = (low_df.ppf(0.90) - center) / (center * frac)
        high_std_units = (high_df.ppf(0.90) - center) / (center * frac)

        self.assertGreater(low_std_units, high_std_units)

    def test_wider_pace_spread_composes_to_wider_band(self):
        center = 100.0
        cv = 0.20
        cv_est = 0.10

        tight_paces = [1.0, 1.05, 0.95]
        wide_paces = [0.5, 1.0, 1.5]

        def cv_pace(paces):
            n = len(paces)
            return (statistics.stdev(paces) / np.sqrt(n)) / statistics.mean(paces)

        tight_frac = np.sqrt(cv_est**2 + cv_pace(tight_paces)**2 + cv**2)
        wide_frac = np.sqrt(cv_est**2 + cv_pace(wide_paces)**2 + cv**2)

        tight = lognormal_marginals.get_distribution("DTSA5002", center, tight_frac, 10.0)
        wide = lognormal_marginals.get_distribution("DTSA5002", center, wide_frac, 10.0)

        self.assertGreater(wide_frac, tight_frac)
        self.assertGreater(wide.ppf(0.90) - wide.ppf(0.10), tight.ppf(0.90) - tight.ppf(0.10))

    def test_project_heavy_log_t_preserves_right_skew(self):
        dist = lognormal_marginals.get_distribution("DTSA5511", 100.0, 0.20, 20.0)

        p10 = dist.ppf(0.10)
        p50 = dist.ppf(0.50)
        p90 = dist.ppf(0.90)

        self.assertGreater(p90 - p50, p50 - p10)


if __name__ == "__main__":
    unittest.main()
