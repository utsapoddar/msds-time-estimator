"""Focus-ratio and ADHD variance adjustments for the MSDS time estimator.

Focus-ratio presets sourced from LITERATURE.md (Unsworth 2012, Theobald 2025,
Tucha 2017, Henning 2022). The reference baseline is a generic neurotypical
student profile, not a named individual.
"""

import pandas as pd

# Population midpoints of ranges reported in LITERATURE.md.
NEUROTYPICAL = 0.70          # midpoint of 0.65-0.75
ADHD_UNMEDICATED = 0.50      # midpoint of 0.45-0.55
ADHD_MEDICATED = 0.65        # midpoint of 0.60-0.70
REFERENCE_RATIO = 0.70       # generic neurotypical reference baseline


def apply_focus_ratio(predictions_df, user_ratio, reference_ratio=REFERENCE_RATIO):
    """Scale predicted_days and sd by reference_ratio / user_ratio.

    Lower user focus ratio => more calendar time needed to bank the same
    focused hours. Returns a new DataFrame; input is not mutated.
    """
    scale = reference_ratio / user_ratio
    out = predictions_df.copy()
    out['predicted_days'] = out['predicted_days'] * scale
    out['sd'] = out['sd'] * scale
    return out


def adhd_variance_multiplier(base_cv, has_adhd, medicated=False):
    """Scale CV for ADHD-driven time-on-task variance (Tucha 2017: medium ES, steeper decline)."""
    if not has_adhd:
        return base_cv
    return base_cv * (1.15 if medicated else 1.4)


if __name__ == "__main__":
    # Toy predictions for 3 courses at S=1.0, CV=0.20.
    demo = pd.DataFrame([
        {'course_id': 'DTSA5001', 'expected_days': 67, 'predicted_days': 67.0, 'sd': 13.4},
        {'course_id': 'DTSA5511', 'expected_days': 85, 'predicted_days': 85.0, 'sd': 17.0},
        {'course_id': 'DTSA5301', 'expected_days': 15, 'predicted_days': 15.0, 'sd': 3.0},
    ])

    nt = apply_focus_ratio(demo, user_ratio=NEUROTYPICAL)
    adhd = apply_focus_ratio(demo, user_ratio=ADHD_UNMEDICATED)

    base_cv = 0.20
    cv_nt = adhd_variance_multiplier(base_cv, has_adhd=False)
    cv_adhd = adhd_variance_multiplier(base_cv, has_adhd=True, medicated=False)

    print("Before (generic neurotypical reference):")
    print(demo.to_string(index=False))
    print(f"\nNeurotypical user (ratio={NEUROTYPICAL}, CV={cv_nt:.3f}):")
    print(nt.round(1).to_string(index=False))
    print(f"\nADHD unmedicated (ratio={ADHD_UNMEDICATED}, CV={cv_adhd:.3f}):")
    print(adhd.round(1).to_string(index=False))
