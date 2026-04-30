"""Weighted decision matrix for MSDS course ordering and elective selection.

Edit the WEIGHTS dict to tune the tradeoff. Criteria are on 1-10 scales:

- career: AI + finance positioning (deep learning, ML, stats, SQL high; ethics, DSaaF low)
- foundation: does it enable later courses? (stats track + intro ML = high)
- light_load: 10 - (predicted_days normalized). High = faster to knock out.
- portfolio: does it produce a showable artifact? (DL, DM project, ML = high)
- admission_gate: 1 if required for program admission (5001/5002/5003), else 0

Scores are hand-assigned, opinionated toward Utsa's stated goals (AI + finance,
physics background, ADHD, side projects competing for time). Tune and re-run.
"""
import pandas as pd

# course_id -> (career, foundation, portfolio) on 1-10
# light_load is computed from predicted_days; admission_gate is hard-coded
SCORES = {
    # Admission trigger / stats foundation
    "DTSA5001": {"career": 7, "foundation": 10, "portfolio": 2, "admission_gate": 1},
    "DTSA5002": {"career": 7, "foundation": 10, "portfolio": 2, "admission_gate": 1},
    "DTSA5003": {"career": 8, "foundation": 10, "portfolio": 3, "admission_gate": 1},
    # Statistical modeling
    "DTSA5011": {"career": 8, "foundation": 9,  "portfolio": 5, "admission_gate": 0},
    "DTSA5012": {"career": 6, "foundation": 6,  "portfolio": 4, "admission_gate": 0},
    "DTSA5013": {"career": 7, "foundation": 6,  "portfolio": 5, "admission_gate": 0},
    # Algorithms / DSA
    "DTSA5501": {"career": 8, "foundation": 8,  "portfolio": 4, "admission_gate": 0},
    "DTSA5502": {"career": 7, "foundation": 7,  "portfolio": 4, "admission_gate": 0},
    "DTSA5503": {"career": 8, "foundation": 6,  "portfolio": 5, "admission_gate": 0},
    # Data mining
    "DTSA5504": {"career": 6, "foundation": 7,  "portfolio": 6, "admission_gate": 0},
    "DTSA5505": {"career": 6, "foundation": 6,  "portfolio": 6, "admission_gate": 0},
    "DTSA5506": {"career": 8, "foundation": 3,  "portfolio": 10, "admission_gate": 0},
    # Machine learning
    "DTSA5509": {"career": 10, "foundation": 9, "portfolio": 9, "admission_gate": 0},
    "DTSA5510": {"career": 9, "foundation": 7,  "portfolio": 8, "admission_gate": 0},
    "DTSA5511": {"career": 10, "foundation": 8, "portfolio": 10, "admission_gate": 0},
    # Databases
    "DTSA5733": {"career": 5, "foundation": 5,  "portfolio": 5, "admission_gate": 0},
    "DTSA5734": {"career": 8, "foundation": 6,  "portfolio": 7, "admission_gate": 0},
    "DTSA5735": {"career": 4, "foundation": 3,  "portfolio": 3, "admission_gate": 0},
    # Vital skills
    "DTSA5301": {"career": 3, "foundation": 4,  "portfolio": 2, "admission_gate": 0},
    "DTSA5302": {"career": 4, "foundation": 3,  "portfolio": 4, "admission_gate": 0},
    "DTSA5303": {"career": 3, "foundation": 2,  "portfolio": 2, "admission_gate": 0},
    "DTSA5304": {"career": 7, "foundation": 4,  "portfolio": 7, "admission_gate": 0},
}

# Tune these weights
WEIGHTS = {
    "career": 3.0,
    "foundation": 2.0,
    "light_load": 1.0,
    "portfolio": 2.0,
    "admission_gate": 1000.0,  # effectively forces these to top
}

# Predicted days from earlier model run at conc=2 (30 hpw, ADHD unmed, math=7/cod=5/wr=6)
PREDICTED_DAYS = {
    "DTSA5001": 47.2, "DTSA5002": 28.7, "DTSA5003": 35.8,
    "DTSA5011": 47.6, "DTSA5012": 42.8, "DTSA5013": 44.5,
    "DTSA5501": 34.4, "DTSA5502": 37.2, "DTSA5503": 40.3,
    "DTSA5504": 21.9, "DTSA5505": 25.5, "DTSA5506": 42.1,
    "DTSA5509": 44.1, "DTSA5510": 38.5, "DTSA5511": 60.5,
    "DTSA5733": 38.5, "DTSA5734": 28.6, "DTSA5735": 19.5,
    "DTSA5301": 11.8, "DTSA5302": 21.2, "DTSA5303": 26.2, "DTSA5304": 15.7,
}


def compute_matrix():
    max_days = max(PREDICTED_DAYS.values())
    min_days = min(PREDICTED_DAYS.values())
    rows = []
    for cid, s in SCORES.items():
        days = PREDICTED_DAYS[cid]
        # light_load on 1-10 scale: shortest course = 10, longest = 1
        light_load = 10 - 9 * (days - min_days) / (max_days - min_days)
        row = {
            "course_id": cid,
            "days": days,
            "career": s["career"],
            "foundation": s["foundation"],
            "light_load": round(light_load, 1),
            "portfolio": s["portfolio"],
            "admission_gate": s["admission_gate"],
        }
        row["score"] = sum(WEIGHTS[k] * row[k] for k in WEIGHTS)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


if __name__ == "__main__":
    df = compute_matrix()
    cols = ["rank", "course_id", "score", "days", "career", "foundation",
            "light_load", "portfolio", "admission_gate"]
    print("\nWeighted course decision matrix (tune WEIGHTS in this file to re-rank):\n")
    print(df[cols].to_string(index=False))
    print(f"\nTotal courses scored: {len(df)}")
    print(f"Top 24 selected for degree (if 30 credits = 30 courses at 1cr each, need 8 more electives + capstone).")
