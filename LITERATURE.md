# Literature: Reported Study Hours vs Actual Focused Time

Ryan's `expected_hours` (from his MSDS README) is almost certainly logged desk time,
not pure focused time. To translate that into calendar hours *you* need to book,
we need a **focus ratio** = (actual focused time) / (reported study time).

## Neurotypical baseline

- **Theobald et al. 2025** — *Study longer or study effectively? Better study
  strategies can compensate for less study time and predict goal achievement and
  lower negative affect.* British Journal of Educational Psychology.
  231 university students, 30-day daily diary. Perceived study time and study
  strategies (planning, monitoring, concentration, procrastination) **interact**:
  with better concentration / lower procrastination, students hit the same goal
  in less time. Negative affect spikes when students study long hours with low
  concentration — consistent with low-focus hours being partly wasted.
  - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12068007/)
  - [Wiley](https://bpspsychub.onlinelibrary.wiley.com/doi/10.1111/bjep.12725)

- **Unsworth, McMillan, Brewer & Spillers 2012** — diary study of attention
  failures. 76% of reported attention lapses occur in class/study contexts.
  Task-unrelated thought rates rise from ~25% early in a lecture to ~44% by the
  end, bounding the off-task fraction for sustained sessions.
  - Summarized in: [Mind wandering and education (Frontiers 2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3730052/)

## ADHD modifier

- **Henning, Summerfeldt & Parker 2022** — *ADHD and Academic Success in
  University Students: The Important Role of Impaired Attention.* J. Atten.
  Disord. University students with ADHD show lower GPA, fewer study-skill
  strategies, slower program progress, and shorter persistence. Impaired
  attention is the mediating mechanism.
  - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8859654/)

- **Tucha et al. 2017** — *Sustained attention in adult ADHD: time-on-task
  effects of various measures of attention.* J. Neural Transm. Adults with ADHD
  show **medium effect-size** deficits in alertness, selective attention, and
  divided attention, with steeper time-on-task performance decline than
  controls.
  - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5281679/)
  - [PubMed](https://pubmed.ncbi.nlm.nih.gov/26206605/)

- **Advokat, Lane & Luo 2011** (and related coping-strategy review) — ADHD
  college students consistently report lower productive study time per reported
  hour; the gap narrows with medication and/or explicit study-skill strategies.
  - [PMC review](https://pmc.ncbi.nlm.nih.gov/articles/PMC6406620/)

## Suggested focus-ratio parameterization

| Population                                  | Focus ratio (actual / reported) | Basis                                   |
|---------------------------------------------|---------------------------------|-----------------------------------------|
| Neurotypical grad student                   | **0.65 – 0.75**                 | Unsworth mind-wandering + Theobald variance |
| ADHD, unmedicated, no explicit strategies   | **0.45 – 0.55**                 | Tucha medium ES + steeper time-on-task decline |
| ADHD, medicated or strong strategies        | **0.60 – 0.70**                 | Henning — gap narrows with study-skill interventions |

### How to apply

To convert Ryan's logged hours into the **calendar hours you actually need to
book**:

    calendar_hours = expected_hours / focus_ratio

Example: DTSA 5001 at 48 logged hours, ADHD unmedicated (ratio 0.50):
    48 / 0.50 = 96 calendar hours booked to accumulate 48 focused hours.

### Caveats

- Ratios are point estimates over ranges reported in medium-effect-size studies;
  individual variance is large. Treat as priors, not ground truth.
- Ryan's numbers may already reflect *some* off-task time (he logged while
  working, not while perfectly focused), so the ratio applied here is
  conservative. If his log is closer to "actual focused time" the ratio pulls
  your calendar estimate up further.
- The model's anchor-course calibration (`S` from your observed DTSA 5001
  hours) partially absorbs your personal focus ratio **if** you record your
  anchor hours the same way you'll record future courses. Be consistent.
