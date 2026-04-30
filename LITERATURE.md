# Literature: Reported Study Hours vs Actual Focused Time

The model's course priors and observations are in reported study hours. Reported
study time is not the same thing as sustained focused time, so the model exposes
a focus-ratio parameter:

```text
focus ratio = actual focused time / reported or available study time
```

The focus ratio is used to convert focused-hour estimates into realistic
calendar-day estimates for a specific student.

## Neurotypical baseline

- **Theobald et al. 2025** - *Study longer or study effectively? Better study
  strategies can compensate for less study time and predict goal achievement and
  lower negative affect.* British Journal of Educational Psychology.
  231 university students, 30-day daily diary. Perceived study time and study
  strategies such as planning, monitoring, concentration, and procrastination
  interact: with better concentration and lower procrastination, students can
  reach similar goals in less time. Negative affect rises when students study
  long hours with low concentration, consistent with low-focus hours being partly
  wasted.
  - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12068007/)
  - [Wiley](https://bpspsychub.onlinelibrary.wiley.com/doi/10.1111/bjep.12725)

- **Unsworth, McMillan, Brewer & Spillers 2012** - diary study of attention
  failures. A large share of reported attention lapses occur in class/study
  contexts. Task-unrelated thought rates rise over sustained sessions, bounding
  the off-task fraction for long study blocks.
  - Summarized in: [Mind wandering and education (Frontiers 2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3730052/)

## ADHD modifier

- **Henning, Summerfeldt & Parker 2022** - *ADHD and Academic Success in
  University Students: The Important Role of Impaired Attention.* J. Atten.
  Disord. University students with ADHD show lower GPA, fewer study-skill
  strategies, slower program progress, and shorter persistence. Impaired
  attention is the mediating mechanism.
  - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8859654/)

- **Tucha et al. 2017** - *Sustained attention in adult ADHD: time-on-task
  effects of various measures of attention.* J. Neural Transm. Adults with ADHD
  show medium effect-size deficits in alertness, selective attention, and divided
  attention, with steeper time-on-task performance decline than controls.
  - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5281679/)
  - [PubMed](https://pubmed.ncbi.nlm.nih.gov/26206605/)

- **Advokat, Lane & Luo 2011** and related coping-strategy review - ADHD college
  students consistently report lower productive study time per reported hour;
  the gap narrows with medication and/or explicit study-skill strategies.
  - [PMC review](https://pmc.ncbi.nlm.nih.gov/articles/PMC6406620/)

## Suggested focus-ratio parameterization

| Population | Focus ratio (focused / reported) | Basis |
| --- | ---: | --- |
| Neurotypical grad student | 0.65 to 0.75 | Mind-wandering and study-strategy literature |
| ADHD, unmedicated, no explicit strategies | 0.45 to 0.55 | Adult ADHD attention deficits and time-on-task decline |
| ADHD, medicated or strong strategies | 0.60 to 0.70 | Attention gap narrows with medication and study-skill interventions |

## How to apply

The prediction pipeline estimates focused course hours first, then applies the
focus ratio at the calendar-day stage:

```text
calendar_days = focused_hours * 7 / target_hours_per_week * (0.70 / focus_ratio)
```

Example: if a course is estimated at 48 focused hours and the student can put in
10 reported hours/week with a focus ratio of 0.50:

```text
48 * 7 / 10 * (0.70 / 0.50) = 47.0 calendar days
```

## Caveats

- Ratios are priors over broad populations, not individually fitted constants.
- A student's completed-course anchors are more important than the default focus
  ratio once they have real course history in the model.
- Be consistent: record anchor-course hours the same way future available study
  hours will be entered.
