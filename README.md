# GTA VI Trophy Rarity Forecaster — v3

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Laika-lgtm/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Laika-lgtm/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](main.py)

Forecasts each Grand Theft Auto VI trophy's earn rate on Exophase at
2026-11-26T00:00:00Z as five quantiles (5%, 25%, 50%, 75%, 95%),
scored by pinball loss on the logit scale. Entry for the Make.fun
competition `gta6-trophy-rarity-forecast-01`.

## The model

A trophy's earn rate is **earners / players**, so the trend branch
forecasts the two sides separately instead of extrapolating the rate
itself:

- **log(earners) per trophy**, OLS against `x = ln(1 + hours since the
  release day began)`. Earners are `rate x players / 100` at each
  *distinct refresh state* — consecutive identical reads collapse into
  one state, timed at its last read, because Exophase refreshes in
  steps and the identical reads between refreshes carry no new signal.
- **log(players) once, globally**, also OLS against `x`, but
  *saturating*: the slope used is the slower of the full-history slope
  and the second-half slope, clipped at zero. Launch-rush growth
  slows; it never accelerates past its own long-run average.
- **Weak shrinkage to the recent median.** The earners slope is shrunk
  toward the cross-trophy median slope, but the median is taken from
  the *second half* of the history (the recent regime — the
  launch-night dilution dominates the early history), and the weight is
  `w = d / (d + 0.5)` where `d` is the number of distinct refresh
  states: half a pseudo-observation of pull. In earners space trophies
  are genuinely heterogeneous (story trophies climb while collectibles
  dilute), so the median is only a weak pull.
- **Forecast:** earners and players are extrapolated from their last
  observed values by their slopes over `dx = x_target - x_last`, and
  the median rate is `earners / players x 100`. A trophy at 100% stays
  100%.

**Spread** (logit scale) grows with the extrapolation distance:
`s = sqrt(resid^2 + (slope_se * dx)^2 + (0.10 * dx)^2)`, floored at
0.08 and capped at 0.80. The 0.10 random-walk coefficient is a *design
choice*, not a measured quantity (the house baseline uses
`0.5 * sqrt(days)` for comparison). The cap is raised from the usual
0.60 because the final horizon is ~4.5 days, far longer than the
90-minute practice horizon.

**Flat history (dilution branch).** When every read is the same page,
as on launch night, no trend exists to fit. The branch then forecasts
the one refresh the rules themselves publish: players 638 -> 3,868
and trophies earned 943 -> 4,276. The 3,230 new players all hold the
100% trophy (it stays 100%); the remaining 103 new earners are split
among the other trophies in proportion to their old rates. Its spread
is a fixed 0.25 logit, chosen by simulating the allocation of those
103 earners (multinomial and overdispersed draws around the
proportional weights; 0.25 was near-optimal across them).

## Why v3, and what it fixes

The previous entry's trend branch shrank *logit-rate* slopes toward
their cross-trophy median. That median is dominated by dilution (most
trophies fell from ~1.1% to ~0.01% at the first refresh), so on a
two-state synthetic case built from real reads it dragged rising
trophies *down*: Chapter 1 ("Paradise Lost") was forecast at 7.48%
from 11.98% today, "Beat Cal" at 7.94% from 13.30%. A house model
forecast that reasons narratively will predict those story trophies
rising — a model predicting them falling loses on ~10 story /
early-game trophies. v3 fixes the mechanism, not a number: earners /
players decomposition (the dilution lives in the players fit, where
it belongs), a recent-regime median, and weak shrinkage.

## Validation (measured 21 Nov 2026, before submission)

- **Synthetic 2-state case** (practice history + live Exophase rates
  from 20 Nov 18:50 UTC, target 26 Nov): Chapter 1 median **21.13%**
  (band 17.67–25.06%), Beat Cal **19.72%**, Breakfast Run **4.99%** —
  all inside the [10%, 50%] plausibility band and all *above* their
  current rates (11.98% / 13.30% / 4.26%). The previous entry
  forecast all three *below* current.
- **Practice cases p00/p01** (dilution branch): loss **0.0556**
  against *reconstructed* practice answers — an estimate, not the
  sealed answers, so it is not directly comparable to the previous
  entry's sealed board loss of 0.153198. The sealed v3 practice loss
  is whatever the board shows after submission. Typical miss 0.01pp,
  90% coverage 1.00 on the reconstruction.
- **Determinism:** two runs on the practice cases produce
  byte-identical outputs. Runtime ~0.2 s per run, far inside the 8 s
  cap. Standard library only, no network.

## Honest limits

The final gain is **unmeasurable before 26 Nov**: there is no ground
truth for the target yet. v3 is justified by the mechanical defect
above, which is proven on the synthetic case, not by a measured score
improvement. The prize needs a final loss below the best house entry;
whether v3 achieves that will only be known after the target's rates
are recorded.

## Usage

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` file in the input directory and writes one
`*.out` file per case (`{"trophies": [{"id": n, "q": [q05, q25, q50,
q75, q95]}, ...]}`) to the output directory.
