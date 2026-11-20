# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Laika-lgtm/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Laika-lgtm/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecast, for every Grand Theft Auto VI trophy on PlayStation 5, the percent
of Exophase-tracked players who will have earned it at 2026-11-26T00:00:00Z,
a week after launch — as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.
An entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## The model

A **jump-diffusion in logit space**, one per trophy:

- **Continuous drift** — the tracked cohort keeps earning trophies, so
  `logit(rate)` drifts. The drift is estimated from stable-player stretches
  of the history (segments between refreshes), span-weighted and shrunk
  toward zero.
- **Refresh jumps** — Exophase refreshes its counts on its own schedule
  (about one per 4.5 hours on launch night). Each refresh floods in players
  who just started: they hold the instant trophy and almost nothing else,
  so every other rate drops in a step. Refreshes arrive as a Poisson
  process whose rate is estimated from the jumps in the history; each
  refresh multiplies a trophy's rate by a factor `g < 1` whose distribution
  is estimated from the jumps seen, shrunk toward a pooled value across
  trophies. Only recent (last-24h) jumps count — refresh multipliers shrink
  as the tracked base grows.
- **Priors** (from the competition's own description of the launch-night
  recording, not from any answer): a refresh multiplies tracked players
  ~5x, and a typical rate is cut to ~0.25x, because 3,230 new players
  brought 3,333 trophies of which 3,230 were the one 100% trophy. The
  priors only bite when a case's history shows no refresh yet.
- The forecast is a **Poisson mixture** over the number of refreshes before
  the target, each component normal in logit space, so the quantiles are
  honestly asymmetric: a long lower tail when a refresh is likely. Quantiles
  are inverted from the mixture CDF by bisection on a fixed grid — fully
  deterministic, no randomness, two runs write byte-identical output.

## Practice check (estimated)

The sealed practice answers are not in the kit, so the practice score below
is scored against answers reconstructed from the aggregate refresh numbers
the rules publish (638 → 3,868 players, 943 → 4,276 trophies earned, the
100% trophy unchanged). That reconstruction reproduces the published
baseline figures (loss ~1.51, typical miss ~0.68pp, coverage 0) at
1.41 / 0.68pp / 0, so it is a usable calibration target, not the sealed
score.

| model | loss | typical miss | 90% coverage |
|---|---|---|---|
| this forecaster | **0.87** | 0.64pp | 1.00 |
| persistence baseline (starter) | 1.41 | 0.68pp | 0.00 |

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `<id>.in` in the
input directory and writes `<id>.out` — `{"trophies": [{"id": n, "q":
[q05, q25, q50, q75, q95]}, ...]}` — to the output directory. About 0.3 s
per case, far inside the 8 s cap; output ~2.7 KB, far inside the 64 KB cap.

## License

MIT — see [LICENSE](LICENSE).
