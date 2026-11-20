# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Laika-lgtm/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Laika-lgtm/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for the [gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) on make.fun: it reads Exophase's recorded reads of Grand Theft Auto VI's PlayStation 5 trophy page and forecasts every trophy's earn rate at 2026-11-26T00:00:00Z as five quantiles (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## The model

Exophase refreshes its counts on its own schedule, hours apart, so the page is stale between refreshes and jumps at each one. A trophy's rate is earners / players, so a refresh moves it two ways at once: new players dilute it, existing players earning more push it up.

- **Refresh states.** The history is collapsed to its distinct states — consecutive identical reads are one stale stretch, not many observations.
- **Trend branch (the final case).** For each trophy, logit(rate) is fit against ln(1 + hours since release) over the refresh states only, the slope shrunk toward the cross-trophy median slope, then anchored at the last observed rate and carried forward to the target. The spread is the fit's residual scatter plus the slope's standard error times the extrapolation distance — tight, because refreshes are near-deterministic once the trend is known.
- **Dilution branch (flat history, the practice cases).** No refresh has been seen yet, so the next refresh is forecast from the one the rules record on launch night: 638 → 3,868 players tracked, 4,276 trophies earned. Rates dilute by the player ratio, except the trophy everybody holds, which stays at 100%. The refreshed earned total pins how many earners the other trophies gain between them; those extra earners are spread evenly, and each trophy's spread follows their Poisson scatter.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every `*.in` case in the input directory and writes one `<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`. Two runs on the same case write byte-identical output.

## Verification (local, practice reconstruction)

The practice answers are not in the kit, so they are reconstructed three ways from the rules' own numbers: (a) pure dilution, every non-100% rate × 638/3868; (b) the same plus ~2 extra earners per trophy, matching the rules' refreshed total of 4,276 trophies earned; (c) a 0.25x multiplier, the reconstruction the previous build used (it gives persistence 1.35, near that build's 1.41). The house baseline scores 1.69 / 1.37 / 1.27 across the three, bracketing its published ~1.51, so the true answers lie somewhere in this family. On the two practice cases the new program scores:

| reconstruction | this forecaster | persistence | old program |
|---|---|---|---|
| (a) 0.165x dilution | **0.1645** | 1.7767 | 0.8732 |
| (b) +2 earners | **0.0495** | 1.4594 | 0.8732 |
| (c) 0.25x | **0.0796** | 1.3542 | 0.8732 |

A reconstruction is an assumption, not the sealed answer, so these numbers compare models on equal footing; they do not prove the final score. What they show is that the new model beats the old one under every reconstruction of the practice, by 5–18×, and that the practice board — where the leaders sit near 0.0002 — rewards knowing the refresh, not the final week's trend. The final is decided by the trend branch, which the practice cannot test.

## License

MIT — see [LICENSE](LICENSE).
