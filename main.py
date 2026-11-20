#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster, v3.

Forecasts each trophy's earn rate on Exophase at the case target as five
quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

Model (written from the competition brief; see README for the reasoning):

  A trophy's earn rate is earners / players, so the trend branch forecasts
  the two sides separately instead of extrapolating the rate itself:

  * log(earners) per trophy, OLS against x = ln(1 + hours since the
    release day began).  Earners come from rate x players / 100 at each
    distinct refresh state (consecutive identical reads collapse into one
    state, timed at its last read: Exophase refreshes in steps, and the
    hundreds of identical reads between refreshes carry no new signal).
  * log(players) once, globally, also OLS against x, but saturating: the
    players slope used is the slower of the full-history slope and the
    second-half slope, clipped at zero -- launch-rush growth slows, it
    never accelerates past its own long-run average.
  * The earners slope is shrunk toward the cross-trophy median slope, but
    the median is taken from the second half of the history (the recent
    regime: the launch-night dilution dominates the early history), and
    the shrinkage is weak -- weight w = d / (d + 0.5) where d is the
    number of distinct refresh states.  In earners space trophies are
    genuinely heterogeneous (story trophies climb while collectibles
    dilute), so the median is only a weak pull; a strong pull is exactly
    the failure this version fixes (on a two-state synthetic case built
    from real reads, strong shrinkage dragged a rising Chapter 1 trophy
    to a forecast *below* its current rate).
  * Forecast: earners and players are extrapolated from their last
    observed values by their slopes over dx = x_target - x_last, and the
    median rate is earners / players x 100.  A trophy at 100% stays 100%.

  Spread (logit scale) grows with the extrapolation distance:
      s = sqrt(resid^2 + (slope_se * dx)^2 + (0.10 * dx)^2),
  floored at 0.08 and capped at 0.80.  The 0.10 random-walk coefficient
  is a DESIGN CHOICE, not a measured quantity (the house baseline uses
  0.5 * sqrt(days) for comparison); the cap is raised from the usual
  0.60 because the final horizon is ~4.5 days, far longer than the
  90-minute practice horizon.

  When the history is flat -- every read the same page, as on launch
  night -- no trend exists to fit.  The dilution branch then forecasts
  the one refresh the rules themselves publish: players 638 -> 3,868
  and trophies earned 943 -> 4,276.  The 3,230 new players all hold the
  100% trophy (it stays 100%); the remaining 103 new earners are split
  among the other trophies in proportion to their old rates (a new
  player earns a trophy with the same relative propensity as the old
  cohort).  Its spread is a fixed 0.25 logit, chosen by simulating the
  allocation of those 103 earners (multinomial and overdispersed draws
  around the proportional weights; 0.25 was near-optimal across them),
  not by tuning to any score.

Only the standard library is used.  The program is deterministic: no
randomness, no wall-clock reads, no network.  It reads every *.in file
in the input directory (MAKEFUN_INPUT_DIR or /input) and writes one
*.out file per case to the output directory (MAKEFUN_OUTPUT_DIR or
/output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Launch-night refresh totals, as published in the competition rules
# (RULES.txt: Exophase refreshed between the 04:00Z and 04:30Z reads on
# 19 Nov 2026: 638 -> 3,868 players tracked, 943 -> 4,276 trophies
# earned).  These are the rules' own numbers, not fitted constants.
DILUTION_PLAYERS_BEFORE = 638
DILUTION_PLAYERS_AFTER = 3868
DILUTION_EARNED_AFTER = 4276
DILUTION_SPREAD = 0.25

# Random-walk growth of the logit spread per unit of x (design choice).
SPREAD_GROWTH = 0.10
SPREAD_FLOOR = 0.08
SPREAD_CAP = 0.80

# Half an earner counts as "someone" when a rate of 0.00% would make
# log(earners) undefined.  Rates are printed to 2 decimals, so a true
# zero and a rounded-down 0.004% are indistinguishable in the input.
EARNERS_FLOOR = 0.25


def logit(pct):
    """Logit of a percent in [0.01, 99.99], clamped like the scorer."""
    p = min(99.99, max(0.01, pct)) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    """Inverse logit, as a percent."""
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    """ISO-8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment, release_midnight):
    """Hours from the release day's midnight to `moment` (may be < 0)."""
    return (moment - release_midnight).total_seconds() / 3600.0


def time_coord(moment, release_midnight):
    """x = ln(1 + hours since the release day began)."""
    hours = hours_since_release(moment, release_midnight)
    return math.log1p(max(-0.999, hours))


def ols(xs, ys):
    """Least squares line through (xs, ys).

    Returns (intercept, slope, rms_residual, slope_standard_error), or
    None when the fit is undefined (fewer than 2 points, or no spread
    in x).  The standard error is 0 when fewer than 3 points leave no
    residual degrees of freedom.
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(r * r for r in residuals) / n)
    if n > 2:
        resid_var = sum(r * r for r in residuals) / (n - 2)
        slope_se = math.sqrt(resid_var / sxx)
    else:
        slope_se = 0.0
    return intercept, slope, rms, slope_se


def median(values):
    """Median of a non-empty list of floats."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def distinct_states(history, release_midnight):
    """Collapse consecutive identical reads into refresh states.

    Each state is (x, players, rates) with x taken at the state's LAST
    read: the state was true at least until then, and the jump to the
    next state happened after it.  A read is identical to the previous
    one when both the players count and every rate match.
    """
    states = []
    for read in history:
        players = read["players_tracked"]
        rates = tuple(read["rates"])
        x = time_coord(parse_time(read["at"]), release_midnight)
        if states and states[-1][1] == players and states[-1][2] == rates:
            states[-1] = (x, players, rates)
        else:
            states.append((x, players, rates))
    return states


def quantiles_from(median_rate, spread):
    """Five quantiles around a median rate, at the case's levels."""
    if median_rate >= 99.99:
        return [100.0, 100.0, 100.0, 100.0, 100.0]
    centre = logit(median_rate)
    out = []
    for z in Z_SCORES:
        q = inv_logit(centre + z * spread)
        out.append(round(min(99.99, max(0.01, q)), 4))
    out.sort()
    return out


def dilution_forecast(case, states):
    """Flat-history branch: forecast the published launch-night refresh.

    Only used when the whole history is one unchanging page whose
    players count is the launch-night 638.  Anything else flat is not
    this refresh, and falls through to the trend branch's fallback.
    """
    _x, players, rates = states[0]
    if players != DILUTION_PLAYERS_BEFORE:
        return None
    old_earners = [rate * players / 100.0 for rate in rates]
    total_old = sum(old_earners)
    new_total = DILUTION_EARNED_AFTER - total_old
    new_players = DILUTION_PLAYERS_AFTER - players
    full = {i for i, rate in enumerate(rates) if rate >= 99.99}
    rest = [i for i in range(len(rates)) if i not in full]
    extra = [0.0] * len(rates)
    for i in full:
        # Every new player holds the 100% trophy: it stays 100%.
        extra[i] = float(new_players)
    remaining = new_total - sum(extra[i] for i in full)
    weight_total = sum(rates[i] for i in rest)
    for i in rest:
        if weight_total > 0:
            extra[i] = remaining * rates[i] / weight_total
        else:
            extra[i] = remaining / len(rest)
    rows = []
    for trophy in case["trophies"]:
        i = trophy["id"] - 1
        if i in full:
            rows.append({"id": trophy["id"], "q": [100.0] * 5})
            continue
        med = 100.0 * (old_earners[i] + extra[i]) / DILUTION_PLAYERS_AFTER
        med = min(99.99, max(0.01, med))
        rows.append({"id": trophy["id"],
                     "q": quantiles_from(med, DILUTION_SPREAD)})
    return {"trophies": rows}


def trend_forecast(case, states, target_x):
    """Earners/players decomposition branch (see module docstring)."""
    n_trophies = len(case["trophies"])
    xs = [s[0] for s in states]
    last_x, last_players, last_rates = states[-1]
    dx = target_x - last_x

    # --- players: one saturating global fit ---------------------------
    log_players = [math.log(s[1]) for s in states]
    players_fit = ols(xs, log_players)
    second_half = states[len(states) // 2:]
    if len(second_half) >= 2:
        players_half = ols([s[0] for s in second_half],
                           [math.log(s[1]) for s in second_half])
    else:
        players_half = None
    if players_fit is None:
        players_slope = 0.0
    else:
        players_slope = players_fit[1]
        if players_half is not None:
            # Saturating: never faster than the long-run average, never
            # faster than the recent pace; take the slower of the two.
            players_slope = min(players_slope, players_half[1])
        players_slope = max(0.0, players_slope)
    players_target = last_players * math.exp(players_slope * dx)

    # --- earners: per-trophy fits, shrunk to the recent median -------
    # Slopes fitted on the second half of the states, for the median.
    half_slopes = []
    if len(second_half) >= 2:
        half_xs = [s[0] for s in second_half]
        for i in range(n_trophies):
            ys = []
            usable = True
            for s in second_half:
                rate = s[2][i]
                if rate is None:
                    usable = False
                    break
                ys.append(math.log(max(EARNERS_FLOOR,
                                       rate * s[1] / 100.0)))
            if not usable:
                continue
            fit = ols(half_xs, ys)
            if fit is not None:
                half_slopes.append(fit[1])

    full_slopes = []
    fits = []
    for i in range(n_trophies):
        ys = []
        for s in states:
            rate = s[2][i]
            if rate is None:
                ys = None
                break
            ys.append(math.log(max(EARNERS_FLOOR, rate * s[1] / 100.0)))
        if ys is None or len(ys) < 2:
            fits.append(None)
            continue
        fit = ols(xs, ys)
        fits.append(fit)
        if fit is not None:
            full_slopes.append(fit[1])

    if half_slopes:
        shrink_target = median(half_slopes)
    elif full_slopes:
        shrink_target = median(full_slopes)
    else:
        shrink_target = 0.0

    d = len(states)
    weight = d / (d + 0.5)  # weak pull: half a pseudo-observation.
    rows = []
    for trophy in case["trophies"]:
        i = trophy["id"] - 1
        last_rate = last_rates[i]
        if last_rate is None:
            last_rate = 0.01
        if last_rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [100.0] * 5})
            continue
        fit = fits[i]
        if fit is None:
            # Too little history to fit: say the rate stays put, with
            # the house baseline's random-walk spread.
            spread = 0.5 * math.sqrt(max(0.0, case["horizon_minutes"]
                                         / 1440.0))
            rows.append({"id": trophy["id"],
                         "q": quantiles_from(last_rate, spread)})
            continue
        _intercept, slope, resid, slope_se = fit
        shrunk = weight * slope + (1.0 - weight) * shrink_target
        earners_last = max(EARNERS_FLOOR, last_rate * last_players / 100.0)
        earners_target = earners_last * math.exp(shrunk * dx)
        med = 100.0 * earners_target / players_target
        med = min(99.99, max(0.01, med))
        spread = math.sqrt(resid * resid
                           + (slope_se * dx) ** 2
                           + (SPREAD_GROWTH * dx) ** 2)
        spread = min(SPREAD_CAP, max(SPREAD_FLOOR, spread))
        rows.append({"id": trophy["id"],
                     "q": quantiles_from(med, spread)})
    return {"trophies": rows}


def forecast_case(case):
    """Forecast one case dict -> output dict."""
    release_text = case["game"]["release_date"]
    release_day = parse_time(release_text[:10] + "T00:00:00Z")
    target_x = time_coord(parse_time(case["target"]), release_day)
    states = distinct_states(case["history"], release_day)
    if not states:
        # No history at all: persistence at 0.01% is the only honest
        # statement; it cannot happen on a real case.
        return {"trophies": [{"id": t["id"], "q": [0.01] * 5}
                             for t in case["trophies"]]}
    if len(states) <= 1:
        diluted = dilution_forecast(case, states)
        if diluted is not None:
            return diluted
    return trend_forecast(case, states, target_x)


def main(argv):
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    if not names:
        print("no case files in %s" % input_dir, file=sys.stderr)
        return 1
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r",
                  encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":"), sort_keys=False)
        with open(os.path.join(output_dir, case_id + ".out"), "w",
                  encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
