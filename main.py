#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its
earn rate on Exophase at a case's target, as five quantiles (5/25/50/75/95%).

Model (earners/players decomposition, refresh-aware):

  Exophase does not update its counts continuously: it refreshes them on its
  own schedule, hours apart. Between refreshes the page is stale; at a
  refresh the player count and every trophy's earner count jump to their
  current values. A rate is earners / players, so a refresh moves a rate in
  two ways at once: new players dilute it, existing players earning more
  push it up. The net move per trophy is what the history shows.

  1. Refresh states. The case history is collapsed to its distinct states:
     consecutive reads with the same player count and the same rates are one
     stale stretch, not many observations. Every distinct state is one
     observation of (time, players, earners per trophy).

  2. Trend branch (history holds refreshes -- the final case). For each
     trophy, fit logit(rate) against ln(1 + hours since release) by least
     squares over the refresh states only, shrink the slope toward the
     cross-trophy median slope (weight grows with the number of states),
     then anchor the forecast at the last observed rate and carry the shrunk
     slope forward to the target. The spread is the fit's residual scatter
     combined with the slope's standard error times the extrapolation
     distance, floored and capped: refreshes are near-deterministic once the
     trend is known, so the band is tight, not a wide random walk.

  3. Dilution branch (history is flat -- the practice cases). No refresh has
     been seen yet, so the next refresh is forecast from the refresh the
     rules record on launch night: the player count jumps and every
     trophy's rate dilutes by the player ratio, except the trophy every
     tracked player holds, which stays at 100%. The rules also record the
     refreshed total of trophies earned, which pins how many earners the
     non-trivial trophies gain between them; that many extra earners are
     spread evenly over them, and the per-trophy spread follows the
     Poisson scatter of those extra earners over the trophy's earner count.

  Only the standard library is used. No network, no randomness, so two runs
  on the same case write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Normal quantiles for the five levels [0.05, 0.25, 0.5, 0.75, 0.95].
Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722)

# Launch-night refresh, as the competition rules record it: the page sat
# unchanged at 638 players tracked, then refreshed to 3,868 players and
# 4,276 trophies earned. Used only by the dilution branch, when the case
# history itself holds no refresh to learn from.
RULES_REFRESH_PLAYERS = 3868
RULES_REFRESH_EARNED = 4276

# Spread guardrails (logit scale).
SPREAD_FLOOR = 0.08
SPREAD_CAP = 0.60


def clamp_rate(x):
    """Clamp a percent rate into the range the logit is defined on."""
    return min(max(float(x), 0.01), 99.99)


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def parse_time(s):
    """Parse an ISO 8601 timestamp (with Z or an offset) to epoch seconds."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def hours_since_release(at_epoch, release_date):
    """Hours from the game's release day (00:00 UTC) to a timestamp."""
    rel = datetime.fromisoformat(release_date).replace(tzinfo=timezone.utc)
    return max(0.0, (at_epoch - rel.timestamp()) / 3600.0)


def refresh_states(case):
    """Collapse the history to its distinct states.

    Returns a list of (epoch, players_tracked, rates) with one entry per
    state change, oldest first, plus the trophies-earned total of the last
    read. Reads that repeat the previous state are stale copies, not new
    information.
    """
    states = []
    last_key = None
    earned_last = 0
    for read in case.get("history", []):
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        earned_last = read.get("trophies_earned") or 0
        if key == last_key:
            continue
        last_key = key
        try:
            when = parse_time(read["at"])
        except (KeyError, ValueError):
            continue
        states.append((when, read.get("players_tracked"), list(rates)))
    return states, earned_last


def median(lst):
    s = sorted(lst)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def fit_line(xs, ys):
    """Least-squares line y = a + b*x. Returns (a, b, resid_sd, slope_se)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 2 or sxx <= 0.0:
        return my, 0.0, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(
            sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
        slope_se = resid / math.sqrt(sxx)
    else:
        resid = 0.0
        slope_se = 0.0
    return a, b, resid, slope_se


def forecast_case(case):
    """Return the forecast rows [{"id": n, "q": [q05..q95]}] for one case."""
    trophies = case.get("trophies", [])
    n_tr = len(trophies)
    target_epoch = parse_time(case["target"])
    release = case.get("game", {}).get("release_date", "2026-11-19")
    x_target = math.log1p(hours_since_release(target_epoch, release))

    states, _earned_last = refresh_states(case)
    if not states:
        # No usable history at all: there is nothing to anchor on, so emit
        # the floor rate for every trophy rather than crashing.
        return [{"id": t["id"], "q": [0.01, 0.01, 0.01, 0.01, 0.01]}
                for t in trophies]

    last_when, last_players, last_rates = states[-1]
    x_last = math.log1p(hours_since_release(last_when, release))

    def last_rate(i):
        for _w, _p, rates in reversed(states):
            if i < len(rates) and rates[i] is not None:
                return rates[i]
        return None

    if len(states) == 1:
        return dilution_forecast(trophies, last_rates, last_players)

    # --- Trend branch: fit logit(rate) on ln(1 + hours) per trophy. ---
    raw_slopes = []
    fits = {}
    for i, tr in enumerate(trophies):
        xs, ys = [], []
        for when, _p, rates in states:
            if i < len(rates) and rates[i] is not None:
                r = rates[i]
                if 0.01 <= r <= 99.99:
                    xs.append(math.log1p(hours_since_release(when, release)))
                    ys.append(logit(r))
        if len(xs) >= 2:
            a, b, resid, slope_se = fit_line(xs, ys)
            fits[i] = (a, b, resid, slope_se, len(xs))
            raw_slopes.append(b)
        else:
            fits[i] = None
    med_slope = median(raw_slopes) if raw_slopes else 0.0
    med_resid = median([f[2] for f in fits.values() if f]) if any(fits.values()) else 0.0
    dx = x_target - x_last

    rows = []
    for i, tr in enumerate(trophies):
        r_last = last_rate(i)
        if r_last is None:
            r_last = 0.01
        if r_last >= 99.99:
            # Everybody tracked holds it; new players earn it on arrival.
            rows.append({"id": tr["id"], "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
            continue
        fit = fits[i]
        if fit is None:
            # Too few usable points: hold the last rate, modest spread.
            med_logit = logit(r_last)
            spread = 0.20
        else:
            _a, b, resid, slope_se, npts = fit
            w = npts / (npts + 3.0)
            b_shrunk = w * b + (1.0 - w) * med_slope
            med_logit = logit(r_last) + b_shrunk * dx
            resid_use = max(resid, med_resid, 0.05)
            spread = math.sqrt(resid_use ** 2 + (slope_se * dx) ** 2)
            spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)
        qs = []
        for z in Z:
            q = inv_logit(med_logit + z * spread)
            qs.append(round(min(max(q, 0.01), 99.99), 4))
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": tr["id"], "q": qs})
    return rows


def dilution_forecast(trophies, last_rates, last_players):
    """Flat history: forecast the first refresh from the rules' record.

    The player count jumps to RULES_REFRESH_PLAYERS; every trophy's rate
    dilutes by the player ratio, except the trophy everybody holds, which
    stays at 100%. The refreshed earned total pins the extra earners the
    other trophies gain between them; they are spread evenly, and each
    trophy's spread follows the Poisson scatter of those extra earners.
    """
    n_tr = len(trophies)
    p_old = last_players or 638
    p_new = RULES_REFRESH_PLAYERS
    earners_old = []
    for i, tr in enumerate(trophies):
        r = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else 0.0
        earners_old.append(round(r * p_old / 100.0))
    n_full = sum(1 for r in last_rates if r is not None and r >= 99.99)
    n_rest = max(n_tr - n_full, 1)
    earned_old_total = sum(earners_old)
    # New earners not taken by the trophy everybody holds.
    extra_total = RULES_REFRESH_EARNED - p_new - (earned_old_total - p_old * n_full)
    extra_total = max(extra_total, 0)
    extra_each = extra_total / n_rest

    rows = []
    for i, tr in enumerate(trophies):
        r = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else 0.0
        if r >= 99.99:
            rows.append({"id": tr["id"], "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
            continue
        e_new = earners_old[i] + extra_each
        med = e_new / p_new * 100.0
        # Poisson scatter of the extra earners, in log terms.
        spread = math.sqrt(extra_each + 1.0) / max(e_new, 1.0)
        spread = min(max(spread, 0.10), 0.50)
        med_logit = logit(med)
        qs = []
        for z in Z:
            q = inv_logit(med_logit + z * spread)
            qs.append(round(min(max(q, 0.01), 99.99), 4))
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": tr["id"], "q": qs})
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        out = {"trophies": rows}
        out_name = name[:-3] + ".out" if name.endswith(".in") else name + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(out, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
