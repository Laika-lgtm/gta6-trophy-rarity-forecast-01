"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Model: a jump-diffusion in logit space, per trophy.

  logit(rate) drifts as the tracked cohort earns trophies (continuous drift,
  estimated from stable-player stretches of the history) and drops in steps
  whenever Exophase refreshes its counts (tracked players jump, the new
  cohort holds fewer trophies each, so every rate falls). Refreshes arrive
  as a Poisson process whose rate is estimated from the jumps seen in the
  history; each refresh multiplies a trophy's rate by a factor g < 1, whose
  distribution is estimated from the jumps seen, shrunk toward a pooled
  value across trophies.

  Priors, from the competition's own description of the launch-night
  recording (not from any answer): refreshes run about one per 4.5 hours
  (the page was flat from 23:48Z to the refresh between 04:00Z and 04:30Z
  on 19 Nov), and a refresh floods in players who just started: they hold
  the instant trophy (rate ~100%) and almost nothing else (638 -> 3,868
  players, 943 -> 4,276 trophies earned, 3,230 of the 3,333 new trophies
  the one 100% trophy), so a typical rate is cut to ~0.25x at a refresh.
  The priors only bite when a case's history shows no refresh yet; a case
  with refreshes in its history is driven by what it shows.

  The forecast for a trophy is a Poisson mixture over the number of
  refreshes before the target, each component normal in logit space, so the
  quantiles are honestly asymmetric: a long lower tail when a refresh is
  likely, a tighter upper side. Quantiles are inverted from the mixture CDF
  by bisection on a fixed grid: fully deterministic, no randomness, two runs
  write byte-identical output.

Standard library only. Run: python3 main.py with MAKEFUN_INPUT_DIR /
MAKEFUN_OUTPUT_DIR, else /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- priors from the launch-night recording described in the rules --------
REFRESH_INTERVAL_HOURS = 4.5      # ~one refresh per 4.5 h on launch night
PRIOR_REFRESH_MULT = 5.0          # typical tracked-player multiplier of a refresh
PRIOR_LOG_G = -1.39               # ln(~0.25): typical rate multiplier at a refresh
PRIOR_LOG_G_VAR = 0.25            # spread of ln(g) across refreshes/trophies
SIGMA_RW_PER_SQRT_DAY = 0.5       # random-walk spread, logit per sqrt(day)
MAX_DRIFT_PER_HOUR = 0.10         # cap on |continuous drift|, logit/hour
MAX_DRIFT_TOTAL = 6.0             # cap on |drift * horizon|, logit


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    if n % 2:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


def mean(values):
    return sum(values) / len(values) if values else 0.0


def var(values, m=None):
    if len(values) < 2:
        return 0.0
    if m is None:
        m = mean(values)
    return sum((v - m) ** 2 for v in values) / (len(values) - 1)


def cohort_ratio(rate):
    """New cohort's earn rate relative to the tracked cohort's, as a
    function of the trophy's rate. A refresh floods in players who just
    started: they hold the instant trophy (rate ~100%) and almost nothing
    else. The launch-night refresh proves it: 3,230 new players brought
    3,333 trophies, 3,230 of them the one 100% trophy, ~103 shared across
    the other 54. So the ratio is ~1 at rate 100 and ~0.05 at rate ~1%:
    (rate/100)**0.6 reproduces that aggregate."""
    return min(1.0, (rate / 100.0) ** 0.6)


def prior_log_g(rate):
    """ln of the rate multiplier at a refresh, before any refresh is seen."""
    g = (1.0 + (PRIOR_REFRESH_MULT - 1.0) * cohort_ratio(rate)) / PRIOR_REFRESH_MULT
    return math.log(g)


class TrophySeries:
    """One trophy's reads: times (hours since first read), players, rates."""

    def __init__(self, rates):
        # rates: list of (hours, players, rate) with rate possibly None
        self.points = [(h, p, r) for (h, p, r) in rates if r is not None]
        self.last_rate = self.points[-1][2] if self.points else 0.0
        self.last_players = self.points[-1][1] if self.points else 0
        self.span_hours = self.points[-1][0] - self.points[0][0] if len(self.points) > 1 else 0.0

    def refresh_indices(self):
        """Indices i>0 where tracked players grew (a refresh landed)."""
        out = []
        for i in range(1, len(self.points)):
            if self.points[i][1] > self.points[i - 1][1]:
                out.append(i)
        return out

    def jump_sizes(self):
        """logit change across each refresh interval."""
        out = []
        for i in self.refresh_indices():
            out.append(logit(self.points[i][2]) - logit(self.points[i - 1][2]))
        return out

    def drift(self):
        """Continuous drift of logit(rate) per hour, from stable-player
        stretches (segments between refreshes). Segment endpoint slopes,
        span-weighted, shrunk toward 0."""
        refresh = set(self.refresh_indices())
        # split into segments of constant players
        segments = []
        start = 0
        for i in range(1, len(self.points)):
            if i in refresh:
                segments.append((start, i - 1))
                start = i
        segments.append((start, len(self.points) - 1))
        num = 0.0
        den = 0.0
        for a, b in segments:
            span = self.points[b][0] - self.points[a][0]
            if span >= 1.0:
                slope = (logit(self.points[b][2]) - logit(self.points[a][2])) / span
                num += span * slope
                den += span
        if den == 0:
            return 0.0, 0.0
        mu = num / (den + 2.0)  # shrink toward 0 with 2 pseudo-hours
        mu = max(-MAX_DRIFT_PER_HOUR, min(MAX_DRIFT_PER_HOUR, mu))
        # residual scatter of logit around the segment lines
        resid = []
        for a, b in segments:
            span = self.points[b][0] - self.points[a][0]
            if span >= 1.0:
                slope = (logit(self.points[b][2]) - logit(self.points[a][2])) / span
                for k in range(a, b + 1):
                    pred = logit(self.points[a][2]) + slope * (self.points[k][0] - self.points[a][0])
                    resid.append(logit(self.points[k][2]) - pred)
        return mu, math.sqrt(var(resid)) if len(resid) > 1 else 0.0


def mixture_quantile(components, level):
    """Invert the CDF of a mixture of normals (list of (weight, mean, sd))
    at `level` by bisection. components must sum to ~1 in weight."""
    lo = min(m for _, m, _ in components) - 12.0
    hi = max(m for _, m, _ in components) + 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = 0.0
        for w, m, s in components:
            cdf += w * norm_cdf((mid - m) / s)
        if cdf < level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_trophy(series, horizon_hours, pooled):
    """pooled: (pooled_log_g, pooled_log_g_var, pooled_drift, lam) shared
    across trophies. Returns five quantiles in percent."""
    r_last = series.last_rate
    if r_last >= 99.995:
        # Held by everyone through every refresh seen: stays 100%.
        return [100, 100, 100, 100, 100]
    pooled_log_g, pooled_var, pooled_drift, lam = pooled
    horizon_days = horizon_hours / 24.0

    mu, resid_sd = series.drift()
    if series.span_hours < 1.0:
        mu = pooled_drift
    # drift contribution, capped
    drift_total = mu * horizon_hours
    drift_total = max(-MAX_DRIFT_TOTAL, min(MAX_DRIFT_TOTAL, drift_total))

    # refresh jump distribution for this trophy: its own RECENT jumps,
    # shrunk to the pooled value (refreshes shrink as the base grows)
    t_end = series.points[-1][0]
    jumps = [dj for (i, dj) in
             ((i, logit(series.points[i][2]) - logit(series.points[i - 1][2]))
              for i in series.refresh_indices())
             if series.points[i][0] >= t_end - 24.0]
    if not jumps:
        jumps = series.jump_sizes()
    if jumps:
        m_obs = mean(jumps)
        m_g = (len(jumps) * m_obs + 2.0 * pooled_log_g) / (len(jumps) + 2.0)
    else:
        m_g = prior_log_g(r_last)
    v_g = pooled_var

    # diffusion: random walk + drift uncertainty + read noise
    sigma_diff = math.sqrt(
        (SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
        + (0.20 * math.sqrt(horizon_days)) ** 2
        + (resid_sd * 0.5) ** 2
        + 0.01
    )

    # Poisson mixture over the number of refreshes before the target
    mean_n = lam * horizon_hours
    comps = []
    n = 0
    w_total = 0.0
    while True:
        # Poisson pmf, computed iteratively to stay deterministic
        if n == 0:
            w = math.exp(-mean_n)
        else:
            w = comps_w_prev * mean_n / n
        comps_w_prev = w
        if w < 1e-6 and n > mean_n + 6:
            break
        if n > 60:
            break
        comps.append((w, logit(r_last) + drift_total + n * m_g,
                      math.sqrt(n * v_g + sigma_diff ** 2)))
        w_total += w
        n += 1
    comps = [(w / w_total, m, s) for (w, m, s) in comps]

    qs = []
    for level in LEVELS:
        lq = mixture_quantile(comps, level)
        qs.append(round4(100.0 * sigmoid(lq)))
    # enforce non-decreasing after rounding
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return [min(100, max(0, q)) for q in qs]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    horizon_hours = case["horizon_minutes"] / 60.0
    t0 = parse_time(case["history"][0]["at"])

    series_list = []
    for i, trophy in enumerate(case["trophies"]):
        pts = []
        for read in case["history"]:
            rate = read["rates"][i] if i < len(read["rates"]) else None
            pts.append(((parse_time(read["at"]) - t0).total_seconds() / 3600.0,
                        read["players_tracked"], rate))
        series_list.append(TrophySeries(pts))

    # pooled statistics across trophies, from RECENT refreshes: refresh
    # multipliers shrink as the tracked base grows (x6 on launch night,
    # far smaller later), so jumps from the last 24 h of history are the
    # regime the target lives in; fall back to all jumps if too few.
    all_jumps = []
    recent_jumps = []
    t_end = max((s.points[-1][0] for s in series_list), default=0.0)
    for s in series_list:
        for i in s.refresh_indices():
            dj = logit(s.points[i][2]) - logit(s.points[i - 1][2])
            all_jumps.append(dj)
            if s.points[i][0] >= t_end - 24.0:
                recent_jumps.append(dj)
    jumps_for_pool = recent_jumps if len(recent_jumps) >= 3 else all_jumps
    pooled_log_g = mean(jumps_for_pool) if jumps_for_pool else PRIOR_LOG_G
    pooled_var = var(jumps_for_pool, pooled_log_g) if len(jumps_for_pool) > 1 else PRIOR_LOG_G_VAR
    pooled_var = max(0.04, min(1.0, pooled_var))
    drifts = [s.drift()[0] for s in series_list]
    pooled_drift = median(drifts)

    # refresh rate: distinct refresh slots in the recent window, smoothed
    # toward the launch-night prior (one per 4.5 h)
    slots = set()
    for s in series_list:
        for i in s.refresh_indices():
            if s.points[i][0] >= t_end - 24.0:
                slots.add(round(s.points[i][0], 6))
    span_recent = min(max((s.span_hours for s in series_list), default=0.0), 24.0)
    lam = (len(slots) + 1.0) / (span_recent + REFRESH_INTERVAL_HOURS) if span_recent > 0 else 1.0 / REFRESH_INTERVAL_HOURS
    pooled = (pooled_log_g, pooled_var, pooled_drift, lam)

    rows = []
    for trophy, s in zip(case["trophies"], series_list):
        rows.append({"id": trophy["id"], "q": forecast_trophy(s, horizon_hours, pooled)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
