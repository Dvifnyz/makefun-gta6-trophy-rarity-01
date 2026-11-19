// GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).
//
// A statistical forecaster written from scratch for this competition. It reads one
// case (the house's reads of Exophase's GTA VI PS5 trophy page up to the checkpoint)
// and forecasts, per trophy, five quantiles of its earn rate at the target.
//
// Model, per trophy, on the logit scale (logit(p) = ln(p / (100 - p)) for a rate p
// in percent, clamped to [0.01, 99.99] as the scorer clamps):
//   1. DISTINCT READS. Consecutive reads that print the same rates carry no new
//      information (Exophase refreshes on its own schedule, hours apart, so the page
//      sits flat between refreshes). Only reads where the trophy's rate changed
//      enter the fit, so a long flat stretch cannot outweigh the real moves.
//   2. TREND. A recency-weighted least-squares line of logit(rate) against
//      ln(1 + hours since release): launch progress is fast at first and slows, so
//      log-time is the natural clock, and the line saturates in linear time. The
//      slope is shrunk toward the mean slope of the trophy's tier (platinum, gold,
//      silver, bronze): rare trophies have few holders and noisy slopes, and the
//      tier carries the shared shape of how far players get. The line is read at
//      the target; the move from the checkpoint is capped at 2 logit units, so no
//      steep early slope can run away over a 4.5-day horizon.
//   3. SPREAD. Three honest sources of uncertainty, combined in quadrature:
//        - process noise: 0.35 * sqrt(horizon in days) on the logit scale (a
//          random walk's widening; smaller than the house baseline's 0.5 because
//          the trend below already carries the systematic drift);
//        - parameter uncertainty: the fit's residual spread times the standard
//          prediction leverage of extrapolating to the target;
//        - staleness: the longer the page has sat unchanged before the checkpoint,
//          the more likely a refresh lands inside the horizon. A refresh dilutes
//          rates (newly tracked players hold fewer trophies), so staleness widens
//          the LOWER tail more than the upper and nudges the median down a little.
//          Where the history shows refreshes, its size is scaled by the noise the
//          fits saw around the trend over the typical gap between refreshes;
//          where no refresh was ever observed, a fixed prior stands in.
//      A trophy already at 100% stays there with a tight band: it cannot rise,
//      and the launch-night refresh left it at 100%.
//   4. QUANTILES. q_tau = inv_logit(median + z_tau * s_side), z the normal
//      quantiles (-1.645, -0.674, 0, 0.674, 1.645), the lower side using the wider
//      downward spread. Rounded to 4 decimals, clamped to [0.01, 99.99].
//
// Nothing here names a trophy or a rate from outside the case file: every number
// the model uses (slopes, residuals, staleness, tier means) is computed from the
// case's own history. Deterministic: pure functions of the input, no clock, no
// randomness. Standard library only, no network.
//
// Run locally: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

interface Trophy {
  id: number;
  tier: string;
}
interface Case {
  levels: number[];
  at: string;
  horizon_minutes: number;
  target: string;
  game: { release_date?: string };
  trophies: Trophy[];
  history: Array<{ at: string; players_tracked: number; rates: Array<number | null> }>;
}

interface Point {
  x: number;
  y: number;
  w: number;
}

/** Recency-weighted least squares of y on x. Returns the line and fit diagnostics. */
function fitLine(pts: Point[]): {
  a: number;
  b: number;
  xbar: number;
  sxx: number;
  residSd: number;
  n: number;
} {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (const p of pts) {
    sw += p.w;
    sx += p.w * p.x;
    sy += p.w * p.y;
  }
  const xbar = sx / sw;
  const ybar = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (const p of pts) {
    sxx += p.w * (p.x - xbar) * (p.x - xbar);
    sxy += p.w * (p.x - xbar) * (p.y - ybar);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = ybar - b * xbar;
  let rss = 0;
  for (const p of pts) {
    const r = p.y - (a + b * p.x);
    rss += p.w * r * r;
  }
  // Residual spread, weighted; a single point leaves no residual to measure.
  const residSd = pts.length > 2 ? Math.sqrt(rss / sw) : 0;
  return { a, b, xbar, sxx, residSd, n: pts.length };
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const releaseMs = Date.parse(c.game.release_date ?? c.history[0].at);
  const atMs = Date.parse(c.at);
  const targetMs = Date.parse(c.target);
  const horizonDays = c.horizon_minutes / 1440;
  const xNow = Math.log(1 + Math.max(0, (atMs - releaseMs) / 3_600_000));
  const xTarget = Math.log(1 + Math.max(0, (targetMs - releaseMs) / 3_600_000));

  // --- Global staleness: when did any trophy's rate last change? ---
  // The page sits flat between Exophase refreshes; the longer it has sat still
  // before the checkpoint, the more of the horizon a refresh can fall in.
  // changeTimes lists every read at which at least one rate moved.
  const changeTimes: number[] = [Date.parse(c.history[0].at)];
  for (let h = 1; h < c.history.length; h++) {
    const prev = c.history[h - 1].rates;
    const cur = c.history[h].rates;
    for (let i = 0; i < cur.length; i++) {
      if (cur[i] !== null && prev[i] !== null && cur[i] !== prev[i]) {
        changeTimes.push(Date.parse(c.history[h].at));
        break;
      }
    }
  }
  const lastChangeMs = changeTimes[changeTimes.length - 1];
  const staleHours = Math.max(0, (atMs - lastChangeMs) / 3_600_000);
  // Median gap between refreshes, when at least three were observed.
  let gapH: number | null = null;
  if (changeTimes.length >= 3) {
    const gaps: number[] = [];
    for (let k = 1; k < changeTimes.length; k++) {
      gaps.push((changeTimes[k] - changeTimes[k - 1]) / 3_600_000);
    }
    gaps.sort((a, b) => a - b);
    gapH = gaps[Math.floor(gaps.length / 2)];
  }

  // --- Per-trophy trend fits (distinct reads only, recency-weighted). ---
  interface Fit {
    anchor: number; // fitted logit at the checkpoint
    b: number; // raw slope
    bShrunk: number;
    residSd: number;
    xbar: number;
    sxx: number;
    n: number;
    lastRate: number;
  }
  const fits: Fit[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const pts: Point[] = [];
    let prevRate: number | null = null;
    for (const h of c.history) {
      const r = h.rates[i];
      if (r === null || r === undefined) continue;
      // Keep a read only when this trophy's rate moved (or it is the first).
      if (prevRate !== null && Math.abs(r - prevRate) < 1e-9) continue;
      prevRate = r;
      const tMs = Date.parse(h.at);
      const x = Math.log(1 + Math.max(0, (tMs - releaseMs) / 3_600_000));
      // Recency weight: half-life 24 h, so launch-night dynamics fade.
      const w = Math.pow(0.5, Math.max(0, atMs - tMs) / 86_400_000);
      pts.push({ x, y: logit(r), w });
    }
    const lastRate = c.history[c.history.length - 1].rates[i] ?? 0;
    if (pts.length >= 3) {
      const f = fitLine(pts);
      fits.push({
        anchor: f.a + f.b * xNow,
        b: f.b,
        bShrunk: f.b,
        residSd: f.residSd,
        xbar: f.xbar,
        sxx: f.sxx,
        n: f.n,
        lastRate,
      });
    } else {
      fits.push({
        anchor: logit(lastRate),
        b: 0,
        bShrunk: 0,
        residSd: 0,
        xbar: xNow,
        sxx: 0,
        n: pts.length,
        lastRate,
      });
    }
  }

  // --- Staleness allowance: how far can the page have moved since its last
  // refresh? A refresh dilutes rates (newly tracked players hold fewer trophies),
  // so this widens the lower tail and nudges the median down. When the history
  // shows refreshes, the allowance is scaled by the noise the fits actually saw
  // around the trend (residPool, the median residual spread) over the typical
  // gap between refreshes: the move the page can have made while it sat still.
  // With no refresh ever observed (a flat launch night), a fixed prior stands in.
  const residPool = (() => {
    const rs = fits.filter((f) => f.n >= 3).map((f) => f.residSd);
    if (rs.length === 0) return 0;
    rs.sort((a, b) => a - b);
    return rs[Math.floor(rs.length / 2)];
  })();
  const sStale =
    gapH !== null
      ? Math.min(1.2, Math.max(residPool, 0.05) * Math.sqrt(staleHours / gapH))
      : Math.min(1.2, 0.35 * Math.log(1 + staleHours / 3));

  // --- Shrink each slope toward its tier's mean slope. ---
  const tierSlopes = new Map<string, number[]>();
  const allSlopes: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    if (fits[i].n >= 3) {
      const tier = c.trophies[i].tier;
      if (!tierSlopes.has(tier)) tierSlopes.set(tier, []);
      tierSlopes.get(tier)!.push(fits[i].b);
      allSlopes.push(fits[i].b);
    }
  }
  const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
  const overallMean = allSlopes.length > 0 ? mean(allSlopes) : 0;
  for (let i = 0; i < c.trophies.length; i++) {
    const tier = c.trophies[i].tier;
    const pool = tierSlopes.get(tier) ?? [];
    const tierMean = pool.length > 0 ? mean(pool) : overallMean;
    const f = fits[i];
    if (f.n >= 3) {
      const nEff = Math.min(f.n, 8);
      f.bShrunk = (nEff * f.b + 2 * tierMean) / (nEff + 2);
    }
  }

  // --- Forecast each trophy. ---
  // Process noise: the logit random-walk scale per square-root day. The house
  // baseline uses 0.5, which must also cover the drift its persistence median
  // ignores; with a trend model carrying the drift, the remaining noise is
  // smaller, and 0.35 matches the residual scale the fits see on rich history.
  const sRw = 0.35 * Math.sqrt(horizonDays);
  return c.trophies.map((t, i) => {
    const f = fits[i];
    let median = f.anchor;
    let sLo: number;
    let sHi: number;
    if (f.lastRate >= 99.99) {
      // Already earned by everyone tracked: it stays there, tight band.
      median = logit(99.99);
      sLo = 0.08;
      sHi = 0.08;
    } else {
      // Extrapolate the (shrunk) trend, capped at a 2-logit move either way.
      const move = Math.max(-2, Math.min(2, f.bShrunk * (xTarget - xNow)));
      median = f.anchor + move;
      // Parameter uncertainty: residual spread times extrapolation leverage.
      const sParam =
        f.n >= 3
          ? Math.min(
              0.8,
              Math.max(0.1, Math.min(1.2, f.residSd)) *
                Math.sqrt(1 / f.n + (xTarget - f.xbar) ** 2 / Math.max(f.sxx, 1e-9)),
            )
          : 0.15;
      const sTot = Math.sqrt(sRw * sRw + sParam * sParam + sStale * sStale);
      sLo = Math.sqrt(sTot * sTot + sStale * sStale); // refreshes dilute: wider below
      sHi = sTot;
      // The same staleness that widens the lower tail nudges the median down.
      median -= 0.5 * sStale;
      if (f.lastRate >= 99) sHi = Math.min(sHi, 0.15);
    }
    const q: number[] = [];
    let running = -Infinity;
    for (let j = 0; j < 5; j++) {
      const s = Z[j] < 0 ? sLo : Z[j] > 0 ? sHi : 0;
      let v = round4(clampPct(invLogit(median + Z[j] * s)));
      if (v < running) v = running;
      running = v;
      q.push(v);
    }
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (!c.history || c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
