// polymarket-nvidia-ceo-forecast-01 entry (Node 22, standard library only, deterministic).
//
// Ensemble of three regime-matched pooled empirical quantile forecasters, averaged quantile by quantile. The median
// stays near the last price; the spread comes from the empirical distribution of h-minute price changes in the case's
// own 72 hours, pooled over every market and weighted by recency and by a Gaussian kernel in log scale and log price
// (a market is compared with moments and markets that looked like it). The members differ only in how a market's
// scale (typical size of a move) is measured and in the geometry the changes are measured in:
//   A: mean |1-minute move| over 60 / 360 / 1440 minutes, in price units, plus a floor and 2% of the price;
//   B: high-low range over 15 / 60 / 240 minutes (scaled to a per-minute rate), same floor and price term;
//   C: mean |1-minute move| over 60 / 360 / 1440 minutes of the variance-stabilised price 2*asin(sqrt(p)).
// The quantile at horizon h is the last price plus the weighted normalised quantile times the scale now, interpolated
// in sqrt(h) between 14 measured horizons.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** Horizons (minutes) measured directly; rows in between interpolate linearly in sqrt(h). */
const HORIZONS = [1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135, 180, 240];
const RECENCY_TAU = 720; // minutes; exponential recency weight of a sample (by the minute its change completes)
const KERNEL_LOG_SCALE = 0.6; // kernel bandwidth, natural-log units of the scale
const KERNEL_LOG_PRICE = 2; // kernel bandwidth, natural-log units of (price + PRICE_OFFSET)
const PRICE_OFFSET = 0.005;
const KERNEL_FLOOR = 1e-6; // keeps the weights of a far-out market from all vanishing
const MIN_START = 30; // skip origins with too little history for a scale
const STRIDE = 3; // origins sampled every 3 minutes
const FALLBACK_SIGMA = 0.0025; // baseline random-walk spread, used only if a case has too little data

type Term = { kind: 'mad' | 'range'; window: number };
interface Member { asin: boolean; terms: Term[]; floor: number; priceTerm: number; }
const MEMBERS: Member[] = [
  { asin: false, terms: [60, 360, 1440].map((window) => ({ kind: 'mad', window })), floor: 0.0005, priceTerm: 0.02 },
  { asin: false, terms: [15, 60, 240].map((window) => ({ kind: 'range', window })), floor: 0.0005, priceTerm: 0.02 },
  { asin: true, terms: [60, 360, 1440].map((window) => ({ kind: 'mad', window })), floor: 0.005, priceTerm: 0 },
];

interface CaseMarket { id: string; prices: number[]; }
interface Case { horizon_minutes: number; quantiles: number[]; markets: CaseMarket[]; }
interface MarketInfo { p: number[]; z: Float64Array; n: number; terms: Float64Array[]; }

const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;
const toZ = (p: number, asin: boolean): number => (asin ? 2 * Math.asin(Math.sqrt(Math.min(1, Math.max(0, p)))) : p);
const fromZ = (z: number, asin: boolean): number => (asin ? Math.sin(Math.min(Math.PI, Math.max(0, z)) / 2) ** 2 : z);

/** Causal series of one scale term at every minute (in z units per minute). */
function termSeries(z: Float64Array, term: Term): Float64Array {
  const n = z.length, a = new Float64Array(n), W = term.window;
  if (term.kind === 'mad') {
    const cum = new Float64Array(n);
    for (let i = 1; i < n; i++) cum[i] = cum[i - 1]! + Math.abs(z[i]! - z[i - 1]!);
    for (let t = 0; t < n; t++) { const s = Math.max(0, t - W); a[t] = (cum[t]! - cum[s]!) / Math.max(1, t - s); }
  } else {
    for (let t = 0; t < n; t++) {
      const s = Math.max(0, t - W);
      let hi = -Infinity, lo = Infinity;
      for (let i = s; i <= t; i++) { const v = z[i]!; if (v > hi) hi = v; if (v < lo) lo = v; }
      a[t] = (hi - lo) / Math.sqrt(Math.max(1, t - s)) / Math.sqrt(W);
    }
  }
  return a;
}

/** One member's forecast rows (price units) for every market: [market][minute h-1][level]. */
function memberForecast(c: Case, mem: Member): number[][][] {
  const infos: MarketInfo[] = c.markets.map((m) => {
    if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
    const z = Float64Array.from(m.prices, (p) => toZ(p, mem.asin));
    return { p: m.prices, z, n: m.prices.length, terms: mem.terms.map((t) => termSeries(z, t)) };
  });
  const scaleAt = (M: MarketInfo, t: number): number => {
    let v = 0;
    for (const a of M.terms) v += a[t]! / mem.terms.length;
    return v + mem.floor + mem.priceTerm * (mem.asin ? M.z[t]! : M.p[t]!);
  };
  const H = HORIZONS.length;
  // offsets[market][horizon index][level]: quantile offsets from the last price in z units.
  const offsets: (number[] | null)[][] = c.markets.map(() => HORIZONS.map(() => null));
  HORIZONS.forEach((h, hi) => {
    const V: number[] = [], W: number[] = [], LS: number[] = [], LP: number[] = [];
    for (const M of infos) {
      for (let t = MIN_START; t + h < M.n; t += STRIDE) {
        const s = scaleAt(M, t);
        V.push((M.z[t + h]! - M.z[t]!) / s);
        W.push(Math.exp(-(M.n - 1 - (t + h)) / RECENCY_TAU));
        LS.push(Math.log(s));
        LP.push(Math.log(M.p[t]! + PRICE_OFFSET));
      }
    }
    if (V.length === 0) return;
    const order = Int32Array.from(V.keys()).sort((a, b) => V[a]! - V[b]! || a - b);
    infos.forEach((M, mi) => {
      const t = M.n - 1, s0 = scaleAt(M, t), ls0 = Math.log(s0), lp0 = Math.log(M.p[t]! + PRICE_OFFSET);
      const w = new Float64Array(V.length);
      let total = 0;
      for (let k = 0; k < V.length; k++) {
        const da = (LS[k]! - ls0) / KERNEL_LOG_SCALE, dp = (LP[k]! - lp0) / KERNEL_LOG_PRICE;
        w[k] = W[k]! * (Math.exp(-0.5 * da * da - 0.5 * dp * dp) + KERNEL_FLOOR);
        total += w[k]!;
      }
      const out: number[] = [];
      let acc = 0, pos = 0;
      for (const q of QUANTILES) {
        const target = q * total;
        while (pos < order.length - 1 && acc + w[order[pos]!]! < target) { acc += w[order[pos]!]!; pos++; }
        out.push(V[order[pos]!]! * s0);
      }
      offsets[mi]![hi] = out;
    });
  });

  return c.markets.map((_, mi) => {
    const M = infos[mi]!, zx = M.z[M.n - 1]!;
    const tab = offsets[mi]!;
    const rows: number[][] = [];
    for (let h = 1; h <= c.horizon_minutes; h++) {
      let off: number[];
      if (tab.some((o) => o === null)) off = Z.map((z) => z * FALLBACK_SIGMA * Math.sqrt(h));
      else if (h <= HORIZONS[0]!) off = tab[0]!;
      else {
        let hi = 1;
        while (hi < H - 1 && h > HORIZONS[hi]!) hi++;
        const a = Math.sqrt(HORIZONS[hi - 1]!), b = Math.sqrt(HORIZONS[hi]!), f = Math.min(1, (Math.sqrt(h) - a) / (b - a));
        off = tab[hi - 1]!.map((v, j) => v + f * (tab[hi]![j]! - v));
      }
      const row = tab.some((o) => o === null) ? off.map((d) => Math.min(1, Math.max(0, M.p[M.n - 1]! + d))) : off.map((d) => Math.min(1, Math.max(0, fromZ(zx + d, mem.asin))));
      for (let j = 1; j < 5; j++) if (row[j]! < row[j - 1]!) row[j] = row[j - 1]!;
      rows.push(row);
    }
    return rows;
  });
}

function forecastCase(c: Case): { id: string; q: number[][] }[] {
  const members = MEMBERS.map((mem) => memberForecast(c, mem));
  return c.markets.map((m, mi) => {
    const q: number[][] = [];
    for (let h = 0; h < c.horizon_minutes; h++) {
      const row = QUANTILES.map((_, j) => price6(members.reduce((s, F) => s + F[mi]![h]![j]!, 0) / members.length));
      for (let j = 1; j < 5; j++) if (row[j]! < row[j - 1]!) row[j] = row[j - 1]!;
      q.push(row);
    }
    return { id: m.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
for (const file of readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort()) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file}: unexpected quantiles`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts: forecastCase(c) })}\n`);
}
