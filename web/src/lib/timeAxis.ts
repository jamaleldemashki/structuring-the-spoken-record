// Axis helpers for the Timeline, so it fits whatever span an archive covers.

const STEPS = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000];

/**
 * A lifespan axis around the given [start, end] ranges (negative years are BCE),
 * padded and snapped to a round step that gives at most about ten ticks.
 */
export function lifeAxis(ranges: [number, number][], now = new Date().getFullYear()): { lo: number; hi: number; ticks: number[] } {
  if (!ranges.length) return { lo: now - 100, hi: now, ticks: [now - 100, now - 50, now] };
  const min = Math.min(...ranges.map((r) => r[0])), max = Math.max(...ranges.map((r) => r[1]));
  const pad = Math.max(5, (max - min) * 0.04);
  const step = STEPS.find((s) => (max - min + 2 * pad) / s <= 10) ?? 1000;
  const lo = Math.floor((min - pad) / step) * step, hi = Math.ceil((max + pad) / step) * step;
  return { lo, hi, ticks: Array.from({ length: Math.round((hi - lo) / step) + 1 }, (_, i) => lo + i * step) };
}

/** Publication years to label, thinned so labels about 34px wide never collide. */
export function yearLabels(first: number, last: number, width: number): number[] {
  const count = last - first + 1;
  const every = Math.max(1, Math.ceil(34 / (width / count)));
  return Array.from({ length: count }, (_, i) => first + i).filter((y) => (y - first) % every === 0);
}
