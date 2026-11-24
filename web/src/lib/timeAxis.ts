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

/**
 * The lifespans left after setting aside those far outside the rest, by Tukey's outer fences:
 * more than three interquartile ranges beyond the quartiles. One legendary figure thousands of
 * years before everyone else would otherwise squeeze all the others into dots.
 */
export function withoutOutliers(ranges: [number, number][]): [number, number][] {
  if (ranges.length < 8) return ranges;
  const quartile = (sorted: number[], q: number) => sorted[Math.floor((sorted.length - 1) * q)];
  const fence = (values: number[]) => {
    const sorted = [...values].sort((a, b) => a - b);
    const q1 = quartile(sorted, 0.25), q3 = quartile(sorted, 0.75), spread = 3 * (q3 - q1);
    return [q1 - spread, q3 + spread];
  };
  const [lo] = fence(ranges.map((r) => r[0])), [, hi] = fence(ranges.map((r) => r[1]));
  return ranges.filter((r) => r[0] >= lo && r[1] <= hi);
}

const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** Months counted from year 0, so "2015-07" and the next month differ by one. */
export const monthIndex = (month: string) => Number(month.slice(0, 4)) * 12 + Number(month.slice(5, 7)) - 1;

export const monthKey = (index: number) => `${Math.floor(index / 12)}-${String((index % 12) + 1).padStart(2, "0")}`;

export const formatMonth = (month: string) => `${MONTH_NAMES[Number(month.slice(5, 7)) - 1]} ${month.slice(0, 4)}`;

/** Every month from the first to the last, both included. */
export function monthsBetween(first: string, last: string): string[] {
  const a = monthIndex(first);
  return Array.from({ length: monthIndex(last) - a + 1 }, (_, i) => monthKey(a + i));
}

const MONTH_STEPS = [1, 2, 3, 6, 12, 24, 60, 120];

/**
 * Months to label on the publication axis, so labels never collide. A short podcast run
 * gets month labels ("Jul 2015"), a long one only year labels on each January.
 */
export function monthLabels(first: string, last: string, width: number): { month: string; text: string }[] {
  const months = monthsBetween(first, last);
  const perMonth = width / months.length;
  const step = MONTH_STEPS.find((s) => s * perMonth >= (s < 12 ? 62 : 36)) ?? 120;
  if (step < 12) {
    return months.filter((_, i) => i % step === 0).map((month) => ({ month, text: formatMonth(month) }));
  }
  const years = step / 12;
  return months.filter((m) => m.endsWith("-01") && Number(m.slice(0, 4)) % years === 0).map((month) => ({ month, text: month.slice(0, 4) }));
}
