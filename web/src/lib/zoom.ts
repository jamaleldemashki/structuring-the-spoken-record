// Zooming with the wheel, the same way for a mouse and a trackpad.
//
// A trackpad sends many small steps per gesture, a mouse a few large ones, and a Mac mouse turned
// slowly only very small ones. Browsers report these in different units too. So the step is
// converted to pixels first, and a step after a pause counts for at least one notch's worth of
// scrolling. Otherwise a slow turn of the wheel would hardly move the view.

const PER_PIXEL = 0.004;
/** A pinch on a trackpad reports much smaller numbers than scrolling. */
const PER_PIXEL_PINCH = 0.012;
const LONGEST_STEP = 240;
const SHORTEST_STEP_AFTER_PAUSE = 40;
const PAUSE_MS = 80;

export interface WheelStep {
  deltaY: number;
  /** 0 pixels, 1 lines, 2 pages, as in WheelEvent. */
  deltaMode: number;
  /** Browsers report a trackpad pinch as a wheel event with the Ctrl key held. */
  ctrlKey: boolean;
}

/** By how much one wheel event multiplies the zoom. */
export function wheelFactor(ev: WheelStep, sinceLastMs: number): number {
  let pixels = ev.deltaY * (ev.deltaMode === 1 ? 33 : ev.deltaMode === 2 ? 400 : 1);
  if (!pixels) return 1;
  if (ev.ctrlKey) return Math.exp(-Math.sign(pixels) * Math.min(Math.abs(pixels), LONGEST_STEP) * PER_PIXEL_PINCH);
  let size = Math.min(Math.abs(pixels), LONGEST_STEP);
  if (sinceLastMs >= PAUSE_MS) size = Math.max(size, SHORTEST_STEP_AFTER_PAUSE);
  return Math.exp(-Math.sign(pixels) * size * PER_PIXEL);
}

/**
 * Glides the zoom to its target over a few frames, around the point under the pointer. New wheel
 * events only move the target, so fast scrolling never queues up. `apply` draws one zoom level.
 */
export function zoomGlide(current: () => number, apply: (level: number, at: { x: number; y: number }) => void,
  limits: () => [number, number], instant: () => boolean) {
  let target = 0, at = { x: 0, y: 0 }, frame = 0, last = 0;
  const step = () => {
    const now = current();
    const gap = Math.log(target / now);
    const done = Math.abs(gap) < 0.003 || instant();
    apply(done ? target : now * Math.exp(gap * 0.35), at);
    frame = done ? 0 : requestAnimationFrame(step);
  };
  return {
    wheel(ev: WheelStep, point: { x: number; y: number }) {
      const time = performance.now();
      const [lo, hi] = limits();
      // A gesture that starts from rest starts from where the view is now.
      target = Math.min(hi, Math.max(lo, (frame ? target : current()) * wheelFactor(ev, time - last)));
      last = time;
      at = point;
      frame ||= requestAnimationFrame(step);
    },
    active: () => frame !== 0,
    stop() { cancelAnimationFrame(frame); frame = 0; },
  };
}
