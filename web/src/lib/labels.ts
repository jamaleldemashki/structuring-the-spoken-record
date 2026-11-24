// Which labels to write. Text stays the same size on screen at every zoom, so only some of it
// fits. The most important names are written first, and a label that would touch one already
// placed is left out. Zooming in makes room, so more names appear with every step.

export interface LabelCandidate {
  id: string;
  /** Position and radius of the node on screen, in pixels. */
  x: number;
  y: number;
  r: number;
  text: string;
  /** Font size in pixels. */
  font: number;
  /** Higher goes first. */
  priority: number;
}

const WIDEST = 150;
/** Average width of a character in this typeface, as a share of the font size. */
const CHARACTER = 0.56;
const GAP = 3;

/** The ids to label, for a screen of this size. Nodes off screen get none. */
export function chooseLabels(list: LabelCandidate[], width: number, height: number): Set<string> {
  const chosen = new Set<string>();
  const cell = 60;
  const grid = new Map<string, [number, number, number, number][]>();
  const cells = (x0: number, y0: number, x1: number, y1: number) => {
    const out: string[] = [];
    for (let i = Math.floor(x0 / cell); i <= Math.floor(x1 / cell); i++)
      for (let j = Math.floor(y0 / cell); j <= Math.floor(y1 / cell); j++) out.push(`${i},${j}`);
    return out;
  };
  for (const c of [...list].sort((a, b) => b.priority - a.priority || a.id.localeCompare(b.id))) {
    if (c.x < 0 || c.x > width || c.y < 0 || c.y > height) continue;
    const w = Math.min(WIDEST, c.text.length * c.font * CHARACTER) + 2 * GAP, h = c.font + 2 * GAP;
    const x0 = c.x - w / 2, x1 = c.x + w / 2, y0 = c.y + c.r, y1 = y0 + h;
    const near = cells(x0, y0, x1, y1);
    if (near.some((key) => grid.get(key)?.some(([a, b, d, e]) => x0 < d && x1 > a && y0 < e && y1 > b))) continue;
    chosen.add(c.id);
    for (const key of near) grid.set(key, [...(grid.get(key) ?? []), [x0, y0, x1, y1]]);
  }
  return chosen;
}
