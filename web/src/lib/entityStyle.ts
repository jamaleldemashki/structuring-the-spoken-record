// Colour and wording per entity type, taken from the archive's site.json.
// Styles go by position in the configured order, not by type name, so a podcast with
// other types works the same way.
//
// Every topic is round, as in Obsidian, so the layout keeps nodes apart exactly. On a graph any
// two colours can stand side by side, and only three hues stay apart that way for every kind of
// colour vision, in both themes (checked with a palette validator; a fourth hue failed in every
// candidate). So the first three types get the hues, the fourth a dark neutral, and the fifth on
// are drawn as a hollow ring. Those share the ring, the legend and the details name them.
import type { Site } from "./types.ts";

/** Topics are round. Episodes are markers on the time axis and keep a shape of their own. */
export type Shape = "ellipse" | "tag";

/** Episodes are drawn in the graph too. They are not an entity type of the archive. */
export const EPISODE_TYPE = "__episode";

export interface TypeStyle {
  type: string;
  label: string;
  plural: string;
  visibleByDefault: boolean;
  /** CSS custom property name, without the leading dashes. */
  token: string;
  shape: Shape;
  /** Drawn as a ring instead of a filled circle. */
  ring: boolean;
}

const TOKENS = ["hue-1", "hue-2", "hue-3", "neutral-1"];
const RING = "neutral-2";

let styles = new Map<string, TypeStyle>();

export function configureTypes(site: Pick<Site, "entity_types">) {
  styles = new Map(site.entity_types.map((t, i) => [t.type, {
    type: t.type, label: t.label, plural: t.plural, visibleByDefault: t.visible_by_default,
    token: TOKENS[i] ?? RING, shape: "ellipse", ring: i >= TOKENS.length,
  }]));
}

/** Types in display order. */
export const allTypes = (): TypeStyle[] => [...styles.values()];

export function typeStyle(type: string): TypeStyle {
  if (type === EPISODE_TYPE) return { type, label: "Episode", plural: "episodes", visibleByDefault: true, token: "episode", shape: "tag", ring: false };
  return styles.get(type) ?? { type, label: type, plural: type, visibleByDefault: false, token: RING, shape: "ellipse", ring: true };
}

export const typeVar = (type: string) => `var(--${typeStyle(type).token})`;

export function cssToken(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
}
