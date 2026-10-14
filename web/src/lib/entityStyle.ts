// Colour, shape and wording per entity type, taken from the archive's site.json.
// Styles go by position in the configured order, not by type name, so a podcast with
// other types works the same way. The first three types get the three hues, the rest get
// neutrals, and each type has its own node shape so colour is not the only cue.
import type { Site } from "./types.ts";

export type Shape = "ellipse" | "round-rectangle" | "rectangle" | "diamond" | "triangle" | "round-hexagon" | "hexagon" | "vee";

export interface TypeStyle {
  type: string;
  label: string;
  plural: string;
  visibleByDefault: boolean;
  /** CSS custom property name, without the leading dashes. */
  token: string;
  shape: Shape;
}

const TOKENS = ["hue-1", "hue-2", "hue-3", "neutral-1", "neutral-2", "neutral-3"];
const SHAPES: Shape[] = ["ellipse", "round-rectangle", "rectangle", "diamond", "triangle", "round-hexagon", "hexagon", "vee"];

let styles = new Map<string, TypeStyle>();

export function configureTypes(site: Pick<Site, "entity_types">) {
  styles = new Map(site.entity_types.map((t, i) => [t.type, {
    type: t.type, label: t.label, plural: t.plural, visibleByDefault: t.visible_by_default,
    token: TOKENS[Math.min(i, TOKENS.length - 1)], shape: SHAPES[i % SHAPES.length],
  }]));
}

/** Types in display order. */
export const allTypes = (): TypeStyle[] => [...styles.values()];

export function typeStyle(type: string): TypeStyle {
  return styles.get(type) ?? { type, label: type, plural: type, visibleByDefault: false, token: "neutral-3", shape: "round-hexagon" };
}

export const typeVar = (type: string) => `var(--${typeStyle(type).token})`;

export function cssToken(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
}
