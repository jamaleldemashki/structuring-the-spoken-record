// Every view has an address, so the browser's Back button works and a view can be bookmarked or
// sent to someone. The address sits behind the #, so a static host needs no rules for it.
//
//   #/                          the home page
//   #/search/love               the results for a text
//   #/map                       the whole map
//   #/map/<id>                  an exploration that started from this topic
//   #/connection/<id>/<id>      the chain between two topics
//   #/episodes/<id>/<passage>   an episode, optionally at a passage
//   #/timeline                  the timeline
//
// ?open=<id> adds the topic or episode shown in the details panel, on any of them. On an exploration
// ?with=<id> names the topic explored together with the first one.

export type Route = { open: string | null } & (
  | { view: "start" }
  | { view: "results"; text: string }
  | { view: "explore"; root: string | null; with?: string | null }
  | { view: "connection"; from: string; to: string }
  | { view: "episodes"; episode: string | null; passage: string | null }
  | { view: "timeline" }
);

const part = encodeURIComponent;

export function writeRoute(r: Route): string {
  const path = r.view === "results" ? `search/${part(r.text)}`
    : r.view === "explore" ? (r.root ? `map/${part(r.root)}` : "map")
    : r.view === "connection" ? `connection/${part(r.from)}/${part(r.to)}`
    : r.view === "episodes" ? ["episodes", r.episode, r.episode && r.passage].filter(Boolean).map((s, i) => (i ? part(s as string) : s)).join("/")
    : r.view === "timeline" ? "timeline"
    : "";
  const query = [r.view === "explore" && r.with ? `with=${part(r.with)}` : "", r.open ? `open=${part(r.open)}` : ""].filter(Boolean).join("&");
  return `#/${path}${query ? `?${query}` : ""}`;
}

/** An address that makes no sense gives the home page, not an error. */
export function readRoute(hash: string): Route {
  const [path, query] = hash.replace(/^#\/?/, "").split("?");
  const params = new URLSearchParams(query ?? ""), open = params.get("open"), together = params.get("with");
  const [head, ...rest] = path.split("/").map((s) => { try { return decodeURIComponent(s); } catch { return s; } });
  if (head === "search" && rest[0]) return { view: "results", text: rest[0], open };
  if (head === "map") return { view: "explore", root: rest[0] || null, ...(together ? { with: together } : {}), open };
  if (head === "connection" && rest[0] && rest[1]) return { view: "connection", from: rest[0], to: rest[1], open };
  if (head === "episodes") return { view: "episodes", episode: rest[0] || null, passage: rest[1] || null, open };
  if (head === "timeline") return { view: "timeline", open };
  return { view: "start", open: null };
}

/** A new page goes into the browser's history. Selecting something on the same page does not. */
export const samePage = (a: string, b: string) => a.split("?")[0] === b.split("?")[0];
