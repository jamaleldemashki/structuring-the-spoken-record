"""Vault notes for suggested passage connections, kept apart from the relations."""

import json
from collections import defaultdict


def add_passage_discovery(config, files, episodes, passages, entities, episode_paths, entity_paths, discovery):
    """Show the source text, both rankings and a small graph, with no made-up topic labels.

    Only MPNet links go into the passage graph. The overlap baseline stays in the comparison
    note, which is excluded from the graph. Obsidian edges have no type, and incoming
    suggestions show up too.
    """
    from spoken_record.vault import (episode_link, entity_link, frontmatter, markdown_text,
                                     passage_time, safe_filename)
    episode_by_id = {e.episode_id: e for e in episodes}
    passage_by_id = {p.passage_id: p for p in passages}
    entity_by_id = {e.corpus_entity_id: e for e in entities}
    input_by_id = {row["passage_id"]: row for row in discovery["inputs"]}
    # Graph nodes get readable names, the passage IDs stay in the frontmatter. Episode paths
    # are already unique and passage numbers count within the episode.
    paths = {p.passage_id: "40_Discussions/" + safe_filename(
        f"{episode_paths[p.episode_id].split('/')[-1]} — Passage {p.index+1}", 160) for p in passages}
    if len({name.casefold() for name in paths.values()}) != len(paths):
        raise ValueError("discovery passage filenames are not unique")
    by_source = defaultdict(list)
    for row in discovery["connections"]:
        by_source[row["source_passage_id"]].append(row)

    def link(pid):
        """Readable episode/passage label that links to the stable note path."""
        p = passage_by_id[pid]
        return f"[[{paths[pid]}|{safe_filename(episode_paths[p.episode_id].split('/')[-1])} · Passage {p.index+1}]]"

    def excerpt(pid):
        """Opening of the retrieval input, copied exactly (no generated summary)."""
        text = input_by_id[pid]["input_text"].strip()
        return markdown_text(text[:320] + ("…" if len(text) > 320 else ""))

    review_path = f"{config.review_folder}/Passage discovery comparison"
    review = [frontmatter("review", {"discovery_run_id": discovery["run_id"]}),
        "# Passage discovery: two retrieval methods", "",
        "Each method ranks cross-episode passages separately. Cosine and Jaccard scores are "
        "similarities, not probabilities of truth or relevance. Counts do not establish retrieval quality.", ""]
    review.extend(f"- {key.replace('_', ' ')}: {value}" for key, value in discovery["counts"].items())
    index = [frontmatter("index", {}), "# Explore related discussions", "",
        "Open a passage below to read its source and follow suggested discussions in other episodes. "
        "Suggestions may connect contrasting viewpoints; they do not imply agreement or influence.", "",
        "Repeated introductions and broad subject overlap can receive high similarity scores. "
        "Inspect the passages before treating a suggestion as a useful thematic connection.", "",
        "For the passage-only graph, set the global Graph filter to `path:\"40_Discussions\"`. "
        "Links in that view are selected MPNet suggestions. Rankings are directional, while the "
        "graph also shows incoming links; node degree can exceed the per-source suggestion limit.", "",
        f"[[{review_path}|Compare MPNet with the shared-entity baseline]]", ""]
    for p in sorted(passages, key=lambda row: (episode_by_id[row.episode_id].title.casefold(), row.index, row.passage_id)):
        rows = sorted(by_source[p.passage_id], key=lambda r: (r["method"], r["rank"]))
        mpnet = [row for row in rows if row["method"] == "mpnet_cosine"]
        baseline = [row for row in rows if row["method"] == "shared_entity_jaccard"]
        index.append(f"- {link(p.passage_id)} — {len(mpnet)} MPNet suggestion(s), {passage_time(p)}")
        lines = [frontmatter("discussion_passage", {"passage_id": p.passage_id,
            "episode_id": p.episode_id, "discovery_run_id": discovery["run_id"],
            "suggestion_count": len(mpnet)}), f"# Passage {p.index+1} · {markdown_text(episode_by_id[p.episode_id].title)}", "",
            f"{episode_link(episode_by_id[p.episode_id], episode_paths, p, naming=config)}", "",
            "Times are approximate source-segment timings when available.", "",
            "## Source passage", "", markdown_text(p.text), "", "## Suggested related discussions", "",
            "Ranked by MPNet cosine similarity. These are retrieval suggestions, not factual relations "
            "or proof of a shared position.", ""]
        if input_by_id[p.passage_id]["excluded_spans"]:
            lines.append("Reviewed credit spans were omitted from retrieval; the source passage above remains unchanged.")
        if not mpnet:
            lines.append("No cross-episode suggestion met the configured minimum similarity.")
        for row in mpnet:
            target = row["target_passage_id"]
            lines.extend([f"### {row['rank']}. {link(target)}", "",
                f"Cosine similarity: {row['score']:.3f}. "
                f"{episode_link(episode_by_id[passage_by_id[target].episode_id], episode_paths, passage_by_id[target], naming=config)}", "",
                excerpt(target), ""])
            shared = [entity_link(entity_by_id[cid], entity_paths) for cid in row["shared_entity_ids"]]
            lines.append("Observed shared entities: " + (", ".join(shared) if shared else "none; this is an embedding-only suggestion.") + "\n")
        lines.extend([f"[[{review_path}#^compare-{p.passage_id}|Compare both methods for this passage]]", "",
                      f"[[{config.index_folder}/Explore related discussions|All discussion passages]]", ""])
        files[f"{paths[p.passage_id]}.md"] = "\n".join(lines).encode()
        # Link back from the episode note. Its text and anchor stay the same.
        files[f"{episode_paths[p.episode_id]}.md"] += (
            f"\nRelated-discussion navigation: {link(p.passage_id)}\n").encode()
        review.extend(["", f"## {link(p.passage_id)}", "", f"^compare-{p.passage_id}", "", excerpt(p.passage_id), ""])
        for method, candidates in (("MPNet cosine", mpnet), ("Shared-entity Jaccard", baseline)):
            review.extend([f"### {method}", ""])
            if not candidates:
                review.append("No suggestion meets this method's threshold.")
            for row in candidates:
                review.extend([f"- {row['rank']}. {link(row['target_passage_id'])} — {row['score']:.3f}",
                               f"  - {excerpt(row['target_passage_id'])}"])
                shared = [entity_link(entity_by_id[cid], entity_paths) for cid in row["shared_entity_ids"]]
                review.append("  - Observed shared entities: " + (", ".join(shared) if shared else "none"))
    files[f"{review_path}.md"] = "\n".join(review).encode()
    files[f"{config.index_folder}/Explore related discussions.md"] = "\n".join(index).encode()
    for path in ("START HERE.md", f"{config.index_folder}/Home.md"):
        files[path] += (f"\n## Related discussions\n\n"
            f"[[{config.index_folder}/Explore related discussions|Explore passages across episodes]]\n\n"
            "Passage similarity is separate from named co-mentions and provisional semantic relations.\n").encode()
    graph = json.loads(files[".obsidian/graph.json"])
    graph["colorGroups"].insert(0, {"query": 'path:"40_Discussions"', "color": {"a": 1, "rgb": 4897144}})
    files[".obsidian/graph.json"] = (json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
