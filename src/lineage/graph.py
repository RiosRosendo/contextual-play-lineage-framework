"""Module A -- Play lineage graph, per the project spec section 4: events connected
by possession continuity (NetworkX). When a goal occurs, traverse the graph
backward; if a "probable unflagged foul" node is found within the same
uninterrupted possession sequence, raise a review alert.

Skeleton-phase note: the foul detector's classifier head is untrained (see
src/events/foul_detector/model.py), so this module treats every detected
foul-type event as a "probable foul" node regardless of its (currently not
meaningful) probability score -- the physical contact heuristic that
produced the event is itself the section-3 "simplified formula" for this
pass. `foul_probability` is still carried on the node for when the
classifier is trained and a real confidence gate becomes meaningful.
"""
from __future__ import annotations

import networkx as nx


def build_lineage_graph(events: list[dict]) -> nx.DiGraph:
    g = nx.DiGraph()
    prev_node = None
    chain_id = 0

    for i, e in enumerate(events):
        g.add_node(i, chain_id=chain_id, **e)
        if prev_node is not None:
            g.add_edge(prev_node, i)
        prev_node = i
        if e["type"] == "turnover":
            chain_id += 1  # events after this belong to the new possessing team's chain

    return g


def find_review_alerts(g: nx.DiGraph) -> list[dict]:
    """For every goal node, walks predecessors backward within the same
    possession chain looking for unflagged foul nodes."""
    alerts = []
    for node, data in g.nodes(data=True):
        if data["type"] != "goal":
            continue
        chain_id = data["chain_id"]
        foul_nodes = []
        visited = set()
        stack = list(g.predecessors(node))
        while stack:
            p = stack.pop()
            if p in visited:
                continue
            visited.add(p)
            pdata = g.nodes[p]
            if pdata["chain_id"] != chain_id:
                continue  # crossed a possession-change boundary, stop this branch
            if pdata["type"] == "foul" and not pdata.get("is_flagged", False):
                foul_nodes.append(p)
            stack.extend(g.predecessors(p))

        if foul_nodes:
            alerts.append({
                "goal_node": node,
                "goal_event": data,
                "foul_nodes": foul_nodes,
                "foul_events": [g.nodes[n] for n in foul_nodes],
            })
    return alerts
