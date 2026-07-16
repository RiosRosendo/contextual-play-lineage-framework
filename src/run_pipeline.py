"""Full end-to-end skeleton pipeline: Layer 1 -> Layer 2 -> Layer 3 -> Layer 4,
writing through Module B (state store) and finishing with Module A (lineage
graph + review alerts) and Module C (grounded explanation). This is the
single script that validates the whole architecture runs without errors on
one short clip, per the project spec section 3.
"""
from __future__ import annotations

from src.assistant.explain import explain_review_alert
from src.events.pipeline import run_events
from src.lineage.graph import build_lineage_graph, find_review_alerts
from src.metrics.pipeline import run_metrics
from src.perception.pipeline import run_perception
from src.predictive.pipeline import run_predictive
from src.state_store.store import MatchStateStore


def run_pipeline(video_path: str = "data/raw/synthetic_match_clip.mp4", backend: str = "color",
                  external_goal_events: list[dict] | None = None) -> dict:
    perception_df = run_perception(video_path, backend=backend)
    metrics = run_metrics(perception_df)
    events = run_events(metrics, external_goal_events=external_goal_events)
    predictive = run_predictive(metrics, events)
    enriched_events = predictive["events"]

    store = MatchStateStore()
    store.write_frame_data(metrics["player_time_df"])
    store.write_aggregate("track_summary", metrics["track_summary_df"])
    store.write_aggregate("possession", metrics["possession_df"])
    store.write_aggregate("formation", metrics["formation_df"])
    store.write_events(enriched_events)

    graph = build_lineage_graph(enriched_events)
    alerts = find_review_alerts(graph)
    store.close()

    return {
        "perception_df": perception_df,
        "metrics": metrics,
        "events": enriched_events,
        "predictive": predictive,
        "graph": graph,
        "review_alerts": alerts,
    }


if __name__ == "__main__":
    result = run_pipeline()
    n_rows = len(result["perception_df"])
    n_tracks = result["perception_df"]["track_id"].nunique()
    n_events = len(result["events"])
    n_alerts = len(result["review_alerts"])

    print(f"Layer 1: {n_rows} frame-rows, {n_tracks} tracks")
    print(f"Layer 2: possession % = {result['metrics']['possession_pct']}")
    print(f"Layer 3: {n_events} events detected "
          f"({sum(1 for e in result['events'] if e['type'] == 'foul')} foul candidates)")
    print(f"Layer 4: pitch control grid shape = {result['predictive']['pitch_control_at_goal'].shape}")
    print(f"Module A: {n_alerts} review alert(s) raised")

    for alert in result["review_alerts"]:
        print("\n" + "=" * 70)
        print(explain_review_alert(alert))
