"""Module B -- Match State Store, per CLAUDE.md section 4: three tiers
(frame-level in-memory, aggregate-level and event-level in DuckDB). Other
modules only ever call the `query_*` methods here; only the pipeline runner
calls `write_*`, so this module never depends on Layers 2-4 or Modules A/C
(no reverse dependency, per CLAUDE.md section 9).
"""
from __future__ import annotations

import io
import json

import duckdb
import pandas as pd
import polars as pl


class MatchStateStore:
    def __init__(self, db_path: str = "data/processed/match_state.duckdb"):
        self.con = duckdb.connect(db_path)
        self.frame_df: pl.DataFrame | None = None
        self._init_schema()

    def _init_schema(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id BIGINT, type VARCHAR, time_s DOUBLE, team VARCHAR, payload JSON
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS aggregates (
                name VARCHAR, payload JSON
            )
        """)

    # --- writers (pipeline-only) -------------------------------------------------
    def write_frame_data(self, player_time_df: pd.DataFrame) -> None:
        self.frame_df = pl.from_pandas(player_time_df)

    def write_aggregate(self, name: str, df: pd.DataFrame) -> None:
        self.con.execute("DELETE FROM aggregates WHERE name = ?", [name])
        self.con.execute(
            "INSERT INTO aggregates SELECT ?, ? ",
            [name, df.to_json(orient="records")],
        )

    def write_event(self, event_id: int, event: dict) -> None:
        payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in event.items()}
        self.con.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
            [event_id, event.get("type"), event.get("time_s"), event.get("team"), json.dumps(payload, default=str)],
        )

    def write_events(self, events: list[dict]) -> None:
        self.con.execute("DELETE FROM events")
        for i, e in enumerate(events):
            self.write_event(i, e)

    # --- readers (everyone else) -------------------------------------------------
    def query_frame_data(self) -> pl.DataFrame:
        if self.frame_df is None:
            raise RuntimeError("No frame data written yet")
        return self.frame_df

    def query_events(self, event_type: str | None = None) -> pd.DataFrame:
        if event_type:
            df = self.con.execute("SELECT * FROM events WHERE type = ? ORDER BY time_s", [event_type]).df()
        else:
            df = self.con.execute("SELECT * FROM events ORDER BY time_s").df()
        if not df.empty:
            df["payload"] = df["payload"].apply(json.loads)
        return df

    def query_aggregate(self, name: str) -> pd.DataFrame | None:
        row = self.con.execute("SELECT payload FROM aggregates WHERE name = ?", [name]).fetchone()
        if row is None:
            return None
        return pd.read_json(io.StringIO(row[0]), orient="records")

    def close(self):
        self.con.close()
