# contextual-play-lineage-framework
Research framework for spatiotemporal player/ball tracking and contextual foul analysis in football, using computer vision and graph-based play-lineage modeling to trace unflagged fouls through to their downstream consequences (e.g. goals). Not affiliated with FIFA/IFAB.

## Architecture

The pipeline is organized into four sequential layers plus three cross-cutting modules, designed around a single broadcast camera angle (additional angles are treated as optional, never required):

- **Layer 1 -- Perception:** player/ball/referee detection, team identification, tracking, and pitch calibration. Outputs real-world (x, y) positions per frame.
- **Layer 2 -- Physical and spatial metrics:** speed, acceleration, distance covered, heatmaps, possession, formation -- derived mathematically from Layer 1.
- **Layer 3 -- Discrete events:** shots, passes, turnovers, and a foul detector that fuses a video-encoder branch with a trajectory sequence-encoder branch to reason about the play context leading up to contact.
- **Layer 4 -- Predictive models:** pass probability, expected goals/threat, and pitch control.
- **Module A -- Play lineage graph:** links events by possession continuity and traces goals backward for unflagged fouls in the same possession sequence.
- **Module B -- Match State Store:** three-tier storage (frame-level, aggregate-level, event-level) that every other module reads from.
- **Module C -- Explanation assistant:** retrieval-augmented generation that grounds foul/goal review explanations in the IFAB Laws of the Game.

## Stack

Python 3.11, OpenCV, Ultralytics YOLOv8, PyTorch, scikit-learn, pandas/Polars, DuckDB, NetworkX, FAISS.

## Status

Currently at the "full skeleton" stage: every layer and module runs end-to-end on a locally generated synthetic test clip, using simplified/heuristic models rather than trained ones. See `PROGRESS.md` and `TODO.md` (kept locally, not part of this repo) for the detailed development log.

## Running the pipeline

```bash
python -m venv venv
venv\Scripts\activate   # venv/bin/activate on Linux/Mac
pip install -r requirements.txt

python -m src.perception.synthetic_clip   # generates a short synthetic test clip
python -m src.run_pipeline                # runs the full pipeline on it
```

`src/run_pipeline.py` runs perception through prediction, writes results into the match state store, builds the play-lineage graph, and prints any review alerts it finds along with a grounded explanation.
