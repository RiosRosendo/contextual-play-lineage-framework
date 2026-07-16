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

Every layer and module runs end-to-end on a locally generated synthetic test clip, and Layer 1 has since been validated against real broadcast footage and is being incrementally deepened (starting with detection quality, then tracking).

## Data & licensing

- The synthetic test clip is generated locally (`src/perception/synthetic_clip.py`) and carries no external licensing constraints.
- Real-footage validation used two openly-licensed research datasets, neither of which is redistributed in this repo:
  - [SoccerSum](https://zenodo.org/records/10612084) (Simula) -- **CC BY-NC-ND 4.0** (non-commercial, no derivatives). Used only to evaluate detection quality against its ground truth; not used to validate tracking (its "sequences" turned out to have large real-time gaps between frames, not continuous motion).
  - [SoccerTrack v2](https://github.com/AtomScott/SoccerTrack-v2) -- CC BY 4.0.
- `weights/soccersum_yolov8n_ball.pt` (a YOLOv8n ball detector fine-tuned on SoccerSum) is **not committed to this repo**. A model fine-tuned on CC BY-NC-ND data is arguably a derivative work, and SoccerSum's license prohibits distributing derivatives -- so that weight file stays local-only. `src/perception/yolo_detector.py` falls back cleanly to plain pretrained COCO weights when it's absent, which is what a fresh clone of this repo will always use unless you regenerate the fine-tuned weights yourself (`python -m notebooks.finetune_ball_detector prepare` then `train`, against your own SoccerSum download).

## Running the pipeline

```bash
python -m venv venv
venv\Scripts\activate   # venv/bin/activate on Linux/Mac
pip install -r requirements.txt

python -m src.perception.synthetic_clip   # generates a short synthetic test clip
python -m src.run_pipeline                # runs the full pipeline on it
```

`src/run_pipeline.py` runs perception through prediction, writes results into the match state store, builds the play-lineage graph, and prints any review alerts it finds along with a grounded explanation.
