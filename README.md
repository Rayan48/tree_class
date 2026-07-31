# Tree Classifier — OOP in Action 🌳

A tree-image classification app built to double as a **live teaching tool
for OOP concepts**, aimed at ML beginners. Every concept below exists for a
real reason in the pipeline — not as a contrived example.

## The idea

The project already had 5 different ways to turn a tree image into features
for the model: edge detection, HSL shadow, PCA, raw pixels, and T-connected
pixel structure. That's a natural **Strategy pattern** — one interface, many
interchangeable implementations — so the app is built around it.

## OOP concepts, mapped to code

| Concept | Where |
|---|---|
| Interface / Abstract Class | `extractors/base.py` — `BaseExtractor` (ABC) |
| Inheritance | `extractors/edge.py`, `shadow_hsl.py`, `pca.py`, `raw.py`, `t_connected.py` all extend `BaseExtractor` |
| Polymorphism | `pipeline.py` calls `extractor.run(image)` without knowing the concrete class — the frontend dropdown picks the class at runtime |
| Exception Handling | `exceptions.py` — custom `InvalidImageError`, `FeatureExtractionError`, `ModelLoadError`, all caught cleanly in `app.py` |
| Multithreading + Collections | `pipeline.py` — `ClassificationPipeline.run_all()` uses `ThreadPoolExecutor` and an `OrderedDict` to run all 5 extractors concurrently |

## Project structure

```
tree_class/
├── extractors/
│   ├── base.py          # Abstract base class
│   ├── edge.py           # Edge detection extractor
│   ├── shadow_hsl.py     # HSL shadow extractor
│   ├── pca.py             # PCA extractor
│   ├── raw.py             # Raw image extractor
│   └── t_connected.py    # T-connected pixel extractor
├── exceptions.py          # Custom exception hierarchy
├── model_wrapper.py        # TreeClassifier — wraps plant_model.keras
├── pipeline.py             # ClassificationPipeline — orchestration + threading
├── app.py                  # Streamlit frontend
├── requirements.txt
└── data/                   # CSVs go here (edge_detection_features.csv, etc.)
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `plant_model.keras` in the project root, and your CSVs in `data/`.

## Known TODO before this matches your original training pipeline

The `shadow_hsl.py`, `pca.py`, and `t_connected.py` extractors are
**placeholder implementations** — replace their `extract()` bodies with the
exact logic that generated `hsl_shadow_features.csv`,
`pca_image_features_with_labels.csv`, and `t_connected_image_features.csv`
so predictions stay consistent with what the model was trained on.

Also verify: each extractor currently outputs a different feature vector
shape. Confirm which shape `plant_model.keras` actually expects before
wiring up `run_all()` for real predictions — otherwise you'll get silent
wrong results for 4 out of 5 methods.
