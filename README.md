# CLIP–DINOv2 complementarity for industrial anomaly detection

Code and per-category results for the paper *CLIP–DINOv2 complementarity for industrial anomaly detection: a systematic study of the understanding–optimization gap* (Machine Vision and Applications).

This is the working research tree, published as-is: analysis scripts and their JSON outputs live side by side in `code/`, the way they actually ran.

## Layout

- `code/` — all analysis scripts, plus the JSON outputs of the submission-era analyses (each script reads/writes next to itself).
- `code/AnomalyDINO/results_*/` — AnomalyDINO per-category image scores (CSV): MVTec AD / VisA / BTAD for reference seeds 0–9; MVTec-LOCO and DAGM for seed 0.
- `code/AnomalyCLIP/results/scores_*/`, `.../loco_split/` — AnomalyCLIP per-category scores (CSV) for MVTec AD, VisA, BTAD, DAGM, and the LOCO structural/logical split.
- `results/` — outputs of the revision-round analyses: pixel-level multi-seed test, cluster-robust bootstraps, bootstrap stability table, seed-averaged metrics, power analysis, streaming normalization, reference-strategy comparison, the 6-backbone panel (`backbone_panel_full.json`, `backbone_panel_fusion_matrix.json`, per-image score dumps in `backbone_panel_scores/`), the full 30-category EfficientAD benchmark at the 15-epoch and 70k-step budgets (aggregates plus per-image scores in `efficientad_scores_full/`), and the VLM logical-anomaly baseline (`vlm_loco_v3_apriori/`).
- `prompts/` — AnomalyCLIP learned object-agnostic prompts (final epoch).
- `unsupervised_strategies_table.md` — definitions of all 102 unsupervised fusion-weight strategies.
- `SEEDS.md`, `CHECKPOINTS.md`.

## Data

The benchmarks are public and not redistributed here: MVTec AD, VisA, BTAD, MVTec LOCO, DAGM 2007. The CSV/JSON files in this repository are derived per-image *scores and metrics*, released for reproducibility; the underlying images remain subject to each dataset's own license (MVTec AD and MVTec LOCO are CC BY-NC-SA 4.0 — non-commercial). The MIT license covers the code.

## Reproducing

The analysis layer runs from the shipped score CSVs and JSONs with numpy / scipy / scikit-learn / pandas (Python 3.10):

    cd code
    python score_fusion.py
    python hierarchical_stats_seedavg.py
    python multiseed_aggregate.py
    python cluster_bootstrap_fusion_N45.py

Scripts write their outputs next to themselves (`code/`) or into `../results/`, matching the shipped files.

Not shipped (too large): the pixel-level anomaly maps. `pixel_fusion.py`, `pixel_seed0_metrics.py` and `pixel_multiseed_analysis.py` need them; regenerate with `run_multiseed_maps.py` against the public datasets. Score regeneration from images also needs the public AnomalyDINO and AnomalyCLIP repositories and the backbones in `CHECKPOINTS.md` (GPU); the run drivers are in `code/` (`run_multiseed_categories.py`, `run_dagm_dino.py`, `build_dagm_layout.py`). The EfficientAD benchmark (`efficientad_benchmark_full.py`) trains per-category student–teacher models from scratch (GPU, long); its aggregates and per-image scores are shipped. The reference-strategy comparison (`refstrategy_eval.py`) additionally needs the medoid/farthest score variants, regenerable with the same drivers.

The VLM logical-anomaly baseline: `vlm_loco_pilot.py` and `vlm_loco_v2.py` are the initial generic-question runs; `vlm_loco_v3.py` is the final count/composition question set (designed from the LOCO defect specification before the run, executed once, reported as-is in the paper). All three are kept for transparency; `results/vlm_loco_v3_apriori/` holds the reported run.

## Citation

To be added on publication.
