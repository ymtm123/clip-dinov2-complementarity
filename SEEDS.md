# Reference-image seeds

AnomalyDINO is 1-shot: one normal image per category is drawn at random as the memory bank. Seeds 0–9 are used. Seed 0 is the configuration reported in the main tables; seeds 0–9 are aggregated for the robustness numbers (§6.4).

Selection is deterministic given the dataset and seed (driver: `code/run_multiseed_categories.py`). Dataset images are not redistributed, so the selected reference per category is reproduced by running the driver against the public datasets rather than listed here. Per-category scores for all ten seeds are in `code/AnomalyDINO/results_*/`.
