"""
AnomalyDINO 1-shot image scores for DAGM (10 classes), seed-0, on the DAGM->MVTec shim.
Mirrors run_multiseed_categories.py; DAGM is an unknown dataset so get_dataset_info falls back to the
generic 'agnostic_no_mask' preset (same path BTAD used). Scores only (save_patch_dists=False).

Run with the vlm_ad python (build_dagm_layout.py must have run first):
  python run_dagm_dino.py
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import get_dataset_info
from src.detection import run_anomaly_detection
from src.backbones import get_model

DAGM_ROOT = "data/DAGM_2007/dagm_mvtec"
CATEGORIES = [f"Class{i}" for i in range(1, 11)]
MODEL_NAME = "dinov2_vitl14"
RESOLUTION = 448
PREPROCESS = "agnostic"
SHOT = 1
TAG = "dagm_all"


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    print(f"Loading {MODEL_NAME} @ {RESOLUTION} ...")
    model = get_model(MODEL_NAME, "cuda", smaller_edge_size=RESOLUTION)

    objects_all, object_anomalies, masking_default, rotation_default = get_dataset_info(
        "DAGM", PREPROCESS, data_path=DAGM_ROOT)
    objects = [o for o in CATEGORIES if o in objects_all]
    missing = set(CATEGORIES) - set(objects_all)
    if missing:
        print(f"  [WARN] not found: {missing}")

    results_dir = f"results_DAGM/{MODEL_NAME}_{RESOLUTION}/{SHOT}-shot_preprocess={PREPROCESS}_{TAG}"
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "measurements_seed=0.csv")
    if os.path.exists(out_csv):
        print(f"  [skip] {out_csv} exists"); return

    tmp = out_csv + ".partial"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Object", "Sample", "Anomaly_Score", "MemoryBank_Time", "Inference_Time"])
        for obj in objects:
            t0 = time.time()
            anomaly_scores, t_mem, t_inf = run_anomaly_detection(
                model, obj, data_root=DAGM_ROOT, n_ref_samples=SHOT,
                object_anomalies=object_anomalies, plots_dir=results_dir,
                save_examples=False, knn_metric="L2_normalized", knn_neighbors=1,
                faiss_on_cpu=True, masking=masking_default[obj],
                mask_ref_images=False, rotation=rotation_default[obj], seed=0,
                save_patch_dists=False, save_tiffs=False)
            for sample in anomaly_scores:
                w.writerow([obj, sample, f"{anomaly_scores[sample]:.5f}", f"{t_mem:.5f}", f"{t_inf[sample]:.5f}"])
            print(f"  {obj}: {len(anomaly_scores)} samples in {time.time()-t0:.1f}s")
    os.replace(tmp, out_csv)
    print(f"  Saved {out_csv}")


if __name__ == "__main__":
    main()
