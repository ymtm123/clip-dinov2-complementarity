"""
T3 §4a: Pixel-level multi-seed AnomalyDINO maps (seeds 1..9) for all 30 categories.
==================================================================================

run_multiseed_categories.py generated the seed 1..9 ANOMALY SCORES (measurements_seed=K.csv)
with save_patch_dists=False, so only seed-0 has pixel anomaly maps on disk. This driver
re-runs the EXACT same configs with save_patch_dists=True to write the per-seed pixel maps
to anomaly_maps/seed={seed}/{obj}/test/{atype}/{id}.npy (the path pixel_fusion.py reads), so
the image-vs-pixel reversal-rate CI and the DEFINITIVE Contribution-2/D verdict can be computed
on seed-averaged PIXEL oracle alpha (currently seed-0 only).

Key differences from run_multiseed_categories.py:
  * save_patch_dists=True (the whole point).
  * Resumability gates on MAP-dir completeness, NOT measurements_seed=K.csv existence — those CSVs
    already exist, so a CSV-existence skip (the original at line 67) would skip every seed and
    generate NO maps. We do NOT rewrite the CSVs (scores already validated, deterministic).
  * seed-0 maps already exist and are never touched (seeds default to 1..9).

Run from this directory with the vlm_ad python:
  python run_multiseed_maps.py --seeds 1 2 3 4 5 6 7 8 9
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import get_dataset_info
from src.detection import run_anomaly_detection
from src.backbones import get_model

D9 = "data"

CONFIGS = [
    dict(dataset="MVTec", data_root=f"{D9}/MVTec_AD", tag="weak_cats",
         categories=["capsule", "screw", "transistor"]),
    dict(dataset="MVTec", data_root=f"{D9}/MVTec_AD", tag="strong_cats",
         categories=["bottle", "cable", "carpet", "grid", "hazelnut", "leather",
                     "metal_nut", "pill", "tile", "toothbrush", "wood", "zipper"]),
    dict(dataset="VisA", data_root=f"{D9}/VisA/VisA_pytorch/1cls", tag="all",
         categories=["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
                     "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]),
    dict(dataset="BTAD", data_root=f"{D9}/BTAD/BTech_Dataset_transformed", tag="btad_all",
         categories=["01", "02", "03"]),
]

MODEL_NAME = "dinov2_vitl14"
RESOLUTION = 448
PREPROCESS = "agnostic"
SHOT = 1


def count_test_images(data_root, obj):
    test_dir = os.path.join(data_root, obj, "test")
    n = 0
    if os.path.isdir(test_dir):
        for atype in os.listdir(test_dir):
            ad = os.path.join(test_dir, atype)
            if os.path.isdir(ad):
                n += sum(1 for f in os.listdir(ad)
                         if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    return n


def count_maps(map_root):
    n = 0
    for dirpath, _, files in os.walk(map_root):
        n += sum(1 for f in files if f.endswith(".npy"))
    return n


def run_config(model, cfg, seeds, faiss_on_cpu=True):
    objects_all, object_anomalies, masking_default, rotation_default = get_dataset_info(
        cfg["dataset"], PREPROCESS, data_path=cfg["data_root"])
    objects = [o for o in cfg["categories"] if o in objects_all]
    missing = set(cfg["categories"]) - set(objects_all)
    if missing:
        print(f"  [WARN] categories not found for {cfg['dataset']}/{cfg['tag']}: {missing}")

    results_dir = f"results_{cfg['dataset']}/{MODEL_NAME}_{RESOLUTION}/{SHOT}-shot_preprocess={PREPROCESS}_{cfg['tag']}"
    os.makedirs(results_dir, exist_ok=True)

    for seed in seeds:
        for obj in objects:
            map_root = os.path.join(results_dir, "anomaly_maps", f"seed={seed}", obj, "test")
            want = count_test_images(cfg["data_root"], obj)
            have = count_maps(map_root) if os.path.isdir(map_root) else 0
            if have >= want and want > 0:
                print(f"  [skip] {cfg['dataset']}/{cfg['tag']}/{obj} seed={seed} (maps complete {have}/{want})")
                continue
            print(f"  === {cfg['dataset']}/{cfg['tag']}/{obj} seed={seed} (maps {have}/{want}) ===")
            t0 = time.time()
            # save_patch_dists=True writes the .npy maps; we discard the returned scores (CSV
            # already exists and is deterministic, so we do not rewrite it).
            run_anomaly_detection(
                model, obj, data_root=cfg["data_root"], n_ref_samples=SHOT,
                object_anomalies=object_anomalies, plots_dir=results_dir,
                save_examples=False, knn_metric="L2_normalized", knn_neighbors=1,
                faiss_on_cpu=faiss_on_cpu, masking=masking_default[obj],
                mask_ref_images=False, rotation=rotation_default[obj], seed=seed,
                save_patch_dists=True, save_tiffs=False)
            print(f"     done in {time.time()-t0:.1f}s -> {map_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--configs", nargs="+", type=int, default=None,
                    help="indices into CONFIGS to run (default: all)")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[-1])
    print(f"Loading {MODEL_NAME} @ {RESOLUTION} on {args.device} ...")
    model = get_model(MODEL_NAME, "cuda", smaller_edge_size=RESOLUTION)

    cfgs = CONFIGS if args.configs is None else [CONFIGS[i] for i in args.configs]
    print(f"Seeds: {args.seeds} | configs: {[c['dataset']+'/'+c['tag'] for c in cfgs]} | save_patch_dists=True")
    for cfg in cfgs:
        run_config(model, cfg, args.seeds)
    print("\nALL DONE (maps for seeds " + ",".join(map(str, args.seeds)) + ").")


if __name__ == "__main__":
    main()
