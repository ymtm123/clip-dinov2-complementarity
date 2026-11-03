"""Multi-seed AnomalyDINO reference-image robustness ."""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import get_dataset_info
from src.detection import run_anomaly_detection
from src.backbones import get_model

D9 = "data"

# The four configs that produced seed 0 (derived from the existing result dirs).
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
        measurements_file = os.path.join(results_dir, f"measurements_seed={seed}.csv")
        if os.path.exists(measurements_file):
            print(f"  [skip] {cfg['dataset']}/{cfg['tag']} seed={seed} (exists)")
            continue
        print(f"\n  === {cfg['dataset']}/{cfg['tag']} seed={seed} ({len(objects)} cats) ===")
        t0 = time.time()
        # write to a temp then rename, so an interrupted seed never leaves a partial CSV
        tmp = measurements_file + ".partial"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Object", "Sample", "Anomaly_Score", "MemoryBank_Time", "Inference_Time"])
            for obj in objects:
                anomaly_scores, t_mem, t_inf = run_anomaly_detection(
                    model, obj, data_root=cfg["data_root"], n_ref_samples=SHOT,
                    object_anomalies=object_anomalies, plots_dir=results_dir,
                    save_examples=False, knn_metric="L2_normalized", knn_neighbors=1,
                    faiss_on_cpu=faiss_on_cpu, masking=masking_default[obj],
                    mask_ref_images=False, rotation=rotation_default[obj], seed=seed,
                    save_patch_dists=False, save_tiffs=False)
                for sample in anomaly_scores:
                    w.writerow([obj, sample, f"{anomaly_scores[sample]:.5f}",
                                f"{t_mem:.5f}", f"{t_inf[sample]:.5f}"])
                print(f"     {obj}: {len(anomaly_scores)} samples")
        os.replace(tmp, measurements_file)
        print(f"  done {cfg['dataset']}/{cfg['tag']} seed={seed} in {time.time()-t0:.1f}s -> {measurements_file}")


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
    print(f"Seeds: {args.seeds} | configs: {[c['dataset']+'/'+c['tag'] for c in cfgs]}")
    for cfg in cfgs:
        run_config(model, cfg, args.seeds)
    print("\nALL DONE.")


if __name__ == "__main__":
    main()
