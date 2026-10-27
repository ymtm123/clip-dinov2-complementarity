"""EfficientAD FULL-BENCHMARK reference (image+pixel AUROC + latency)."""
import os
import csv
import json
import time
import argparse
import warnings
import numpy as np
import torch
from pathlib import Path

warnings.filterwarnings("ignore")

from anomalib.data import MVTecAD, Folder
from anomalib.models import EfficientAd
from anomalib.engine import Engine

D9 = "data"
MVTEC_ROOT = f"{D9}/MVTec_AD"
VISA_ROOT = f"{D9}/VisA/VisA_pytorch/1cls"
BTAD_ROOT = f"{D9}/BTAD/BTech_Dataset_transformed"

OUTDIR = Path(__file__).resolve().parent.parent / "results"
PERCATDIR = OUTDIR / "efficientad_full"
SCOREDIR = OUTDIR / "efficientad_scores_full"
SUFFIX = ""   # set from --out_suffix in main() to isolate the full-training (70k-step) run
IMAGENETTE = str(Path(__file__).resolve().parent / "datasets" / "imagenette")

MVTEC_CATS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
              "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
VISA_CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
             "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
BTAD_CATS = ["01", "02", "03"]

DATASETS = {"mvtec": MVTEC_CATS, "visa": VISA_CATS, "btad": BTAD_CATS}
DSLABEL = {"mvtec": "MVTec AD", "visa": "VisA", "btad": "BTAD"}


def make_datamodule(ds, cat):
    """Build the dataset-appropriate anomalib datamodule (full test split)."""
    if ds == "mvtec":
        return MVTecAD(root=MVTEC_ROOT, category=cat, train_batch_size=1,
                       eval_batch_size=16, num_workers=0)
    if ds == "visa":
        return Folder(name=f"visa_{cat}", root=f"{VISA_ROOT}/{cat}",
                      normal_dir="train/good", abnormal_dir="test/bad",
                      normal_test_dir="test/good", mask_dir="ground_truth/bad",
                      val_split_mode="same_as_test",
                      train_batch_size=1, eval_batch_size=16, num_workers=0)
    if ds == "btad":
        return Folder(name=f"btad_{cat}", root=f"{BTAD_ROOT}/{cat}",
                      normal_dir="train/ok", abnormal_dir="test/ko",
                      normal_test_dir="test/ok", mask_dir="ground_truth/ko",
                      val_split_mode="same_as_test",
                      train_batch_size=1, eval_batch_size=16, num_workers=0)
    raise ValueError(ds)


def measure_latency(model, datamodule, device, n_warmup=10, n_iter=100):
    """Per-image forward-pass latency (ms) on the test set, single image."""
    model.eval().to(device)
    dl = datamodule.test_dataloader()
    imgs = []
    for batch in dl:
        x = batch.image if hasattr(batch, "image") else batch["image"]
        for i in range(x.shape[0]):
            imgs.append(x[i:i + 1].to(device))
            if len(imgs) >= n_warmup + n_iter:
                break
        if len(imgs) >= n_warmup + n_iter:
            break
    with torch.no_grad():
        for i in range(min(n_warmup, len(imgs))):
            _ = model.model(imgs[i])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        k = 0
        for i in range(n_warmup, len(imgs)):
            _ = model.model(imgs[i])
            k += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    if k < 1:  # too few test images to time
        return None, None, k
    ms = 1000 * dt / k
    return round(ms, 3), (round(1000 / ms, 1) if ms > 0 else None), k


def _extract_auroc(metrics, want_pixel):
    """Pull image- or pixel-level AUROC from anomalib's test() metric dict."""
    if not hasattr(metrics, "items"):
        return None
    for k, v in metrics.items():
        kl = k.lower()
        if "auroc" not in kl:
            continue
        is_pixel = "pixel" in kl
        if want_pixel and is_pixel:
            return round(float(v) * 100, 2)
        if (not want_pixel) and (("image" in kl or "img" in kl) and not is_pixel):
            return round(float(v) * 100, 2)
    return None


def run_category(ds, cat, max_epochs=None, max_steps=None):
    dm = make_datamodule(ds, cat)
    model = EfficientAd(imagenet_dir=IMAGENETTE, model_size="small")
    # max_steps gives a UNIFORM step budget across categories (EfficientAd.configure_optimizers
    # uses num_steps=max_steps and decays the LR at 0.95*num_steps); must NOT also pass a finite
    # max_epochs, else num_steps=min(max_steps, max_epochs*len(train)) caps small categories.
    if max_steps is not None:
        engine = Engine(max_steps=max_steps, accelerator="gpu", devices=1,
                        logger=False, enable_model_summary=False, enable_progress_bar=False)
    else:
        engine = Engine(max_epochs=max_epochs, accelerator="gpu", devices=1,
                        logger=False, enable_model_summary=False, enable_progress_bar=False)
    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=dm)
    train_s = time.perf_counter() - t0

    test_res = engine.test(model=model, datamodule=dm)
    metrics = test_res[0] if isinstance(test_res, list) and test_res else (test_res or {})
    print(f"  [debug] {ds}/{cat} test metric keys: {list(metrics.keys()) if hasattr(metrics,'keys') else type(metrics)}")
    image_auroc = _extract_auroc(metrics, want_pixel=False)
    pixel_auroc = _extract_auroc(metrics, want_pixel=True)

    # per-image scores via predict (for the cascade)
    preds = engine.predict(model=model, datamodule=dm)
    rows = []
    if preds:
        for batch in preds:
            scores = getattr(batch, "pred_score", None)
            labels = getattr(batch, "gt_label", None)
            paths = getattr(batch, "image_path", None)
            if scores is None:
                continue
            scores = scores.detach().cpu().numpy().ravel()
            labels = labels.detach().cpu().numpy().ravel() if labels is not None else [None] * len(scores)
            paths = paths if paths is not None else [None] * len(scores)
            for p, s, l in zip(paths, scores, labels):
                rel = ""
                if p is not None:
                    pp = str(p).replace("\\", "/")
                    rel = pp.split("/test/")[-1] if "/test/" in pp else os.path.basename(pp)
                rows.append([rel, float(s), int(l) if l is not None else ""])
    SCOREDIR.mkdir(parents=True, exist_ok=True)
    with open(SCOREDIR / f"{ds}__{cat}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Sample", "Score", "Label"]); w.writerows(rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lat_ms, img_s, n_timed = measure_latency(model, dm, device)

    rec = dict(dataset=DSLABEL[ds], dataset_key=ds, category=cat,
               image_auroc=image_auroc, pixel_auroc=pixel_auroc,
               train_seconds=round(train_s, 1), latency_ms_per_image=lat_ms,
               throughput_img_s=img_s, n_predictions=len(rows),
               n_latency_timed=n_timed, max_epochs=max_epochs, max_steps=max_steps)
    print(f"  [{ds}/{cat}] img_AUROC={image_auroc} pix_AUROC={pixel_auroc} "
          f"train={train_s:.0f}s latency={lat_ms}ms/img ({img_s} img/s) n_pred={len(rows)}")
    return rec


def aggregate(max_epochs, max_steps=None):
    """Combine all per-category JSONs into results/efficientad_benchmark_full{SUFFIX}.json."""
    recs = []
    for ds, cats in DATASETS.items():
        for cat in cats:
            p = PERCATDIR / f"{ds}__{cat}.json"
            if p.exists():
                recs.append(json.load(open(p)))
    ok = [r for r in recs if r.get("image_auroc") is not None]
    okp = [r for r in recs if r.get("pixel_auroc") is not None]
    errored = [f"{r['dataset_key']}/{r['category']}" for r in recs if r.get("image_auroc") is None]
    n_expected = sum(len(v) for v in DATASETS.values())

    def by_ds(key, rows):
        out = {}
        for ds in DATASETS:
            rs = [r for r in rows if r.get("dataset_key") == ds and r.get(key) is not None]
            if rs:
                out[DSLABEL[ds]] = dict(n=len(rs), mean=round(float(np.mean([r[key] for r in rs])), 2))
        return out

    budget_note = (("70,000-step official EfficientAD budget (full training); supersedes the 15-epoch "
                    "reduced run in results/efficientad_full/.") if max_steps is not None else
                   ("15-epoch reduced budget. The original 6-category run "
                    "(efficientad_benchmark.py) used 20 epochs; full EfficientAD training for all 30 "
                    "categories (--max_steps 70000) supersedes these numbers."))
    summary = dict(
        experiment="EfficientAD FULL benchmark (image+pixel AUROC + latency)",
        hardware="RTX 4090", max_epochs=max_epochs, max_steps=max_steps,
        epoch_budget_note=budget_note,
        n_expected=n_expected, n_errored=len(errored), errored_categories=errored,
        n_categories=len(recs), n_with_image_auroc=len(ok), n_with_pixel_auroc=len(okp),
        mean_image_auroc=round(float(np.mean([r["image_auroc"] for r in ok])), 2) if ok else None,
        mean_pixel_auroc=round(float(np.mean([r["pixel_auroc"] for r in okp])), 2) if okp else None,
        mean_latency_ms=round(float(np.mean([r["latency_ms_per_image"] for r in ok])), 3) if ok else None,
        mean_throughput_img_s=round(float(np.mean([r["throughput_img_s"] for r in ok])), 1) if ok else None,
        image_auroc_by_dataset=by_ds("image_auroc", ok),
        pixel_auroc_by_dataset=by_ds("pixel_auroc", okp),
        per_category=recs,
    )
    with open(OUTDIR / f"efficientad_benchmark_full{SUFFIX}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== AGGREGATE{SUFFIX} ({len(ok)}/{n_expected} with image AUROC; {len(recs)} attempted; "
          f"{len(errored)} errored{': ' + ', '.join(errored) if errored else ''}) ===")
    print(f"  mean image AUROC = {summary['mean_image_auroc']}  by ds: {summary['image_auroc_by_dataset']}")
    print(f"  mean pixel AUROC = {summary['mean_pixel_auroc']}  by ds: {summary['pixel_auroc_by_dataset']}")
    print(f"  mean latency     = {summary['mean_latency_ms']} ms/img")
    print(f"  Saved: results/efficientad_benchmark_full{SUFFIX}.json")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mvtec", "visa", "btad", "all"], default="all")
    ap.add_argument("--categories", nargs="+", default=None,
                    help="optional explicit category subset within --dataset")
    ap.add_argument("--max_epochs", type=int, default=15)
    ap.add_argument("--max_steps", type=int, default=None,
                    help="uniform step budget (e.g. 70000 for the official EfficientAD budget); "
                         "when set, overrides --max_epochs and max_epochs is NOT passed to the Engine")
    ap.add_argument("--out_suffix", type=str, default="",
                    help="suffix for the output dirs/json (e.g. _70kstep) to isolate the full-training run")
    ap.add_argument("--force", action="store_true", help="re-run categories even if a JSON exists")
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()

    # route outputs to suffixed dirs so the 15-epoch run is never overwritten by the 70k-step run
    global PERCATDIR, SCOREDIR, SUFFIX
    SUFFIX = args.out_suffix
    PERCATDIR = OUTDIR / f"efficientad_full{SUFFIX}"
    SCOREDIR = OUTDIR / f"efficientad_scores_full{SUFFIX}"
    max_epochs = None if args.max_steps is not None else args.max_epochs  # exactly one budget lever

    PERCATDIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(max_epochs, args.max_steps); return

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    plan = []
    for ds in targets:
        cats = args.categories if (args.categories and args.dataset == ds) else DATASETS[ds]
        for cat in cats:
            plan.append((ds, cat))

    budget = f"max_steps={args.max_steps}" if args.max_steps is not None else f"max_epochs={max_epochs}"
    print(f"EfficientAD FULL benchmark | {len(plan)} categories | {budget} | out_suffix='{SUFFIX}'")
    for i, (ds, cat) in enumerate(plan, 1):
        outp = PERCATDIR / f"{ds}__{cat}.json"
        if outp.exists() and not args.force:
            r = json.load(open(outp))
            if r.get("image_auroc") is not None:
                print(f"[{i}/{len(plan)}] SKIP {ds}/{cat} (done: img={r.get('image_auroc')} pix={r.get('pixel_auroc')})")
                continue
            # JSON exists but holds an error record -> re-run rather than count it as done
            print(f"[{i}/{len(plan)}] RETRY {ds}/{cat} (prior attempt errored: {r.get('error')})")
        print(f"[{i}/{len(plan)}] RUN  {ds}/{cat} ...")
        try:
            rec = run_category(ds, cat, max_epochs=max_epochs, max_steps=args.max_steps)
            json.dump(rec, open(outp, "w"), indent=2)
        except Exception as e:
            import traceback; traceback.print_exc()
            json.dump(dict(dataset=DSLABEL[ds], dataset_key=ds, category=cat, error=str(e)),
                      open(outp, "w"), indent=2)
        aggregate(max_epochs, args.max_steps)  # refresh aggregate after each category

    aggregate(max_epochs, args.max_steps)


if __name__ == "__main__":
    main()
