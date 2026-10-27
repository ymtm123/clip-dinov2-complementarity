"""EfficientAD reference benchmark (AUROC + latency)."""
import os
import csv
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import EfficientAd
from anomalib.engine import Engine

ROOT = "data/MVTec_AD"
OUTDIR = Path(__file__).resolve().parent.parent / "results"
SCOREDIR = OUTDIR / "efficientad_scores"
IMAGENETTE = str(Path(__file__).resolve().parent / "datasets" / "imagenette")


def measure_latency(model, datamodule, device, n_warmup=10, n_iter=100):
    """Per-image forward-pass latency (ms) on the test set, single image."""
    model.eval().to(device)
    dl = datamodule.test_dataloader()
    imgs = []
    for batch in dl:
        x = batch.image if hasattr(batch, "image") else batch["image"]
        for i in range(x.shape[0]):
            imgs.append(x[i:i+1].to(device))
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
    ms = 1000 * dt / max(k, 1)
    return round(ms, 3), round(1000 / ms, 1) if ms > 0 else None, k


def run_category(category, max_epochs):
    dm = MVTecAD(root=ROOT, category=category, train_batch_size=1, eval_batch_size=16, num_workers=0)
    model = EfficientAd(imagenet_dir=IMAGENETTE, model_size="small")
    engine = Engine(max_epochs=max_epochs, accelerator="gpu", devices=1,
                    logger=False, enable_model_summary=False, enable_progress_bar=False)
    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=dm)
    train_s = time.perf_counter() - t0

    test_res = engine.test(model=model, datamodule=dm)
    metrics = test_res[0] if isinstance(test_res, list) and test_res else (test_res or {})
    print(f"  [debug] test metric keys: {list(metrics.keys()) if hasattr(metrics,'keys') else type(metrics)}")
    auroc = None
    for k, v in (metrics.items() if hasattr(metrics, "items") else []):
        if "auroc" in k.lower() and ("image" in k.lower() or "img" in k.lower()):
            auroc = round(float(v) * 100, 2)
    if auroc is None:  # fallback: any AUROC metric
        for k, v in (metrics.items() if hasattr(metrics, "items") else []):
            if "auroc" in k.lower():
                auroc = round(float(v) * 100, 2)

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
    with open(SCOREDIR / f"{category}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Sample", "Score", "Label"]); w.writerows(rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lat_ms, img_s, n_timed = measure_latency(model, dm, device)

    print(f"  [{category}] AUROC={auroc} train={train_s:.0f}s latency={lat_ms}ms/img ({img_s} img/s) n_pred={len(rows)}")
    return dict(category=category, image_auroc=auroc, train_seconds=round(train_s, 1),
               latency_ms_per_image=lat_ms, throughput_img_s=img_s, n_predictions=len(rows),
               max_epochs=max_epochs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["bottle"])
    ap.add_argument("--max_epochs", type=int, default=20)
    args = ap.parse_args()

    print(f"EfficientAD benchmark | categories={args.categories} | max_epochs={args.max_epochs}")
    results = []
    for c in args.categories:
        try:
            results.append(run_category(c, args.max_epochs))
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append(dict(category=c, error=str(e)))

    ok = [r for r in results if r.get("image_auroc") is not None]
    summary = dict(
        experiment="EfficientAD reference benchmark",
        hardware="RTX 4090", n_categories=len(ok),
        mean_image_auroc=round(float(np.mean([r["image_auroc"] for r in ok])), 2) if ok else None,
        mean_latency_ms=round(float(np.mean([r["latency_ms_per_image"] for r in ok])), 3) if ok else None,
        mean_throughput_img_s=round(float(np.mean([r["throughput_img_s"] for r in ok])), 1) if ok else None,
        per_category=results,
    )
    with open(OUTDIR / "efficientad_benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: mean AUROC={summary['mean_image_auroc']}, mean latency={summary['mean_latency_ms']}ms/img")
    print(f"Saved: results/efficientad_benchmark.json")


if __name__ == "__main__":
    main()
