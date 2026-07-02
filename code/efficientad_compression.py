"""T3 §4f(ii) / : measured edge compression (FP16 / INT8) + async-pipelined inference for EfficientAD."""
import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from efficientad_benchmark_full import make_datamodule, _extract_auroc, IMAGENETTE, OUTDIR
from anomalib.models import EfficientAd
from anomalib.engine import Engine

DEVICE = torch.device("cuda")


from contextlib import nullcontext


def fwd_latency(net, imgs, n_warmup=10, autocast=False, async_pipe=False):
    """FP16 is measured via torch.autocast (robust for EfficientAd's mixed-precision internals — a
    blanket .half() breaks on float buffers/normalization ops). async_pipe times the H2D copy too."""
    net.eval()

    def prep(x):
        return x.to(DEVICE, non_blocking=True) if async_pipe else x

    ctx = torch.autocast("cuda", dtype=torch.float16) if autocast else nullcontext()
    with torch.no_grad(), ctx:
        for i in range(min(n_warmup, len(imgs))):
            _ = net(prep(imgs[i]))
        torch.cuda.synchronize()
        t0 = time.perf_counter(); k = 0
        for i in range(n_warmup, len(imgs)):
            _ = net(prep(imgs[i])); k += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return round(1000 * dt / max(k, 1), 3)


def collect_imgs(dm, n, to_device=True):
    out = []
    for batch in dm.test_dataloader():
        x = batch.image if hasattr(batch, "image") else batch["image"]
        for i in range(x.shape[0]):
            t = x[i:i + 1]
            out.append(t.to(DEVICE) if to_device else t.pin_memory())
            if len(out) >= n:
                return out
    return out


def eval_auroc(engine, model, dm):
    res = engine.test(model=model, datamodule=dm)
    m = res[0] if isinstance(res, list) and res else (res or {})
    return _extract_auroc(m, False), _extract_auroc(m, True)


def run_category(ds, cat, max_epochs):
    dm = make_datamodule(ds, cat)
    model = EfficientAd(imagenet_dir=IMAGENETTE, model_size="small")
    engine = Engine(max_epochs=max_epochs, accelerator="gpu", devices=1, logger=False,
                    enable_model_summary=False, enable_progress_bar=False)
    engine.fit(model=model, datamodule=dm)
    img32, pix32 = eval_auroc(engine, model, dm)
    net = model.model.to(DEVICE).eval()

    imgs = collect_imgs(dm, 110, to_device=True)
    rec = dict(dataset=ds, category=cat, image_auroc_fp32=img32, pixel_auroc_fp32=pix32)
    rec["latency_fp32_ms"] = fwd_latency(net, imgs)
    rec["latency_async_fp32_ms"] = fwd_latency(net, collect_imgs(dm, 110, to_device=False), async_pipe=True)
    import copy
    # FP16 latency via autocast on the same FP32 net (robust; no in-place half())
    try:
        rec["latency_fp16_ms"] = fwd_latency(net, imgs, autocast=True)
        rec["fp16_speedup"] = round(rec["latency_fp32_ms"] / rec["latency_fp16_ms"], 2)
    except Exception as e:
        rec["latency_fp16_ms"] = None; rec["fp16_error"] = str(e)[:120]
    # FP16 AUROC via autocast on the UNCHANGED FP32 model
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            img16, pix16 = eval_auroc(engine, model, dm)
        rec["image_auroc_fp16"] = img16; rec["pixel_auroc_fp16"] = pix16
    except Exception as e:
        rec["image_auroc_fp16"] = None; rec["fp16_auroc_error"] = str(e)[:120]
    # INT8 (dynamic) — attempt on a CPU copy; conv-heavy net usually not accelerated -> disclose
    try:
        qnet = torch.quantization.quantize_dynamic(copy.deepcopy(net).float().cpu(), {torch.nn.Linear}, dtype=torch.qint8)
        cpu_imgs = [im.float().cpu() for im in imgs[:40]]
        qnet.eval()
        with torch.no_grad():
            for i in range(5):
                _ = qnet(cpu_imgs[i])
            t0 = time.perf_counter(); k = 0
            for i in range(5, len(cpu_imgs)):
                _ = qnet(cpu_imgs[i]); k += 1
            dt = time.perf_counter() - t0
        rec["int8_dynamic_cpu_latency_ms"] = round(1000 * dt / max(k, 1), 3)
        rec["int8_note"] = ("dynamic INT8 quantizes only Linear layers; EfficientAd is conv-dominated so "
                            "this does NOT meaningfully accelerate it (measured on CPU). Static INT8 of the "
                            "student-teacher conv stack needs a custom calibrated qconfig + fused conv-bn — "
                            "DISCLOSED as out of scope; FP16 is the recommended edge precision.")
    except Exception as e:
        rec["int8_dynamic_cpu_latency_ms"] = None
        rec["int8_note"] = f"dynamic INT8 not applicable / errored ({str(e)[:80]}); disclosed. FP16 is the edge precision."
    print(f"  [{ds}/{cat}] fp32 {rec['latency_fp32_ms']}ms (img {img32}/pix {pix32}) | "
          f"fp16 {rec.get('latency_fp16_ms')}ms x{rec.get('fp16_speedup')} (img {rec.get('image_auroc_fp16')}) | "
          f"async {rec['latency_async_fp32_ms']}ms", flush=True)
    del model, net; torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["bottle", "carpet", "grid", "hazelnut", "cable", "transistor"],
                    help="MVTec categories (representative easy+hard subset)")
    ap.add_argument("--dataset", default="mvtec")
    ap.add_argument("--max_epochs", type=int, default=15)
    args = ap.parse_args()
    recs = []
    for cat in args.categories:
        try:
            recs.append(run_category(args.dataset, cat, args.max_epochs))
        except Exception as e:
            import traceback; traceback.print_exc()
            recs.append(dict(dataset=args.dataset, category=cat, error=str(e)))
    ok = [r for r in recs if r.get("latency_fp32_ms")]
    summary = dict(experiment="EfficientAD edge compression (FP16/INT8/async)",
                   hardware="RTX 4090", n_categories=len(ok),
                   mean_latency_fp32_ms=round(float(np.mean([r["latency_fp32_ms"] for r in ok])), 3) if ok else None,
                   mean_latency_fp16_ms=round(float(np.mean([r["latency_fp16_ms"] for r in ok if r.get("latency_fp16_ms")])), 3) if ok else None,
                   mean_fp16_speedup=round(float(np.mean([r["fp16_speedup"] for r in ok if r.get("fp16_speedup")])), 2) if ok else None,
                   mean_latency_async_ms=round(float(np.mean([r["latency_async_fp32_ms"] for r in ok])), 3) if ok else None,
                   mean_image_auroc_fp32=round(float(np.mean([r["image_auroc_fp32"] for r in ok if r.get("image_auroc_fp32")])), 2) if ok else None,
                   mean_image_auroc_fp16=round(float(np.mean([r["image_auroc_fp16"] for r in ok if r.get("image_auroc_fp16")])), 2) if ok else None,
                   per_category=recs)
    json.dump(summary, open(os.path.join(OUTDIR, "efficientad_compression.json"), "w"), indent=2)
    print(f"\n  mean latency fp32 {summary['mean_latency_fp32_ms']} -> fp16 {summary['mean_latency_fp16_ms']} "
          f"(x{summary['mean_fp16_speedup']}) | async {summary['mean_latency_async_ms']} ms")
    print(f"  image AUROC fp32 {summary['mean_image_auroc_fp32']} -> fp16 {summary['mean_image_auroc_fp16']}")
    print(f"  Saved: results/efficientad_compression.json")


if __name__ == "__main__":
    main()
