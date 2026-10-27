"""FULL-benchmark setting-axis table + hybrid cascade."""
import os
import csv
import json
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from score_fusion import load_anomalyclip_scores, load_anomalydino_scores
from fine_grained_alpha_sweep import load_btad_dino_scores

RES = os.path.join(os.path.dirname(BASE), "results")
BENCH_JSON = os.path.join(RES, "efficientad_benchmark_full.json")
SCOREDIR = os.path.join(RES, "efficientad_scores_full")
SPEED_JSON = os.path.join(BASE, "inference_speed_results.json")

ALPHA_DINO = 0.65  # manuscript fixed-fusion DINOv2 weight
QS = [50, 60, 70, 80, 90]
DS_KEYS = {"mvtec": "MVTec AD", "visa": "VisA", "btad": "BTAD"}


def stemkey(rel):
    rel = rel.replace("\\", "/")
    d = os.path.dirname(rel)
    s = os.path.splitext(os.path.basename(rel))[0]
    return (d + "/" + s).lower().lstrip("/")


def mm(x):
    x = np.asarray(x, float); lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(x)


def load_fusion_by_dataset():
    """Return {ds_key: {cat: {stemkey: (clip_raw, dino_raw, label)}}} for all 3 datasets."""
    ad = os.path.join(BASE, "AnomalyDINO"); cl = os.path.join(BASE, "AnomalyCLIP", "results")
    out = {}
    # MVTec (weak + strong DINO splits)
    clip = load_anomalyclip_scores(os.path.join(cl, "scores_mvtec", "anomalyclip_scores.csv"))
    dino = {}
    for sub in ("agnostic_weak_cats", "agnostic_strong_cats"):
        dino.update(load_anomalydino_scores(os.path.join(ad, "results_MVTec", "dinov2_vitl14_448", f"1-shot_preprocess={sub}", "measurements_seed=0.csv")))
    out["mvtec"] = _merge(clip, dino)
    # VisA
    clip = load_anomalyclip_scores(os.path.join(cl, "scores_visa", "anomalyclip_scores.csv"))
    dino = load_anomalydino_scores(os.path.join(ad, "results_VisA", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv"))
    out["visa"] = _merge(clip, dino)
    # BTAD
    clip = load_anomalyclip_scores(os.path.join(cl, "scores_btad", "anomalyclip_scores.csv"))
    dino = load_btad_dino_scores(os.path.join(ad, "results_BTAD", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_btad_all", "measurements_seed=0.csv"))
    out["btad"] = _merge(clip, dino)
    return out


def _merge(clip, dino):
    cats = {}
    for cat in set(clip) & set(dino):
        cidx = {stemkey(p): i for i, p in enumerate(clip[cat]["paths"])}
        didx = {stemkey(p): i for i, p in enumerate(dino[cat]["paths"])}
        d = {}
        for k in set(cidx) & set(didx):
            ci, di = cidx[k], didx[k]
            d[k] = (clip[cat]["scores"][ci], dino[cat]["scores"][di], int(clip[cat]["labels"][ci]))
        if d:
            cats[cat] = d
    return cats


def load_effad_scores(ds, cat):
    p = os.path.join(SCOREDIR, f"{ds}__{cat}.csv")
    if not os.path.exists(p):
        return None
    d = {}
    with open(p) as f:
        r = csv.reader(f); next(r)
        for row in r:
            if not row or row[1] == "":
                continue
            d[stemkey(row[0])] = (float(row[1]), int(row[2]) if row[2] != "" else None)
    return d


def get_fusion_latency():
    if not os.path.exists(SPEED_JSON):
        return None, "inference_speed_results.json missing — run inference_speed.py"
    s = json.load(open(SPEED_JSON))
    if "total" in s and s["total"].get("parallel_ms"):
        return float(s["total"]["parallel_ms"]), "measured parallel (max(CLIP,DINO)+fusion overhead)"
    # fall back to components
    dino = s.get("anomalydino", {}).get("mean_ms"); clip = s.get("anomalyclip", {}).get("mean_ms")
    if dino and clip:
        ov = s.get("fusion_overhead", {}).get("pixel_fusion_per_image_ms", 0.0)
        return round(max(dino, clip) + ov, 2), "computed max(CLIP,DINO)+overhead from components"
    return None, "no usable latency fields in inference_speed_results.json"


def main():
    if not os.path.exists(BENCH_JSON):
        print(f"[WAIT] {BENCH_JSON} not found — run efficientad_benchmark_full.py first."); return
    bench = json.load(open(BENCH_JSON))
    eff_lat = bench.get("mean_latency_ms")
    per_cat_eff = {(r["dataset_key"], r["category"]): r for r in bench["per_category"] if r.get("image_auroc") is not None}

    fusion_lat, lat_note = get_fusion_latency()
    fusion = load_fusion_by_dataset()

    lat_ok = (eff_lat is not None) and (fusion_lat is not None)
    setting_rows, cascade_rows, excluded = [], [], []
    for (ds, cat), effrec in sorted(per_cat_eff.items()):
        eff = load_effad_scores(ds, cat)
        fus = fusion.get(ds, {}).get(cat)
        if eff is None or not fus:
            excluded.append(dict(dataset_key=ds, category=cat,
                                 reason="no EfficientAD scores" if eff is None else "no fusion scores"))
            continue
        common = [k for k in fus if k in eff]
        if len(common) < 10:
            excluded.append(dict(dataset_key=ds, category=cat, reason=f"<10 aligned samples ({len(common)})"))
            continue
        cl = mm([fus[k][0] for k in common]); dn = mm([fus[k][1] for k in common])
        fused = (1 - ALPHA_DINO) * cl + ALPHA_DINO * dn
        y = np.array([fus[k][2] for k in common])
        eff_s = np.array([eff[k][0] for k in common])
        if len(np.unique(y)) < 2:
            excluded.append(dict(dataset_key=ds, category=cat, reason="single-class labels in aligned set"))
            continue
        auroc_eff = roc_auc_score(y, eff_s) * 100
        auroc_fuse = roc_auc_score(y, fused) * 100
        setting_rows.append(dict(dataset=DS_KEYS[ds], dataset_key=ds, category=cat,
                                 effad_image_auroc=round(auroc_eff, 1), fusion_image_auroc=round(auroc_fuse, 1),
                                 effad_pixel_auroc=effrec.get("pixel_auroc"),
                                 effad_latency_ms=effrec.get("latency_ms_per_image"), n=len(common)))
        normal_eff = eff_s[y == 0]
        for q in QS:
            tau = np.percentile(normal_eff, q)
            escalate = eff_s > tau
            esc = float(escalate.mean())
            final = np.where(escalate, fused, -1.0 + 0.001 * (eff_s - eff_s.min()))
            casc = roc_auc_score(y, final) * 100
            # latency unavailable -> no cascade latency
            mean_lat = round((1 - esc) * eff_lat + esc * fusion_lat, 1) if lat_ok else None
            cascade_rows.append(dict(dataset_key=ds, category=cat, q=q, escalation_rate=round(esc, 3),
                                     cascade_auroc=round(casc, 1), mean_latency_ms=mean_lat))

    def agg_setting(rows):
        if not rows:
            return None
        return dict(n=len(rows),
                    mean_effad_image_auroc=round(float(np.mean([r["effad_image_auroc"] for r in rows])), 2),
                    mean_fusion_image_auroc=round(float(np.mean([r["fusion_image_auroc"] for r in rows])), 2),
                    mean_effad_pixel_auroc=round(float(np.mean([r["effad_pixel_auroc"] for r in rows if r["effad_pixel_auroc"] is not None])), 2))

    def agg_cascade(rows):
        out = {}
        for q in QS:
            rs = [r for r in rows if r["q"] == q]
            if rs:
                lats = [r["mean_latency_ms"] for r in rs if r["mean_latency_ms"] is not None]
                out[q] = dict(mean_escalation=round(float(np.mean([r["escalation_rate"] for r in rs])), 3),
                              mean_cascade_auroc=round(float(np.mean([r["cascade_auroc"] for r in rs])), 2),
                              mean_latency_ms=(round(float(np.mean(lats)), 1) if lats else None))
        return out

    n_expected = len(per_cat_eff)  # categories EfficientAD scored (image_auroc not None)
    out = dict(experiment="EfficientAD FULL setting-axis + hybrid cascade",
               n_categories=len(setting_rows), n_expected=n_expected, excluded=excluded,
               setting_axis_basis="EfficientAD image AUROC here is recomputed on the CLIP/DINOv2 fusion-overlap "
                                  "common-sample subset (NOT the full anomalib test split); the full-test EfficientAD "
                                  "mean is in efficientad_benchmark_full.json mean_image_auroc.",
               alpha_dino=ALPHA_DINO,
               efficientad_latency_ms=eff_lat, fusion_latency_ms=fusion_lat, fusion_latency_note=lat_note,
               setting_axis=dict(
                   overall=agg_setting(setting_rows),
                   by_dataset={DS_KEYS[ds]: agg_setting([r for r in setting_rows if r["dataset_key"] == ds]) for ds in DS_KEYS},
                   efficientad=dict(target_data="per-category normal training set (distillation)", cross_category="no (re-train per product)", latency_ms=eff_lat),
                   clip_dino_fusion=dict(target_data="none (zero/few-shot)", cross_category="yes", latency_ms=fusion_lat)),
               cascade=dict(overall=agg_cascade(cascade_rows),
                            by_dataset={DS_KEYS[ds]: agg_cascade([r for r in cascade_rows if r["dataset_key"] == ds]) for ds in DS_KEYS}),
               per_category=setting_rows)
    json.dump(out, open(os.path.join(RES, "efficientad_analysis_full.json"), "w"), indent=2)

    print("=" * 90)
    print(f"  FULL SETTING-AXIS ({len(setting_rows)} categories) + HYBRID CASCADE")
    print("=" * 90)
    ov = out["setting_axis"]["overall"]
    print(f"  EfficientAD: image {ov['mean_effad_image_auroc']}  pixel {ov['mean_effad_pixel_auroc']}  @ {eff_lat} ms/img (per-cat distillation, no cross-category)")
    print(f"  CLIP-DINOv2 fusion: image {ov['mean_fusion_image_auroc']}  @ {fusion_lat} ms/img ({lat_note})")
    for ds in DS_KEYS:
        s = out["setting_axis"]["by_dataset"][DS_KEYS[ds]]
        if s:
            print(f"     {DS_KEYS[ds]:<9}: effad img {s['mean_effad_image_auroc']} / pix {s['mean_effad_pixel_auroc']}  | fusion img {s['mean_fusion_image_auroc']}  (n={s['n']})")
    print(f"\n  Hybrid cascade (overall): {'q':>4} {'escalation':>11} {'cascade AUROC':>14} {'mean latency':>13}")
    for q, v in out["cascade"]["overall"].items():
        print(f"     {q:>4} {v['mean_escalation']:>11} {v['mean_cascade_auroc']:>14} {v['mean_latency_ms']} ms")
    if len(setting_rows) != n_expected or excluded:
        print(f"\n  [WARN] setting-axis covers {len(setting_rows)}/{n_expected} EfficientAD-scored categories; "
              f"excluded: {excluded}")
    else:
        print(f"\n  [OK] setting-axis covers all {len(setting_rows)} EfficientAD-scored categories (none excluded).")
    print(f"\n  Saved: results/efficientad_analysis_full.json")


if __name__ == "__main__":
    main()
