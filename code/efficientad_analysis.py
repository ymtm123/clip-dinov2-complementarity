"""Setting-axis comparison and hybrid fast-screen cascade."""
import os
import csv
import json
import numpy as np
import sys
from collections import defaultdict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fusion import load_anomalyclip_scores, load_anomalydino_scores, align_samples

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(BASE), "results")
EFFAD_JSON = os.path.join(RES, "efficientad_benchmark.json")
EFFAD_SCORES = os.path.join(RES, "efficientad_scores")


def load_fusion_mvtec():
    """Per-category fused (CLIP+DINO, min-max, alpha=0.65) per-image scores, keyed by subdir/file."""
    AD = os.path.join(BASE, "AnomalyDINO")
    clip = load_anomalyclip_scores(os.path.join(BASE, "AnomalyCLIP", "results", "scores_mvtec", "anomalyclip_scores.csv"))
    dino = {}
    dino.update(load_anomalydino_scores(os.path.join(AD, "results_MVTec", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_weak_cats", "measurements_seed=0.csv")))
    dino.update(load_anomalydino_scores(os.path.join(AD, "results_MVTec", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_strong_cats", "measurements_seed=0.csv")))
    aligned = align_samples(clip, dino)
    out = {}
    for cat, d in aligned.items():
        cl = np.asarray(d["clip_scores"], float); dn = np.asarray(d["dino_scores"], float)
        def mm(x):
            lo, hi = x.min(), x.max(); return (x - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(x)
        fused = 0.35 * mm(cl) + 0.65 * mm(dn)
        paths = d.get("paths")
        # align_samples may not return paths; rebuild from clip raw if needed
        out[cat] = dict(fused=fused, labels=np.asarray(d["labels"]), paths=paths)
    return out, clip, dino


def load_effad_scores(cat):
    p = os.path.join(EFFAD_SCORES, f"{cat}.csv")
    if not os.path.exists(p):
        return None
    d = {}
    with open(p) as f:
        r = csv.reader(f); next(r)
        for row in r:
            d[row[0].replace("\\", "/")] = (float(row[1]), int(row[2]) if row[2] != "" else None)
    return d


def main():
    if not os.path.exists(EFFAD_JSON):
        print(f"[WAIT] {EFFAD_JSON} not found yet (EfficientAD run in progress)."); return
    effad = json.load(open(EFFAD_JSON))
    eff_lat = effad.get("mean_latency_ms")

    # fusion latency
    isr = json.load(open(os.path.join(BASE, "inference_speed_results.json")))
    fusion_lat = float(isr["parallel_ms"])

    # rebuild per-image fusion scores keyed by subdir/file
    AD = os.path.join(BASE, "AnomalyDINO")
    clip = load_anomalyclip_scores(os.path.join(BASE, "AnomalyCLIP", "results", "scores_mvtec", "anomalyclip_scores.csv"))
    dino = {}
    dino.update(load_anomalydino_scores(os.path.join(AD, "results_MVTec", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_weak_cats", "measurements_seed=0.csv")))
    dino.update(load_anomalydino_scores(os.path.join(AD, "results_MVTec", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_strong_cats", "measurements_seed=0.csv")))

    def mm(x):
        x = np.asarray(x, float); lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(x)

    setting_rows = []
    cascade_rows = []
    cats_done = [r["category"] for r in effad["per_category"] if r.get("image_auroc") is not None]

    for cat in cats_done:
        eff = load_effad_scores(cat)
        if eff is None or cat not in clip or cat not in dino:
            continue
        # build fusion per-sample dict keyed by subdir/file
        cpaths = [p.replace("\\", "/") for p in clip[cat]["paths"]]
        dpaths = {p.replace("\\", "/"): i for i, p in enumerate(dino[cat]["paths"])}
        common = [p for p in cpaths if p in dpaths and p in eff]
        if len(common) < 10:
            continue
        ci = {p: i for i, p in enumerate(cpaths)}
        cl = mm([clip[cat]["scores"][ci[p]] for p in common])
        dn = mm([dino[cat]["scores"][dpaths[p]] for p in common])
        fused = 0.35 * cl + 0.65 * dn
        y = np.array([clip[cat]["labels"][ci[p]] for p in common])
        eff_s = np.array([eff[p][0] for p in common])
        if len(np.unique(y)) < 2:
            continue
        auroc_eff = roc_auc_score(y, eff_s) * 100
        auroc_fuse = roc_auc_score(y, fused) * 100
        setting_rows.append(dict(category=cat, effad_auroc=round(auroc_eff, 1), fusion_auroc=round(auroc_fuse, 1),
                                 n=len(common)))

        # cascade: escalate images whose EfficientAD score is above the q-th percentile of NORMAL effad scores
        normal_eff = eff_s[y == 0]
        for q in [50, 60, 70, 80, 90]:
            tau = np.percentile(normal_eff, q)
            escalate = eff_s > tau
            esc_rate = float(escalate.mean())
            # final score: fusion for escalated; for non-escalated (confident normal) rank below all escalated
            final = np.where(escalate, fused, -1.0 + 0.001 * (eff_s - eff_s.min()))
            auroc_casc = roc_auc_score(y, final) * 100 if len(np.unique(y)) > 1 else float("nan")
            mean_lat = (1 - esc_rate) * eff_lat + esc_rate * fusion_lat
            cascade_rows.append(dict(category=cat, q=q, escalation_rate=round(esc_rate, 3),
                                     cascade_auroc=round(auroc_casc, 1), mean_latency_ms=round(mean_lat, 1)))

    # aggregate cascade by q
    casc_by_q = {}
    for q in [50, 60, 70, 80, 90]:
        rs = [r for r in cascade_rows if r["q"] == q]
        if rs:
            casc_by_q[q] = dict(mean_escalation=round(float(np.mean([r["escalation_rate"] for r in rs])), 3),
                                mean_cascade_auroc=round(float(np.mean([r["cascade_auroc"] for r in rs])), 2),
                                mean_latency_ms=round(float(np.mean([r["mean_latency_ms"] for r in rs])), 1))
    mean_eff = round(float(np.mean([r["effad_auroc"] for r in setting_rows])), 2) if setting_rows else None
    mean_fuse = round(float(np.mean([r["fusion_auroc"] for r in setting_rows])), 2) if setting_rows else None

    out = dict(experiment="EfficientAD setting-axis + hybrid cascade",
               n_categories=len(setting_rows),
               setting_axis=dict(
                   efficientad=dict(mean_auroc_subset=mean_eff, latency_ms=eff_lat,
                                    target_data="per-category normal training set (distillation)",
                                    cross_category="no (re-train per product)"),
                   clip_dino_fusion=dict(mean_auroc_subset=mean_fuse, latency_ms=fusion_lat,
                                         target_data="none (zero/few-shot)", cross_category="yes"),
               ),
               per_category=setting_rows, cascade_by_q=casc_by_q,
               fusion_alone_latency_ms=fusion_lat, effad_latency_ms=eff_lat)
    with open(os.path.join(RES, "efficientad_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 80)
    print("  SETTING-AXIS (subset) + HYBRID CASCADE")
    print("=" * 80)
    print(f"  EfficientAD: mean AUROC {mean_eff} @ {eff_lat} ms/img (per-category distillation, no cross-category)")
    print(f"  CLIP-DINOv2 fusion: mean AUROC {mean_fuse} @ {fusion_lat} ms/img (zero/few-shot, cross-category)")
    print(f"\n  Hybrid cascade (EfficientAD screen -> fusion confirm), averaged over {len(setting_rows)} categories:")
    print(f"  {'screen q':>9} {'escalation':>11} {'cascade AUROC':>14} {'mean latency':>13}")
    for q, v in casc_by_q.items():
        print(f"  {q:>9} {v['mean_escalation']:>11} {v['mean_cascade_auroc']:>14} {v['mean_latency_ms']:>11} ms")
    print(f"\n  (fusion-alone latency {fusion_lat} ms; EfficientAD-alone {eff_lat} ms)")
    print(f"  Saved: results/efficientad_analysis.json")


if __name__ == "__main__":
    main()
