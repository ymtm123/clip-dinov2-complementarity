"""Logical vs structural complementarity on MVTec LOCO."""

import os
import sys
import csv
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
ALPHAS = [round(i / 100.0, 2) for i in range(0, 101)]
CATS = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']

DINO_CSV = os.path.join(BASE, "AnomalyDINO", "results_MVTEC_LOCO", "dinov2_vitl14_448",
                        "1-shot_preprocess=agnostic_loco", "measurements_seed=0.csv")
CLIP_CSV = os.path.join(BASE, "AnomalyCLIP", "results", "loco_split", "loco_clip_scores.csv")


def atype(sample):
    s = sample.replace('\\', '/')
    if s.startswith('good/') or '/good/' in s or s.split('/')[0] == 'good':
        return 'good'
    if 'logical' in s:
        return 'logical'
    if 'structural' in s:
        return 'structural'
    return 'good'


def load_dino():
    d = defaultdict(dict)  # cat -> sample -> score
    with open(DINO_CSV) as f:
        r = csv.reader(f); next(r)
        for row in r:
            d[row[0]][row[1].replace('\\', '/')] = float(row[2])
    return d


def load_clip():
    d = defaultdict(dict)
    with open(CLIP_CSV) as f:
        r = csv.reader(f); next(r)
        for row in r:
            d[row[0]][row[1].replace('\\', '/')] = float(row[2])
    return d


def auroc(y, s):
    return roc_auc_score(y, s) * 100 if len(set(y)) > 1 else float('nan')


def regime_eval(clip, dino):
    """For each category and regime (structural/logical), align good+anom samples,
    return per-model AUROC and fusion alpha-curve."""
    rows = {}
    for cat in CATS:
        if cat not in clip or cat not in dino:
            continue
        common = set(clip[cat]) & set(dino[cat])
        by_type = defaultdict(list)
        for s in common:
            by_type[atype(s)].append(s)
        for regime, anom_key in [('structural', 'structural'), ('logical', 'logical')]:
            samples = by_type['good'] + by_type[anom_key]
            if not by_type[anom_key] or not by_type['good']:
                continue
            y = np.array([0] * len(by_type['good']) + [1] * len(by_type[anom_key]))
            cl = np.array([clip[cat][s] for s in samples])
            dn = np.array([dino[cat][s] for s in samples])
            # min-max norm
            def mm(x):
                lo, hi = x.min(), x.max(); return (x - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(x)
            cln, dnn = mm(cl), mm(dn)
            curve = {a: auroc(y, (1 - a) * cln + a * dnn) for a in ALPHAS}
            oa = max(ALPHAS, key=lambda x: curve[x])
            rows[(cat, regime)] = dict(clip=auroc(y, cl), dino=auroc(y, dn),
                                       fused065=curve[0.65], oracle_alpha=oa, oracle=curve[oa],
                                       n_good=len(by_type['good']), n_anom=len(by_type[anom_key]))
    return rows


def main():
    if not os.path.exists(DINO_CSV):
        print(f"[WAIT] DINO LOCO scores not found yet: {DINO_CSV}"); return
    if not os.path.exists(CLIP_CSV):
        print(f"[WAIT] CLIP LOCO per-sample scores not found yet: {CLIP_CSV}\n"
              f"       Run: AnomalyCLIP/test_loco_split.py with --data_path <LOCO> --checkpoint epoch_15.pth"); return

    clip, dino = load_clip(), load_dino()
    rows = regime_eval(clip, dino)

    def agg(regime, key):
        vals = [rows[k][key] for k in rows if k[1] == regime and not np.isnan(rows[k].get(key, np.nan)) if isinstance(rows[k][key], (int, float))]
        return float(np.mean(vals)) if vals else float('nan')

    out = {"experiment": "LOCO logical/structural complementarity",
           "per_category": {f"{c}|{r}": v for (c, r), v in rows.items()},
           "summary": {}}

    print("=" * 88)
    print("  LOCO-1/3: CLIP vs DINOv2 by anomaly regime (image AUROC), and 2x2 map")
    print("=" * 88)
    print(f"  {'Category':<20} {'Regime':<11} {'CLIP':>6} {'DINOv2':>7} {'best':>6} {'Fuse.65':>8} {'a*':>5} {'Oracle':>7}")
    for cat in CATS:
        for regime in ['structural', 'logical']:
            if (cat, regime) in rows:
                v = rows[(cat, regime)]
                best = max(v['clip'], v['dino'])
                print(f"  {cat:<20} {regime:<11} {v['clip']:>6.1f} {v['dino']:>7.1f} {best:>6.1f} "
                      f"{v['fused065']:>8.1f} {v['oracle_alpha']:>5} {v['oracle']:>7.1f}")
    print("  " + "-" * 80)
    for regime in ['structural', 'logical']:
        cm, dm = agg(regime, 'clip'), agg(regime, 'dino')
        fm, om = agg(regime, 'fused065'), agg(regime, 'oracle')
        a_list = [rows[k]['oracle_alpha'] for k in rows if k[1] == regime]
        bestm = float(np.mean([max(rows[k]['clip'], rows[k]['dino']) for k in rows if k[1] == regime]))
        out["summary"][regime] = dict(clip_mean=round(cm, 1), dino_mean=round(dm, 1),
                                      best_single_mean=round(bestm, 1), fused065_mean=round(fm, 1),
                                      oracle_mean=round(om, 1), mean_oracle_alpha=round(float(np.mean(a_list)), 3),
                                      fusion_gain_vs_best=round(fm - bestm, 2))
        print(f"  {'MEAN':<20} {regime:<11} {cm:>6.1f} {dm:>7.1f} {bestm:>6.1f} {fm:>8.1f} "
              f"{np.mean(a_list):>5.2f} {om:>7.1f}   gain={fm-bestm:+.2f}")

    with open(os.path.join(os.path.dirname(BASE), "results", "loco_complementarity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Saved: results/loco_complementarity.json")

    s = out["summary"]
    print("\n  --- verdict ---")
    if 'structural' in s and 'logical' in s:
        print(f"  CLIP: structural {s['structural']['clip_mean']} vs logical {s['logical']['clip_mean']}")
        print(f"  DINOv2: structural {s['structural']['dino_mean']} vs logical {s['logical']['dino_mean']}")
        print(f"  optimal alpha: structural {s['structural']['mean_oracle_alpha']} vs logical {s['logical']['mean_oracle_alpha']} "
              f"(regime-specific?)")
        print(f"  fusion gain vs best single: structural {s['structural']['fusion_gain_vs_best']:+} / "
              f"logical {s['logical']['fusion_gain_vs_best']:+} pp")


if __name__ == "__main__":
    main()
