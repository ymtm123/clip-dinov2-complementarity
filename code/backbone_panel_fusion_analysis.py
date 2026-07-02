"""C7 / : 15-pair backbone-fusion MATRIX + paradigm analysis from the panel score dumps."""
import os
import sys
import json
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(BASE), "results")
DUMP = os.path.join(RES, "backbone_panel_scores")
PANEL = os.path.join(RES, "backbone_panel_full.json")
AXIS = {"CLIP": "contrastive-language", "SigLIP": "contrastive-language", "DINOv2": "ssl-spatial",
        "DINOv2-reg": "ssl-spatial", "MAE": "ssl-spatial", "WRN50": "supervised"}
ALPHAS = [round(i / 20, 2) for i in range(21)]


def label_of(relpath):
    return 0 if relpath.replace("\\", "/").split("/")[-2] in ("good", "ok") else 1


def mm(x):
    x = np.asarray(x, float); lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(x)


def main():
    backbones = [b for b in AXIS if os.path.exists(os.path.join(DUMP, f"{b}.json"))]
    if len(backbones) < 2:
        print(f"Need >=2 backbone dumps in {DUMP}; found {backbones}"); return
    dumps = {b: json.load(open(os.path.join(DUMP, f"{b}.json"))) for b in backbones}
    cats = sorted(set.intersection(*[set(dumps[b]) for b in backbones]))
    print(f"Backbones: {backbones} | common categories: {len(cats)}")

    # single-backbone seed-0 AUROC per cat
    single = {b: {} for b in backbones}
    for b in backbones:
        for ck in cats:
            d = dumps[b][ck]; keys = sorted(d)
            y = np.array([label_of(k) for k in keys]); s = np.array([d[k] for k in keys])
            single[b][ck] = roc_auc_score(y, s) * 100 if len(np.unique(y)) > 1 else np.nan

    pairs = list(itertools.combinations(backbones, 2))
    matrix = {}
    for a, b in pairs:
        eq_gains, or_gains = [], []
        for ck in cats:
            da, db = dumps[a][ck], dumps[b][ck]
            keys = sorted(set(da) & set(db))
            if len(keys) < 10:
                continue
            y = np.array([label_of(k) for k in keys])
            if len(np.unique(y)) < 2:
                continue
            na, nb = mm([da[k] for k in keys]), mm([db[k] for k in keys])
            best = max(roc_auc_score(y, na), roc_auc_score(y, nb)) * 100
            eq = roc_auc_score(y, 0.5 * na + 0.5 * nb) * 100
            orc = max(roc_auc_score(y, (1 - al) * na + al * nb) for al in ALPHAS) * 100
            eq_gains.append(eq - best); or_gains.append(orc - best)
        matrix[f"{a}x{b}"] = dict(
            cross_paradigm=AXIS[a] != AXIS[b],
            mean_equal_gain=round(float(np.mean(eq_gains)), 3), mean_oracle_gain=round(float(np.mean(or_gains)), 3),
            frac_equal_positive=round(float(np.mean([g > 0 for g in eq_gains])), 2), n_cats=len(eq_gains))

    cross = [m for m in matrix.values() if m["cross_paradigm"]]
    within = [m for m in matrix.values() if not m["cross_paradigm"]]
    out = dict(experiment="15-pair backbone-fusion matrix",
               n_backbones=len(backbones), n_pairs=len(pairs), n_categories=len(cats),
               note="equal_gain = AUROC@0.5 - best single (FIXED fusion, can be <0); oracle_gain = best-alpha "
                    "- best single (>=0). Cross-paradigm = the two backbones span different representational "
                    "axes (contrastive-language / ssl-spatial / supervised).",
               n_pairs_net_positive_equal=int(sum(m["mean_equal_gain"] > 0 for m in matrix.values())),
               cross_paradigm_mean_equal_gain=round(float(np.mean([m["mean_equal_gain"] for m in cross])), 3) if cross else None,
               within_paradigm_mean_equal_gain=round(float(np.mean([m["mean_equal_gain"] for m in within])), 3) if within else None,
               cross_paradigm_mean_oracle_gain=round(float(np.mean([m["mean_oracle_gain"] for m in cross])), 3) if cross else None,
               within_paradigm_mean_oracle_gain=round(float(np.mean([m["mean_oracle_gain"] for m in within])), 3) if within else None,
               pair_matrix=matrix)
    json.dump(out, open(os.path.join(RES, "backbone_panel_fusion_matrix.json"), "w"), indent=2)

    print(f"\n  15-pair fusion (equal alpha=0.5 gain vs best single):")
    print(f"  {'pair':<18}{'cross?':>7}{'eq_gain':>9}{'oracle':>8}{'%cats+':>8}")
    for k, m in sorted(matrix.items(), key=lambda x: -x[1]["mean_equal_gain"]):
        print(f"  {k:<18}{'Y' if m['cross_paradigm'] else 'n':>7}{m['mean_equal_gain']:>9.2f}{m['mean_oracle_gain']:>8.2f}{m['frac_equal_positive']:>8}")
    print(f"\n  pairs with net positive equal-fusion gain: {out['n_pairs_net_positive_equal']}/{len(pairs)}")
    print(f"  cross-paradigm mean equal gain {out['cross_paradigm_mean_equal_gain']} vs within {out['within_paradigm_mean_equal_gain']}")
    print(f"  cross-paradigm mean ORACLE gain {out['cross_paradigm_mean_oracle_gain']} vs within {out['within_paradigm_mean_oracle_gain']}")
    print(f"\n  Saved: results/backbone_panel_fusion_matrix.json")


if __name__ == "__main__":
    main()
