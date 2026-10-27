"""Cluster-robust (dataset-stratified) bootstrap of the headline CIs."""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seed_avg_fusion_metrics import load_all_datasets
from leave_one_dataset_out_alpha import build_curves, global_alpha, oracle, ALPHAS, BASE_DIR
import practical_normalization as pn

RES = os.path.join(os.path.dirname(BASE_DIR), "results")
NB = 10000
RNG = np.random.RandomState(20260628)


def per_category_table():
    """Assemble {cat: (dataset, reversal, fixed065_regret, transfer_regret, calib_regret)}."""
    curves, cat2ds = build_curves(load_all_datasets(0))
    cats = list(curves)
    orc = {c: oracle(curves, c)[1] for c in cats}
    fixed065 = {c: orc[c] - curves[c][0.65] for c in cats}
    names = sorted(set(cat2ds.values()))
    transfer = {}
    for held in names:
        aT, _ = global_alpha(curves, [c for c in cats if cat2ds[c] != held])
        for c in cats:
            if cat2ds[c] == held:
                transfer[c] = orc[c] - curves[c][aT]

    bst = json.load(open(os.path.join(RES, "bootstrap_stability_table.json"), encoding="utf-8"))
    reversal = {r["category"]: int(r["reversal"]) for r in bst["categories"]}

    cc = pn.aggregate(pn.load_30())["calib_zscore"]
    glob_a = max(pn.ALPHAS, key=lambda a: np.mean([cc[c][a] for c in cc]))
    calib = {c: (max(cc[c].values()) - cc[c][glob_a]) for c in cc}

    table = {}
    for c in cats:
        if c in reversal and c in calib:
            table[c] = dict(dataset=cat2ds[c], reversal=reversal[c],
                            fixed065=fixed065[c], transfer=transfer[c], calib=calib[c])
    missing = [c for c in cats if c not in table]
    return table, cat2ds, missing


def flat_ci(vals, stat=np.mean):
    v = np.asarray(vals, float)
    s = [stat(v[RNG.choice(len(v), len(v), replace=True)]) for _ in range(NB)]
    return float(stat(v)), [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def cluster_ci(vals_by_ds, stat=np.mean):
    """Two-stage: resample datasets w/ replacement, then categories within each drawn dataset."""
    dsn = list(vals_by_ds)
    point = stat(np.concatenate([vals_by_ds[d] for d in dsn]))
    s = []
    for _ in range(NB):
        chosen = RNG.choice(len(dsn), len(dsn), replace=True)
        samp = []
        for di in chosen:
            arr = vals_by_ds[dsn[di]]
            samp.append(arr[RNG.choice(len(arr), len(arr), replace=True)])
        s.append(stat(np.concatenate(samp)))
    return float(point), [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def main():
    table, cat2ds, missing = per_category_table()
    cats = list(table)
    print(f"Assembled {len(cats)} categories" + (f" (missing {missing})" if missing else ""))
    metrics = ["reversal", "fixed065", "transfer", "calib"]
    label = {"reversal": "reversal_rate", "fixed065": "fixed_065_regret_pp",
             "transfer": "lodo_transfer_regret_pp", "calib": "calib_zscore_regret_pp"}

    out = {"experiment": "cluster-robust (dataset-stratified) bootstrap of headline CIs",
           "n_categories": len(cats), "n_clusters": 3, "n_boot": NB,
           "note": ("seed-0 per-category values; flat = i.i.d. category bootstrap (naive), "
                    "cluster_robust = two-stage dataset->category bootstrap. With only 3 dataset "
                    "clusters the cluster-robust CI is coarse/wide, "
                    "reflecting the dataset cluster structure."),
           "metrics": {}}
    print(f"\n  {'metric':<26}{'point':>8}   {'flat 95% CI':>20}   {'cluster-robust 95% CI':>22}")
    for m in metrics:
        by_cat = {c: table[c][m] for c in cats}
        by_ds = {}
        for c in cats:
            by_ds.setdefault(cat2ds[c], []).append(table[c][m])
        by_ds = {d: np.asarray(v, float) for d, v in by_ds.items()}
        pf, cf = flat_ci(list(by_cat.values()))
        pc, cc_ = cluster_ci(by_ds)
        out["metrics"][label[m]] = dict(point=round(pf, 4),
                                        flat_ci95=[round(cf[0], 4), round(cf[1], 4)],
                                        cluster_robust_ci95=[round(cc_[0], 4), round(cc_[1], 4)],
                                        flat_width=round(cf[1] - cf[0], 4),
                                        cluster_robust_width=round(cc_[1] - cc_[0], 4),
                                        width_inflation=round((cc_[1] - cc_[0]) / (cf[1] - cf[0]), 2) if cf[1] - cf[0] > 1e-9 else None)
        print(f"  {label[m]:<26}{pf:>8.3f}   [{cf[0]:>7.3f},{cf[1]:>7.3f}]   [{cc_[0]:>9.3f},{cc_[1]:>9.3f}]")
    json.dump(out, open(os.path.join(RES, "cluster_bootstrap_headline_ci.json"), "w"), indent=2)
    print("\n  Saved: results/cluster_bootstrap_headline_ci.json")


if __name__ == "__main__":
    main()
