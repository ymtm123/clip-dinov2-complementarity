"""Cluster-robust bootstrap of the FIXED alpha=0.65 fusion gain, N=30 -> N=45."""
import os
import sys
import json
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from power_sensitivity_analysis import core_rows, dagm_rows, loco_structural_rows
from cluster_bootstrap_headline_ci import flat_ci, cluster_ci, NB

RES = os.path.join(os.path.dirname(BASE), "results")


def _stage(rows, label):
    vals = {i: r["fixed065_gain"] for i, r in enumerate(rows)}
    by_ds = {}
    for i, r in enumerate(rows):
        by_ds.setdefault(r["dataset"], []).append(r["fixed065_gain"])
    by_ds = {d: np.asarray(v, float) for d, v in by_ds.items()}
    pf, cf = flat_ci(list(vals.values()))
    pc, cc = cluster_ci(by_ds)
    return dict(label=label, N=len(rows), n_clusters=len(by_ds),
                point=round(pf, 4),
                flat_ci95=[round(cf[0], 4), round(cf[1], 4)], flat_width=round(cf[1] - cf[0], 4),
                cluster_robust_ci95=[round(cc[0], 4), round(cc[1], 4)], cluster_robust_width=round(cc[1] - cc[0], 4),
                per_cluster={d: dict(n=len(v), mean=round(float(v.mean()), 3)) for d, v in by_ds.items()})


def main():
    core = core_rows(); dagm = dagm_rows(); loco = loco_structural_rows()
    print(f"core={len(core)}  DAGM={len(dagm)}  LOCO-structural={len(loco)}  (n_boot={NB})")
    # N30 (3 clusters, starting point); N40 (4 clusters core+DAGM = same traditional-AD regime = the
    # same traditional-AD regime); N45 (5 clusters +LOCO = cross-regime, where LOCO's
    # heterogeneity widens the cluster-robust CI to include 0).
    stages = [_stage(core, "N30_3clusters_core")]
    if dagm:
        stages.append(_stage(core + dagm, "N40_4clusters_core+DAGM_sameRegime"))
    if dagm or loco:
        stages.append(_stage(core + dagm + loco, "N45_5clusters_+LOCO_crossRegime"))
    out = dict(experiment="cluster-robust bootstrap of fixed alpha=0.65 fusion gain, N=30->N=45",
               effect="fixed alpha=0.65 fusion gain (AUROC@0.65 - best single), seed-0, image-level",
               n_boot=NB,
               note="Cluster-robust = resample datasets then categories within. N=45 adds DAGM (10) + "
                    "MVTec-LOCO structural (5) as 2 extra clusters. LOCO is its own cluster so its "
                    "heterogeneity (negative fixed-fusion gain) is absorbed into the between-"
                    "dataset variance. 5 clusters give a less-coarse cluster CI than the 3-cluster case.",
               stages={s["label"]: s for s in stages})
    json.dump(out, open(os.path.join(RES, "cluster_bootstrap_fusion_N45.json"), "w"), indent=2)
    print(f"\n  {'stage':<18}{'N':>4}{'clusters':>9}{'point':>9}{'flat 95% CI':>22}{'cluster-robust 95% CI':>26}")
    for s in stages:
        print(f"  {s['label']:<18}{s['N']:>4}{s['n_clusters']:>9}{s['point']:>9.3f}"
              f"   [{s['flat_ci95'][0]:>6.3f},{s['flat_ci95'][1]:>6.3f}]"
              f"     [{s['cluster_robust_ci95'][0]:>7.3f},{s['cluster_robust_ci95'][1]:>7.3f}]")
        print(f"        per-cluster: {s['per_cluster']}")
    print(f"\n  Saved: results/cluster_bootstrap_fusion_N45.json")


if __name__ == "__main__":
    main()
