"""transfer fuse dominance."""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leave_one_dataset_out_alpha import load_all_datasets, build_curves, oracle, ALPHAS, BASE_DIR


def global_alpha_on(curves, cats):
    best_a, best_m = 0.5, -1.0
    for a in ALPHAS:
        m = np.mean([curves[c][a] for c in cats])
        if m > best_m:
            best_m, best_a = m, a
    return best_a


def main():
    datasets = load_all_datasets()
    curves, cat2ds = build_curves(datasets)
    cats = list(curves.keys())
    ds_names = list(datasets.keys())
    by_ds = {ds: [c for c in cats if cat2ds[c] == ds] for ds in ds_names}
    oracle_auroc = {c: oracle(curves, c)[1] for c in cats}

    def clip(c): return curves[c][0.0]
    def dino(c): return curves[c][1.0]

    per_fold = {}
    pooled = {"fused_T": [], "floor_dino": [], "floor_clip": [], "ceiling": [], "best_single_percat": []}
    for held in ds_names:
        train = [c for c in cats if cat2ds[c] != held]
        a_T = global_alpha_on(curves, train)
        hc = by_ds[held]
        fused = np.mean([curves[c][a_T] for c in hc])
        dn = np.mean([dino(c) for c in hc])
        cl = np.mean([clip(c) for c in hc])
        orc = np.mean([oracle_auroc[c] for c in hc])
        bsp = np.mean([max(clip(c), dino(c)) for c in hc])
        decide = fused - dn
        optimize = orc - fused
        per_fold[held] = {
            "transferred_alpha": a_T, "n": len(hc),
            "floor_dino_only": round(float(dn), 3),
            "fused_transfer": round(float(fused), 3),
            "ceiling_oracle": round(float(orc), 3),
            "value_decide_to_fuse_pp": round(float(decide), 3),
            "value_optimize_weight_pp": round(float(optimize), 3),
            "decide_share_pct": round(float(100 * decide / (orc - dn)), 1) if orc > dn else None,
        }
        for c in hc:
            pooled["fused_T"].append(curves[c][a_T]); pooled["floor_dino"].append(dino(c))
            pooled["floor_clip"].append(clip(c)); pooled["ceiling"].append(oracle_auroc[c])
            pooled["best_single_percat"].append(max(clip(c), dino(c)))

    P = {k: float(np.mean(v)) for k, v in pooled.items()}
    decide = P["fused_T"] - P["floor_dino"]
    optimize = P["ceiling"] - P["fused_T"]
    decide_share = 100 * decide / (P["ceiling"] - P["floor_dino"])

    out = {
        "experiment": "decision-to-fuse dominance under LODO transfer",
        "n_categories": len(cats),
        "pooled": {
            "floor_dino_only": round(P["floor_dino"], 3),
            "floor_clip_only": round(P["floor_clip"], 3),
            "fused_transfer": round(P["fused_T"], 3),
            "ceiling_oracle": round(P["ceiling"], 3),
            "value_decide_to_fuse_pp": round(decide, 3),
            "value_optimize_weight_pp": round(optimize, 3),
            "decide_to_fuse_share_pct": round(decide_share, 1),
            "decide_over_optimize_ratio": round(decide / optimize, 1) if optimize > 0 else None,
        },
        "per_fold": per_fold,
    }
    with open(os.path.join(os.path.dirname(BASE_DIR), "results", "transfer_fuse_dominance.json"), 'w') as f:
        json.dump(out, f, indent=2)

    print("=" * 70)
    print("  DECISION-TO-FUSE DOMINANCE UNDER TRANSFER")
    print("=" * 70)
    print(f"  POOLED (30 cats, label-free transfer):")
    print(f"    floor (DINOv2-only)        {P['floor_dino']:.3f}")
    print(f"    fused @ transferred alpha  {P['fused_T']:.3f}")
    print(f"    ceiling (per-cat oracle)   {P['ceiling']:.3f}")
    print(f"    value of DECIDING to fuse  +{decide:.3f} pp  (zero held-out labels)")
    print(f"    value of OPTIMIZING weight +{optimize:.3f} pp  (needs per-cat labels)")
    print(f"    -> decision-to-fuse share  {decide_share:.1f}%   (ratio {decide/optimize:.1f} : 1)")
    print(f"\n  {'Held-out':<8} {'aT':>5} {'decide(pp)':>11} {'optimize(pp)':>13} {'decide share':>13}")
    for held in ds_names:
        f = per_fold[held]
        print(f"  {held:<8} {f['transferred_alpha']:>5} {f['value_decide_to_fuse_pp']:>11} "
              f"{f['value_optimize_weight_pp']:>13} {str(f['decide_share_pct'])+'%':>13}")
    print("\n  Saved: results/transfer_fuse_dominance.json")


if __name__ == "__main__":
    main()
