"""C7 / : analyze the 6-backbone same-kNN panel."""
import os
import json
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
J = os.path.join(os.path.dirname(BASE), "results", "backbone_panel_knn.json")
AXIS = {"CLIP": "contrastive-language", "SigLIP": "contrastive-language",
        "DINOv2": "ssl-spatial", "DINOv2-reg": "ssl-spatial", "MAE": "ssl-spatial",
        "WRN50": "supervised"}
RNG = np.random.RandomState(42)


def main():
    if not os.path.exists(J):
        print(f"[WAIT] {J} not found yet (panel run in progress).")
        return
    M = json.load(open(J))
    backbones = [b for b in M if not b.startswith("_")]
    cats = sorted([c for c in M[backbones[0]] if not c.startswith("_")
                   and all(M[b].get(c) is not None for b in backbones)])
    A = np.array([[M[b][c] for c in cats] for b in backbones])  # [6, n_cat]
    n = len(backbones)

    corr = np.corrcoef(A)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    within = [corr[i, j] for (i, j) in pairs if AXIS[backbones[i]] == AXIS[backbones[j]]]
    cross = [corr[i, j] for (i, j) in pairs if AXIS[backbones[i]] != AXIS[backbones[j]]]
    obs = np.mean(within) - np.mean(cross)

    labels = [AXIS[b] for b in backbones]
    cnt = 0
    for _ in range(20000):
        perm = RNG.permutation(labels)
        w = [corr[i, j] for (i, j) in pairs if perm[i] == perm[j]]
        c = [corr[i, j] for (i, j) in pairs if perm[i] != perm[j]]
        if w and c and (np.mean(w) - np.mean(c)) >= obs:
            cnt += 1
    p_perm = (cnt + 1) / 20001

    win_axis = {"contrastive-language": 0, "ssl-spatial": 0, "supervised": 0}
    for k in range(len(cats)):
        wb = backbones[int(np.argmax(A[:, k]))]
        win_axis[AXIS[wb]] += 1

    ci, di = backbones.index("CLIP"), backbones.index("DINOv2")
    clip_wins = int(np.sum(A[ci] > A[di]))

    out = {
        "experiment": "6-backbone same-kNN panel",
        "backbones": backbones, "n_categories": len(cats),
        "mean_auroc_per_backbone": {b: round(float(A[i].mean()), 2) for i, b in enumerate(backbones)},
        "corr_matrix": {backbones[i]: {backbones[j]: round(float(corr[i, j]), 3) for j in range(n)} for i in range(n)},
        "within_axis_mean_corr": round(float(np.mean(within)), 3),
        "cross_axis_mean_corr": round(float(np.mean(cross)), 3),
        "within_minus_cross": round(float(obs), 3),
        "permutation_p": round(float(p_perm), 4),
        "winner_by_axis": win_axis,
        "clip_vs_dinov2_clip_wins": f"{clip_wins}/{len(cats)}",
    }
    with open(os.path.join(os.path.dirname(BASE), "results", "backbone_panel_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 78)
    print("  6-BACKBONE SAME-kNN PANEL")
    print("=" * 78)
    print("  mean AUROC: " + ", ".join(f"{b}={A[i].mean():.1f}" for i, b in enumerate(backbones)))
    print(f"\n  within-axis mean corr = {np.mean(within):.3f}  vs  cross-axis = {np.mean(cross):.3f}")
    print(f"  within - cross = {obs:.3f}  (permutation p = {p_perm:.4f})")
    print(f"  per-category winner by axis: {win_axis}")
    print(f"  embedded CLIP vs DINOv2: CLIP wins {clip_wins}/{len(cats)}")
    print(f"\n  Saved: results/backbone_panel_analysis.json")


if __name__ == "__main__":
    main()
