"""Taxonomy LOO directional accuracy + majority-class baseline on."""

import os
import sys
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_fusion import load_anomalyclip_scores, align_samples
import multiseed_aggregate as msa

BASE = os.path.dirname(os.path.abspath(__file__))
SEEDS = msa.SEEDS


def build_by_seed():
    """seed -> {cat: oracle_alpha} (0.01 grid, min-max, image level), via the
    exact multiseed_aggregate machinery."""
    clip_data = {ds: load_anomalyclip_scores(p) for ds, p in msa.CLIP.items()}
    by_seed = {}
    cat_ds = {}
    for s in SEEDS:
        merged = {}
        for ds in ["MVTec", "VisA", "BTAD"]:
            aligned = align_samples(clip_data[ds], msa.load_dino_seed(ds, s))
            m = msa.per_cat_metrics(aligned)
            for c in m:
                cat_ds[c] = ds
            merged.update(m)
        by_seed[s] = {c: merged[c]["oracle_alpha"] for c in merged}
    return by_seed, cat_ds


def loo_directional(alpha, types):
    """Leave-one-category-out type-based directional accuracy + majority baseline.

    alpha : {cat: oracle_alpha*}      types : {cat: 'texture'|'object'}
    For each held-out cat: predict DINOv2 if the mean alpha* of the OTHER same-type
    categories exceeds 0.5, else CLIP; compare to the cat's own preference
    (alpha*>0.5 => DINOv2). Majority-class baseline = always predict DINOv2,
    correct on exactly the DINOv2-preferred categories.
    """
    cats = list(alpha.keys())
    n = len(cats)
    n_correct = 0
    for c in cats:
        others = [alpha[o] for o in cats if o != c and types[o] == types[c]]
        pred = "dino" if (others and np.mean(others) > 0.5) else "clip"
        true = "dino" if alpha[c] > 0.5 else "clip"
        n_correct += (pred == true)
    n_dino_pref = sum(1 for c in cats if alpha[c] > 0.5)  # majority-baseline correct count
    return n_correct, n_dino_pref, n


def group_mean_alpha(alpha, types):
    out = {}
    for grp in ("texture", "object"):
        vals = [alpha[c] for c in alpha if types[c] == grp]
        out[grp] = dict(mean=round(float(np.mean(vals)), 3),
                        sd=round(float(np.std(vals, ddof=1)), 3), n=len(vals))
    return out


def ci95(vals):
    v = np.array(vals, dtype=float)
    m, sd = float(v.mean()), float(v.std(ddof=1))
    half = 1.96 * sd / np.sqrt(len(v))
    return dict(mean=round(m, 3), sd=round(sd, 3),
                ci95=[round(m - half, 3), round(m + half, 3)],
                range=[round(float(v.min()), 1), round(float(v.max()), 1)])


def binom_one_sided(successes, n, p0):
    """One-sided binomial P(X >= successes | p0) — matches the manuscript's
    'taxonomy beats the majority baseline?' test framing (H0: p = baseline rate)."""
    return float(stats.binomtest(successes, n, p0, alternative="greater").pvalue)


def main():
    by_seed, cat_ds = build_by_seed()
    cats = sorted(by_seed[0].keys())
    tax = json.load(open(os.path.join(BASE, "category_taxonomy_results.json")))["taxonomy"]
    types = {c: tax[c]["type"] for c in cats}
    n = len(cats)
    assert n == 30, f"expected 30 categories, got {n}"
    print(f"{n} categories; textures={sum(1 for c in cats if types[c]=='texture')}, "
          f"objects={sum(1 for c in cats if types[c]=='object')}")

    # avg-alpha per category
    avg_alpha = {c: float(np.mean([by_seed[s][c] for s in SEEDS])) for c in cats}

    # (1) per-seed
    per_seed_acc, per_seed_base = [], []
    for s in SEEDS:
        nc, nd, _ = loo_directional(by_seed[s], types)
        per_seed_acc.append(nc)
        per_seed_base.append(nd)

    # (2) avg-alpha point estimate
    nc_avg, nd_avg, _ = loo_directional(avg_alpha, types)

    # (3) seed-0 anchor (this pipeline)
    nc0, nd0, _ = loo_directional(by_seed[0], types)

    # significance of taxonomy vs majority baseline (avg-alpha operationalization)
    p_avg = binom_one_sided(nc_avg, n, nd_avg / n)
    p_seed0 = binom_one_sided(nc0, n, nd0 / n)

    acc_ci = ci95(per_seed_acc)
    base_ci = ci95(per_seed_base)

    print("\n=== LOO type-based directional accuracy & majority-class baseline ===")
    print(f"(1) per-seed  : accuracy  {acc_ci['mean']:.1f}/30  CI{acc_ci['ci95']}  "
          f"({acc_ci['mean']/n*100:.0f}%)   range {acc_ci['range']}")
    print(f"               baseline  {base_ci['mean']:.1f}/30  CI{base_ci['ci95']}  "
          f"({base_ci['mean']/n*100:.0f}%)   range {base_ci['range']}")
    print(f"(2) avg-alpha : accuracy  {nc_avg}/30 ({nc_avg/n*100:.0f}%)   "
          f"baseline {nd_avg}/30 ({nd_avg/n*100:.0f}%)   "
          f"margin {nc_avg-nd_avg:+d}   one-sided binomial p={p_avg:.3f}")
    print(f"(3) seed-0    : accuracy  {nc0}/30 ({nc0/n*100:.0f}%)   "
          f"baseline {nd0}/30 ({nd0/n*100:.0f}%)   margin {nc0-nd0:+d}   p={p_seed0:.3f}")

    gm_avg = group_mean_alpha(avg_alpha, types)
    gm_seed0 = group_mean_alpha(by_seed[0], types)
    print("\n=== group-mean oracle alpha* (texture vs object) ===")
    print(f"  avg-alpha : texture {gm_avg['texture']['mean']:.3f} (n={gm_avg['texture']['n']}), "
          f"object {gm_avg['object']['mean']:.3f} (n={gm_avg['object']['n']})")
    print(f"  seed-0    : texture {gm_seed0['texture']['mean']:.3f}, object {gm_seed0['object']['mean']:.3f}")

    out = {
        "experiment": "Step C: taxonomy LOO directional accuracy on seed-averaged alpha*",
        "n_seeds": len(SEEDS), "n_categories": n,
        "convention_note": "alpha* = image-level oracle on the 0.01 grid, per-category min-max "
                           "(identical machinery to multiseed_aggregate E1/E2/E3).",
        "per_seed": {
            "directional_accuracy_count": acc_ci,
            "majority_baseline_count": base_ci,
            "raw_accuracy_by_seed": per_seed_acc,
            "raw_baseline_by_seed": per_seed_base,
        },
        "avg_alpha": {
            "directional_accuracy": {"count": nc_avg, "pct": round(nc_avg / n * 100, 1)},
            "majority_baseline": {"count": nd_avg, "pct": round(nd_avg / n * 100, 1)},
            "margin": nc_avg - nd_avg,
            "binomial_p_one_sided": round(p_avg, 4),
            "group_mean_alpha": gm_avg,
        },
        "seed0_anchor": {
            "directional_accuracy": {"count": nc0, "pct": round(nc0 / n * 100, 1)},
            "majority_baseline": {"count": nd0, "pct": round(nd0 / n * 100, 1)},
            "margin": nc0 - nd0,
            "binomial_p_one_sided": round(p_seed0, 4),
            "group_mean_alpha": gm_seed0,
        },
        "avg_alpha_per_category": {c: round(avg_alpha[c], 3) for c in cats},
        "type_per_category": types,
    }
    outpath = os.path.join(os.path.dirname(BASE), "results", "taxonomy_multiseed_directional.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
