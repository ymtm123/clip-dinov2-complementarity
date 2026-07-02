"""
T3 §4a deliverable: pixel-level MULTI-SEED analysis (the definitive Contribution-2 / D verdict).
================================================================================================

run_multiseed_maps.py generates the seed-1..9 pixel anomaly maps (seed-0 already on disk). This
script consumes all 10 seeds to deliver what T1/§3b could only do for seed-0:

  1. Seed-AVERAGED pixel oracle alpha* per category (image-side CLIP maps are seed-independent;
     DINOv2 maps are seed-specific, so the pixel alpha* varies by seed).
  2. Seed-ROBUST image-vs-pixel reversal RATE with a 95% CI over the 10 DINO seeds (the central
     image-vs-pixel claim, previously seed-0 only).
  3. DEFINITIVE association-D verdict: re-run the within-dataset permutation test for
     reversal->fusion-gain using the SEED-AVERAGED pixel alpha (replaces the hybrid seed-avg-image
     + seed-0-pixel D of hierarchical_stats_seedavg, n_reversal hybrid p=0.034).

Reuses validated machinery: pixel_seed0_metrics (build_category_arrays / category_curve / ALPHAS,
256x256 maps, alpha-0.02 grid, per-cat min-max, unbiased NEG_CAP) for the pixel curves;
hierarchical_stats_seedavg.seed_avg_table for the image-side seed-averaged alpha* + fusion gain;
hierarchical_stats.perm_twogroup for the permutation test.

Run with the effad (or any numpy/scipy/sklearn/PIL/cv2) python AFTER run_multiseed_maps.py:
  python src/pixel_multiseed_analysis.py
"""
import os
import sys
import json
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import pixel_seed0_metrics as PS
from pixel_fusion import load_anomalyclip_maps, load_anomalydino_maps, align_maps
from hierarchical_stats import perm_twogroup
from hierarchical_stats_seedavg import seed_avg_table

RES = os.path.join(os.path.dirname(BASE), "results")
D9 = "data"
SEEDS = list(range(10))
AD = os.path.join(BASE, "AnomalyDINO")
DSNAME = {"mvtec": "MVTec AD", "visa": "VisA", "btad": "BTAD"}


def _dino_dirs(ds):
    if ds == "mvtec":
        return [os.path.join(AD, "results_MVTec", "dinov2_vitl14_448", f"1-shot_preprocess={s}")
                for s in ("agnostic_weak_cats", "agnostic_strong_cats")]
    if ds == "visa":
        return [os.path.join(AD, "results_VisA", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_all")]
    return [os.path.join(AD, "results_BTAD", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_btad_all")]


def _load_dino_seed(results_dir, seed):
    """Seed-parameterised copy of pixel_fusion.load_anomalydino_maps (which hardcodes seed=0)."""
    from collections import defaultdict
    data = defaultdict(list)
    am = os.path.join(results_dir, "anomaly_maps", f"seed={seed}")
    if not os.path.isdir(am):
        return {}
    for obj in sorted(os.listdir(am)):
        td = os.path.join(am, obj, "test")
        if not os.path.isdir(td):
            continue
        for atype in sorted(os.listdir(td)):
            ad = os.path.join(td, atype)
            if not os.path.isdir(ad):
                continue
            for f in sorted(os.listdir(ad)):
                if f.endswith(".npy"):
                    data[obj].append(dict(map=np.load(os.path.join(ad, f)).astype(np.float32),
                                          anomaly_type=atype, img_id=os.path.splitext(f)[0],
                                          is_normal=(atype in ("good", "ok"))))
    return dict(data)


CLIP_ROOT = {"mvtec": f"{D9}/MVTec_AD", "visa": f"{D9}/VisA/VisA_pytorch/1cls",
             "btad": f"{D9}/BTAD/BTech_Dataset_transformed"}


def pixel_alpha_for_seed(seed):
    """{category: pixel oracle alpha*} for one DINO seed (seed-indep CLIP maps + seed-s DINO maps)."""
    out, cov = {}, {}
    for ds in ("mvtec", "visa", "btad"):
        clip = load_anomalyclip_maps(os.path.join(BASE, "AnomalyCLIP", "results", f"maps_{ds}"), CLIP_ROOT[ds], ds)
        dino = {}
        for d in _dino_dirs(ds):
            dino.update(_load_dino_seed(d, seed))
        if not dino:
            continue
        aligned = align_maps(clip, dino, target_size=(256, 256), dino_sigma=4)
        cats = PS.build_category_arrays(aligned, ds)
        for c, rec in cats.items():
            cur = PS.category_curve(rec, PS.ALPHAS)
            out[c] = max(PS.ALPHAS, key=lambda a: cur[a])
            cov[c] = DSNAME[ds]
    return out, cov


def ci95(xs):
    a = np.asarray(xs, float); m = a.mean()
    if len(a) < 2:
        return float(m), float(m), float(m)
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(m), float(m - h), float(m + h)


def main():
    print("Computing per-seed pixel oracle alpha (seeds 0..9) ...")
    alpha_by_seed, cat_ds = {}, {}
    avail = []
    for s in SEEDS:
        a, cov = pixel_alpha_for_seed(s)
        if not a:
            print(f"  seed {s}: NO maps found (run run_multiseed_maps.py for this seed) — skipping")
            continue
        alpha_by_seed[s] = a; cat_ds.update(cov); avail.append(s)
        print(f"  seed {s}: pixel alpha* for {len(a)} categories")
    if not avail:
        print("No pixel maps for any seed; aborting."); return

    cats = sorted(set.intersection(*[set(alpha_by_seed[s]) for s in avail]))
    seed_avg_pix = {c: float(np.mean([alpha_by_seed[s][c] for s in avail])) for c in cats}

    # image-side seed-averaged alpha* + fusion gain (the manuscript convention), reused verbatim
    img_df = seed_avg_table().set_index("category")
    img_alpha = {c: float(img_df.loc[c, "alpha_img"]) for c in cats if c in img_df.index}
    fusion_gain = {c: float(img_df.loc[c, "fusion_gain"]) for c in cats if c in img_df.index}

    def reversal(ai, ap):
        return int((ai > 0.5 and ap < 0.5) or (ai < 0.5 and ap > 0.5))

    # (2) seed-robust reversal RATE: fixed seed-avg image alpha vs each seed's pixel alpha
    common = [c for c in cats if c in img_alpha]
    rate_by_seed = []
    for s in avail:
        revs = [reversal(img_alpha[c], alpha_by_seed[s][c]) for c in common]
        rate_by_seed.append(float(np.mean(revs)))
    rate_m, rate_lo, rate_hi = ci95(rate_by_seed)

    # (3) DEFINITIVE D: seed-avg image alpha vs seed-AVG pixel alpha
    import pandas as pd
    rows = [dict(category=c, dataset=cat_ds[c], fusion_gain=fusion_gain[c],
                 alpha_img=img_alpha[c], alpha_pix=seed_avg_pix[c],
                 reversal=reversal(img_alpha[c], seed_avg_pix[c])) for c in common]
    dfr = pd.DataFrame(rows)
    rev = dfr[dfr.reversal == 1].fusion_gain; non = dfr[dfr.reversal == 0].fusion_gain
    _, mw_p = stats.mannwhitneyu(rev, non, alternative="two-sided") if len(rev) and len(non) else (None, None)
    _, perm_p, nds = perm_twogroup(dfr, "fusion_gain", "reversal")

    out = dict(
        experiment="T3 §4a pixel multi-seed: seed-averaged pixel alpha*, seed-robust reversal rate, definitive D",
        seeds_available=avail, n_categories=len(common),
        note="image alpha* and fusion gain are seed-averaged (hierarchical_stats_seedavg convention); "
             "pixel alpha* is now also seed-averaged over the available DINO seeds (was seed-0 in T1/§3b). "
             "Definitive D replaces the hybrid seed-avg-image + seed-0-pixel reversal of "
             "hierarchical_stats_seedavg (which gave within-ds perm p=0.034, n_reversal=12).",
        reversal_rate_seed_robust=dict(mean=round(rate_m, 4), ci95=[round(rate_lo, 4), round(rate_hi, 4)],
                                       per_seed=[round(r, 4) for r in rate_by_seed],
                                       n_reversal_seedavg=int(dfr.reversal.sum())),
        D_reversal_to_gain=dict(n_reversal=int(len(rev)), n_nonreversal=int(len(non)),
                                reversal_mean_gain=round(float(rev.mean()), 3) if len(rev) else None,
                                nonreversal_mean_gain=round(float(non.mean()), 3) if len(non) else None,
                                pooled_mannwhitney_p=round(float(mw_p), 4) if mw_p is not None else None,
                                perm_within_dataset_p=round(float(perm_p), 4) if not np.isnan(perm_p) else None,
                                n_datasets=nds),
        seed_avg_pixel_alpha={c: round(seed_avg_pix[c], 3) for c in common},
        image_alpha={c: round(img_alpha[c], 3) for c in common},
    )
    json.dump(out, open(os.path.join(RES, "pixel_multiseed_analysis.json"), "w"), indent=2)
    print("\n" + "=" * 80)
    print(f"  PIXEL MULTI-SEED ({len(avail)} seeds, {len(common)} cats)")
    print("=" * 80)
    r = out["reversal_rate_seed_robust"]
    print(f"  seed-robust reversal RATE: {r['mean']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]  "
          f"(seed-avg n_reversal={r['n_reversal_seedavg']}/{len(common)}); per-seed {r['per_seed']}")
    d = out["D_reversal_to_gain"]
    print(f"  DEFINITIVE D (reversal->gain, seed-avg pixel): within-ds perm p={d['perm_within_dataset_p']} "
          f"(n_rev={d['n_reversal']}, rev gain {d['reversal_mean_gain']} vs non {d['nonreversal_mean_gain']})")
    print(f"\n  Saved: results/pixel_multiseed_analysis.json")


if __name__ == "__main__":
    main()
