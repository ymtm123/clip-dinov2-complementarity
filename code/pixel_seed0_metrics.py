"""Pixel-level (seed-0) re-runs of the four image-level headline analyses."""
import os
import sys
import json
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import pandas as pd

try:  # Windows console defaults to cp932; force UTF-8 so unicode in prints never crashes the run
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from pixel_fusion import load_anomalyclip_maps, load_anomalydino_maps, align_maps
from hierarchical_stats import perm_paired_vs0, perm_twogroup, icc_oneway

RESULTS = os.path.join(os.path.dirname(BASE), "results")
D9 = "data"

ALPHAS = [round(i / 50.0, 2) for i in range(51)]
ALPHAS_CALIB = [round(i / 20.0, 2) for i in range(21)]
DSNAME = {"mvtec": "MVTec AD", "visa": "VisA", "btad": "BTAD"}
IMG_ALPHA_KEY = {"mvtec": "mvtec_image", "visa": "visa_image", "btad": "btad_image"}
RNG = np.random.RandomState(42)
NEG_CAP = 400_000   # uniform sub-sample cap on negative pixels per category (unbiased for AUROC)
CALIB_SPLITS = 10
CALIB_FRAC = 0.5
MIN_CALIB = 5       # match practical_normalization.py image-side calibration-size floor


# ----------------------------------------------------------------------------
# Load seed-0 aligned pixel maps for all three datasets
# ----------------------------------------------------------------------------
def load_dataset(ds):
    ad = os.path.join(BASE, "AnomalyDINO")
    clip_dir = os.path.join(BASE, "AnomalyCLIP", "results", f"maps_{ds}")
    if ds == "mvtec":
        root = f"{D9}/MVTec_AD"
        clip = load_anomalyclip_maps(clip_dir, root, "mvtec")
        dino = {}
        for sub in ("agnostic_weak_cats", "agnostic_strong_cats"):
            d = load_anomalydino_maps(os.path.join(ad, "results_MVTec", "dinov2_vitl14_448", f"1-shot_preprocess={sub}"))
            dino.update(d)
    elif ds == "visa":
        root = f"{D9}/VisA/VisA_pytorch/1cls"
        clip = load_anomalyclip_maps(clip_dir, root, "visa")
        dino = load_anomalydino_maps(os.path.join(ad, "results_VisA", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_all"))
    elif ds == "btad":
        root = f"{D9}/BTAD/BTech_Dataset_transformed"
        clip = load_anomalyclip_maps(clip_dir, root, "btad")
        dino = load_anomalydino_maps(os.path.join(ad, "results_BTAD", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_btad_all"))
    else:
        raise ValueError(ds)
    aligned = align_maps(clip, dino, target_size=(256, 256), dino_sigma=4)
    return aligned


def _minmax_pool(maps):
    allp = np.concatenate([m.ravel() for m in maps])
    lo, hi = float(allp.min()), float(allp.max())
    sc = hi - lo if hi - lo > 1e-10 else 1.0
    return [(m - lo) / sc for m in maps]


def build_category_arrays(aligned, ds):
    """Per category: transductively min-max-normalised, flattened CLIP/DINO pixel
    arrays + binary gt + per-map is_normal, with negatives capped (unbiased AUROC)."""
    cats = {}
    for obj, items in aligned.items():
        if not items:
            continue
        clip_maps = [it["clip_map"] for it in items]
        dino_maps = [it["dino_map"] for it in items]
        gt_masks = [(it["gt_mask"] > 0.5).astype(np.int8) for it in items]
        npos = int(sum(int(m.sum()) for m in gt_masks))
        nneg = int(sum(int((1 - m).sum()) for m in gt_masks))
        if npos == 0 or nneg == 0:
            continue
        clip_n = _minmax_pool(clip_maps)
        dino_n = _minmax_pool(dino_maps)
        cflat = np.concatenate([m.ravel() for m in clip_n]).astype(np.float32)
        dflat = np.concatenate([m.ravel() for m in dino_n]).astype(np.float32)
        yflat = np.concatenate([m.ravel() for m in gt_masks]).astype(np.int8)
        # uniform sub-sample of negatives (keeps all positives) -> unbiased AUROC, bounded cost
        neg_idx = np.where(yflat == 0)[0]
        if len(neg_idx) > NEG_CAP:
            keep_neg = RNG.choice(neg_idx, size=NEG_CAP, replace=False)
            pos_idx = np.where(yflat == 1)[0]
            sel = np.concatenate([pos_idx, keep_neg]); sel.sort()
            cflat, dflat, yflat = cflat[sel], dflat[sel], yflat[sel]
        cats[obj] = dict(dataset=DSNAME[ds], clip=cflat, dino=dflat, y=yflat,
                         n_pos=npos, n_neg=nneg)
    return cats


def auroc(y, s):
    return roc_auc_score(y, s) * 100.0


def category_curve(rec, alphas):
    c, d, y = rec["clip"], rec["dino"], rec["y"]
    return {a: auroc(y, (1 - a) * c + a * d) for a in alphas}


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def analyse_regret_band(curves, cat2ds):
    cats = list(curves)
    orc = {c: max(curves[c].values()) for c in cats}
    pooled = {a: float(np.mean([orc[c] - curves[c][a] for c in cats])) for a in ALPHAS}
    pmin_a = min(ALPHAS, key=lambda a: pooled[a]); pmin = pooled[pmin_a]
    band45 = [a for a in ALPHAS if pooled[a] <= pmin + 0.45]
    band20 = [a for a in ALPHAS if pooled[a] <= pmin + 0.20]
    optd = {}
    for ds in DSNAME.values():
        cs = [c for c in cats if cat2ds[c] == ds]
        if not cs:
            continue
        pc = {a: float(np.mean([orc[c] - curves[c][a] for c in cs])) for a in ALPHAS}
        optd[ds] = min(ALPHAS, key=lambda a: pc[a])
    # The step-0.02 grid cannot land on the manuscript's alpha=0.65 fixed weight; report the two
    # bracketing grid points and a linear interpolation, NOT a 0.64 value mislabeled as 0.65.
    reg064, reg066 = pooled[0.64], pooled[0.66]
    return dict(pooled_opt_alpha=pmin_a, pooled_min_regret=round(pmin, 4),
                regret_050=round(pooled[0.5], 4),
                regret_064=round(reg064, 4), regret_066=round(reg066, 4),
                regret_065_interp=round(0.5 * (reg064 + reg066), 4),
                regret_at_alpha={str(a): round(pooled[a], 4) for a in ALPHAS},
                band45=[min(band45), max(band45)], band20=[min(band20), max(band20)],
                per_dataset_optimum=optd)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def analyse_hierarchical(curves, cat2ds, tax, img_alpha):
    i50 = 0.5
    rows = []
    for c in curves:
        cur = curves[c]
        clip, dino, equal = cur[0.0], cur[1.0], cur[i50]
        best = max(clip, dino)
        ai = max(ALPHAS, key=lambda a: cur[a])           # pixel oracle alpha
        aimg = img_alpha.get(c)
        rows.append(dict(category=c, dataset=cat2ds[c], clip=clip, dino=dino, equal=equal,
                         best_single=best, dino_minus_clip=dino - clip, fusion_gain=equal - best,
                         alpha_pix=ai, alpha_img=aimg,
                         reversal=(None if aimg is None else int((ai > 0.5 and aimg < 0.5) or (ai < 0.5 and aimg > 0.5))),
                         type=tax.get(c, {}).get("type")))
    df = pd.DataFrame(rows)
    out = {}
    _, p = stats.wilcoxon(df.dino_minus_clip); _, pp = perm_paired_vs0(df, "dino_minus_clip"); icc = icc_oneway(df.dino_minus_clip, df.dataset)
    out["A_clip_vs_dino"] = dict(pooled_wilcoxon_p=round(float(p), 4), pooled_mean_diff=round(float(df.dino_minus_clip.mean()), 3),
                                 perm_within_dataset_p=round(float(pp), 4), between_dataset_var_share_icc=round(icc, 3))
    _, p = stats.wilcoxon(df.fusion_gain); _, pp = perm_paired_vs0(df, "fusion_gain"); icc = icc_oneway(df.fusion_gain, df.dataset)
    out["B_fusion_vs_best_single"] = dict(pooled_wilcoxon_p=round(float(p), 4), pooled_mean_gain=round(float(df.fusion_gain.mean()), 3),
                                          perm_within_dataset_p=round(float(pp), 4), between_dataset_var_share_icc=round(icc, 3))
    dft = df.dropna(subset=["type", "alpha_pix"]).copy(); dft["type_bin"] = (dft.type == "object").astype(int)
    obj = dft[dft.type == "object"].alpha_pix; tex = dft[dft.type == "texture"].alpha_pix
    _, p = stats.mannwhitneyu(obj, tex, alternative="two-sided"); _, pp, nds = perm_twogroup(dft, "alpha_pix", "type_bin"); icc = icc_oneway(dft.alpha_pix, dft.dataset)
    out["C_type_to_alpha"] = dict(pooled_mannwhitney_p=round(float(p), 4), object_mean_alpha=round(float(obj.mean()), 3),
                                  texture_mean_alpha=round(float(tex.mean()), 3), n_object=int(len(obj)), n_texture=int(len(tex)),
                                  perm_within_dataset_p=(None if np.isnan(pp) else round(float(pp), 4)),
                                  n_datasets_with_both_types=nds, between_dataset_var_share_icc=round(icc, 3))
    dfr = df.dropna(subset=["reversal"]).copy()
    rev = dfr[dfr.reversal == 1].fusion_gain; non = dfr[dfr.reversal == 0].fusion_gain
    if len(rev) and len(non):
        _, p = stats.mannwhitneyu(rev, non, alternative="two-sided"); _, pp, nds = perm_twogroup(dfr, "fusion_gain", "reversal")
        out["D_reversal_to_gain"] = dict(pooled_mannwhitney_p=round(float(p), 4), reversal_mean_gain=round(float(rev.mean()), 3),
                                         nonreversal_mean_gain=round(float(non.mean()), 3), n_reversal=int(len(rev)), n_nonreversal=int(len(non)),
                                         perm_within_dataset_p=(None if np.isnan(pp) else round(float(pp), 4)),
                                         note="reversal = image oracle alpha (taxonomy, seed-0) vs pixel oracle alpha (seed-0) disagree about which model wins")
    return out, df


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def _norm_apply(x, loc, sc):
    return (x - loc) / (sc if sc > 1e-10 else 1.0)


def analyse_calibration(cats_by_obj):
    """transductive_minmax (paper) vs calib_zscore (fit on NORMAL maps only).
    Per category, split the per-map normal set into calib/eval halves; eval set =
    held-out normal maps + all anomaly maps. Average over CALIB_SPLITS random splits."""
    # Four protocols mirroring practical_normalization.py (image side): two transductive
    # (params fit on the eval batch incl. anomalies = paper baseline) and two deployment-legal
    # (params fit on NORMAL calibration maps only).
    protocols = ["transductive_minmax", "transductive_zscore", "calib_zscore", "calib_robust_minmax"]
    cat_curves = {p: {} for p in protocols}
    for obj, items in cats_by_obj.items():
        # rebuild per-map structure with is_normal so we can split by image
        maps = items["_maps"]  # list of dict(clip, dino, y, is_normal) flattened per map
        normal_ix = [i for i, m in enumerate(maps) if m["is_normal"]]
        anom_ix = [i for i, m in enumerate(maps) if not m["is_normal"]]
        if len(normal_ix) < 4 or not anom_ix:
            continue
        k = max(MIN_CALIB, int(round(len(normal_ix) * CALIB_FRAC)))  # match image-side MIN_CALIB floor
        k = min(k, len(normal_ix) - 2)
        acc = {p: [] for p in protocols}
        for _ in range(CALIB_SPLITS):
            calib = list(RNG.choice(normal_ix, size=k, replace=False))
            eval_norm = [i for i in normal_ix if i not in set(calib)]
            eval_ix = eval_norm + anom_ix
            cal_c = np.concatenate([maps[i]["clip"] for i in calib])
            cal_d = np.concatenate([maps[i]["dino"] for i in calib])
            ev_c = np.concatenate([maps[i]["clip"] for i in eval_ix])
            ev_d = np.concatenate([maps[i]["dino"] for i in eval_ix])
            ev_y = np.concatenate([maps[i]["y"] for i in eval_ix])
            if len(np.unique(ev_y)) < 2:
                continue
            # uniform sub-sample of eval negatives (unbiased AUROC, bounded cost)
            neg = np.where(ev_y == 0)[0]
            if len(neg) > NEG_CAP:
                pos = np.where(ev_y == 1)[0]
                keep = np.concatenate([pos, RNG.choice(neg, size=NEG_CAP, replace=False)]); keep.sort()
                ev_c, ev_d, ev_y = ev_c[keep], ev_d[keep], ev_y[keep]
            for p in protocols:
                if p == "transductive_minmax":
                    cl = _norm_apply(ev_c, ev_c.min(), ev_c.max() - ev_c.min())
                    dn = _norm_apply(ev_d, ev_d.min(), ev_d.max() - ev_d.min())
                elif p == "transductive_zscore":
                    cl = _norm_apply(ev_c, ev_c.mean(), ev_c.std())
                    dn = _norm_apply(ev_d, ev_d.mean(), ev_d.std())
                elif p == "calib_zscore":
                    cl = _norm_apply(ev_c, cal_c.mean(), cal_c.std())
                    dn = _norm_apply(ev_d, cal_d.mean(), cal_d.std())
                else:  # calib_robust_minmax
                    clo, chi = np.percentile(cal_c, 1), np.percentile(cal_c, 99)
                    dlo, dhi = np.percentile(cal_d, 1), np.percentile(cal_d, 99)
                    cl = _norm_apply(ev_c, clo, chi - clo); dn = _norm_apply(ev_d, dlo, dhi - dlo)
                acc[p].append([auroc(ev_y, (1 - a) * cl + a * dn) for a in ALPHAS_CALIB])
        for p in protocols:
            if acc[p]:
                cat_curves[p][obj] = dict(zip(ALPHAS_CALIB, np.mean(acc[p], axis=0)))
    out = {}
    for p in protocols:
        cc = cat_curves[p]
        if not cc:
            continue
        glob_a = max(ALPHAS_CALIB, key=lambda a: np.mean([cc[c][a] for c in cc]))
        regret = float(np.mean([max(cc[c].values()) - cc[c][glob_a] for c in cc]))
        best_single = float(np.mean([max(cc[c][0.0], cc[c][1.0]) for c in cc]))
        fixed = float(np.mean([cc[c][glob_a] for c in cc]))
        out[p] = dict(global_alpha=glob_a, regret_pp=round(regret, 3),
                      fusion_gain_vs_best=round(fixed - best_single, 3),
                      n_categories=len(cc), deployment_legal=p.startswith("calib"))
    return out


def attach_per_map(cats_by_obj, aligned, ds):
    """Add the per-map flattened (clip,dino,y,is_normal) list needed by the calibration-normalization analysis.
    We keep RAW (un-normalised) maps so each protocol estimates its own
    params; maps are downsampled to 128x128 (stride 2) to bound the split sweep
    cost — AUROC is robust to this resolution and this analysis is a relative (regret)
    comparison across normalisation protocols."""
    for obj, items in aligned.items():
        if obj not in cats_by_obj:
            continue
        maps = []
        for it in items:
            c = it["clip_map"][::2, ::2].astype(np.float32).ravel()
            d = it["dino_map"][::2, ::2].astype(np.float32).ravel()
            y = (it["gt_mask"][::2, ::2] > 0.5).astype(np.int8).ravel()
            maps.append(dict(clip=c, dino=d, y=y, is_normal=bool(it["is_normal"])))
        cats_by_obj[obj]["_maps"] = maps


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def loo_directional(alpha, types):
    cats = list(alpha)
    nc = 0
    for c in cats:
        others = [alpha[o] for o in cats if o != c and types[o] == types[c]]
        pred = "dino" if (others and np.mean(others) > 0.5) else "clip"
        true = "dino" if alpha[c] > 0.5 else "clip"
        nc += (pred == true)
    nd = sum(1 for c in cats if alpha[c] > 0.5)
    return nc, nd, len(cats)


def analyse_directional(df, tax):
    alpha = {r.category: r.alpha_pix for r in df.itertuples()}
    types = {c: tax.get(c, {}).get("type") for c in alpha}
    alpha = {c: a for c, a in alpha.items() if types.get(c) in ("object", "texture")}
    types = {c: types[c] for c in alpha}
    nc, nd, n = loo_directional(alpha, types)
    p = float(stats.binomtest(nc, n, nd / n, alternative="greater").pvalue)
    tex = [alpha[c] for c in alpha if types[c] == "texture"]
    obj = [alpha[c] for c in alpha if types[c] == "object"]
    return dict(directional_accuracy={"count": nc, "n": n, "pct": round(nc / n * 100, 1)},
                majority_baseline={"count": nd, "pct": round(nd / n * 100, 1)},
                margin=nc - nd, binomial_p_one_sided=round(p, 4),
                texture_mean_alpha=round(float(np.mean(tex)), 3),
                object_mean_alpha=round(float(np.mean(obj)), 3),
                n_texture=len(tex), n_object=len(obj))


# ----------------------------------------------------------------------------
def main():
    tax_full = json.load(open(os.path.join(BASE, "category_taxonomy_results.json")))
    tax = tax_full["taxonomy"]
    img_alpha = {}
    for ds in DSNAME:
        img_alpha.update(tax_full["oracle_alphas"][IMG_ALPHA_KEY[ds]])

    all_cats = {}     # obj -> rec (capped flattened arrays)
    cat2ds = {}
    curves = {}
    print("Loading + curving seed-0 pixel maps (256x256, alpha grid step 0.02)...")
    for ds in ["mvtec", "visa", "btad"]:
        aligned = load_dataset(ds)
        cats = build_category_arrays(aligned, ds)
        attach_per_map(cats, aligned, ds)
        for c, rec in cats.items():
            all_cats[c] = rec
            cat2ds[c] = DSNAME[ds]
            curves[c] = category_curve(rec, ALPHAS)
        print(f"  {DSNAME[ds]:<9}: {len(cats)} categories curved (running total {len(curves)})")

    print(f"\nTotal categories with anomaly pixels: {len(curves)}")

    rb = analyse_regret_band(curves, cat2ds)
    hier, df = analyse_hierarchical(curves, cat2ds, tax, img_alpha)
    calib = analyse_calibration(all_cats)
    dirn = analyse_directional(df, tax)

    out = dict(
        experiment="pixel-level (seed-0) re-runs of the four image-level headline analyses",
        level="pixel", seed=0, n_categories=len(curves),
        alpha_convention="alpha = DINOv2 weight; fused = (1-a)*CLIP + a*DINOv2; per-category min-max (transductive) except the calibration protocol",
        map_resolution="256x256", alpha_grid_step=0.02, neg_pixel_cap=NEG_CAP,
        calib_map_resolution="128x128 (stride-2 of the 256x256 aligned maps; bounds the split sweep, AUROC robust)",
        calib_alpha_grid_step=0.05, calib_n_splits=CALIB_SPLITS, calib_floor=MIN_CALIB,
        calib_note="The calibration protocol sweeps a coarser alpha grid (0.05) at 128x128 than the rest of the pixel pipeline "
                 "(0.02 at 256x256); it is a relative (regret) comparison across normalisation protocols.",
        regret_band=rb,
        hierarchical=hier,
        normalization=calib,
        taxonomy_directional=dirn,
        per_category=df.round(3).to_dict(orient="records"),
    )
    outp = os.path.join(RESULTS, "pixel_seed0_metrics.json")
    json.dump(out, open(outp, "w"), indent=2)

    # ---- console summary ----
    print("\n" + "=" * 86)
    print("  PIXEL-LEVEL (seed-0) SUMMARY  — image-side seed-0 in brackets for reference")
    print("=" * 86)
    print(f"  pixel regret-vs-alpha: pooled min regret {rb['pooled_min_regret']}pp at a*={rb['pooled_opt_alpha']}; "
          f"band<=0.45 {rb['band45']}, band<=0.20 {rb['band20']}")
    print(f"             per-dataset optimum {rb['per_dataset_optimum']}")
    print(f"  pixel A clip-vs-dino : within-ds perm p={hier['A_clip_vs_dino']['perm_within_dataset_p']} (mean {hier['A_clip_vs_dino']['pooled_mean_diff']})")
    print(f"             B fusion>best    : within-ds perm p={hier['B_fusion_vs_best_single']['perm_within_dataset_p']} (mean {hier['B_fusion_vs_best_single']['pooled_mean_gain']})")
    print(f"             C type->alpha    : within-ds perm p={hier['C_type_to_alpha']['perm_within_dataset_p']} (obj {hier['C_type_to_alpha']['object_mean_alpha']} vs tex {hier['C_type_to_alpha']['texture_mean_alpha']})")
    if "D_reversal_to_gain" in hier:
        d = hier["D_reversal_to_gain"]
        print(f"             D reversal->gain : within-ds perm p={d['perm_within_dataset_p']} (n_rev={d['n_reversal']}, rev {d['reversal_mean_gain']} vs non {d['nonreversal_mean_gain']})")
    print(f"  pixel calibration: " + "; ".join(f"{p} a*={v['global_alpha']} regret={v['regret_pp']}{' (legal)' if v['deployment_legal'] else ''}" for p, v in calib.items()))
    print(f"  pixel directional: {dirn['directional_accuracy']['count']}/{dirn['directional_accuracy']['n']} "
          f"vs baseline {dirn['majority_baseline']['count']} margin {dirn['margin']:+d} binom p={dirn['binomial_p_one_sided']}; "
          f"obj a={dirn['object_mean_alpha']} tex a={dirn['texture_mean_alpha']}")
    print(f"\n  Saved: {outp}")


if __name__ == "__main__":
    main()
