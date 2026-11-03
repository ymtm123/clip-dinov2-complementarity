"""Seed-averaged hierarchical re-analysis over the 10 reference-image seeds."""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leave_one_dataset_out_alpha import build_curves, ALPHAS, BASE_DIR
from seed_avg_fusion_metrics import load_all_datasets
from hierarchical_stats import perm_paired_vs0, perm_twogroup, icc_oneway

PREF = {'MVTec': 'mvtec', 'VisA': 'visa', 'BTAD': 'btad'}
DSNAME = {'MVTec': 'MVTec AD', 'VisA': 'VisA', 'BTAD': 'BTAD'}


def seed_avg_table(seeds=range(10)):
    acc, cat2ds, nseen = {}, {}, {}
    for s in seeds:
        curves, c2d = build_curves(load_all_datasets(s))
        for c in curves:
            cat2ds[c] = c2d[c]
            arr = np.array([curves[c][a] for a in ALPHAS])
            acc[c] = acc.get(c, np.zeros(len(ALPHAS))) + arr
            nseen[c] = nseen.get(c, 0) + 1
    tax = json.load(open(os.path.join(BASE_DIR, 'category_taxonomy_results.json'), encoding='utf-8'))
    oalpha, taxo = tax['oracle_alphas'], tax['taxonomy']
    i50 = ALPHAS.index(0.5)
    rows = []
    for c, csum in acc.items():
        avg = csum / nseen[c]
        clip, dino, equal = float(avg[0]), float(avg[-1]), float(avg[i50])
        best = max(clip, dino)
        ai = ALPHAS[int(np.argmax(avg))]
        ds = cat2ds[c]; pref = PREF[ds]
        ap = oalpha.get(f"{pref}_pixel", {}).get(c)
        t = taxo.get(c, {})
        rows.append(dict(category=c, dataset=DSNAME[ds], clip=clip, dino=dino, equal=equal,
                         best_single=best, dino_minus_clip=dino - clip, fusion_gain=equal - best,
                         alpha_img=ai, alpha_pix=ap,
                         reversal=(None if ap is None else int((ai > 0.5 and ap < 0.5) or (ai < 0.5 and ap > 0.5))),
                         type=t.get('type')))
    return pd.DataFrame(rows)


def main():
    df = seed_avg_table()
    print("Loaded (seed-averaged) " + str(len(df)) + " categories: " +
          ", ".join(f"{d}={(df.dataset == d).sum()}" for d in df.dataset.unique()))
    out = {"n_categories": len(df),
           "note": "seed-AVERAGED over 10 reference-image seeds (image-side oracle alpha seed-averaged; pixel-side alpha seed-0 pending pixel multi-seed). Permutation N=20000, RNG seed 42.",
           "associations": {}}

    _, p = stats.wilcoxon(df.dino_minus_clip); obs, pp = perm_paired_vs0(df, 'dino_minus_clip'); icc = icc_oneway(df.dino_minus_clip, df.dataset)
    out["associations"]["A_clip_vs_dino"] = {"pooled_wilcoxon_p": round(float(p), 4), "pooled_mean_diff": round(float(df.dino_minus_clip.mean()), 3),
                                             "perm_within_dataset_p": round(float(pp), 4), "between_dataset_var_share_icc": round(icc, 3),
                                             "per_dataset_mean_diff": {d: round(float(df[df.dataset == d].dino_minus_clip.mean()), 2) for d in df.dataset.unique()}}

    _, p = stats.wilcoxon(df.fusion_gain); obs, pp = perm_paired_vs0(df, 'fusion_gain'); icc = icc_oneway(df.fusion_gain, df.dataset)
    out["associations"]["B_fusion_vs_best_single"] = {"pooled_wilcoxon_p": round(float(p), 4), "pooled_mean_gain": round(float(df.fusion_gain.mean()), 3),
                                                      "perm_within_dataset_p": round(float(pp), 4), "between_dataset_var_share_icc": round(icc, 3)}

    dft = df.dropna(subset=['type', 'alpha_img']).copy(); dft['type_bin'] = (dft.type == 'object').astype(int)
    obj = dft[dft.type == 'object'].alpha_img; tex = dft[dft.type == 'texture'].alpha_img
    _, p = stats.mannwhitneyu(obj, tex, alternative='two-sided'); obs, pp, nds = perm_twogroup(dft, 'alpha_img', 'type_bin'); icc = icc_oneway(dft.alpha_img, dft.dataset)
    out["associations"]["C_type_to_alpha"] = {"pooled_mannwhitney_p": round(float(p), 4), "object_mean_alpha": round(float(obj.mean()), 3), "texture_mean_alpha": round(float(tex.mean()), 3),
                                              "n_object": int(len(obj)), "n_texture": int(len(tex)),
                                              "perm_within_dataset_p": None if np.isnan(pp) else round(float(pp), 4), "n_datasets_with_both_types": nds,
                                              "between_dataset_var_share_icc": round(icc, 3)}

    dfr = df.dropna(subset=['reversal']).copy()
    rev = dfr[dfr.reversal == 1].fusion_gain; nonrev = dfr[dfr.reversal == 0].fusion_gain
    _, p = stats.mannwhitneyu(rev, nonrev, alternative='two-sided'); obs, pp, nds = perm_twogroup(dfr, 'fusion_gain', 'reversal')
    out["associations"]["D_reversal_to_gain"] = {"pooled_mannwhitney_p": round(float(p), 4), "reversal_mean_gain": round(float(rev.mean()), 3), "nonreversal_mean_gain": round(float(nonrev.mean()), 3),
                                                 "n_reversal": int(len(rev)), "n_nonreversal": int(len(nonrev)),
                                                 "perm_within_dataset_p": None if np.isnan(pp) else round(float(pp), 4)}

    json.dump(out, open(os.path.join(BASE_DIR, 'hierarchical_stats_seedavg_results.json'), 'w'), indent=2)
    print("\n  SEED-AVERAGED HIERARCHICAL (seed-0 within-ds p in parens):")
    ref = {"A_clip_vs_dino": "0.0001", "B_fusion_vs_best_single": "0.0058", "C_type_to_alpha": "0.0053", "D_reversal_to_gain": "0.0546"}
    for k, a in out["associations"].items():
        pooled = a.get("pooled_wilcoxon_p", a.get("pooled_mannwhitney_p")); wp = a["perm_within_dataset_p"]
        print(f"  {k:<26} pooled={pooled}  within-ds perm={wp}   (seed-0 {ref[k]})")
    print(f"  Saved hierarchical_stats_seedavg_results.json")


if __name__ == "__main__":
    main()
