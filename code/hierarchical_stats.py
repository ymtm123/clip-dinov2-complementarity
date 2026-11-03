"""Hierarchical / dataset-grouped re-analysis of the headline associations."""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.RandomState(42)
N_PERM = 20000

DS_FULL = {"MVTec AD": "mvtec", "VisA": "visa", "BTAD": "btad"}


def load_table():
    ev = json.load(open(os.path.join(BASE, "comprehensive_eval_results.json"), encoding="utf-8"))
    tax_all = json.load(open(os.path.join(BASE, "category_taxonomy_results.json"), encoding="utf-8"))
    tax, oalpha = tax_all["taxonomy"], tax_all["oracle_alphas"]
    rows = []
    for ds_full, pref in DS_FULL.items():
        d = ev[ds_full]
        clip = d["CLIP only"]["per_category"]
        dino = d["DINOv2 only"]["per_category"]
        equal = d["Equal (a=0.5)"]["per_category"]
        for c in clip:
            ai = oalpha.get(f"{pref}_image", {}).get(c)
            ap = oalpha.get(f"{pref}_pixel", {}).get(c)
            t = tax.get(c, {})
            rows.append(dict(
                category=c, dataset=ds_full,
                clip=clip[c], dino=dino[c], equal=equal[c],
                best_single=max(clip[c], dino[c]),
                dino_minus_clip=dino[c] - clip[c],
                fusion_gain=equal[c] - max(clip[c], dino[c]),
                alpha_img=ai, alpha_pix=ap,
                reversal=(None if (ai is None or ap is None)
                          else int((ai > 0.5 and ap < 0.5) or (ai < 0.5 and ap > 0.5))),
                type=t.get("type"),
            ))
    return pd.DataFrame(rows)


def icc_oneway(values, groups):
    """Between-group variance share via one-way ANOVA (ICC1)."""
    df = pd.DataFrame({"y": values, "g": groups})
    grand = df["y"].mean()
    gm = df.groupby("g")["y"].agg(["mean", "count"])
    ss_between = float((gm["count"] * (gm["mean"] - grand) ** 2).sum())
    ss_total = float(((df["y"] - grand) ** 2).sum())
    return ss_between / ss_total if ss_total > 0 else 0.0


def mixedlm_p(df, formula, group="dataset", coef=None):
    try:
        import statsmodels.formula.api as smf
        m = smf.mixedlm(formula, df, groups=df[group]).fit(reml=False, method="lbfgs")
        name = coef if coef else "Intercept"
        return float(m.pvalues.get(name, np.nan)), float(m.params.get(name, np.nan))
    except Exception as e:
        return np.nan, np.nan


def perm_paired_vs0(df, col):
    """Within-dataset sign-flip permutation that the dataset-averaged mean != 0."""
    sub = df.dropna(subset=[col])
    per_ds = [sub[sub.dataset == d][col].values for d in sub.dataset.unique()]
    obs = np.mean([v.mean() for v in per_ds])
    cnt = 0
    for _ in range(N_PERM):
        signs = [RNG.choice([-1, 1], size=len(v)) for v in per_ds]
        stat = np.mean([(s * v).mean() for s, v in zip(signs, per_ds)])
        if abs(stat) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (N_PERM + 1)


def perm_twogroup(df, value_col, group_col):
    """Within-dataset label-permutation that dataset-averaged group-mean-diff != 0.
    group_col is binary-ish; effect = mean(value|g=1) - mean(value|g=0) averaged over datasets."""
    sub = df.dropna(subset=[value_col, group_col])
    datasets = [d for d in sub.dataset.unique()
                if sub[(sub.dataset == d)][group_col].nunique() == 2]
    if not datasets:
        return np.nan, np.nan, 0

    def effect(frame):
        diffs = []
        for d in datasets:
            s = frame[frame.dataset == d]
            g1 = s[s[group_col] == s[group_col].max()][value_col].mean()
            g0 = s[s[group_col] == s[group_col].min()][value_col].mean()
            diffs.append(g1 - g0)
        return np.mean(diffs)

    obs = effect(sub)
    cnt = 0
    work = sub.copy()
    for _ in range(N_PERM):
        permuted = work.copy()
        for d in datasets:
            mask = permuted.dataset == d
            permuted.loc[mask, group_col] = RNG.permutation(permuted.loc[mask, group_col].values)
        if abs(effect(permuted)) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (N_PERM + 1), len(datasets)


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def main():
    df = load_table()
    print(f"Loaded {len(df)} categories: " +
          ", ".join(f"{d}={ (df.dataset==d).sum() }" for d in df.dataset.unique()))
    out = {"n_categories": len(df), "note": "seed-0 inputs; superseded by the seed-averaged rerun (hierarchical_stats_seedavg.py)",
           "associations": {}}

    # ---------- A. CLIP vs DINOv2 ----------
    naive_stat, naive_p = stats.wilcoxon(df["dino_minus_clip"])
    obs, pp = perm_paired_vs0(df, "dino_minus_clip")
    mp, mb = mixedlm_p(df, "dino_minus_clip ~ 1")
    icc = icc_oneway(df["dino_minus_clip"], df["dataset"])
    per_ds = {d: round(df[df.dataset == d]["dino_minus_clip"].mean(), 2) for d in df.dataset.unique()}
    out["associations"]["A_clip_vs_dino"] = {
        "pooled_wilcoxon_p": round(float(naive_p), 4), "pooled_mean_diff": round(float(df.dino_minus_clip.mean()), 3),
        "perm_within_dataset_p": round(float(pp), 4), "dataset_avg_effect": round(float(obs), 3),
        "mixedlm_intercept_p": None if np.isnan(mp) else round(mp, 4),
        "between_dataset_var_share_icc": round(icc, 3), "per_dataset_mean_diff": per_ds,
    }

    # ---------- B. Equal fusion vs best single ----------
    naive_stat, naive_p = stats.wilcoxon(df["fusion_gain"])
    obs, pp = perm_paired_vs0(df, "fusion_gain")
    mp, mb = mixedlm_p(df, "fusion_gain ~ 1")
    icc = icc_oneway(df["fusion_gain"], df["dataset"])
    out["associations"]["B_fusion_vs_best_single"] = {
        "pooled_wilcoxon_p": round(float(naive_p), 4), "pooled_mean_gain": round(float(df.fusion_gain.mean()), 3),
        "perm_within_dataset_p": round(float(pp), 4), "dataset_avg_gain": round(float(obs), 3),
        "mixedlm_intercept_p": None if np.isnan(mp) else round(mp, 4),
        "between_dataset_var_share_icc": round(icc, 3),
    }

    # ---------- C. Product type -> image oracle alpha ----------
    dft = df.dropna(subset=["type", "alpha_img"]).copy()
    dft["type_bin"] = (dft["type"] == "object").astype(int)
    obj = dft[dft.type == "object"]["alpha_img"]; tex = dft[dft.type == "texture"]["alpha_img"]
    naive_u, naive_p = stats.mannwhitneyu(obj, tex, alternative="two-sided")
    obs, pp, nds = perm_twogroup(dft, "alpha_img", "type_bin")
    mp, mb = mixedlm_p(dft, "alpha_img ~ C(type)", coef="C(type)[T.texture]")
    icc = icc_oneway(dft["alpha_img"], dft["dataset"])
    out["associations"]["C_type_to_alpha"] = {
        "pooled_mannwhitney_p": round(float(naive_p), 4),
        "object_mean_alpha": round(float(obj.mean()), 3), "texture_mean_alpha": round(float(tex.mean()), 3),
        "n_object": int(len(obj)), "n_texture": int(len(tex)),
        "perm_within_dataset_p": None if np.isnan(pp) else round(float(pp), 4),
        "dataset_avg_obj_minus_tex": None if np.isnan(obs) else round(float(obs), 3),
        "n_datasets_with_both_types": nds,
        "mixedlm_type_p": None if np.isnan(mp) else round(mp, 4),
        "between_dataset_var_share_icc": round(icc, 3),
    }

    # ---------- D. Reversal -> fusion gain ----------
    dfr = df.dropna(subset=["reversal"]).copy()
    rev = dfr[dfr.reversal == 1]["fusion_gain"]; nonrev = dfr[dfr.reversal == 0]["fusion_gain"]
    naive_u, naive_p = stats.mannwhitneyu(rev, nonrev, alternative="two-sided")
    obs, pp, nds = perm_twogroup(dfr, "fusion_gain", "reversal")
    mp, mb = mixedlm_p(dfr, "fusion_gain ~ C(reversal)", coef="C(reversal)[T.1]")
    out["associations"]["D_reversal_to_gain"] = {
        "pooled_mannwhitney_p": round(float(naive_p), 4),
        "reversal_mean_gain": round(float(rev.mean()), 3), "nonreversal_mean_gain": round(float(nonrev.mean()), 3),
        "n_reversal": int(len(rev)), "n_nonreversal": int(len(nonrev)),
        "perm_within_dataset_p": None if np.isnan(pp) else round(float(pp), 4),
        "dataset_avg_rev_minus_nonrev": None if np.isnan(obs) else round(float(obs), 3),
        "n_datasets_with_both": nds,
        "mixedlm_reversal_p": None if np.isnan(mp) else round(mp, 4),
    }

    with open(os.path.join(BASE, "hierarchical_stats_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---------- console ----------
    print("\n" + "=" * 92)
    print("  HIERARCHICAL RE-ANALYSIS: naive (pooled) vs within-dataset permutation vs mixed model")
    print("=" * 92)
    labels = {"A_clip_vs_dino": "A. CLIP vs DINOv2 (diff)",
              "B_fusion_vs_best_single": "B. Equal fusion vs best single",
              "C_type_to_alpha": "C. Product type -> oracle alpha",
              "D_reversal_to_gain": "D. Reversal -> fusion gain"}
    for k, lab in labels.items():
        a = out["associations"][k]
        np_ = a.get("pooled_wilcoxon_p", a.get("pooled_mannwhitney_p"))
        pp_ = a.get("perm_within_dataset_p")
        mp_ = a.get("mixedlm_intercept_p", a.get("mixedlm_type_p", a.get("mixedlm_reversal_p")))
        icc = a.get("between_dataset_var_share_icc")
        print(f"\n  {lab}")
        print(f"     pooled p          = {np_}  ({sig(np_) if np_ is not None else '?'})")
        print(f"     within-dataset perm p = {pp_}  ({sig(pp_) if isinstance(pp_,float) else '?'})  <-- primary grouped test")
        print(f"     mixed-model p     = {mp_}  ({sig(mp_) if isinstance(mp_,float) else 'n/a (3 groups)'})")
        if icc is not None:
            print(f"     between-dataset variance share (ICC) = {icc}")
    print("\n  Saved: hierarchical_stats_results.json")


if __name__ == "__main__":
    main()
