""": per-category bootstrap-stability table over ALL 30 categories."""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reviewer_response_experiments as rr

ALPHA_GRID = np.arange(0.0, 1.01, 0.05)
NB = 1000
PIXKEY = {"MVTec AD": "mvtec_pixel", "VisA": "visa_pixel", "BTAD": "btad_pixel"}
IMGKEY = {"MVTec AD": "mvtec_image", "VisA": "visa_image", "BTAD": "btad_image"}
BASE = os.path.dirname(os.path.abspath(rr.__file__))
AD = os.path.join(BASE, "AnomalyDINO")
CL = os.path.join(BASE, "AnomalyCLIP", "results")


def load_30():
    ds = {}
    mv = {}
    mv.update(rr.load_anomalydino_scores(os.path.join(AD, "results_MVTec", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_weak_cats", "measurements_seed=0.csv")))
    mv.update(rr.load_anomalydino_scores(os.path.join(AD, "results_MVTec", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_strong_cats", "measurements_seed=0.csv")))
    ds["MVTec AD"] = rr.align_samples(rr.load_anomalyclip_scores(os.path.join(CL, "scores_mvtec", "anomalyclip_scores.csv")), mv)
    ds["VisA"] = rr.align_samples(rr.load_anomalyclip_scores(os.path.join(CL, "scores_visa", "anomalyclip_scores.csv")),
                                  rr.load_anomalydino_scores(os.path.join(AD, "results_VisA", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv")))
    ds["BTAD"] = rr.align_samples(rr.load_anomalyclip_scores(os.path.join(CL, "scores_btad", "anomalyclip_scores.csv")),
                                  rr.load_anomalydino_scores(os.path.join(AD, "results_BTAD", "dinov2_vitl14_448", "1-shot_preprocess=agnostic_btad_all", "measurements_seed=0.csv")))
    return ds


def main():
    tax = json.load(open(os.path.join(BASE, "category_taxonomy_results.json"), encoding="utf-8"))
    ds = load_30()
    total = sum(len(v) for v in ds.values())
    print(f"Loaded {total} categories: " + ", ".join(f"{k}={len(v)}" for k, v in ds.items()))
    rng = np.random.RandomState(42)
    rows = []
    for dname, dd in ds.items():
        for cat, d in sorted(dd.items()):
            y = np.asarray(d['labels']); cl = np.asarray(d['clip_scores'], float); dn = np.asarray(d['dino_scores'], float)
            n = len(y)
            if n < 10 or len(np.unique(y)) < 2:
                continue
            a_img, _ = rr.find_oracle_alpha(y, rr.normalize_minmax(cl), rr.normalize_minmax(dn), ALPHA_GRID)
            a_pix = tax["oracle_alphas"].get(PIXKEY[dname], {}).get(cat, 0.5)
            a_img_lock = tax["oracle_alphas"].get(IMGKEY[dname], {}).get(cat, a_img)  # locked image alpha = manuscript reversal definition (13/30)
            boots = []
            for _ in range(NB):
                idx = rng.choice(n, n, replace=True)
                if len(np.unique(y[idx])) < 2:
                    continue
                ba, _ = rr.find_oracle_alpha(y[idx], rr.normalize_minmax(cl[idx]), rr.normalize_minmax(dn[idx]), ALPHA_GRID)
                boots.append(ba)
            boots = np.array(boots)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            crosses = bool(lo < 0.5 < hi)
            rev = bool((a_img_lock > 0.5 and a_pix < 0.5) or (a_img_lock < 0.5 and a_pix > 0.5))  # strict: matches manuscript 13/30 (excludes exactly-0.5)
            dom = float(np.mean((boots > 0.5) == (a_img_lock > 0.5)))
            rows.append(dict(category=cat, dataset=dname, alpha_img=round(float(a_img_lock), 3), alpha_img_recomputed=round(float(a_img), 3), alpha_pix=round(float(a_pix), 3),
                             ci=[round(float(lo), 3), round(float(hi), 3)], ci_width=round(float(hi - lo), 3),
                             crosses_05=crosses, reversal=rev, dominance_stability=round(dom, 3),
                             stable_reversal=bool(rev and not crosses)))
    n_rev = sum(r['reversal'] for r in rows)
    stable = [r for r in rows if r['stable_reversal']]
    arr = np.array([r['reversal'] for r in rows], float); rng2 = np.random.RandomState(7)
    rates = [arr[rng2.choice(len(arr), len(arr), replace=True)].mean() for _ in range(2000)]
    ci = np.percentile(rates, [2.5, 97.5])
    out = dict(n_total=len(rows), n_reversal=n_rev, reversal_rate=round(n_rev / len(rows), 3),
               reversal_rate_ci=[round(float(ci[0]), 3), round(float(ci[1]), 3)],
               n_stable_reversal=len(stable),
               stable_reversal_categories=[f"{r['category']} ({r['dataset']})" for r in stable],
               note="seed-0 scores, all 30 categories (MVTec from the weak+strong splits); pixel alpha seed-0; alpha grid 0.05; 1000 bootstrap resamples of the test set",
               categories=rows)
    json.dump(out, open(os.path.join(os.path.dirname(BASE), "results", "bootstrap_stability_table.json"), "w"), indent=2)
    print(f"\n  Reversals: {n_rev}/{len(rows)} ({100*n_rev/len(rows):.0f}%), rate CI [{ci[0]:.2f},{ci[1]:.2f}]")
    print(f"  STABLE reversals (CI excludes 0.5): {len(stable)}/{n_rev}")
    print("  Named stable-reversal categories: " + ", ".join(out["stable_reversal_categories"]))
    print("\n  Per-category (reversals only):")
    print(f"    {'category':16s}{'ds':10s}{'a_img':>6}{'a_pix':>6}{'CI':>16}{'cross0.5':>9}{'stable':>7}")
    for r in rows:
        if r['reversal']:
            print(f"    {r['category']:16s}{r['dataset']:10s}{r['alpha_img']:>6}{r['alpha_pix']:>6}"
                  f"{str(r['ci']):>16}{'Y' if r['crosses_05'] else 'N':>9}{'YES' if r['stable_reversal'] else '-':>7}")
    print("\n  Saved: results/bootstrap_stability_table.json")


if __name__ == "__main__":
    main()
