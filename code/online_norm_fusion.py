"""Online / streaming-normalization variant of the fixed fusion."""
import os
import sys
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import practical_normalization as pn
from seed_avg_practical_norm import load_30_seed
from score_fusion import compute_auroc

ALPHAS = pn.ALPHAS
WARMUP = 10          # pre-deployment normal calibration prefix
N_ORDERS = 5         # stream orderings per (category, seed)
EMA_LAMBDA = 0.01
PROTOCOLS = ["running_zscore", "ema_zscore", "calib_zscore"]


def _causal_z(x_stream, warm):
    """Running (Welford) causal z-scores for a 1-D stream, warm-started on `warm`."""
    n = len(warm)
    mean = float(np.mean(warm)) if n else 0.0
    M2 = float(np.var(warm) * n) if n else 0.0
    cnt = n
    out = np.empty(len(x_stream), float)
    for i, x in enumerate(x_stream):
        sd = np.sqrt(M2 / cnt) if cnt > 1 and M2 > 0 else 1.0
        out[i] = (x - mean) / (sd if sd > 1e-9 else 1.0)
        cnt += 1
        d = x - mean
        mean += d / cnt
        M2 += d * (x - mean)
    return out


def _ema_z(x_stream, warm, lam=EMA_LAMBDA):
    """EMA causal z-scores, warm-started on `warm`."""
    mean = float(np.mean(warm)) if len(warm) else 0.0
    var = float(np.var(warm)) if len(warm) else 1.0
    out = np.empty(len(x_stream), float)
    for i, x in enumerate(x_stream):
        sd = np.sqrt(var) if var > 0 else 1.0
        out[i] = (x - mean) / (sd if sd > 1e-9 else 1.0)
        d = x - mean
        mean += lam * d
        var = (1 - lam) * (var + lam * d * d)
    return out


def _static_z(x_stream, warm):
    """Frozen warm-up-prefix z-score (deployment-legal calib reference)."""
    mu = float(np.mean(warm)); sd = float(np.std(warm))
    return (x_stream - mu) / (sd if sd > 1e-9 else 1.0)


NORM = {"running_zscore": _causal_z, "ema_zscore": _ema_z, "calib_zscore": _static_z}


def per_cat_curves(clip, dino, y, rng):
    """{protocol: {alpha: mean AUROC over N_ORDERS stream orders}} for one category."""
    normals = np.where(y == 0)[0]
    anoms = np.where(y == 1)[0]
    k = min(WARMUP, max(2, len(normals) - 2))
    if len(normals) - k < 1 or len(anoms) < 1:
        return None
    acc = {p: [] for p in PROTOCOLS}
    for _ in range(N_ORDERS):
        warm_idx = rng.choice(normals, size=k, replace=False)
        eval_idx = np.array([i for i in range(len(y)) if i not in set(warm_idx)])
        rng.shuffle(eval_idx)                       # simulate arrival order of the live stream
        yi = y[eval_idx]
        if len(np.unique(yi)) < 2:
            continue
        for p in PROTOCOLS:
            cz = NORM[p](clip[eval_idx], clip[warm_idx])
            dz = NORM[p](dino[eval_idx], dino[warm_idx])
            acc[p].append([compute_auroc(yi, (1 - a) * cz + a * dz) for a in ALPHAS])
    out = {}
    for p in PROTOCOLS:
        if acc[p]:
            out[p] = dict(zip(ALPHAS, np.mean(acc[p], axis=0)))
    return out


def summarize(cat_curves):
    res = {}
    for p in PROTOCOLS:
        curves = {c: cat_curves[c][p] for c in cat_curves if p in cat_curves[c]}
        cats = list(curves)
        glob_a = max(ALPHAS, key=lambda a: np.mean([curves[c][a] for c in cats]))
        oracle = np.mean([max(curves[c].values()) for c in cats])
        best = np.mean([max(curves[c][0.0], curves[c][1.0]) for c in cats])
        fixed = np.mean([curves[c][glob_a] for c in cats])
        regret = np.mean([max(curves[c].values()) - curves[c][glob_a] for c in cats])
        res[p] = dict(global_alpha=glob_a, regret_pp=float(regret),
                      fusion_gain_vs_best=float(fixed - best), oracle=float(oracle))
    return res


def ci95(xs):
    a = np.asarray(xs, float); m = a.mean()
    if len(a) < 2:
        return float(m), float(m), float(m)
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(m), float(m - h), float(m + h)


def main():
    fields = ["regret_pp", "fusion_gain_vs_best", "global_alpha", "oracle"]
    acc = {p: {f: [] for f in fields} for p in PROTOCOLS}
    for seed in range(10):
        table = load_30_seed(seed)
        rng = np.random.RandomState(1000 + seed)
        cat_curves = {}
        for cat, (ds, clip, dino, y) in table.items():
            r = per_cat_curves(np.asarray(clip, float), np.asarray(dino, float), np.asarray(y), rng)
            if r:
                cat_curves[cat] = r
        summ = summarize(cat_curves)
        for p in PROTOCOLS:
            for f in fields:
                acc[p][f].append(summ[p][f])
        print(f"  seed {seed}: running regret={summ['running_zscore']['regret_pp']:.3f}  "
              f"ema regret={summ['ema_zscore']['regret_pp']:.3f}  calib regret={summ['calib_zscore']['regret_pp']:.3f}")

    out = {"experiment": "online/streaming normalization variant of fixed fusion", 
           "warmup_normals": WARMUP, "n_stream_orders": N_ORDERS, "ema_lambda": EMA_LAMBDA,
           "method": "causal per-sample z-score (running Welford / EMA), warm-started on a normal prefix; "
                     "seed-averaged over 10 DINO seeds x 5 stream orders, t(9) 95% CI",
           "protocols": {}}
    for p in PROTOCOLS:
        d = {}
        for f in fields:
            m, lo, hi = ci95(acc[p][f])
            d[f] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "seed0": round(float(acc[p][f][0]), 4)}
        d["streaming"] = p != "calib_zscore"
        out["protocols"][p] = d
    json.dump(out, open(os.path.join(os.path.dirname(pn.BASE), "results", "online_norm_fusion.json"), "w"), indent=2)
    print("\n" + "=" * 86)
    print("  ONLINE/STREAMING NORMALIZATION (mean [95% CI]; seed0 in parens)")
    print("=" * 86)
    for p in PROTOCOLS:
        r = out["protocols"][p]["regret_pp"]; g = out["protocols"][p]["fusion_gain_vs_best"]; a = out["protocols"][p]["global_alpha"]
        tag = " (streaming)" if out["protocols"][p]["streaming"] else " (static calib reference)"
        print(f"  {p:<16} a*={a['mean']:.2f}  regret={r['mean']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]  "
              f"gain={g['mean']:+.3f} (seed0 regret {r['seed0']}){tag}")
    print("\n  Saved: results/online_norm_fusion.json")


if __name__ == "__main__":
    main()
