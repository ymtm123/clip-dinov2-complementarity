"""Regret-vs-alpha curves ."""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leave_one_dataset_out_alpha import load_all_datasets, build_curves, oracle, ALPHAS, BASE_DIR

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 300,
})


def regret_curve(curves, cats, oracle_auroc):
    """Mean regret over `cats` for each alpha."""
    return [float(np.mean([oracle_auroc[c] - curves[c][a] for c in cats])) for a in ALPHAS]


def main():
    datasets = load_all_datasets()
    curves, cat2ds = build_curves(datasets)
    all_cats = list(curves.keys())
    ds_names = list(datasets.keys())
    by_ds = {ds: [c for c in all_cats if cat2ds[c] == ds] for ds in ds_names}
    oracle_auroc = {c: oracle(curves, c)[1] for c in all_cats}

    pooled = regret_curve(curves, all_cats, oracle_auroc)
    per_ds = {ds: regret_curve(curves, by_ds[ds], oracle_auroc) for ds in ds_names}

    def at(curve, a):
        return curve[ALPHAS.index(round(a, 2))]

    def argmin_alpha(curve):
        i = int(np.argmin(curve))
        return ALPHAS[i], curve[i]

    # plateau: contiguous alphas whose pooled regret is within 0.1pp of the pooled minimum
    pmin_a, pmin = argmin_alpha(pooled)

    def band_within(thresh):
        """Contiguous alpha range around the optimum whose pooled regret <= min+thresh."""
        ok = [a for a, r in zip(ALPHAS, pooled) if r <= pmin + thresh]
        return (round(min(ok), 2), round(max(ok), 2))

    within01 = [a for a, r in zip(ALPHAS, pooled) if r <= pmin + 0.1]
    plateau = (min(within01), max(within01))
    band_010 = band_within(0.10)
    band_020 = band_within(0.20)
    band_045 = band_within(0.45)
    # max regret across the [0.5,0.7] band (validates the manuscript's "within X pp" claim)
    band = [r for a, r in zip(ALPHAS, pooled) if 0.5 <= a <= 0.7]
    band_max = max(band)
    band_spread = band_max - min(band)

    summary = {
        "experiment": "regret-vs-alpha",
        "n_categories": len(all_cats),
        "pooled_optimal_alpha": pmin_a,
        "pooled_min_regret_pp": round(pmin, 3),
        "pooled_regret_at_0.50_pp": round(at(pooled, 0.50), 3),
        "pooled_regret_at_0.65_pp": round(at(pooled, 0.65), 3),
        "pooled_regret_at_0.70_pp": round(at(pooled, 0.70), 3),
        "plateau_within_0.1pp_of_min": [round(plateau[0], 2), round(plateau[1], 2)],
        "alpha_range_within_0.10pp_of_optimum": list(band_010),
        "alpha_range_within_0.20pp_of_optimum": list(band_020),
        "alpha_range_within_0.45pp_of_optimum": list(band_045),
        "band_0.5_0.7_max_regret_pp": round(band_max, 3),
        "band_0.5_0.7_spread_pp": round(band_spread, 3),
        "per_dataset": {
            ds: {
                "optimal_alpha": argmin_alpha(per_ds[ds])[0],
                "min_regret_pp": round(argmin_alpha(per_ds[ds])[1], 3),
                "regret_at_0.65_pp": round(at(per_ds[ds], 0.65), 3),
                "regret_at_0.70_pp": round(at(per_ds[ds], 0.70), 3),
                "drift_cost_0.65_vs_0.70_pp": round(abs(at(per_ds[ds], 0.65) - at(per_ds[ds], 0.70)), 3),
            } for ds in ds_names
        },
    }

    out_json = os.path.join(os.path.dirname(BASE_DIR), "results", "regret_vs_alpha.json")
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.axvspan(0.5, 0.7, color='0.90', zorder=0, label=r'$\alpha\in[0.5,0.7]$ band')
    colors = {'MVTec': '#1f77b4', 'VisA': '#ff7f0e', 'BTAD': '#2ca02c'}
    linestyles = {'MVTec': '--', 'VisA': '-.', 'BTAD': ':'}  # grayscale-safe distinction
    markers = {'MVTec': 'o', 'VisA': 's', 'BTAD': 'D'}
    for ds in ds_names:
        ax.plot(ALPHAS, per_ds[ds], color=colors.get(ds, None), ls=linestyles.get(ds, '-'),
                lw=1.6, alpha=0.9, label=ds)
        oa = argmin_alpha(per_ds[ds])[0]
        ax.plot(oa, at(per_ds[ds], oa), marker=markers.get(ds, 'o'), color=colors.get(ds, None), ms=5)
    ax.plot(ALPHAS, pooled, color='black', lw=2.4, label='Pooled (30 cat)')
    ax.axvline(0.65, color='crimson', ls='--', lw=1.2, label=r'fixed $\alpha=0.65$')
    ax.set_xlabel(r'Fusion weight $\alpha$ (0 = CLIP, 1 = DINOv2)')
    ax.set_ylabel('Mean regret vs per-category oracle (pp)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, min(8, max(pooled) * 1.05))
    ax.legend(loc='upper center', ncol=2, frameon=False)
    ax.grid(True, alpha=0.25)

    axin = ax.inset_axes([0.58, 0.30, 0.39, 0.40])
    axin.axvspan(0.5, 0.7, color='0.90', zorder=0)
    for ds in ds_names:
        axin.plot(ALPHAS, per_ds[ds], color=colors.get(ds, None), ls=linestyles.get(ds, '-'),
                  lw=1.4, alpha=0.9)
    axin.plot(ALPHAS, pooled, color='black', lw=2.0)
    axin.axvline(0.65, color='crimson', ls='--', lw=1.0)
    axin.axhline(pmin + 0.45, color='0.4', ls=':', lw=0.9)
    axin.set_xlim(0.45, 0.85)
    axin.set_ylim(0, 1.5)
    axin.set_title(r'plateau zoom ($\leq 0.45$ pp)', fontsize=8)
    axin.tick_params(labelsize=7)
    axin.grid(True, alpha=0.25)
    ax.indicate_inset_zoom(axin, edgecolor='0.5')
    fig.tight_layout()

    fig_dir = os.path.join(os.path.dirname(BASE_DIR), "paper", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"regret_vs_alpha.{ext}"), bbox_inches='tight')
    plt.close(fig)

    # ---- console summary ----
    print("=" * 70)
    print("  REGRET-VS-ALPHA")
    print("=" * 70)
    print(f"  Pooled optimal alpha = {pmin_a} (regret {pmin:.3f} pp)")
    print(f"  Pooled regret at 0.50 / 0.65 / 0.70: "
          f"{at(pooled,0.50):.3f} / {at(pooled,0.65):.3f} / {at(pooled,0.70):.3f} pp")
    print(f"  Alpha range within X pp of the optimum:")
    print(f"     within 0.10 pp: [{band_010[0]}, {band_010[1]}]")
    print(f"     within 0.20 pp: [{band_020[0]}, {band_020[1]}]")
    print(f"     within 0.45 pp: [{band_045[0]}, {band_045[1]}]")
    print(f"  Max pooled regret across alpha in [0.5,0.7]: {band_max:.3f} pp (spread {band_spread:.3f} pp)")
    print(f"\n  {'Dataset':<8} {'opt-alpha':>9} {'minRegret':>10} {'@0.65':>7} {'@0.70':>7} {'drift':>7}")
    for ds in ds_names:
        s = summary['per_dataset'][ds]
        print(f"  {ds:<8} {s['optimal_alpha']:>9} {s['min_regret_pp']:>10} "
              f"{s['regret_at_0.65_pp']:>7} {s['regret_at_0.70_pp']:>7} {s['drift_cost_0.65_vs_0.70_pp']:>7}")
    print(f"\n  Saved: {out_json}")
    print(f"  Saved: {fig_dir}/regret_vs_alpha.pdf (+ .png)")


if __name__ == "__main__":
    main()
