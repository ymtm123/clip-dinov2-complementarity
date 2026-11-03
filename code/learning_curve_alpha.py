"""learning curve alpha."""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leave_one_dataset_out_alpha import load_all_datasets, build_curves, oracle, ALPHAS, BASE_DIR

plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
                     'axes.titlesize': 12, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'legend.fontsize': 9, 'figure.dpi': 300})

RNG = np.random.RandomState(42)
N_DRAWS = 300


def global_alpha_on(curves, cats):
    best_a, best_m = 0.5, -1.0
    for a in ALPHAS:
        m = np.mean([curves[c][a] for c in cats])
        if m > best_m:
            best_m, best_a = m, a
    return best_a


def main():
    datasets = load_all_datasets()
    curves, _ = build_curves(datasets)
    cats = list(curves.keys())
    N = len(cats)
    oracle_auroc = {c: oracle(curves, c)[1] for c in cats}

    # reference: full-data global alpha + its regret over all 30
    a_full = global_alpha_on(curves, cats)
    full_regret = float(np.mean([oracle_auroc[c] - curves[c][a_full] for c in cats]))

    Ks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]
    rows = []
    for K in Ks:
        alphas, regrets = [], []
        for _ in range(N_DRAWS):
            idx = RNG.choice(N, size=K, replace=False)
            train = [cats[i] for i in idx]
            held = [cats[i] for i in range(N) if i not in idx]
            aK = global_alpha_on(curves, train)
            alphas.append(aK)
            regrets.append(np.mean([oracle_auroc[c] - curves[c][aK] for c in held]))
        rows.append({
            "K": K,
            "alpha_mean": round(float(np.mean(alphas)), 3),
            "alpha_std": round(float(np.std(alphas)), 3),
            "heldout_regret_mean_pp": round(float(np.mean(regrets)), 3),
            "heldout_regret_std_pp": round(float(np.std(regrets)), 3),
            "p_alpha_in_055_075": round(float(np.mean([(0.55 <= a <= 0.75) for a in alphas])), 3),
        })

    out = {
        "experiment": "learning curve for global-alpha estimation",
        "n_categories": N, "n_draws_per_K": N_DRAWS,
        "full_data_alpha": a_full, "full_data_regret_pp": round(full_regret, 3),
        "wrong_model_penalty_pp": 6.34,
        "curve": rows,
    }
    with open(os.path.join(os.path.dirname(BASE_DIR), "results", "learning_curve_alpha.json"), 'w') as f:
        json.dump(out, f, indent=2)

    # figure: two panels (alpha estimate, held-out regret) vs K
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    Kx = [r['K'] for r in rows]
    am = np.array([r['alpha_mean'] for r in rows]); asd = np.array([r['alpha_std'] for r in rows])
    rm = np.array([r['heldout_regret_mean_pp'] for r in rows]); rsd = np.array([r['heldout_regret_std_pp'] for r in rows])

    ax1.axhline(a_full, color='crimson', ls='--', lw=1.1, label=f'full-data α={a_full}')
    ax1.fill_between(Kx, am - asd, am + asd, color='#1f77b4', alpha=0.2)
    ax1.plot(Kx, am, '-o', color='#1f77b4', ms=4, label='estimated α (mean ± SD)')
    ax1.set_xlabel('# labeled categories K'); ax1.set_ylabel('Estimated global α')
    ax1.set_ylim(0.45, 0.85); ax1.grid(True, alpha=0.25); ax1.legend(frameon=False)

    ax2.axhline(full_regret, color='crimson', ls='--', lw=1.1, label=f'full-data regret={full_regret:.2f} pp')
    ax2.fill_between(Kx, rm - rsd, rm + rsd, color='#2ca02c', alpha=0.2)
    ax2.plot(Kx, rm, '-o', color='#2ca02c', ms=4, label='held-out regret (mean ± SD)')
    ax2.set_xlabel('# labeled categories K'); ax2.set_ylabel('Held-out regret (pp)')
    ax2.set_ylim(0, max(rm + rsd) * 1.1); ax2.grid(True, alpha=0.25); ax2.legend(frameon=False)
    fig.tight_layout()
    fig_dir = os.path.join(BASE_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"learning_curve_alpha.{ext}"), bbox_inches='tight')
    plt.close(fig)

    print(f"  full-data alpha={a_full}, regret={full_regret:.3f} pp")
    print(f"  {'K':>3} {'alpha(mean±sd)':>16} {'heldout regret(pp)':>20} {'P(a in[.55,.75])':>16}")
    for r in rows:
        print(f"  {r['K']:>3} {r['alpha_mean']:>7}±{r['alpha_std']:<7} "
              f"{r['heldout_regret_mean_pp']:>10}±{r['heldout_regret_std_pp']:<8} {r['p_alpha_in_055_075']:>16}")
    print("  Saved: results/learning_curve_alpha.json + figures/learning_curve_alpha.{pdf,png}")


if __name__ == "__main__":
    main()
