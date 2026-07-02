"""Full enumeration of all unsupervised fusion-weight strategies ."""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(BASE_DIR), "results")
REVDIR = os.path.dirname(BASE_DIR)

CYCLES = [
    ("unsupervised_alpha_discovery.json", "C1", "unsupervised_alpha_discovery.py",
     "Individual-model signals & pseudo-labeling"),
    ("alpha_discovery_cycle2.json", "C2", "alpha_discovery_cycle2.py",
     "Cross-model pseudo-labeling & anomaly-rate priors"),
    ("alpha_discovery_cycle3.json", "C3", "alpha_discovery_cycle3.py",
     "Direct unsupervised metric optimization (separation/Fisher/silhouette)"),
    ("alpha_discovery_cycle4.json", "C4", "alpha_discovery_cycle4.py",
     "Conditional score analysis (gradient/top-k/PCA/cond-variance)"),
    ("alpha_discovery_cycle5.json", "C5", "alpha_discovery_cycle5.py",
     "Non-linear fusion, sample-adaptive weighting & score calibration"),
    ("alpha_discovery_cycle6.json", "C6", "alpha_discovery_cycle6.py",
     "Power-mean family, alpha-ensembles & trimmed fusion"),
]


def main():
    rows = []
    seen = set()
    fixed_auroc = None

    # establish Fixed-0.65 baseline from data
    for fn, _, _, _ in CYCLES:
        d = json.load(open(os.path.join(RESULTS, fn), encoding='utf-8'))
        src = d.get('summary', d)
        f = src.get('Fixed-0.65')
        if isinstance(f, dict) and 'mean_auroc' in f:
            fixed_auroc = f['mean_auroc']
            break

    for fn, tag, script, theme in CYCLES:
        d = json.load(open(os.path.join(RESULTS, fn), encoding='utf-8'))
        src = d.get('summary', d)
        for m, v in src.items():
            if not isinstance(v, dict) or 'mean_auroc' not in v:
                continue
            if m in seen:
                continue
            seen.add(m)
            auroc = v['mean_auroc']
            regret = v.get('mean_regret', v.get('regret'))
            rows.append({
                "method": m,
                "cycle": tag,
                "defined_in": script,
                "theme": theme,
                "mean_auroc": round(auroc, 3),
                "mean_regret_pp": round(regret, 3) if regret is not None else None,
                "mae_alpha_vs_oracle": round(v['mae_alpha'], 3) if 'mae_alpha' in v else None,
                "pearson_r_alpha": round(v['pearson_r'], 3) if 'pearson_r' in v else None,
                "margin_vs_fixed065_pp": round(auroc - fixed_auroc, 4),
            })

    rows.sort(key=lambda r: -r['mean_auroc'])

    out = {
        "experiment": "full unsupervised-strategy enumeration",
        "n_categories": 30,
        "n_distinct_methods": len(rows),
        "fixed_065_auroc": round(fixed_auroc, 3),
        "best_non_fixed": next(r['method'] for r in rows if not r['method'].startswith('Fixed-0')),
        "max_margin_over_fixed_pp": max(r['margin_vs_fixed065_pp']
                                        for r in rows if not r['method'].startswith('Fixed-0')),
        "methods": rows,
    }
    with open(os.path.join(RESULTS, "unsupervised_strategies_full.json"), 'w') as f:
        json.dump(out, f, indent=2)

    # markdown supplement
    lines = []
    lines.append("# Supplementary Table — Unsupervised fusion-weight strategies (all 30 categories)\n")
    lines.append(f"All strategies evaluated across 6 discovery cycles (30 categories). "
                 f"**{len(rows)} distinct methods**; Fixed-α=0.65 baseline AUROC = {fixed_auroc:.3f}. "
                 f"No unsupervised strategy beats the fixed weight by more than "
                 f"{out['max_margin_over_fixed_pp']:+.3f} pp. Precise definitions are in the cited cycle script.\n")
    lines.append("| Rank | Method | Cycle | Mean AUROC | Mean regret (pp) | MAE α vs oracle | Margin vs Fixed-0.65 (pp) | Defined in |")
    lines.append("|---:|---|:--:|---:|---:|---:|---:|---|")
    for i, r in enumerate(rows, 1):
        reg = f"{r['mean_regret_pp']:.3f}" if r['mean_regret_pp'] is not None else "--"
        mae = f"{r['mae_alpha_vs_oracle']:.3f}" if r['mae_alpha_vs_oracle'] is not None else "--"
        mark = " **(fixed)**" if r['method'].startswith('Fixed-0') else ""
        lines.append(f"| {i} | {r['method']}{mark} | {r['cycle']} | {r['mean_auroc']:.3f} | {reg} | {mae} | "
                     f"{r['margin_vs_fixed065_pp']:+.4f} | `{r['defined_in']}` |")
    lines.append("\n## Cycle / paradigm map\n")
    for fn, tag, script, theme in CYCLES:
        n = sum(1 for r in rows if r['cycle'] == tag)
        lines.append(f"- **{tag}** ({n} methods) — {theme}. Source: `src/{script}`.")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(REVDIR, "unsupervised_strategies_table.md"), 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"Distinct methods: {len(rows)} | Fixed-0.65 AUROC {fixed_auroc:.3f} | "
          f"best non-fixed margin {out['max_margin_over_fixed_pp']:+.4f} pp")
    print("Top 5:", [(r['method'], r['mean_auroc']) for r in rows[:5]])
    print("Saved: results/unsupervised_strategies_full.json")
    print("Saved: paper/revision_round1/unsupervised_strategies_table.md")


if __name__ == "__main__":
    main()
