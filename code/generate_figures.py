"""Generate publication-quality figures for the paper."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Style settings for publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

src_dir = Path(__file__).parent
fig_dir = src_dir / 'figures'
fig_dir.mkdir(exist_ok=True)

# Load data
with open(src_dir / 'comprehensive_eval_results.json') as f:
    eval_data = json.load(f)

with open(src_dir / 'category_taxonomy_results.json') as f:
    tax_data = json.load(f)

taxonomy = tax_data['taxonomy']
oracle_alphas = tax_data['oracle_alphas']


# ============================================================
# Figure 1: Complementarity Scatter Plot
# ============================================================

def make_complementarity_scatter():
    """CLIP vs DINOv2 AUROC per category, colored by product type."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)

    datasets = [
        ('MVTec AD', 'mvtec_image', 'MVTec AD (15 categories)'),
        ('VisA', 'visa_image', 'VisA (12 categories)'),
        ('BTAD', 'btad_image', 'BTAD (3 categories)'),
    ]

    color_map = {'texture': '#E24A33', 'object': '#348ABD'}
    marker_map = {'texture': 's', 'object': 'o'}

    for idx, (eval_key, alpha_key, title) in enumerate(datasets):
        ax = axes[idx]

        clip_per = eval_data[eval_key]['CLIP only']['per_category']
        dino_per = eval_data[eval_key]['DINOv2 only']['per_category']

        for cat in clip_per:
            clip_val = clip_per[cat]
            dino_val = dino_per[cat]
            cat_type = taxonomy.get(cat, {}).get('type', 'object')
            color = color_map[cat_type]
            marker = marker_map[cat_type]

            ax.scatter(clip_val, dino_val, c=color, marker=marker,
                       s=60, edgecolors='black', linewidth=0.5, zorder=3)

            # Label points
            offset_x, offset_y = 0.5, 0.5
            # Adjust offsets for overlapping labels
            if cat in ['leather', 'carpet']:
                offset_y = -1.5
            elif cat in ['grid', 'hazelnut']:
                offset_x = -2.0
            elif cat in ['capsule']:
                offset_y = -1.5
            elif cat in ['pcb1', 'pcb2']:
                offset_x = 1.5
            elif cat in ['macaroni1', 'macaroni2']:
                offset_y = -1.5

            ax.annotate(cat, (clip_val, dino_val),
                        textcoords="offset points", xytext=(offset_x, offset_y),
                        fontsize=6.5, alpha=0.8)

        # Diagonal line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]

        # Set consistent limits
        if eval_key == 'MVTec AD':
            lims = [65, 102]
        elif eval_key == 'VisA':
            lims = [55, 102]
        else:
            lims = [82, 102]

        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.8, zorder=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Fill regions
        ax.fill_between(lims, lims, lims[1], alpha=0.05, color='#348ABD', zorder=0)
        ax.fill_between(lims, lims[0], lims, alpha=0.05, color='#E24A33', zorder=0)

        # Labels
        ax.set_xlabel('AnomalyCLIP AUROC (%)')
        if idx == 0:
            ax.set_ylabel('AnomalyDINO AUROC (%)')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        # Add region labels
        ax.text(0.05, 0.92, 'DINOv2 better', transform=ax.transAxes,
                fontsize=8, color='#348ABD', alpha=0.5, style='italic')
        ax.text(0.55, 0.05, 'CLIP better', transform=ax.transAxes,
                fontsize=8, color='#E24A33', alpha=0.5, style='italic')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E24A33',
               markersize=8, markeredgecolor='black', markeredgewidth=0.5, label='Texture'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#348ABD',
               markersize=8, markeredgecolor='black', markeredgewidth=0.5, label='Object'),
    ]
    axes[2].legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(fig_dir / 'complementarity_scatter.pdf', bbox_inches='tight')
    plt.savefig(fig_dir / 'complementarity_scatter.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 1: Complementarity scatter plot saved.")


# ============================================================
# Figure 2: Wrong Model Penalty + Fusion Benefit
# ============================================================

def make_wrong_model_penalty():
    """Bar chart showing penalty for choosing wrong model, sorted by magnitude."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Collect data from all datasets
    penalties = []
    for eval_key, alpha_key in [('MVTec AD', 'mvtec_image'), ('VisA', 'visa_image'), ('BTAD', 'btad_image')]:
        clip_per = eval_data[eval_key]['CLIP only']['per_category']
        dino_per = eval_data[eval_key]['DINOv2 only']['per_category']
        equal_per = eval_data[eval_key]['Equal (a=0.5)']['per_category']
        oracle_per = eval_data[eval_key].get('Oracle', {}).get('per_category', {})

        dataset_name = eval_key.replace(' AD', '')

        for cat in clip_per:
            clip_val = clip_per[cat]
            dino_val = dino_per[cat]
            equal_val = equal_per.get(cat, (clip_val + dino_val) / 2)
            oracle_val = oracle_per.get(cat, max(clip_val, dino_val))

            best_single = max(clip_val, dino_val)
            worst_single = min(clip_val, dino_val)
            penalty = best_single - worst_single
            fusion_gain = equal_val - best_single
            oracle_gap = oracle_val - equal_val

            penalties.append({
                'name': f"{cat} ({dataset_name})" if dataset_name != 'MVTec' else cat,
                'penalty': penalty,
                'fusion_gain': fusion_gain,
                'oracle_gap': oracle_gap,
                'dataset': dataset_name,
            })

    penalties.sort(key=lambda x: -x['penalty'])

    # Plot
    x = np.arange(len(penalties))
    width = 0.6

    colors_penalty = []
    for p in penalties:
        if p['dataset'] == 'MVTec':
            colors_penalty.append('#E24A33')
        elif p['dataset'] == 'VisA':
            colors_penalty.append('#348ABD')
        else:
            colors_penalty.append('#988ED5')

    bars = ax.bar(x, [p['penalty'] for p in penalties], width,
                  color=colors_penalty, alpha=0.7, edgecolor='black', linewidth=0.3)
    hatch_map = {'MVTec': '', 'VisA': '//', 'BTAD': 'xx'}  # grayscale-safe distinction
    for bar, p in zip(bars, penalties):
        bar.set_hatch(hatch_map.get(p['dataset'], ''))

    # Add horizontal lines
    mean_penalty = np.mean([p['penalty'] for p in penalties])
    ax.axhline(y=mean_penalty, color='red', linestyle='--', linewidth=1, alpha=0.7,
               label=f'Mean penalty: {mean_penalty:.1f}pp')
    ax.axhline(y=5.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    # Add Equal fusion regret line (0.76pp = mean regret across 3 datasets)
    ax.axhline(y=0.76, color='green', linestyle='-', linewidth=1.5, alpha=0.7,
               label='Equal fusion regret: 0.76pp')

    ax.set_xticks(x)
    ax.set_xticklabels([p['name'] for p in penalties], rotation=75, ha='right', fontsize=7)
    ax.set_ylabel('Wrong Model Penalty (pp)')
    ax.set_title('Cost of Choosing the Wrong Single Model vs. Risk with Equal Fusion')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E24A33', alpha=0.7, edgecolor='black', label='MVTec AD'),
        Patch(facecolor='#348ABD', alpha=0.7, edgecolor='black', hatch='//', label='VisA'),
        Patch(facecolor='#988ED5', alpha=0.7, edgecolor='black', hatch='xx', label='BTAD'),
        plt.Line2D([0], [0], color='red', linestyle='--', label=f'Mean penalty: {mean_penalty:.1f}pp'),
        plt.Line2D([0], [0], color='green', linestyle='-', linewidth=1.5, label='Equal fusion regret: 0.76pp'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    ax.set_ylim(0, 32)
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_dir / 'wrong_model_penalty.pdf', bbox_inches='tight')
    plt.savefig(fig_dir / 'wrong_model_penalty.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 2: Wrong model penalty chart saved.")


# ============================================================
# Figure 3: Level Reversal Visualization
# ============================================================

def make_level_reversal():
    """Arrow plot showing image-level alpha vs pixel-level alpha for each category."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for idx, (dataset_name, img_key, pix_key) in enumerate([
        ('MVTec AD', 'mvtec_image', 'mvtec_pixel'),
        ('VisA', 'visa_image', 'visa_pixel'),
    ]):
        ax = axes[idx]

        img_alphas = oracle_alphas[img_key]
        pix_alphas = oracle_alphas[pix_key]

        common_cats = sorted(set(img_alphas.keys()) & set(pix_alphas.keys()))

        for cat in common_cats:
            img_a = img_alphas[cat]
            pix_a = pix_alphas[cat]

            cat_type = taxonomy.get(cat, {}).get('type', 'object')
            color = '#E24A33' if cat_type == 'texture' else '#348ABD'

            # Determine if reversal
            img_prefers_dino = img_a > 0.5
            pix_prefers_dino = pix_a > 0.5
            reversal = img_prefers_dino != pix_prefers_dino

            marker = 'D' if reversal else 'o'
            edgecolor = 'red' if reversal else 'black'
            linewidth = 1.5 if reversal else 0.5

            ax.scatter(img_a, pix_a, c=color, marker=marker,
                       s=70, edgecolors=edgecolor, linewidth=linewidth, zorder=3)

            ax.annotate(cat, (img_a, pix_a),
                        textcoords="offset points", xytext=(2, 3),
                        fontsize=7, alpha=0.8)

        # Quadrant lines
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.4)

        # Diagonal
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=0.8)

        # Quadrant labels
        ax.text(0.02, 0.98, 'CLIP\nboth levels', transform=ax.transAxes,
                fontsize=7, color='gray', alpha=0.5, va='top')
        ax.text(0.98, 0.02, 'DINOv2\nboth levels', transform=ax.transAxes,
                fontsize=7, color='gray', alpha=0.5, ha='right')
        ax.text(0.02, 0.02, 'CLIP image\nDINOv2 pixel', transform=ax.transAxes,
                fontsize=7, color='red', alpha=0.4)
        ax.text(0.98, 0.98, 'DINOv2 image\nCLIP pixel', transform=ax.transAxes,
                fontsize=7, color='red', alpha=0.4, ha='right', va='top')

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'Image-level oracle $\alpha^*$ ($\rightarrow$ DINOv2)')
        ax.set_ylabel(r'Pixel-level oracle $\alpha^*$ ($\rightarrow$ DINOv2)')
        ax.set_title(dataset_name)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15)

        # Count reversals (exclude boundary cases where alpha* = 0.5 exactly)
        n_reversal = sum(1 for c in common_cats
                         if img_alphas[c] != 0.5 and pix_alphas[c] != 0.5
                         and (img_alphas[c] > 0.5) != (pix_alphas[c] > 0.5))
        n_total = len(common_cats)

        ax.text(0.5, 0.95, f'Reversals: {n_reversal}/{n_total} ({n_reversal/n_total*100:.0f}%)',
                transform=ax.transAxes, fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E24A33',
               markersize=8, markeredgecolor='black', markeredgewidth=0.5, label='Texture'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#348ABD',
               markersize=8, markeredgecolor='black', markeredgewidth=0.5, label='Object'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
               markersize=8, markeredgecolor='red', markeredgewidth=1.5, label='Reversal'),
    ]
    axes[1].legend(handles=legend_elements, loc='center left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(fig_dir / 'level_reversal.pdf', bbox_inches='tight')
    plt.savefig(fig_dir / 'level_reversal.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 3: Level reversal visualization saved.")


def make_compensatory_pattern():
    """Scatter plot: kNN AUROC vs Method Premium for CLIP and DINOv2, both datasets."""

    # Load cross-dataset premium results
    with open(src_dir / 'cross_dataset_premium_results.json') as f:
        premium_data = json.load(f)

    results = premium_data['experiment_14_cross_dataset_premium']['per_category']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax_idx, (model_name, knn_key, prem_key, color) in enumerate([
        ('CLIP (AnomalyCLIP)', 'clip_knn', 'clip_premium', '#E24A33'),
        ('DINOv2 (AnomalyDINO)', 'dino_knn', 'dino_premium', '#348ABD'),
    ]):
        ax = axes[ax_idx]

        mvtec = [r for r in results if r['dataset'] == 'MVTec AD']
        visa = [r for r in results if r['dataset'] == 'VisA']

        # Plot MVTec points
        x_m = [r[knn_key] for r in mvtec]
        y_m = [r[prem_key] for r in mvtec]
        ax.scatter(x_m, y_m, c=color, marker='o', s=50, alpha=0.8,
                   edgecolors='white', linewidths=0.5, label='MVTec AD', zorder=3)

        # Plot VisA points
        x_v = [r[knn_key] for r in visa]
        y_v = [r[prem_key] for r in visa]
        ax.scatter(x_v, y_v, c=color, marker='s', s=50, alpha=0.5,
                   edgecolors='white', linewidths=0.5, label='VisA', zorder=3)

        # Combined regression line
        x_all = np.array(x_m + x_v)
        y_all = np.array(y_m + y_v)

        from scipy import stats as sp_stats
        slope, intercept, r_val, p_val, _ = sp_stats.linregress(x_all, y_all)

        x_line = np.linspace(x_all.min() - 2, x_all.max() + 2, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)

        # Zero line
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, zorder=1)

        # Labels for notable points
        all_pts = mvtec + visa
        # Label the most extreme negative premium points
        for r in all_pts:
            if r[prem_key] < -5 or r[prem_key] > 25:
                ax.annotate(r['category'][:8],
                            (r[knn_key], r[prem_key]),
                            fontsize=7, ha='center', va='bottom',
                            xytext=(0, 4), textcoords='offset points')

        ax.set_xlabel('kNN AUROC (%)', fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel('Method Premium (pp)', fontsize=11)

        if p_val < 0.001:
            p_str = '$p < 0.001$'
        elif p_val < 0.01:
            p_str = f'$p = {p_val:.3f}$'
        else:
            p_str = f'$p = {p_val:.2f}$'
        ax.set_title(f'{model_name}\n$r = {r_val:.2f}$, {p_str} (N={len(x_all)})', fontsize=11)

        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2, zorder=0)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(fig_dir / f'compensatory_pattern.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] compensatory_pattern.pdf/png saved")


def make_fusion_gain_landscape():
    """Figure 5: Fusion gain vs category difficulty, colored by model balance.

    Shows that fusion helps most when categories are difficult and models
    are balanced (|alpha-0.5| small). Two panels: scatter + bar chart.
    """
    from scipy import stats as sp_stats

    # Build data table for all 30 categories
    rows = []
    for ds_key, alpha_key in [("MVTec AD", "mvtec_image"), ("VisA", "visa_image"), ("BTAD", "btad_image")]:
        ds = eval_data[ds_key]
        categories = list(ds["CLIP only"]["per_category"].keys())
        for cat in categories:
            clip = ds["CLIP only"]["per_category"][cat]
            dino = ds["DINOv2 only"]["per_category"][cat]
            oracle = ds["Oracle"]["per_category"][cat]
            best = max(clip, dino)
            gain = oracle - best
            alpha = oracle_alphas[alpha_key].get(cat, 0.5)
            extremity = abs(alpha - 0.5)
            cat_type = taxonomy.get(cat, {}).get("type", "object")
            rows.append({
                "category": cat, "dataset": ds_key, "type": cat_type,
                "best_single": best, "fusion_gain": gain,
                "oracle_alpha": alpha, "alpha_extremity": extremity
            })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel A: Fusion gain vs best single AUROC ---
    ax = axes[0]
    for r in rows:
        marker = 's' if r["type"] == "texture" else 'o'
        # Color by alpha extremity: darker = more extreme = less balanced
        balance = 1 - r["alpha_extremity"] * 2  # 0=extreme, 1=balanced
        color_val = plt.cm.RdYlGn(balance * 0.8 + 0.1)  # avoid extremes
        ds_marker_map = {"MVTec AD": 'o', "VisA": 's', "BTAD": 'D'}
        m = ds_marker_map[r["dataset"]]
        ax.scatter(r["best_single"], r["fusion_gain"], c=[color_val],
                   marker=m, s=55, edgecolors='black', linewidths=0.3,
                   alpha=0.85, zorder=3)

    # Regression line
    x = np.array([r["best_single"] for r in rows])
    y = np.array([r["fusion_gain"] for r in rows])
    slope, intercept, r_val, p_val, _ = sp_stats.linregress(x, y)
    rho_val, rho_p = sp_stats.spearmanr(x, y)
    x_line = np.linspace(x.min() - 1, x.max() + 1, 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1.5, alpha=0.5, zorder=2)

    ax.set_xlabel('Best Single-Model AUROC (%)')
    ax.set_ylabel('Oracle Fusion Gain (pp)')
    ax.set_title(f'Difficulty predicts fusion gain\n$\\rho = {rho_val:.2f}$, $p < 0.0001$ (N=30)')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.15)

    # Legend for dataset markers
    from matplotlib.lines import Line2D
    legend_ds = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=7, markeredgecolor='black', markeredgewidth=0.3, label='MVTec AD'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=7, markeredgecolor='black', markeredgewidth=0.3, label='VisA'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
               markersize=7, markeredgecolor='black', markeredgewidth=0.3, label='BTAD'),
    ]
    ax.legend(handles=legend_ds, loc='upper right', fontsize=8)

    # Colorbar for balance
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                                norm=plt.Normalize(vmin=0, vmax=0.5))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r'$|\alpha^* - 0.5|$ (model imbalance)', fontsize=9)
    cbar.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # --- Panel B: Cross-dataset fusion improvement consistency ---
    ax2 = axes[1]

    datasets = ["MVTec AD", "VisA", "BTAD"]
    strategies = {
        "CLIP only": '#E24A33',
        "DINOv2 only": '#348ABD',
        "Equal (a=0.5)": '#988ED5',
        "Fixed (a=0.65)": '#8EBA42',
    }

    x_pos = np.arange(len(datasets))
    width = 0.15
    offsets = np.arange(len(strategies)) - len(strategies) / 2 + 0.5

    bar_hatches = ['', '//', '\\\\', 'xx']  # grayscale-safe distinction per strategy
    for i, (strat, color) in enumerate(strategies.items()):
        vals = [eval_data[ds][strat]["mean_auroc"] for ds in datasets]
        label = strat.replace("(a=0.5)", "($\\alpha$=0.5)").replace("(a=0.65)", "($\\alpha$=0.65)")
        ax2.bar(x_pos + offsets[i] * width, vals, width * 0.9,
                label=label, color=color, edgecolor='black', linewidth=0.5,
                hatch=bar_hatches[i % len(bar_hatches)])

    # Oracle line
    oracle_vals = [eval_data[ds]["Oracle"]["mean_auroc"] for ds in datasets]
    ax2.plot(x_pos, oracle_vals, 'k^', markersize=8, zorder=5, label='Oracle')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(datasets)
    ax2.set_ylabel('Mean AUROC (%)')
    ax2.set_title('Cross-dataset consistency\n(ranking preserved: $\\tau \\geq 0.87$, $p \\leq 0.017$)')
    ax2.legend(fontsize=7, loc='lower left', ncol=2)
    ax2.set_ylim(78, 100)
    ax2.grid(True, alpha=0.15, axis='y')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(fig_dir / f'fusion_gain_landscape.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] fusion_gain_landscape.pdf/png saved")


if __name__ == '__main__':
    make_complementarity_scatter()
    make_wrong_model_penalty()
    make_level_reversal()
    make_compensatory_pattern()
    make_fusion_gain_landscape()
    print("\nAll figures generated successfully!")
