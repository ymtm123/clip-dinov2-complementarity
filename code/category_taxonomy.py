"""
Phase 2-2: Category Taxonomy & VFM Dominance Analysis
=====================================================
Classifies 30 categories across 3 datasets by:
- Type: Texture vs Object
- Structure: Fine vs Coarse defects
- Semantic: Color-based vs Shape-based anomalies
- Scale: Small vs Large objects
Analyzes correlation with VFM dominance patterns.
"""

import json
import os
import numpy as np
from scipy import stats
from collections import defaultdict


# Category taxonomy based on domain knowledge
CATEGORY_TAXONOMY = {
    # MVTec AD (15 categories)
    'bottle':     {'dataset': 'MVTec', 'type': 'object', 'structure': 'coarse', 'defect_scale': 'large', 'semantic': 'shape'},
    'cable':      {'dataset': 'MVTec', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'shape'},
    'capsule':    {'dataset': 'MVTec', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'mixed'},
    'carpet':     {'dataset': 'MVTec', 'type': 'texture', 'structure': 'fine',  'defect_scale': 'small', 'semantic': 'color'},
    'grid':       {'dataset': 'MVTec', 'type': 'texture', 'structure': 'fine',  'defect_scale': 'small', 'semantic': 'shape'},
    'hazelnut':   {'dataset': 'MVTec', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'shape'},
    'leather':    {'dataset': 'MVTec', 'type': 'texture', 'structure': 'coarse','defect_scale': 'large', 'semantic': 'color'},
    'metal_nut':  {'dataset': 'MVTec', 'type': 'object', 'structure': 'coarse', 'defect_scale': 'large', 'semantic': 'shape'},
    'pill':       {'dataset': 'MVTec', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'mixed'},
    'screw':      {'dataset': 'MVTec', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'shape'},
    'tile':       {'dataset': 'MVTec', 'type': 'texture', 'structure': 'coarse','defect_scale': 'large', 'semantic': 'color'},
    'toothbrush': {'dataset': 'MVTec', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'shape'},
    'transistor': {'dataset': 'MVTec', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'mixed'},
    'wood':       {'dataset': 'MVTec', 'type': 'texture', 'structure': 'coarse','defect_scale': 'large', 'semantic': 'color'},
    'zipper':     {'dataset': 'MVTec', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'shape'},

    # VisA (12 categories)
    'candle':     {'dataset': 'VisA', 'type': 'object', 'structure': 'coarse', 'defect_scale': 'large', 'semantic': 'mixed'},
    'capsules':   {'dataset': 'VisA', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'color'},
    'cashew':     {'dataset': 'VisA', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'shape'},
    'chewinggum': {'dataset': 'VisA', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'shape'},
    'fryum':      {'dataset': 'VisA', 'type': 'object', 'structure': 'coarse', 'defect_scale': 'large', 'semantic': 'shape'},
    'macaroni1':  {'dataset': 'VisA', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'shape'},
    'macaroni2':  {'dataset': 'VisA', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'shape'},
    'pcb1':       {'dataset': 'VisA', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'mixed'},
    'pcb2':       {'dataset': 'VisA', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'mixed'},
    'pcb3':       {'dataset': 'VisA', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'mixed'},
    'pcb4':       {'dataset': 'VisA', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'mixed'},
    'pipe_fryum': {'dataset': 'VisA', 'type': 'object', 'structure': 'coarse', 'defect_scale': 'large', 'semantic': 'shape'},

    # BTAD (3 categories)
    '01':         {'dataset': 'BTAD', 'type': 'object', 'structure': 'coarse', 'defect_scale': 'large', 'semantic': 'shape'},
    '02':         {'dataset': 'BTAD', 'type': 'texture', 'structure': 'fine',  'defect_scale': 'small', 'semantic': 'color'},
    '03':         {'dataset': 'BTAD', 'type': 'object', 'structure': 'fine',   'defect_scale': 'small', 'semantic': 'mixed'},
}


def load_oracle_alphas(base_dir):
    """Load oracle alphas for all datasets and levels."""
    alphas = {}

    # Image-level: adaptive_fusion_results.json
    adaptive_path = os.path.join(base_dir, "adaptive_fusion_results.json")
    if os.path.exists(adaptive_path):
        with open(adaptive_path) as f:
            data = json.load(f)
        for ds in ['mvtec', 'visa']:
            if ds in data and 'oracle_alphas' in data[ds]:
                alphas[f'{ds}_image'] = data[ds]['oracle_alphas']

    # BTAD image-level
    btad_path = os.path.join(base_dir, "btad_fusion_results.json")
    if os.path.exists(btad_path):
        with open(btad_path) as f:
            data = json.load(f)
        if 'btad_image_level' in data:
            alphas['btad_image'] = data['btad_image_level'].get('oracle_alphas', {})

    # Pixel-level
    pix_path = os.path.join(base_dir, "pixel_fusion_results.json")
    if os.path.exists(pix_path):
        with open(pix_path) as f:
            data = json.load(f)
        for ds in ['mvtec', 'visa', 'btad']:
            if ds in data and 'oracle_alphas' in data[ds]:
                alphas[f'{ds}_pixel'] = data[ds]['oracle_alphas']

    return alphas


def analyze_taxonomy_vs_dominance(alphas):
    """Analyze correlation between category attributes and VFM dominance."""
    print("=" * 80)
    print("  CATEGORY TAXONOMY vs VFM DOMINANCE ANALYSIS")
    print("=" * 80)

    # Collect data for analysis
    for level in ['image', 'pixel']:
        print(f"\n\n{'='*60}")
        print(f"  {level.upper()}-LEVEL ANALYSIS")
        print(f"{'='*60}")

        all_cats = []
        all_alphas = []
        all_attrs = defaultdict(list)

        for ds_key in [f'mvtec_{level}', f'visa_{level}', f'btad_{level}']:
            if ds_key not in alphas:
                continue
            for cat, alpha in alphas[ds_key].items():
                if cat in CATEGORY_TAXONOMY:
                    all_cats.append(cat)
                    all_alphas.append(alpha)
                    attrs = CATEGORY_TAXONOMY[cat]
                    for attr_name, attr_val in attrs.items():
                        if attr_name != 'dataset':
                            all_attrs[attr_name].append(attr_val)

        if not all_cats:
            continue

        all_alphas = np.array(all_alphas)

        # --- Type: Texture vs Object ---
        print(f"\n  --- Type: Texture vs Object ---")
        tex_alphas = [a for a, t in zip(all_alphas, all_attrs['type']) if t == 'texture']
        obj_alphas = [a for a, t in zip(all_alphas, all_attrs['type']) if t == 'object']

        print(f"  Texture ({len(tex_alphas)} cats): mean alpha = {np.mean(tex_alphas):.3f} +/- {np.std(tex_alphas):.3f}")
        print(f"  Object  ({len(obj_alphas)} cats): mean alpha = {np.mean(obj_alphas):.3f} +/- {np.std(obj_alphas):.3f}")

        if len(tex_alphas) >= 2 and len(obj_alphas) >= 2:
            u_stat, u_p = stats.mannwhitneyu(tex_alphas, obj_alphas, alternative='two-sided')
            print(f"  Mann-Whitney U: U={u_stat:.1f}, p={u_p:.4f}")

        tex_clip_dom = sum(1 for a in tex_alphas if a < 0.5)
        tex_dino_dom = sum(1 for a in tex_alphas if a > 0.5)
        obj_clip_dom = sum(1 for a in obj_alphas if a < 0.5)
        obj_dino_dom = sum(1 for a in obj_alphas if a > 0.5)
        print(f"  Texture: CLIP-dom={tex_clip_dom}, DINO-dom={tex_dino_dom}")
        print(f"  Object:  CLIP-dom={obj_clip_dom}, DINO-dom={obj_dino_dom}")

        # --- Structure: Fine vs Coarse ---
        print(f"\n  --- Structure: Fine vs Coarse defects ---")
        fine_alphas = [a for a, s in zip(all_alphas, all_attrs['structure']) if s == 'fine']
        coarse_alphas = [a for a, s in zip(all_alphas, all_attrs['structure']) if s == 'coarse']

        print(f"  Fine   ({len(fine_alphas)} cats): mean alpha = {np.mean(fine_alphas):.3f} +/- {np.std(fine_alphas):.3f}")
        print(f"  Coarse ({len(coarse_alphas)} cats): mean alpha = {np.mean(coarse_alphas):.3f} +/- {np.std(coarse_alphas):.3f}")

        if len(fine_alphas) >= 2 and len(coarse_alphas) >= 2:
            u_stat, u_p = stats.mannwhitneyu(fine_alphas, coarse_alphas, alternative='two-sided')
            print(f"  Mann-Whitney U: U={u_stat:.1f}, p={u_p:.4f}")

        # --- Defect Scale: Small vs Large ---
        print(f"\n  --- Defect Scale: Small vs Large ---")
        small_alphas = [a for a, s in zip(all_alphas, all_attrs['defect_scale']) if s == 'small']
        large_alphas = [a for a, s in zip(all_alphas, all_attrs['defect_scale']) if s == 'large']

        print(f"  Small ({len(small_alphas)} cats): mean alpha = {np.mean(small_alphas):.3f} +/- {np.std(small_alphas):.3f}")
        print(f"  Large ({len(large_alphas)} cats): mean alpha = {np.mean(large_alphas):.3f} +/- {np.std(large_alphas):.3f}")

        if len(small_alphas) >= 2 and len(large_alphas) >= 2:
            u_stat, u_p = stats.mannwhitneyu(small_alphas, large_alphas, alternative='two-sided')
            print(f"  Mann-Whitney U: U={u_stat:.1f}, p={u_p:.4f}")

        # --- Semantic: Color vs Shape vs Mixed ---
        print(f"\n  --- Semantic: Color vs Shape vs Mixed ---")
        color_alphas = [a for a, s in zip(all_alphas, all_attrs['semantic']) if s == 'color']
        shape_alphas = [a for a, s in zip(all_alphas, all_attrs['semantic']) if s == 'shape']
        mixed_alphas = [a for a, s in zip(all_alphas, all_attrs['semantic']) if s == 'mixed']

        if color_alphas:
            print(f"  Color ({len(color_alphas)} cats): mean alpha = {np.mean(color_alphas):.3f} +/- {np.std(color_alphas):.3f}")
        if shape_alphas:
            print(f"  Shape ({len(shape_alphas)} cats): mean alpha = {np.mean(shape_alphas):.3f} +/- {np.std(shape_alphas):.3f}")
        if mixed_alphas:
            print(f"  Mixed ({len(mixed_alphas)} cats): mean alpha = {np.mean(mixed_alphas):.3f} +/- {np.std(mixed_alphas):.3f}")

        if len(color_alphas) >= 2 and len(shape_alphas) >= 2:
            u_stat, u_p = stats.mannwhitneyu(color_alphas, shape_alphas, alternative='two-sided')
            print(f"  Color vs Shape Mann-Whitney U: U={u_stat:.1f}, p={u_p:.4f}")

        # --- Kruskal-Wallis across all semantic groups ---
        groups = [g for g in [color_alphas, shape_alphas, mixed_alphas] if len(g) >= 2]
        if len(groups) >= 2:
            kw_stat, kw_p = stats.kruskal(*groups)
            print(f"  Kruskal-Wallis H={kw_stat:.2f}, p={kw_p:.4f}")

        # Full table
        print(f"\n  --- Full Category Table ({level}-level) ---")
        print(f"  {'Category':<15} {'Dataset':<8} {'Type':<10} {'Struct':<8} {'Scale':<7} {'Semantic':<8} {'Alpha':>6} {'Dominant':>10}")
        print(f"  {'-'*80}")
        for cat, alpha in sorted(zip(all_cats, all_alphas), key=lambda x: x[1]):
            attrs = CATEGORY_TAXONOMY[cat]
            dominant = "DINOv2" if alpha > 0.5 else ("CLIP" if alpha < 0.5 else "equal")
            print(f"  {cat:<15} {attrs['dataset']:<8} {attrs['type']:<10} {attrs['structure']:<8} "
                  f"{attrs['defect_scale']:<7} {attrs['semantic']:<8} {alpha:>5.2f}  {dominant:>10}")


def practical_guidelines(alphas):
    """Generate practical guidelines for VFM selection."""
    print("\n\n" + "=" * 80)
    print("  PRACTICAL GUIDELINES FOR VFM SELECTION")
    print("=" * 80)

    # Aggregate patterns
    patterns = defaultdict(lambda: {'clip_count': 0, 'dino_count': 0, 'alphas': []})

    for level in ['image', 'pixel']:
        for ds_key in [f'mvtec_{level}', f'visa_{level}', f'btad_{level}']:
            if ds_key not in alphas:
                continue
            for cat, alpha in alphas[ds_key].items():
                if cat not in CATEGORY_TAXONOMY:
                    continue
                attrs = CATEGORY_TAXONOMY[cat]
                for attr_name in ['type', 'structure', 'defect_scale', 'semantic']:
                    key = f"{attr_name}={attrs[attr_name]}_{level}"
                    patterns[key]['alphas'].append(alpha)
                    if alpha < 0.5:
                        patterns[key]['clip_count'] += 1
                    elif alpha > 0.5:
                        patterns[key]['dino_count'] += 1

    print("\n  Attribute-Level VFM Preference:")
    print(f"  {'Attribute':<30} {'Level':<8} {'Mean Alpha':>10} {'CLIP':>6} {'DINO':>6} {'Recommendation':<20}")
    print(f"  {'-'*85}")

    for key in sorted(patterns.keys()):
        p = patterns[key]
        attr_part, level = key.rsplit('_', 1)
        mean_alpha = np.mean(p['alphas'])
        if mean_alpha > 0.6:
            rec = "DINOv2-heavy"
        elif mean_alpha < 0.4:
            rec = "CLIP-heavy"
        else:
            rec = "Equal fusion"
        print(f"  {attr_part:<30} {level:<8} {mean_alpha:>10.3f} {p['clip_count']:>6} {p['dino_count']:>6} {rec:<20}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading oracle alphas...")
    alphas = load_oracle_alphas(base_dir)
    print(f"Loaded: {list(alphas.keys())}")
    for k, v in alphas.items():
        print(f"  {k}: {len(v)} categories")

    # Main analysis
    analyze_taxonomy_vs_dominance(alphas)

    # Practical guidelines
    practical_guidelines(alphas)

    # Save taxonomy
    output = {
        'taxonomy': {k: v for k, v in CATEGORY_TAXONOMY.items()},
        'oracle_alphas': {k: v for k, v in alphas.items()},
    }
    output_path = os.path.join(base_dir, "category_taxonomy_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
