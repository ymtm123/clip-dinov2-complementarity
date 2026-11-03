"""Cross-analysis of all experiment results."""
import json
import numpy as np
from pathlib import Path

# Load all results
with open(Path(__file__).parent / 'deep_analysis_cycle3_results.json', 'r') as f:
    c3 = json.load(f)
with open(Path(__file__).parent / 'deep_analysis_cycle2_results.json', 'r') as f:
    c2 = json.load(f)
with open(Path(__file__).parent / 'deep_analysis_results.json', 'r') as f:
    data = json.load(f)

exp9_breakdown = c3['experiment_9_difficulty']['defect_type_breakdown']
exp7 = c2['experiment_7_defect_taxonomy']
k_shot = c3.get('experiment_8_kshot', {})

defect_taxonomy = {
    ('bottle', 'broken_large'): 'structural', ('bottle', 'broken_small'): 'structural',
    ('bottle', 'contamination'): 'foreign_object',
    ('cable', 'bent_wire'): 'geometric', ('cable', 'cable_swap'): 'semantic',
    ('cable', 'combined'): 'mixed', ('cable', 'cut_inner_insulation'): 'geometric',
    ('cable', 'cut_outer_insulation'): 'geometric', ('cable', 'missing_cable'): 'semantic',
    ('cable', 'missing_wire'): 'semantic', ('cable', 'poke_insulation'): 'structural',
    ('capsule', 'crack'): 'structural', ('capsule', 'faulty_imprint'): 'color',
    ('capsule', 'poke'): 'structural', ('capsule', 'scratch'): 'texture_surface',
    ('capsule', 'squeeze'): 'geometric',
    ('carpet', 'color'): 'color', ('carpet', 'cut'): 'structural',
    ('carpet', 'hole'): 'structural', ('carpet', 'metal_contamination'): 'foreign_object',
    ('carpet', 'thread'): 'texture_surface',
    ('grid', 'bent'): 'geometric', ('grid', 'broken'): 'structural',
    ('grid', 'glue'): 'foreign_object', ('grid', 'metal_contamination'): 'foreign_object',
    ('grid', 'thread'): 'texture_surface',
    ('hazelnut', 'crack'): 'structural', ('hazelnut', 'cut'): 'structural',
    ('hazelnut', 'hole'): 'structural', ('hazelnut', 'print'): 'color',
    ('leather', 'color'): 'color', ('leather', 'cut'): 'structural',
    ('leather', 'fold'): 'geometric', ('leather', 'glue'): 'foreign_object',
    ('leather', 'poke'): 'structural',
    ('metal_nut', 'bent'): 'geometric', ('metal_nut', 'color'): 'color',
    ('metal_nut', 'flip'): 'semantic', ('metal_nut', 'scratch'): 'texture_surface',
    ('pill', 'color'): 'color', ('pill', 'combined'): 'mixed',
    ('pill', 'contamination'): 'foreign_object', ('pill', 'crack'): 'structural',
    ('pill', 'faulty_imprint'): 'color', ('pill', 'pill_type'): 'semantic',
    ('pill', 'scratch'): 'texture_surface',
    ('screw', 'manipulated_front'): 'geometric', ('screw', 'scratch_head'): 'texture_surface',
    ('screw', 'scratch_neck'): 'texture_surface', ('screw', 'thread_side'): 'texture_surface',
    ('screw', 'thread_top'): 'texture_surface',
    ('tile', 'crack'): 'structural', ('tile', 'glue_strip'): 'foreign_object',
    ('tile', 'gray_stroke'): 'color', ('tile', 'oil'): 'foreign_object',
    ('tile', 'rough'): 'texture_surface',
    ('toothbrush', 'defective'): 'structural',
    ('transistor', 'bent_lead'): 'geometric', ('transistor', 'cut_lead'): 'geometric',
    ('transistor', 'damaged_case'): 'structural', ('transistor', 'misplaced'): 'semantic',
    ('wood', 'color'): 'color', ('wood', 'combined'): 'mixed',
    ('wood', 'hole'): 'structural', ('wood', 'liquid'): 'foreign_object',
    ('wood', 'scratch'): 'texture_surface',
    ('zipper', 'broken_teeth'): 'structural', ('zipper', 'combined'): 'mixed',
    ('zipper', 'fabric_border'): 'texture_surface', ('zipper', 'fabric_interior'): 'texture_surface',
    ('zipper', 'rough'): 'texture_surface', ('zipper', 'split_teeth'): 'structural',
    ('zipper', 'squeezed_teeth'): 'geometric',
}

# Exclusive detection by anomaly nature
print("=" * 80)
print("Exclusive Detection Rate by Anomaly Nature")
print("=" * 80)

nature_excl = {}
for item in exp9_breakdown:
    defect = item['defect']
    cat, dt = defect.split('/')
    key = (cat, dt)
    if key not in defect_taxonomy:
        continue
    nature = defect_taxonomy[key]
    if nature not in nature_excl:
        nature_excl[nature] = {'clip_only': 0, 'dino_only': 0, 'both': 0, 'neither': 0, 'total': 0}
    nature_excl[nature]['clip_only'] += item.get('clip_only', 0)
    nature_excl[nature]['dino_only'] += item.get('dino_only', 0)
    nature_excl[nature]['both'] += item.get('both', 0)
    nature_excl[nature]['neither'] += item.get('neither', 0)
    nature_excl[nature]['total'] += item['total']

print(f"{'Nature':20s} {'Total':>5s} {'Both%':>6s} {'CLIP%':>6s} {'DINO%':>6s} {'Neither%':>8s} {'Excl%':>6s}")
print("-" * 60)
for nature in ['color', 'texture_surface', 'structural', 'geometric', 'foreign_object', 'semantic', 'mixed']:
    if nature in nature_excl:
        d = nature_excl[nature]
        t = d['total']
        print(f"{nature:20s} {t:5d} {d['both']/t*100:5.1f}% {d['clip_only']/t*100:5.1f}% "
              f"{d['dino_only']/t*100:5.1f}% {d['neither']/t*100:7.1f}% "
              f"{(d['clip_only']+d['dino_only'])/t*100:5.1f}%")

# Comprehensive per-nature summary
print()
print("=" * 80)
print("Comprehensive: Anomaly Nature x VFM Performance")
print("=" * 80)
for nature in ['color', 'texture_surface', 'structural', 'geometric', 'foreign_object', 'semantic']:
    if nature in exp7 and nature in nature_excl:
        e7 = exp7[nature]
        ne = nature_excl[nature]
        t = ne['total']
        print(f"  {nature}:")
        print(f"    AUROC diff = {e7['mean_diff']:+.2f}, CLIP win rate = {e7['clip_win_rate']:.0f}%")
        print(f"    Detection: CLIP-only={ne['clip_only']/t*100:.1f}% DINO-only={ne['dino_only']/t*100:.1f}% "
              f"Both={ne['both']/t*100:.1f}% Neither={ne['neither']/t*100:.1f}%")
        p_val = e7.get('p_value')
        if p_val:
            print(f"    t-test p-value: {p_val}")
        print()

# k-shot by category type
print("=" * 80)
print("k-shot Analysis by Category Type")
print("=" * 80)

texture_cats = ['carpet', 'grid', 'leather', 'tile', 'wood']
object_cats = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

for group_name, group_cats in [('Texture', texture_cats), ('Object', object_cats)]:
    print(f"\n{group_name} categories:")
    for k in ['1', '2', '4', '8']:
        clip_aucs = [k_shot[c][k]['clip_auroc'] for c in group_cats if c in k_shot and k in k_shot[c]]
        dino_aucs = [k_shot[c][k]['dino_auroc'] for c in group_cats if c in k_shot and k in k_shot[c]]
        gains = [k_shot[c][k]['fusion_gain'] for c in group_cats if c in k_shot and k in k_shot[c]]
        clip_w = sum(1 for c in group_cats if c in k_shot and k in k_shot[c] and k_shot[c][k]['winner'] == 'CLIP')
        total = len(clip_aucs)
        print(f"  k={k}: Avg CLIP={np.mean(clip_aucs):.1f} DINO={np.mean(dino_aucs):.1f} "
              f"Gain={np.mean(gains):+.1f} CLIP wins={clip_w}/{total}")

# Key cross-analysis: k-shot improvement rate
print()
print("=" * 80)
print("k-shot Improvement Rate: k=1 to k=8")
print("=" * 80)
print(f"{'Category':15s} {'CLIP 1->8':>10s} {'DINO 1->8':>10s} {'Gap 1':>7s} {'Gap 8':>7s} {'Gap change':>10s}")
print("-" * 60)
for cat in sorted(k_shot.keys()):
    if '1' in k_shot[cat] and '8' in k_shot[cat]:
        clip_imp = k_shot[cat]['8']['clip_auroc'] - k_shot[cat]['1']['clip_auroc']
        dino_imp = k_shot[cat]['8']['dino_auroc'] - k_shot[cat]['1']['dino_auroc']
        gap1 = k_shot[cat]['1']['gap']
        gap8 = k_shot[cat]['8']['gap']
        print(f"{cat:15s} {clip_imp:+10.1f} {dino_imp:+10.1f} {gap1:7.1f} {gap8:7.1f} {gap8-gap1:+10.1f}")
