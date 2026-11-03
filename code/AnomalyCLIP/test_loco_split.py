"""Test AnomalyCLIP on LOCO AD with separate evaluation for structural/logical anomalies."""
import AnomalyCLIP_lib
import torch
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import normalize, get_transform
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm

import os
import random
import json
import numpy as np
from tabulate import tabulate
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_loco_split(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    os.makedirs(save_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)

    # Load meta.json to know which images are logical vs structural
    with open(os.path.join(args.data_path, 'meta.json'), 'r') as f:
        meta = json.load(f)

    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    # Build results with anomaly type tracking
    results = {}
    for obj in obj_list:
        results[obj] = {
            'all': {'gt': [], 'pr': []},
            'structural': {'gt': [], 'pr': []},
            'logical': {'gt': [], 'pr': []},
        }

    sample_rows = []  # per-sample CLIP scores for score-level fusion

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path, weights_only=False)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]
        anomaly = items['anomaly'].item()
        img_path = items['img_path'][0]

        # Determine anomaly type from path
        if 'structural_anomalies' in img_path:
            anom_type = 'structural'
        elif 'logical_anomalies' in img_path:
            anom_type = 'logical'
        else:
            anom_type = 'good'

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            score = text_probs[:, 0, 1].item()

        # per-sample row (relative path subdir/filename, e.g. logical_anomalies/000.png)
        rel = img_path.replace('\\', '/').split('/test/')[-1]
        sample_rows.append([cls_name, rel, f"{score:.6f}", anomaly, anom_type])

        # All: good vs all anomalies
        results[cls_name]['all']['gt'].append(anomaly)
        results[cls_name]['all']['pr'].append(score)

        # Structural: good vs structural only
        if anom_type in ('good', 'structural'):
            results[cls_name]['structural']['gt'].append(anomaly)
            results[cls_name]['structural']['pr'].append(score)

        # Logical: good vs logical only
        if anom_type in ('good', 'logical'):
            results[cls_name]['logical']['gt'].append(anomaly)
            results[cls_name]['logical']['pr'].append(score)

    # Compute metrics
    print("\n" + "=" * 80)
    print("AnomalyCLIP on LOCO AD - Split Results (Image-level AUROC)")
    print("=" * 80)

    table_rows = []
    all_aurocs = {'all': [], 'structural': [], 'logical': []}

    for obj in obj_list:
        row = [obj]
        for atype in ['all', 'structural', 'logical']:
            gt = np.array(results[obj][atype]['gt'])
            pr = np.array(results[obj][atype]['pr'])
            if len(np.unique(gt)) < 2:
                auroc = float('nan')
            else:
                auroc = roc_auc_score(gt, pr)
            row.append(f"{auroc * 100:.1f}" if not np.isnan(auroc) else "N/A")
            if not np.isnan(auroc):
                all_aurocs[atype].append(auroc)
        table_rows.append(row)

    # Add mean row
    mean_row = ['mean']
    for atype in ['all', 'structural', 'logical']:
        if all_aurocs[atype]:
            mean_row.append(f"{np.mean(all_aurocs[atype]) * 100:.1f}")
        else:
            mean_row.append("N/A")
    table_rows.append(mean_row)

    headers = ['Category', 'All AUROC', 'Structural AUROC', 'Logical AUROC']
    print(tabulate(table_rows, headers=headers, tablefmt="pipe"))

    # Save results
    with open(os.path.join(save_path, 'loco_split_results.txt'), 'w') as f:
        f.write(tabulate(table_rows, headers=headers, tablefmt="pipe"))
        f.write("\n")

    print(f"\nResults saved to {save_path}/loco_split_results.txt")

    # per-sample scores for score-level fusion
    import csv as _csv
    with open(os.path.join(save_path, 'loco_clip_scores.csv'), 'w', newline='') as f:
        w = _csv.writer(f)
        w.writerow(["Object", "Sample", "Score", "GT", "AnomType"])
        w.writerows(sample_rows)
    print(f"Per-sample CLIP scores saved to {save_path}/loco_clip_scores.csv ({len(sample_rows)} rows)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default='./results/loco_split')
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='loco')
    parser.add_argument("--features_list", type=int, nargs="+", default=[24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--sigma", type=int, default=4)
    args = parser.parse_args()
    setup_seed(args.seed)
    test_loco_split(args)
