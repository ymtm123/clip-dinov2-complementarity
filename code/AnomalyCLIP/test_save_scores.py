"""
Modified AnomalyCLIP test script that saves raw per-image anomaly scores to CSV.
This is needed for score-level fusion with AnomalyDINO.
"""
import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import normalize
from dataset import Dataset
from tqdm import tqdm
import csv
import os
import random
import numpy as np
from utils import get_transform
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_save_scores(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {}
    for obj in obj_list:
        results[obj] = {'gt_sp': [], 'pr_sp': [], 'img_paths': []}

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for idx, items in enumerate(tqdm(test_dataloader, desc="Running AnomalyCLIP")):
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]
        gt_label = items['anomaly'].detach().cpu().item()

        # Get image path
        img_path = items['img_path'][0] if 'img_path' in items else f"sample_{idx}"

        results[cls_name]['gt_sp'].append(gt_label)
        results[cls_name]['img_paths'].append(img_path)

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            anomaly_score = text_probs[:, 0, 1].detach().cpu().item()
            results[cls_name]['pr_sp'].append(anomaly_score)

    # Save scores to CSV
    os.makedirs(args.save_path, exist_ok=True)
    output_file = os.path.join(args.save_path, "anomalyclip_scores.csv")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Object", "ImagePath", "AnomalyScore", "GroundTruth"])

        for obj in obj_list:
            for i in range(len(results[obj]['pr_sp'])):
                img_path = results[obj]['img_paths'][i]
                score = results[obj]['pr_sp'][i]
                gt = results[obj]['gt_sp'][i]
                writer.writerow([obj, img_path, f"{score:.6f}", int(gt)])

    print(f"\nScores saved to {output_file}")

    # Also print AUROC per category
    print("\n" + "="*60)
    print(f"{'Category':<20} {'AUROC':>10} {'AP':>10} {'Samples':>8}")
    print("-"*50)
    aurocs = []
    for obj in obj_list:
        if len(set(results[obj]['gt_sp'])) > 1:
            auroc = roc_auc_score(results[obj]['gt_sp'], results[obj]['pr_sp']) * 100
            ap = average_precision_score(results[obj]['gt_sp'], results[obj]['pr_sp']) * 100
            aurocs.append(auroc)
            print(f"{obj:<20} {auroc:>9.1f}% {ap:>9.1f}% {len(results[obj]['pr_sp']):>8}")
    print("-"*50)
    print(f"{'Mean':<20} {np.mean(aurocs):>9.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP Score Saver")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default='./results/scores/')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/9_12_4_multiscale/epoch_15.pth')
    parser.add_argument("--dataset", type=str, default='mvtec')
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
    test_save_scores(args)
