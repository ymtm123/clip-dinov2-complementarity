"""
AnomalyCLIP test script that saves pixel-level anomaly maps to .npy files.
Maps are saved in the same directory structure as AnomalyDINO for easy fusion.

Output structure:
  save_path/anomaly_maps/{object}/test/{anomaly_type}/{image_id}.npy
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
import re
import random
import numpy as np
from utils import get_transform
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_save_maps(args):
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

    maps_dir = os.path.join(args.save_path, "anomaly_maps")
    n_saved = 0

    for idx, items in enumerate(tqdm(test_dataloader, desc="Saving anomaly maps")):
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]
        gt_label = items['anomaly'].detach().cpu().item()
        img_path = items['img_path'][0] if 'img_path' in items else f"sample_{idx}"

        results[cls_name]['gt_sp'].append(gt_label)
        results[cls_name]['img_paths'].append(img_path)

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Image-level score
            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            anomaly_score = text_probs[:, 0, 1].detach().cpu().item()
            results[cls_name]['pr_sp'].append(anomaly_score)

            # Pixel-level anomaly map
            anomaly_map_list = []
            for fidx, patch_feature in enumerate(patch_features):
                if fidx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma))
                                       for i in anomaly_map.detach().cpu()], dim=0)
            anomaly_map_np = anomaly_map[0].numpy()  # [H, W]

            # Extract save path from img_path
            # Pattern: .../test/{anomaly_type}/{image_name}.png
            img_path_norm = img_path.replace('\\', '/')
            match = re.search(r'/test/(.+?)(/[^/]+\.\w+)$', img_path_norm)
            if match:
                anomaly_type = match.group(1)
                img_name = os.path.splitext(os.path.basename(img_path_norm))[0]
            else:
                # Fallback
                anomaly_type = "unknown"
                img_name = f"{idx:04d}"

            save_dir = os.path.join(maps_dir, cls_name, "test", anomaly_type)
            os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(save_dir, f"{img_name}.npy")
            np.save(save_file, anomaly_map_np)
            n_saved += 1

    print(f"\nSaved {n_saved} anomaly maps to {maps_dir}")

    # Print AUROC per category
    print("\n" + "=" * 60)
    print(f"{'Category':<20} {'Image AUROC':>12} {'Samples':>8}")
    print("-" * 45)
    aurocs = []
    for obj in obj_list:
        if len(set(results[obj]['gt_sp'])) > 1:
            auroc = roc_auc_score(results[obj]['gt_sp'], results[obj]['pr_sp']) * 100
            aurocs.append(auroc)
            print(f"{obj:<20} {auroc:>11.1f}% {len(results[obj]['pr_sp']):>8}")
    print("-" * 45)
    if aurocs:
        print(f"{'Mean':<20} {np.mean(aurocs):>11.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP Map Saver")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default='./results/maps_mvtec/')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/9_12_4_multiscale/epoch_15.pth')
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--sigma", type=int, default=4)
    args = parser.parse_args()
    setup_seed(args.seed)
    test_save_maps(args)
