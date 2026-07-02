"""
Inference Speed Measurement for AnomalyCLIP, AnomalyDINO, and Fusion
=====================================================================
Aggregates AnomalyDINO timing from existing CSV files,
runs AnomalyCLIP inference timing benchmark,
and measures fusion overhead.
"""

import os
import csv
import time
import numpy as np
from collections import defaultdict


def aggregate_anomalydino_speed(csv_paths):
    """Aggregate AnomalyDINO inference speed from measurement CSVs.

    Each CSV has columns: Object,Sample,Anomaly_Score,MemoryBank_Time,Inference_Time
    MemoryBank_Time is setup time (one-time per category), Inference_Time is per-image.
    """
    all_times = defaultdict(list)
    membank_times = {}

    for csv_path in csv_paths:
        dataset_name = "unknown"
        if "MVTec" in csv_path:
            dataset_name = "MVTec"
        elif "VisA" in csv_path:
            dataset_name = "VisA"
        elif "BTAD" in csv_path:
            dataset_name = "BTAD"

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj = row['Object']
                inf_time = float(row['Inference_Time'])
                mb_time = float(row['MemoryBank_Time'])
                all_times[(dataset_name, obj)].append(inf_time)
                membank_times[(dataset_name, obj)] = mb_time

    return all_times, membank_times


def measure_anomalyclip_speed():
    """Measure AnomalyCLIP inference speed by running a small benchmark.

    Uses the actual AnomalyCLIP pipeline with PromptLearner.
    """
    import sys
    import torch
    anomalyclip_dir = os.path.join(os.path.dirname(__file__), 'AnomalyCLIP')
    sys.path.insert(0, anomalyclip_dir)
    orig_cwd = os.getcwd()
    os.chdir(anomalyclip_dir)

    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform
    from PIL import Image as PILImage
    import torchvision.transforms as transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model (same as test_save_scores.py)
    AnomalyCLIP_parameters = {
        "Prompt_length": 12,
        "learnabel_text_embedding_depth": 9,
        "learnabel_text_embedding_length": 4,
    }
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    # Load checkpoint
    checkpoint_path = os.path.join(anomalyclip_dir, "checkpoints", "9_12_4_multiscale", "epoch_15.pth")
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    # Prepare text features (one-time cost)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
        text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    text_time = time.perf_counter() - t0
    print(f"  Text encoding time: {text_time*1000:.1f} ms (one-time)")

    # Image transform
    preprocess = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    features_list = [6, 12, 18, 24]

    # Use MVTec categories for benchmarking
    data_root = "data/MVTec_AD"
    benchmark_cats = ['bottle', 'cable', 'capsule']

    timings = {}

    # GPU warmup
    for _ in range(5):
        dummy = torch.randn(1, 3, 518, 518).to(device)
        with torch.no_grad():
            model.encode_image(dummy, features_list, DPAM_layer=20)
    torch.cuda.synchronize()

    for cat in benchmark_cats:
        cat_dir = os.path.join(data_root, cat, "test")
        if not os.path.exists(cat_dir):
            continue

        # Collect test image paths
        test_images = []
        for atype in sorted(os.listdir(cat_dir)):
            atype_dir = os.path.join(cat_dir, atype)
            if not os.path.isdir(atype_dir):
                continue
            for f in sorted(os.listdir(atype_dir)):
                if f.lower().endswith(('.png', '.jpg', '.bmp')):
                    test_images.append(os.path.join(atype_dir, f))

        # Per-image inference time
        image_times = []
        n_images = min(len(test_images), 50)

        for img_path in test_images[:n_images]:
            img = PILImage.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                image_features, patch_features = model.encode_image(img_tensor, features_list, DPAM_layer=20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_probs = image_features @ text_features.permute(0, 2, 1)
                text_probs = (text_probs / 0.07).softmax(-1)
                anomaly_score = text_probs[0, :, 1].max().item()
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            image_times.append(t1 - t0)

        # Drop first measurement (warmup effect)
        if len(image_times) > 5:
            image_times = image_times[1:]

        timings[cat] = {
            'text_encoding_time': text_time,
            'per_image_times': image_times,
            'mean_per_image': np.mean(image_times),
            'std_per_image': np.std(image_times),
            'n_images': len(image_times),
        }

        print(f"  {cat}: {np.mean(image_times)*1000:.1f} +/- {np.std(image_times)*1000:.1f} ms/image "
              f"({len(image_times)} images)")

    # Cleanup
    del model, prompt_learner
    torch.cuda.empty_cache()
    os.chdir(orig_cwd)

    return timings


def measure_fusion_overhead():
    """Measure the overhead of score-level and pixel-level fusion operations."""
    # Score-level fusion overhead
    print("\n--- Fusion Overhead Measurement ---")

    # Simulate typical scenario: 2 model scores to fuse
    n_images = 1000
    clip_scores = np.random.randn(n_images)
    dino_scores = np.random.randn(n_images)

    # Min-max normalization
    t0 = time.perf_counter()
    for _ in range(100):
        clip_norm = (clip_scores - clip_scores.min()) / (clip_scores.max() - clip_scores.min() + 1e-10)
        dino_norm = (dino_scores - dino_scores.min()) / (dino_scores.max() - dino_scores.min() + 1e-10)
        fused = 0.5 * clip_norm + 0.5 * dino_norm
    score_fusion_time = (time.perf_counter() - t0) / 100

    print(f"  Score-level fusion ({n_images} images): {score_fusion_time*1000:.3f} ms")
    print(f"  Per-image score fusion: {score_fusion_time/n_images*1e6:.2f} us")

    # Pixel-level fusion overhead
    # Typical map sizes
    map_size = (256, 256)
    from scipy.ndimage import gaussian_filter

    clip_map = np.random.randn(*map_size).astype(np.float32)
    dino_map_small = np.random.randn(32, 32).astype(np.float32)

    import cv2

    # Resize + gaussian + normalize + fuse
    t0 = time.perf_counter()
    n_iter = 100
    for _ in range(n_iter):
        # Resize DINO map
        dino_resized = cv2.resize(dino_map_small, (map_size[1], map_size[0]), interpolation=cv2.INTER_LINEAR)
        # Gaussian smooth
        dino_smooth = gaussian_filter(dino_resized, sigma=4)
        # Normalize both
        c_norm = (clip_map - clip_map.min()) / (clip_map.max() - clip_map.min() + 1e-10)
        d_norm = (dino_smooth - dino_smooth.min()) / (dino_smooth.max() - dino_smooth.min() + 1e-10)
        # Fuse
        fused = 0.5 * c_norm + 0.5 * d_norm
    pixel_fusion_time = (time.perf_counter() - t0) / n_iter

    print(f"  Pixel-level fusion (256x256 per image): {pixel_fusion_time*1000:.3f} ms")

    # AdaFuse signal computation overhead (per category)
    n_scores = 200  # typical test set size per category
    scores_a = np.random.randn(n_scores)
    scores_b = np.random.randn(n_scores)

    from scipy.stats import entropy as scipy_entropy

    t0 = time.perf_counter()
    for _ in range(1000):
        # Compute distribution signals
        s_a = (scores_a - scores_a.min()) / (scores_a.max() - scores_a.min() + 1e-10)
        s_b = (scores_b - scores_b.min()) / (scores_b.max() - scores_b.min() + 1e-10)

        # Signal 1: Score range
        range_a = s_a.max() - s_a.min()
        range_b = s_b.max() - s_b.min()

        # Signal 2: Bimodality coefficient
        from scipy.stats import skew, kurtosis
        bc_a = (skew(s_a)**2 + 1) / (kurtosis(s_a, fisher=False) + 3)
        bc_b = (skew(s_b)**2 + 1) / (kurtosis(s_b, fisher=False) + 3)

        # Signal 3: Entropy of histogram
        hist_a, _ = np.histogram(s_a, bins=20, density=True)
        hist_b, _ = np.histogram(s_b, bins=20, density=True)
        ent_a = scipy_entropy(hist_a + 1e-10)
        ent_b = scipy_entropy(hist_b + 1e-10)

    signal_time = (time.perf_counter() - t0) / 1000

    print(f"  AdaFuse signal computation (per category, {n_scores} images): {signal_time*1000:.3f} ms")

    return {
        'score_fusion_per_batch_ms': score_fusion_time * 1000,
        'score_fusion_per_image_us': score_fusion_time / n_images * 1e6,
        'pixel_fusion_per_image_ms': pixel_fusion_time * 1000,
        'adafuse_signal_per_category_ms': signal_time * 1000,
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 80)
    print("  INFERENCE SPEED MEASUREMENT")
    print("=" * 80)

    # 1. Aggregate AnomalyDINO speed from existing CSVs
    print("\n--- AnomalyDINO Inference Speed (from CSVs) ---")
    dino_csvs = [
        # MVTec: weak_cats + strong_cats = full 15 cats (NOT the buggy 8-cat 'agnostic' file)
        os.path.join(base_dir, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_weak_cats", "measurements_seed=0.csv"),
        os.path.join(base_dir, "AnomalyDINO", "results_MVTec", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_strong_cats", "measurements_seed=0.csv"),
        os.path.join(base_dir, "AnomalyDINO", "results_VisA", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_all", "measurements_seed=0.csv"),
        os.path.join(base_dir, "AnomalyDINO", "results_BTAD", "dinov2_vitl14_448",
                     "1-shot_preprocess=agnostic_btad_all", "measurements_seed=0.csv"),
    ]
    existing_csvs = [p for p in dino_csvs if os.path.exists(p)]

    dino_times, dino_membank = aggregate_anomalydino_speed(existing_csvs)

    print(f"\n  {'Dataset':<10} {'Category':<15} {'Mean(ms)':<10} {'Std(ms)':<10} {'MemBank(ms)':<12} {'N':<6}")
    print("  " + "-" * 65)

    all_dino_inf_times = []
    all_dino_membank_times = []

    for (ds, obj), times in sorted(dino_times.items()):
        mean_t = np.mean(times) * 1000
        std_t = np.std(times) * 1000
        mb_t = dino_membank[(ds, obj)] * 1000
        all_dino_inf_times.extend(times)
        all_dino_membank_times.append(dino_membank[(ds, obj)])
        print(f"  {ds:<10} {obj:<15} {mean_t:<10.1f} {std_t:<10.1f} {mb_t:<12.1f} {len(times):<6}")

    dino_mean_inf = np.mean(all_dino_inf_times) * 1000
    dino_std_inf = np.std(all_dino_inf_times) * 1000
    dino_mean_mb = np.mean(all_dino_membank_times) * 1000
    print(f"\n  Overall AnomalyDINO: {dino_mean_inf:.1f} +/- {dino_std_inf:.1f} ms/image")
    print(f"  Memory bank setup: {dino_mean_mb:.1f} ms/category (one-time)")
    print(f"  Throughput: {1000/dino_mean_inf:.1f} images/sec")

    # 2. Measure AnomalyCLIP inference speed
    print("\n\n--- AnomalyCLIP Inference Speed (GPU Benchmark) ---")
    try:
        clip_timings = measure_anomalyclip_speed()

        all_clip_times = []
        for cat, data in clip_timings.items():
            all_clip_times.extend(data['per_image_times'])

        clip_mean = np.mean(all_clip_times) * 1000
        clip_std = np.std(all_clip_times) * 1000
        print(f"\n  Overall AnomalyCLIP: {clip_mean:.1f} +/- {clip_std:.1f} ms/image")
        print(f"  Throughput: {1000/clip_mean:.1f} images/sec")
    except Exception as e:
        print(f"  [ERROR] Could not measure AnomalyCLIP speed: {e}")
        clip_timings = None
        clip_mean = None
        clip_std = None

    # 3. Measure fusion overhead
    print("\n")
    fusion_overhead = measure_fusion_overhead()

    # 4. Summary
    print("\n\n" + "=" * 80)
    print("  SPEED COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n  {'Component':<35} {'Time (ms)':<15} {'Throughput':<15}")
    print("  " + "-" * 65)
    print(f"  {'AnomalyDINO (per image)':<35} {dino_mean_inf:<15.1f} {1000/dino_mean_inf:<15.1f} img/s")
    if clip_mean:
        print(f"  {'AnomalyCLIP (per image)':<35} {clip_mean:<15.1f} {1000/clip_mean:<15.1f} img/s")
    print(f"  {'Score fusion overhead':<35} {fusion_overhead['score_fusion_per_image_us']/1000:<15.4f} negligible")
    print(f"  {'Pixel fusion overhead (per image)':<35} {fusion_overhead['pixel_fusion_per_image_ms']:<15.3f}")
    print(f"  {'AdaFuse signal (per category)':<35} {fusion_overhead['adafuse_signal_per_category_ms']:<15.3f}")

    if clip_mean:
        total_sequential = dino_mean_inf + clip_mean
        total_parallel = max(dino_mean_inf, clip_mean)
        print(f"\n  Total (sequential): {total_sequential:.1f} ms/image ({1000/total_sequential:.1f} img/s)")
        print(f"  Total (parallel):   {total_parallel:.1f} ms/image ({1000/total_parallel:.1f} img/s)")
        print(f"  Fusion overhead:    <{fusion_overhead['pixel_fusion_per_image_ms']:.1f} ms (negligible)")

    # Save results
    import json
    speed_results = {
        'anomalydino': {
            'mean_ms': round(dino_mean_inf, 2),
            'std_ms': round(dino_std_inf, 2),
            'throughput_img_s': round(1000/dino_mean_inf, 1),
            'membank_setup_ms': round(dino_mean_mb, 2),
            'n_samples': len(all_dino_inf_times),
        },
        'fusion_overhead': {
            'score_fusion_per_image_us': round(fusion_overhead['score_fusion_per_image_us'], 2),
            'pixel_fusion_per_image_ms': round(fusion_overhead['pixel_fusion_per_image_ms'], 3),
            'adafuse_signal_per_category_ms': round(fusion_overhead['adafuse_signal_per_category_ms'], 3),
        }
    }

    if clip_timings:
        speed_results['anomalyclip'] = {
            'mean_ms': round(clip_mean, 2),
            'std_ms': round(clip_std, 2),
            'throughput_img_s': round(1000/clip_mean, 1),
            'n_samples': len(all_clip_times),
            'per_category': {cat: {
                'mean_ms': round(data['mean_per_image'] * 1000, 2),
                'text_encoding_ms': round(data['text_encoding_time'] * 1000, 2),
            } for cat, data in clip_timings.items()},
        }
        speed_results['total'] = {
            'sequential_ms': round(total_sequential, 2),
            'parallel_ms': round(total_parallel, 2),
            'fusion_overhead_ms': round(fusion_overhead['pixel_fusion_per_image_ms'], 3),
        }

    output_path = os.path.join(base_dir, "inference_speed_results.json")
    with open(output_path, 'w') as f:
        json.dump(speed_results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
